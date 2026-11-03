"""Dense-update structure, causal information boundary and native integration."""
import pytest
import torch

from models.layers import CastedLinear
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1
from models.recursive_reasoning.trm_dense import (DensePlasticTRM, DenseTemporaryProjection,
    DenseWeightGenerator, native_a_from_checkpoint)


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config():
    return dict(batch_size=2, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
        puzzle_emb_ndim=32, puzzle_emb_len=1, H_cycles=3, L_cycles=2, H_layers=0,
        L_layers=2, hidden_size=32, expansion=2., num_heads=4, pos_encodings="rope",
        halt_max_steps=4, halt_exploration_prob=.1, forward_dtype="float32",
        no_ACT_continue=True, dense_controller_width=32, dense_controller_heads=4,
        dense_controller_layers=1, dense_tile_size=8, dense_chunk_tiles=17,
        dense_decoder_width=16, dense_relative_cap=.03)


def data():
    return dict(inputs=torch.tensor([[1, 2, 3, 1, 5, 6, 1, 8, 9] * 9,
                                    [2, 1, 4, 5, 1, 7, 8, 1, 10] * 9]),
        labels=torch.arange(81)[None].expand(2, -1) % 9 + 2,
        puzzle_identifiers=torch.zeros(2, dtype=torch.long))


def model():
    torch.manual_seed(17)
    return DensePlasticTRM(config()).eval()


def test_directly_generated_matrices_can_have_full_rank_and_are_norm_bounded():
    torch.manual_seed(8)
    weights = [torch.randn(17, 19), torch.randn(32, 64)]
    generator = DenseWeightGenerator([tuple(w.shape) for w in weights], width=16,
        tile_size=8, chunk_tiles=5, decoder_width=8, relative_cap=.03)
    matrices = generator(torch.randn(2, 16), weights)
    for matrix, weight in zip(matrices, weights):
        assert matrix.shape == (2, *weight.shape)
        assert torch.linalg.matrix_rank(matrix[0]) == min(weight.shape)
        assert torch.all(matrix.flatten(1).norm(dim=1) <= .03 * weight.norm() + 1e-7)
    assert not torch.equal(matrices[0][0], matrices[0][1])


def test_dense_projection_matches_explicit_matrix_update_and_zero_recovers_base():
    torch.manual_seed(2)
    projection = DenseTemporaryProjection(CastedLinear(7, 5, bias=True))
    inputs = torch.randn(2, 3, 7)
    base = projection(inputs)
    projection.delta = torch.zeros(2, 5, 7)
    assert torch.equal(base, projection(inputs))
    projection.delta = torch.randn(2, 5, 7) * .01
    expected = torch.stack([torch.nn.functional.linear(inputs[i],
        projection.base.weight + projection.delta[i], projection.base.bias) for i in range(2)])
    torch.testing.assert_close(projection(inputs), expected)


def test_disabled_dense_model_is_bitwise_native_and_has_no_factor_parameters():
    dense, batch = model(), data()
    native = TinyRecursiveReasoningModel_ACTV1(config()).eval()
    native.load_state_dict(dense.a_state(), strict=True)
    dense.set_mode("off")
    with torch.no_grad():
        carry_dense, carry_native = dense.initial_carry(batch), native.initial_carry(batch)
        for _ in range(4):
            carry_dense, output_dense = dense(carry_dense, batch)
            carry_native, output_native = native(carry_native, batch)
            assert torch.equal(output_dense["logits"], output_native["logits"])
    assert all(not name.endswith((".u", ".v")) for name, _ in dense.named_parameters())
    assert all(port.delta is None for port in dense.ports)


@pytest.mark.parametrize("mode", ["dynamic", "phase", "shuffle", "reset_memory"])
def test_b_gradients_flow_with_full_a_frozen_and_query_labels_cannot_affect_actions(mode):
    dense, batch = model().set_mode(mode), data()
    before = dense.a_state()
    table_before = dense.puzzle_emb.local_weights.detach().clone()
    optimizer = torch.optim.Adam(dense.b_parameters, lr=1e-3)
    _, output = dense(dense.initial_carry(batch), batch)
    output["logits"].square().mean().backward()
    for component in (dense.generator, dense.controller):
        assert sum(p.grad.abs().sum() for p in component.parameters() if p.grad is not None) > 0
    assert all(p.grad is None and not p.requires_grad for p in dense.inner.parameters())
    changed_labels = dict(batch, labels=batch["labels"].flip(1))
    with torch.no_grad():
        # Keep the execution context identical: PyTorch uses different encoder
        # kernels with/without autograd, which can change floating-point bits.
        dense(dense.initial_carry(batch), batch)
        action = [value.clone() for value in dense.last_action]
        dense(dense.initial_carry(changed_labels), changed_labels)
    assert all(torch.equal(a, b) for a, b in zip(action, dense.last_action))
    optimizer.step()
    assert all(torch.equal(value, dense.a_state()[key]) for key, value in before.items())
    assert torch.equal(table_before, dense.puzzle_emb.local_weights)


def test_native_timing_and_constant_late_matrix_with_cleanup():
    dense, batch = DensePlasticTRM(dict(config(), weight_update_cadence="per_act")).eval(), data()
    seen = []
    handle = dense.ports[0].register_forward_pre_hook(lambda module, args: seen.append(
        None if module.delta is None else module.delta.detach().clone()))
    with torch.no_grad():
        dense(dense.initial_carry(batch), batch)
    handle.remove()
    assert len(seen) == 9
    assert all(value is None for value in seen[:6])
    assert all(torch.equal(value, seen[6]) for value in seen[6:])
    assert all(port.delta is None for port in dense.ports)
    assert dense.weight_updates_this_act == 1


def test_batch_isolation_and_fresh_puzzle_memory_reset():
    dense, batch = model(), data()
    with torch.no_grad():
        carry, first = dense(dense.initial_carry(batch), batch)
        first_action = [v.clone() for v in dense.last_action]
        dense(carry, batch)
        assert any(not torch.equal(a, b) for a, b in zip(first_action, dense.last_action))
        dense(dense.initial_carry(batch), batch)
        assert all(torch.equal(a, b) for a, b in zip(first_action, dense.last_action))
        changed = {k: v.clone() for k, v in batch.items()}
        changed["inputs"][1].fill_(1)
        _, second = dense(dense.initial_carry(changed), changed)
    assert torch.equal(first["logits"][0], second["logits"][0])
    assert all(torch.equal(a[0], b[0]) for a, b in zip(first_action, dense.last_action))


def test_joint_mode_has_a_and_b_gradients():
    dense, batch = model().configure_training("joint"), data()
    _, output = dense(dense.initial_carry(batch), batch)
    output["logits"].square().mean().backward()
    assert dense.ports[0].base.weight.grad.abs().sum() > 0
    assert sum(p.grad.abs().sum() for p in dense.b_parameters if p.grad is not None) > 0


def test_legacy_checkpoint_conversion_drops_factors_and_load_is_atomic():
    dense = model()
    original = dense.a_state()
    legacy = {k.replace(".mlp.down_proj.weight", ".mlp.down_proj.base.weight"): v.clone()
              for k, v in original.items()}
    legacy.update({"controller.gates.weight": torch.ones(4),
        "inner.L_level.layers.0.mlp.down_proj.u": torch.ones(32, 4),
        "inner.L_level.layers.0.mlp.down_proj.v": torch.ones(256, 4)})
    converted = native_a_from_checkpoint(legacy)
    assert set(converted) == set(original)
    dense.load_a_state(converted)
    bad = dict(converted, **{"inner.lm_head.weight": torch.zeros(1)})
    with pytest.raises(ValueError):
        dense.load_a_state(bad)
    assert all(torch.equal(v, dense.a_state()[k]) for k, v in original.items())


def test_projection_cleanup_if_a_forward_fails():
    dense, batch = model(), data()
    def fail_when_enabled(module, args):
        if module.delta is not None:
            raise RuntimeError("injected failure")
    handle = dense.ports[0].register_forward_pre_hook(fail_when_enabled)
    with pytest.raises(RuntimeError, match="injected failure"):
        dense(dense.initial_carry(batch), batch)
    handle.remove()
    assert all(port.delta is None for port in dense.ports)


@pytest.mark.parametrize("kwargs", [{"tile_size": 0}, {"chunk_tiles": -1},
    {"relative_cap": float("nan")}, {"relative_cap": -.1}])
def test_invalid_generator_settings_fail_before_running(kwargs):
    with pytest.raises(ValueError):
        DenseWeightGenerator([(4, 4)], **kwargs)


@pytest.mark.parametrize("small_chunk", [1, 7])
def test_larger_generation_chunks_preserve_outputs_and_answer_gradients(small_chunk):
    torch.manual_seed(31)
    weights = [torch.randn(17, 19), torch.randn(32, 64)]
    kwargs = dict(shapes=[tuple(w.shape) for w in weights], width=16,
                  tile_size=8, decoder_width=8, relative_cap=.03)
    reference = DenseWeightGenerator(**kwargs, chunk_tiles=small_chunk)
    combined = DenseWeightGenerator(**kwargs, chunk_tiles=1000)
    combined.load_state_dict(reference.state_dict(), strict=True)
    first = torch.randn(2, 16, requires_grad=True)
    second = first.detach().clone().requires_grad_(True)
    a, b = reference(first, weights), combined(second, weights)
    for expected, actual in zip(a, b):
        torch.testing.assert_close(expected, actual, rtol=1e-5, atol=1e-8)
    sum(value.square().sum() for value in a).backward()
    sum(value.square().sum() for value in b).backward()
    torch.testing.assert_close(first.grad, second.grad, rtol=1e-5, atol=1e-8)
    for expected, actual in zip(reference.parameters(), combined.parameters()):
        torch.testing.assert_close(expected.grad, actual.grad, rtol=1e-5, atol=1e-8)
