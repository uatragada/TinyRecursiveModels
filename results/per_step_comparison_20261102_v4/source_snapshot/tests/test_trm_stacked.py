"""C->B->A gradients, intervention identity, causal ordering and isolation."""
import pytest
import torch

from models.recursive_reasoning.trm_dense import DensePlasticTRM
from models.recursive_reasoning.trm_stacked import StackedDenseTRM


@pytest.fixture(autouse=True)
def single_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config(cross=False):
    return dict(batch_size=2, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
        puzzle_emb_ndim=32, puzzle_emb_len=1, H_cycles=3, L_cycles=2, H_layers=0,
        L_layers=2, hidden_size=32, expansion=2., num_heads=4, pos_encodings="rope",
        halt_max_steps=4, halt_exploration_prob=.1, forward_dtype="float32",
        no_ACT_continue=True, dense_controller_width=32, dense_controller_heads=4,
        dense_controller_layers=1, dense_tile_size=8, dense_chunk_tiles=17,
        dense_decoder_width=16, dense_relative_cap=.03, meta_controller_width=16,
        meta_controller_heads=4, meta_controller_layers=1, meta_history=3,
        meta_tile_size=8, meta_chunk_tiles=17, meta_decoder_width=8, meta_relative_cap=.1,
        meta_cross_attention=cross, meta_cross_heads=4, meta_read_queries=4)


def data():
    return dict(inputs=torch.tensor([[1, 2, 3, 1, 5, 6, 1, 8, 9] * 9,
                                    [2, 1, 4, 5, 1, 7, 8, 1, 10] * 9]),
        labels=torch.arange(81)[None].expand(2, -1) % 9 + 2,
        puzzle_identifiers=torch.zeros(2, dtype=torch.long))


def model(cross=False):
    torch.manual_seed(17)
    return StackedDenseTRM(config(cross)).eval()


@pytest.mark.parametrize("cross", [False, True])
def test_c_off_is_bitwise_equal_to_the_original_dense_ab_model(cross):
    stacked, batch = model(cross).set_meta_mode("off"), data()
    reference = DensePlasticTRM(config(cross)).eval()
    reference.load_state_dict(stacked.dense_state(), strict=True)
    with torch.no_grad():
        a, b = stacked.initial_carry(batch), reference.initial_carry(batch)
        for _ in range(4):
            a, actual = stacked(a, batch)
            b, expected = reference(b, batch)
            assert torch.equal(actual["logits"], expected["logits"])
    assert stacked.last_c_action is None
    assert stacked.b_write_port.delta is None
    assert all(not name.endswith((".u", ".v")) for name, _ in stacked.named_parameters())


def test_c_changes_b_weights_before_b_generates_a_weights():
    stacked, batch = model(), data()
    original = stacked.b_write_port.base.weight.detach().clone()
    seen = []
    handle = stacked.b_write_port.register_forward_pre_hook(lambda module, args: seen.append(
        None if module.delta is None else module.delta.detach().clone()))
    with torch.no_grad():
        stacked(stacked.initial_carry(batch), batch)
        changed = [value.clone() for value in stacked.last_action]
        c_action = stacked.last_c_action.clone()
        stacked.set_meta_mode("off")(stacked.initial_carry(batch), batch)
    handle.remove()
    assert seen and all(value is not None for value in seen[:len(seen) // 2])
    assert all(value is None for value in seen[len(seen) // 2:])
    assert any(not torch.equal(first, second) for first, second in zip(changed, stacked.last_action))
    assert torch.equal(original, stacked.b_write_port.base.weight)
    assert c_action.shape == (2, *original.shape)
    assert torch.linalg.matrix_rank(c_action[0]) == min(original.shape)
    assert torch.all(c_action.flatten(1).norm(dim=1) <= .1 * original.norm() + 1e-7)
    assert stacked.b_write_port.delta is None


@pytest.mark.parametrize("train_b,train_c,train_a", [(True, True, False),
    (False, True, False), (True, True, True)])
def test_loss_gradients_reach_c_through_b_and_optional_joint_modes(train_b, train_c, train_a):
    stacked, batch = model().configure_hierarchy(train_b, train_c, train_a), data()
    before = stacked.a_state()
    b_before = stacked.b_state()
    optimizer = torch.optim.Adam([p for p in stacked.parameters() if p.requires_grad], lr=1e-3)
    _, outputs = stacked(stacked.initial_carry(batch), batch)
    outputs["logits"].square().mean().backward()
    for component in (stacked.meta_controller, stacked.meta_generator):
        assert sum(p.grad.abs().sum() for p in component.parameters() if p.grad is not None) > 0
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in stacked.b_parameters) == train_b
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in stacked.inner.parameters()) == train_a
    optimizer.step()
    if not train_a:
        assert all(torch.equal(v, stacked.a_state()[k]) for k, v in before.items())
    if not train_b:
        assert all(torch.equal(v, stacked.b_state()[k]) for k, v in b_before.items())


def test_query_labels_do_not_change_b_or_c_interventions():
    stacked, batch = model(), data()
    with torch.no_grad():
        stacked(stacked.initial_carry(batch), batch)
        a, c = [v.clone() for v in stacked.last_action], stacked.last_c_action.clone()
        changed = dict(batch, labels=batch["labels"].flip(1))
        stacked(stacked.initial_carry(changed), changed)
    assert torch.equal(c, stacked.last_c_action)
    assert all(torch.equal(first, second) for first, second in zip(a, stacked.last_action))


def test_per_puzzle_c_memory_reset_and_cross_batch_isolation():
    stacked, batch = model(), data()
    with torch.no_grad():
        carry, first = stacked(stacked.initial_carry(batch), batch)
        first_c, first_a = stacked.last_c_action.clone(), [v.clone() for v in stacked.last_action]
        stacked(carry, batch)
        assert not torch.equal(first_c, stacked.last_c_action)
        stacked(stacked.initial_carry(batch), batch)
        assert torch.equal(first_c, stacked.last_c_action)
        changed = {k: v.clone() for k, v in batch.items()}
        changed["inputs"][1].fill_(1)
        _, second = stacked(stacked.initial_carry(changed), changed)
    assert torch.equal(first["logits"][0], second["logits"][0])
    assert torch.equal(first_c[0], stacked.last_c_action[0])
    assert all(torch.equal(a[0], b[0]) for a, b in zip(first_a, stacked.last_action))


@pytest.mark.parametrize("cross", [False, True])
def test_partial_puzzle_reset_matches_a_fresh_slot_without_resetting_other_slots(cross):
    stacked, batch = model(cross), data()
    with torch.no_grad():
        carry, _ = stacked(stacked.initial_carry(batch), batch)
        carry.halted[0] = True
        stacked(carry, batch)
        mixed = stacked.last_c_action.clone()
        stacked(stacked.initial_carry(batch), batch)
    assert torch.equal(mixed[0], stacked.last_c_action[0])
    assert not torch.equal(mixed[1], stacked.last_c_action[1])


def test_c_cleanup_on_failure_inside_b_generator():
    stacked, batch = model(), data()
    def fail(module, args):
        raise RuntimeError("injected B failure")
    handle = stacked.b_write_port.register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="injected B failure"):
        stacked(stacked.initial_carry(batch), batch)
    handle.remove()
    assert stacked.b_write_port.delta is None
    assert all(port.delta is None for port in stacked.ports)


def test_dense_ab_loading_is_exact_and_rejects_invalid_state_atomically():
    stacked = model()
    torch.manual_seed(9)
    original = DensePlasticTRM(config()).eval()
    stacked.load_dense_state(original.state_dict())
    assert all(torch.equal(v, stacked.dense_state()[k]) for k, v in original.state_dict().items())
    bad = dict(original.state_dict())
    bad["generator.decoder.2.weight"] = torch.zeros(1)
    with pytest.raises(ValueError):
        stacked.load_dense_state(bad)
    assert all(torch.equal(v, stacked.dense_state()[k]) for k, v in original.state_dict().items())


@pytest.mark.parametrize("mode", ["phase", "shuffle", "reset_memory"])
def test_c_ablation_modes_are_finite_and_cleanup(mode):
    stacked, batch = model().set_meta_mode(mode), data()
    with torch.no_grad():
        stacked(stacked.initial_carry(batch), batch)
    assert torch.isfinite(stacked.last_c_action).all()
    assert stacked.b_write_port.delta is None
    assert all(port.delta is None for port in stacked.ports)


@pytest.mark.parametrize("grad_enabled", [False, True])
def test_exposing_b_layer_tokens_preserves_native_encoder_computation(grad_enabled):
    stacked = model(True)
    sequence = torch.randn(2, 4, 32)
    with torch.set_grad_enabled(grad_enabled):
        expected = stacked.controller.transformer(sequence)
        actual, trace = stacked.controller.encode_with_trace(sequence)
    assert torch.equal(expected, actual)
    assert len(trace) == 2 and all(not token.requires_grad for token in trace)


def test_c_cross_attention_reads_all_b_stages_with_separate_head_weights():
    stacked, batch = model(True), data()
    with torch.no_grad():
        stacked(stacked.initial_carry(batch), batch)
    tokens = stacked.controller.reasoning_tokens
    weights = stacked.meta_controller.last_attention
    assert tokens.shape == (2, 10, 32)  # input + one block, four slots each, memory + readout
    assert weights.shape == (2, 4, 4, 10)
    torch.testing.assert_close(weights.sum(-1), torch.ones(2, 4, 4))
    assert torch.isfinite(weights).all() and (weights >= 0).all()
    assert torch.equal(stacked.controller.reasoning_stage_ids, torch.tensor([0]*4 + [1]*4 + [2, 3]))


def test_answer_loss_trains_inter_model_attention_through_c_to_b_to_a():
    stacked, batch = model(True), data()
    _, output = stacked(stacked.initial_carry(batch), batch)
    output["logits"].square().mean().backward()
    for parameter in (stacked.meta_controller.cross_read.in_proj_weight,
                      stacked.meta_controller.cross_read.out_proj.weight,
                      stacked.meta_controller.source_projection.weight,
                      stacked.meta_controller.read_queries):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in stacked.inner.parameters())


def test_different_b_parts_change_c_even_with_unchanged_pooled_summary():
    stacked, batch = model(True), data()
    with torch.no_grad():
        stacked(stacked.initial_carry(batch), batch)
        tokens = stacked.controller.reasoning_tokens.clone()
        changed = tokens.clone()
        adjustment = torch.randn_like(tokens[:, 0]) * 3
        changed[:, 0] += adjustment
        changed[:, 4] -= adjustment
        torch.testing.assert_close(tokens.mean(1), changed.mean(1), atol=1e-6, rtol=1e-5)
        def read(values, mode):
            stacked.meta_controller.reset(torch.ones(2, dtype=torch.bool))
            return stacked.meta_controller(stacked.last_b_context, torch.zeros(2, dtype=torch.long),
                tokens=values, stage_ids=stacked.controller.reasoning_stage_ids,
                slot_ids=stacked.controller.reasoning_slot_ids, read_mode=mode).clone()
        first, second = read(tokens, "tokens"), read(changed, "tokens")
        assert (first - second).abs().max() > 1e-5
        torch.testing.assert_close(read(tokens, "pooled"), read(changed, "pooled"), atol=1e-6, rtol=1e-5)


def test_cross_reads_are_label_free_and_isolated_between_puzzles():
    stacked, batch = model(True), data()
    with torch.no_grad():
        _, initial = stacked(stacked.initial_carry(batch), batch)
        weights, action = stacked.meta_controller.last_attention.clone(), stacked.last_c_action.clone()
        changed = dict(batch, labels=batch["labels"].flip(1))
        stacked(stacked.initial_carry(changed), changed)
        assert torch.equal(weights, stacked.meta_controller.last_attention)
        assert torch.equal(action, stacked.last_c_action)
        changed = {key: value.clone() for key, value in batch.items()}
        changed["inputs"][1].fill_(1)
        _, output = stacked(stacked.initial_carry(changed), changed)
    assert torch.equal(weights[0], stacked.meta_controller.last_attention[0])
    assert torch.equal(action[0], stacked.last_c_action[0])
    assert torch.equal(initial["logits"][0], output["logits"][0])


def test_cross_reader_accepts_b_phase_control_without_fabricated_layer_states():
    stacked, batch = model(True).set_mode("phase"), data()
    with torch.no_grad():
        stacked(stacked.initial_carry(batch), batch)
    assert stacked.controller.reasoning_tokens is None
    assert stacked.meta_controller.last_attention.shape == (2, 4, 4, 1)
    assert stacked.b_write_port.delta is None
