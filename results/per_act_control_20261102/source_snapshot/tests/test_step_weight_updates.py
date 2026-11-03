"""Every recurrent call uses a new causal C -> B -> A intervention."""
import pytest
import torch

from models.recursive_reasoning.trm_dense import DensePlasticTRM
from models.recursive_reasoning.trm_stacked import StackedDenseTRM


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def config(**extra):
    return dict(dict(batch_size=2, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
        puzzle_emb_ndim=16, puzzle_emb_len=1, H_cycles=3, L_cycles=2, H_layers=0,
        L_layers=2, hidden_size=16, expansion=2., num_heads=4, pos_encodings="rope",
        halt_max_steps=4, halt_exploration_prob=.1, forward_dtype="float32",
        no_ACT_continue=True, dense_controller_width=16, dense_controller_heads=4,
        dense_controller_layers=1, dense_tile_size=8, dense_chunk_tiles=17,
        dense_decoder_width=8, dense_relative_cap=.03, meta_controller_width=16,
        meta_controller_heads=4, meta_controller_layers=1, meta_history=3,
        meta_tile_size=8, meta_chunk_tiles=17, meta_decoder_width=8, meta_relative_cap=.1,
        meta_cross_attention=True, meta_cross_heads=4, meta_read_queries=4), **extra)


def data():
    return dict(inputs=torch.tensor([[1, 2, 3, 1, 5, 6, 1, 8, 9] * 9,
                                    [2, 1, 4, 5, 1, 7, 8, 1, 10] * 9]),
        labels=torch.arange(81)[None].expand(2, -1) % 9 + 2,
        puzzle_identifiers=torch.zeros(2, dtype=torch.long))


def model(hierarchy=True, **extra):
    torch.manual_seed(17)
    return (StackedDenseTRM if hierarchy else DensePlasticTRM)(config(**extra)).eval()


@pytest.mark.parametrize("hierarchy", [False, True])
@pytest.mark.parametrize("cycles", [(1, 0), (1, 2), (3, 2), (3, 6)])
def test_fresh_effective_weights_at_every_low_and_high_call(hierarchy, cycles):
    instance, batch = model(hierarchy, H_cycles=cycles[0], L_cycles=cycles[1]), data()
    a_actions, c_actions, counts = [], [], []
    def inspect(_module, _args):
        assert all(port.delta is not None for port in instance.ports)
        a_actions.append([port.delta.detach().clone() for port in instance.ports])
        counts.append(instance.weight_updates_this_act)
        if hierarchy:
            assert instance.b_write_port.delta is None  # C's write has already been consumed.
            assert instance.meta_weight_updates_this_act == instance.weight_updates_this_act
            c_actions.append(instance.last_c_action.clone())
    handle = instance.inner.L_level.register_forward_pre_hook(inspect)
    with torch.no_grad():
        instance(instance.initial_carry(batch), batch)
    handle.remove()
    expected = cycles[0] * (cycles[1] + 1)
    assert instance.weight_update_cadence == "per_reasoning_step"
    assert len(a_actions) == instance.reasoning_calls_per_act == expected
    assert counts == list(range(1, expected + 1))
    assert all(not torch.equal(a, b) for first, second in zip(a_actions, a_actions[1:])
               for a, b in zip(first, second))
    assert all(not torch.equal(a, b) for a, b in zip(c_actions, c_actions[1:]))
    assert all(port.delta is None for port in instance.ports)


def test_b_observes_current_low_state_and_previous_step_features_at_every_call():
    instance, batch = model(), data()
    observations, previous_norms = [], torch.zeros(2, 2)
    def reader(_module, args):
        observations.append(tuple(value.detach().clone() for value in args[:7]))
    def reasoning(_module, args):
        nonlocal previous_norms
        low, high, previous_high, probabilities, previous_probs, norms, _ = observations[-1]
        high_step = instance.call % (instance.config.L_cycles + 1) == 0
        expected_low = args[1] if high_step else args[0]
        assert torch.equal(low, expected_low[:, 1:].float())
        assert torch.equal(high, instance.high[:, 1:].float())
        assert torch.equal(norms, previous_norms)
        if len(observations) > 1:
            assert torch.equal(previous_high, observations[-2][1])
            assert torch.equal(previous_probs, observations[-2][3])
        previous_norms = torch.stack([p.delta.detach().square().mean((1, 2)).sqrt()
                                      for p in instance.ports], -1)
    handles = [instance.controller.register_forward_pre_hook(reader),
               instance.inner.L_level.register_forward_pre_hook(reasoning)]
    with torch.no_grad():
        instance(instance.initial_carry(batch), batch)
    for handle in handles:
        handle.remove()
    assert len(observations) == instance.reasoning_calls_per_act


def test_native_truncation_keeps_all_step_actions_and_trains_each_final_cycle_write():
    instance, batch = model(L_cycles=6), data()
    a_actions, c_actions = [], []
    def capture(target):
        def hook(_module, _args, output):
            values = output if isinstance(output, list) else [output]
            for value in values:
                if value.requires_grad:
                    value.retain_grad()
            target.append(values)
        return hook
    handles = [instance.generator.register_forward_hook(capture(a_actions)),
               instance.meta_generator.register_forward_hook(capture(c_actions))]
    _, output = instance(instance.initial_carry(batch), batch)
    output["logits"].square().mean().backward()
    for handle in handles:
        handle.remove()
    assert len(a_actions) == len(c_actions) == 21
    for actions in (a_actions, c_actions):
        assert all(not value.requires_grad for row in actions[:14] for value in row)
        assert all(value.grad is not None and value.grad.abs().sum() > 0
                   for row in actions[14:] for value in row)
    assert all(p.grad is None for p in instance.inner.parameters())
    assert not instance.controller.memory.requires_grad
    assert not instance.meta_controller.state.requires_grad


def test_full_puzzle_and_next_puzzle_have_no_unmodified_or_reused_step():
    instance, batch = model(L_cycles=6, halt_max_steps=16), data()
    first_steps = []
    handle = instance.inner.L_level.register_forward_pre_hook(lambda _m, _a:
        first_steps.append(instance.ports[0].delta.detach().clone()) if instance.call == 1 else None)
    with torch.no_grad():
        carry = instance.initial_carry(batch)
        for index in range(16):
            carry, _ = instance(carry, batch)
            assert carry.steps.tolist() == [index + 1] * 2
            assert instance.weight_updates_this_act == instance.meta_weight_updates_this_act == 21
        assert carry.halted.all()
        instance(carry, batch)
    handle.remove()
    assert len(first_steps) == 17
    assert torch.equal(first_steps[0], first_steps[-1])  # independent new puzzle reset
    assert all(not torch.equal(a, b) for a, b in zip(first_steps[:-1], first_steps[1:-1]))


def test_all_step_actions_are_label_free_and_batch_isolated():
    instance, batch = model(), data()
    def trace(current):
        result = []
        handle = instance.inner.L_level.register_forward_pre_hook(lambda _m, _a: result.append(
            (instance.last_c_action.clone(), *[p.delta.detach().clone() for p in instance.ports])))
        with torch.no_grad():
            instance(instance.initial_carry(current), current)
        handle.remove()
        return result
    first = trace(batch)
    changed_labels = trace(dict(batch, labels=batch["labels"].flip(1)))
    changed_input = {key: value.clone() for key, value in batch.items()}
    changed_input["inputs"][1].fill_(1)
    changed_peer = trace(changed_input)
    assert len(first) == len(changed_labels) == len(changed_peer) == 9
    assert all(torch.equal(a, b) for row_a, row_b in zip(first, changed_labels)
               for a, b in zip(row_a, row_b))
    assert all(torch.equal(a[0], b[0]) for row_a, row_b in zip(first, changed_peer)
               for a, b in zip(row_a, row_b))


def test_mid_puzzle_failure_clears_both_levels_and_retry_is_fresh():
    instance, batch = model(), data()
    def fail(_module, _args):
        if instance.call == 4:
            raise RuntimeError("injected fourth-step failure")
    handle = instance.inner.L_level.register_forward_pre_hook(fail)
    with pytest.raises(RuntimeError, match="fourth-step"):
        instance(instance.initial_carry(batch), batch)
    handle.remove()
    assert all(p.delta is None for p in instance.ports)
    assert instance.b_write_port.delta is None
    assert instance.controller.memory is instance.meta_controller.state is None
    reference = model()
    with torch.no_grad():
        _, actual = instance(instance.initial_carry(batch), batch)
        _, expected = reference(reference.initial_carry(batch), batch)
    assert torch.equal(actual["logits"], expected["logits"])


@pytest.mark.parametrize("cadence", [None, "per_layer", "sometimes", True])
def test_invalid_cadence_fails_before_using_the_model(cadence):
    with pytest.raises(ValueError, match="weight_update_cadence"):
        model(weight_update_cadence=cadence)
