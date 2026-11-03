"""Intervention, information-boundary and isolation checks for the frozen-A pilot."""
import pytest
import torch
from models.recursive_reasoning.trm_plastic import PlasticTinyRecursiveReasoningModel_ACTV1
from local_lab.frozen_control import (FrozenControllerModel, ObservationController,
    PhaseController, constraint_features, gate_oracle, task_loss)
from local_lab.train import state_digest, state_snapshot


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def model(kind="revised"):
    torch.manual_seed(11)
    config = dict(batch_size=2, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
        puzzle_emb_ndim=32, puzzle_emb_len=1, H_cycles=3, L_cycles=6, H_layers=0,
        L_layers=2, hidden_size=32, expansion=2., num_heads=4, pos_encodings="rope",
        halt_max_steps=16, halt_exploration_prob=.1, forward_dtype="float32",
        controller_width=32, controller_heads=4, controller_layers=1,
        controller_ff_multiplier=2, controller_history=4, adapter_rank=4, adapter_alpha=.2)
    core = PlasticTinyRecursiveReasoningModel_ACTV1(config)
    actor = PhaseController() if kind == "phase" else ObservationController(hidden=32, width=32, layers=1, pooled=kind == "pooled")
    return FrozenControllerModel(core, actor, "activation" if kind == "activation" else "weight")


def batch():
    inputs = torch.tensor([[1, 2, 3, 1, 5, 6, 1, 8, 9] * 9,
                           [2, 1, 4, 5, 1, 7, 8, 1, 10] * 9])
    return dict(inputs=inputs, labels=(torch.arange(81)[None].expand(2, -1) % 9 + 2),
                puzzle_identifiers=torch.zeros(2, dtype=torch.long))


def test_constraint_features_respect_rows_columns_boxes_and_no_labels():
    grid = torch.tensor([[(r * 3 + r // 3 + c) % 9 for c in range(9)] for r in range(9)])
    probabilities = torch.nn.functional.one_hot(grid.flatten(), 9).float()[None]
    assert constraint_features(probabilities).abs().max() == 0
    probabilities[0, 1] = probabilities[0, 0]
    residuals = constraint_features(probabilities)
    assert residuals.shape == (1, 81, 27)
    assert residuals.abs().max() > 0
    assert residuals[0, 80].abs().sum() == 0


def test_disabled_interventions_reproduce_native_and_late_replay_matches_zero():
    m, data = model(), batch()
    m.mode = "off"
    with torch.no_grad():
        _, output = m(m.core.initial_carry(data), data)
        replay = m.late_logits(m.late_state, torch.zeros(2, 2, 4))
    assert torch.equal(output["logits"], replay)
    assert m.call == 21
    assert all(port.gate is None for port in m.ports)


@pytest.mark.parametrize("kind", ["phase", "pooled", "revised", "activation"])
def test_actor_gradient_with_no_substrate_changes_and_label_free_actions(kind):
    m, data = model(kind), batch()
    before = state_digest(m.frozen_state())
    buffers = state_digest(m.frozen_buffers())
    optimizer = torch.optim.Adam(m.actor.parameters(), lr=1e-3)
    _, output = m(m.core.initial_carry(data), data)
    action = m.action.detach().clone()
    task_loss(output["logits"], data["inputs"], data["labels"]).mean().backward()
    assert sum(float(p.grad.abs().sum()) for p in m.actor.parameters() if p.grad is not None) > 0
    assert all(p.grad is None and not p.requires_grad for p in m.core.parameters())
    # Fresh carry resets every persistent runtime slot. Query labels cannot alter actions.
    alternative = dict(data, labels=data["labels"].flip(1))
    with torch.no_grad():
        m(m.core.initial_carry(alternative), alternative)
    assert torch.equal(action, m.action)
    optimizer.step()
    assert state_digest(m.frozen_state()) == before
    assert state_digest(m.frozen_buffers()) == buffers


def test_oracle_is_bounded_monotone_and_does_not_change_substrate():
    m, data = model("phase"), batch()
    m.mode = "off"
    before = state_digest(m.frozen_state())
    with torch.no_grad():
        m(m.core.initial_carry(data), data)
    gates, best, zero, history = gate_oracle(m, m.late_state, data["inputs"], data["labels"], steps=3)
    assert gates.abs().max() <= 1
    assert all(a >= b - 1e-7 for a, b in zip(history, history[1:]))
    assert torch.all(task_loss(best, data["inputs"], data["labels"]) <= task_loss(zero, data["inputs"], data["labels"]) + 1e-5)
    assert state_digest(m.frozen_state()) == before
    assert all(p.grad is None for p in m.core.parameters())
    assert "override" not in dict(m.named_parameters())


def test_revised_pooled_activation_parameter_counts_match():
    counts = [sum(p.numel() for p in model(kind).actor.parameters()) for kind in ("revised", "pooled", "activation")]
    assert len(set(counts)) == 1


def test_memory_persists_between_updates_and_fresh_carry_resets():
    m, data = model(), batch()
    with torch.no_grad():
        carry, _ = m(m.core.initial_carry(data), data)
        first = m.action.clone()
        m(carry, data)
        second = m.action.clone()
        m(m.core.initial_carry(data), data)
    assert not torch.equal(first, second)
    assert torch.equal(first, m.action)


def test_action_applies_only_in_last_seven_calls_and_is_constant_there():
    m, data = model(), batch()
    observed = []
    handle = m.ports[0].register_forward_pre_hook(lambda module, args: observed.append(
        None if module.gate is None else module.gate.detach().clone()))
    with torch.no_grad():
        m(m.core.initial_carry(data), data)
    handle.remove()
    assert len(observed) == 21
    assert all(gate is None for gate in observed[:14])
    assert all(torch.equal(gate, observed[14]) for gate in observed[14:])


def test_other_puzzles_cannot_change_a_batch_members_action_or_prediction():
    m, data = model(), batch()
    with torch.no_grad():
        _, first = m(m.core.initial_carry(data), data)
        action = m.action.clone()
        changed = {name: value.clone() for name, value in data.items()}
        changed["inputs"][1].fill_(1)
        changed["labels"][1].fill_(10)
        _, second = m(m.core.initial_carry(changed), changed)
    assert torch.equal(first["logits"][0], second["logits"][0])
    assert torch.equal(action[0], m.action[0])
