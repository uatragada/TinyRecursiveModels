"""Small CPU checks for transplant isolation and a frozen-A adaptation step."""
import copy

import pytest
import torch
import torch.nn.functional as F

from local_lab.adam_atan2 import AdamATan2
from local_lab.probe import (native_a_state, non_a_state, transplant_native_a,
                             shared_native_initial_constants, native_buffer_snapshot,
                             configure_probe, validate_adaptation_partition, checkpoint_steps)
from local_lab.train import state_snapshot, state_digest
from models.recursive_reasoning.trm_plastic import PlasticTinyRecursiveReasoningModel_ACTV1


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def small_model(seed):
    torch.manual_seed(seed)
    return PlasticTinyRecursiveReasoningModel_ACTV1(dict(
        batch_size=2, seq_len=9, puzzle_emb_ndim=32, puzzle_emb_len=1,
        num_puzzle_identifiers=1, vocab_size=11, H_cycles=3, L_cycles=6,
        H_layers=0, L_layers=2, hidden_size=32, expansion=2, num_heads=4,
        pos_encodings="rope", forward_dtype="float32", halt_max_steps=16,
        halt_exploration_prob=.1, controller_width=32, controller_heads=4,
        controller_layers=1, controller_ff_multiplier=2, controller_history=4,
        adapter_rank=4, adapter_alpha=.2))


def test_adaptation_partition_excludes_source_validation_endpoint_and_reuses_data():
    source = {"eval_size": 128, "data_dir": "data/local-bounded-sudoku"}
    cfg = {"adaptation_val_offset": 128, "data_dir": source["data_dir"]}
    validate_adaptation_partition(cfg, source)
    with pytest.raises(ValueError, match="overlaps"):
        validate_adaptation_partition(dict(cfg, adaptation_val_offset=127), source)
    with pytest.raises(ValueError, match="reuse"):
        validate_adaptation_partition(dict(cfg, data_dir="data/some-other-cache"), source)


def test_requested_checkpoint_steps_are_normalized_and_invalid_budgets_rejected():
    assert checkpoint_steps({"steps": 300, "checkpoint_steps": [200, 100]}) == [0, 100, 200, 300]
    assert checkpoint_steps({"steps": 60, "checkpoint_steps": []}) == [0, 60]
    assert checkpoint_steps({"steps": 60}) == [0, 60]
    for requested in ([0], [-1], [301], [100, 100], [True], [1.5], "100"):
        with pytest.raises(ValueError):
            checkpoint_steps({"steps": 300, "checkpoint_steps": requested})


def test_a_transplant_includes_native_table_and_constants_excludes_all_donor_b_and_uv():
    common, donor = state_snapshot(small_model(29)), state_snapshot(small_model(17))
    for name in donor:
        donor[name].fill_(.25 if name in native_a_state(donor) else 7.)
    original_common = {name: value.clone() for name, value in common.items()}
    combined = transplant_native_a(common, donor)
    assert "inner.puzzle_emb.weights" in native_a_state(combined)
    assert "inner.H_init" in native_a_state(combined)
    assert "inner.L_init" in native_a_state(combined)
    assert any(name.endswith(".u") for name in non_a_state(combined))
    for name, value in combined.items():
        assert torch.equal(value, donor[name] if name in native_a_state(combined) else common[name]), name
        assert torch.equal(common[name], original_common[name]), name
        assert value.data_ptr() != common[name].data_ptr()
        assert value.data_ptr() != donor[name].data_ptr()
    assert state_digest(non_a_state(combined)) == state_digest(non_a_state(common))


@pytest.mark.parametrize("failure", ["missing", "shape", "dtype"])
def test_transplant_rejects_bad_a_before_mutating_common(failure):
    common, donor = state_snapshot(small_model(29)), state_snapshot(small_model(17))
    before = state_digest(common)
    key = "inner.q_head.weight"
    if failure == "missing":
        donor.pop(key)
    elif failure == "shape":
        donor[key] = donor[key][:-1]
    else:
        donor[key] = donor[key].double()
    with pytest.raises(ValueError):
        transplant_native_a(common, donor)
    assert state_digest(common) == before


def test_shared_native_initial_constants_checked_not_randomized():
    donor = state_snapshot(small_model(17))
    other = {name: value.clone() for name, value in donor.items()}
    expected = state_digest({name: donor[name] for name in ("inner.H_init", "inner.L_init")})
    assert shared_native_initial_constants({"native": donor, "joint": other}) == expected
    other["inner.H_init"].add_(.01)
    with pytest.raises(ValueError, match="different"):
        shared_native_initial_constants({"native": donor, "joint": other})
    other.pop("inner.H_init")
    with pytest.raises(ValueError, match="missing"):
        shared_native_initial_constants({"native": donor, "joint": other})


def test_one_dynamic_adaptation_step_updates_fresh_b_uv_but_freezes_entire_a_and_native_buffers():
    core = small_model(29)
    active = configure_probe(core)
    assert not core.inner.puzzle_emb.training
    assert not core.inner.puzzle_emb.local_weights.requires_grad
    before = state_snapshot(core)
    buffers_before = state_digest(native_buffer_snapshot(core))
    batch = {"inputs": torch.tensor([[1, 2, 1, 4, 1, 6, 1, 8, 1], [9, 1, 7, 1, 5, 1, 3, 1, 1]]),
             "labels": torch.tensor([[2, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 2]]),
             "puzzle_identifiers": torch.zeros(2, dtype=torch.long)}
    optimizer = AdamATan2(active, lr=1e-4, betas=(.9, .95), weight_decay=1.)
    carry, outputs = core(core.initial_carry(batch), batch)
    loss = F.cross_entropy(outputs["logits"].reshape(-1, 11), batch["labels"].reshape(-1))
    loss.backward()
    assert all(parameter.grad is None for name, parameter in core.inner.named_parameters()
               if not name.endswith((".u", ".v")))
    assert core.inner.puzzle_emb.local_weights.grad is None
    optimizer.step()
    after = state_snapshot(core)
    assert state_digest(native_a_state(after)) == state_digest(native_a_state(before))
    assert state_digest(native_buffer_snapshot(core)) == buffers_before
    assert any(not torch.equal(after[name], before[name]) for name in after if name.startswith("controller."))
    assert any(not torch.equal(after[name], before[name]) for name in after if name.endswith((".u", ".v")))
    assert torch.equal(after["static_gates"], before["static_gates"])
    assert all(adapter.gate is None for adapter in core.adapters)
    assert not carry.inner_carry.z_H.requires_grad and not carry.inner_carry.z_L.requires_grad


@pytest.mark.parametrize("fixed_act", [False, True])
def test_fixed_act_probe_ignores_positive_halt_head_without_disabling_controller_gradients(fixed_act):
    core = small_model(29)
    core.config.halt_exploration_prob = 0.
    configure_probe(core, fixed_act_training=fixed_act)
    with torch.no_grad():
        core.inner.q_head.bias.fill_(10.)
    batch = {"inputs": torch.tensor([[1, 2, 1, 4, 1, 6, 1, 8, 1], [9, 1, 7, 1, 5, 1, 3, 1, 1]]),
             "labels": torch.tensor([[2, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 2]]),
             "puzzle_identifiers": torch.zeros(2, dtype=torch.long)}
    carry, outputs = core(core.initial_carry(batch), batch)
    assert core.training is not fixed_act
    assert outputs["logits"].requires_grad
    assert bool(carry.halted.all()) is not fixed_act
    assert int(carry.steps.min()) == 1
    if fixed_act:
        F.cross_entropy(outputs["logits"].reshape(-1, 11), batch["labels"].reshape(-1)).backward()
        assert core.controller.gates.weight.grad.norm() > 0
        with torch.no_grad():
            for step in range(2, 17):
                carry, _ = core(carry, batch)
                assert bool(carry.halted.all()) == (step == 16)
                assert int(carry.steps.min()) == step
