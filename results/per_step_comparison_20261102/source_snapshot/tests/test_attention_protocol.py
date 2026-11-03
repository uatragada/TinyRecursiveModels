"""Budget calibration and development measurements preserve the training stream."""
import numpy as np
import pytest
import torch

from local_lab.attention_test import checkpoint_steps, evaluate, evaluation_calls
from models.recursive_reasoning.trm_stacked import StackedDenseTRM


@pytest.fixture(autouse=True)
def one_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def model():
    torch.manual_seed(17)
    cfg = dict(batch_size=2, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
        puzzle_emb_ndim=16, puzzle_emb_len=1, H_cycles=1, L_cycles=0, H_layers=0,
        L_layers=2, hidden_size=16, expansion=2., num_heads=4, pos_encodings="rope",
        halt_max_steps=16, halt_exploration_prob=.1, forward_dtype="float32",
        no_ACT_continue=True, dense_controller_width=16, dense_controller_heads=4,
        dense_controller_layers=1, dense_tile_size=8, dense_chunk_tiles=17,
        dense_decoder_width=8, dense_relative_cap=.03, meta_controller_width=16,
        meta_controller_heads=4, meta_controller_layers=1, meta_history=3,
        meta_tile_size=8, meta_chunk_tiles=17, meta_decoder_width=8, meta_relative_cap=.1,
        meta_cross_attention=True, meta_cross_heads=4, meta_read_queries=4)
    return StackedDenseTRM(cfg).eval()


def pool(n=2):
    labels = np.tile(np.arange(81, dtype=np.int64) % 9 + 2, (n, 1))
    inputs = labels.copy()
    inputs[:, ::2] = 1
    return dict(inputs=inputs, labels=labels,
        puzzle_identifiers=np.zeros(n, dtype=np.int64), base_ids=np.arange(n))


def test_timing_plan_counts_checkpoints_and_only_the_predeclared_ablations():
    cfg = dict(batch_size=32, eval_batch_size=16, dev_size=32, test_size=64,
               seeds=[29, 43, 59], dev_every=32, ablations=["c_off", "c_read_shuffle"])
    assert checkpoint_steps(cfg, 96) == [32, 64]
    assert evaluation_calls(cfg, 96) == 12 * (4 * 32 + 64) + 3 * 2 * 64 + 32 + 64
    assert checkpoint_steps(cfg, 32) == []


def test_historical_timing_plan_is_preserved_when_new_options_are_absent():
    cfg = dict(batch_size=32, dev_size=64, test_size=128, seeds=[29, 43, 59])
    assert checkpoint_steps(cfg, 240) == []
    assert evaluation_calls(cfg, 240) == 2784


@pytest.mark.parametrize("variant,b,c", [("normal", 1, 1), ("a_only", 0, 0),
                                         ("c_off", 1, 0), ("c_read_shuffle", 1, 1)])
def test_evaluation_records_actual_writes_for_every_act_and_partial_batch(variant, b, c):
    instance = model()
    row, saved = evaluate(instance, pool(3), {"batch_size": 2, "eval_batch_size": 2},
                          "cross", variant, lambda: None)
    assert row["b_weight_updates_per_act"] == [b] * 32
    assert row["c_weight_updates_per_act"] == [c] * 32
    assert saved["logits"].shape == (3, 81, 11)
    assert row["actor_unchanged"] and row["a_unchanged"]


def test_development_at_episode_boundary_does_not_change_the_next_training_update():
    left, right = model(), model()
    batch = {key: torch.as_tensor(value) for key, value in pool().items() if key != "base_ids"}
    optimizers = [torch.optim.SGD(instance.b_parameters + instance.c_parameters, lr=.01)
                  for instance in (left, right)]
    carries = [instance.initial_carry(batch) for instance in (left, right)]
    for _ in range(16):
        for index, instance in enumerate((left, right)):
            optimizers[index].zero_grad(set_to_none=True)
            carries[index], output = instance(carries[index], batch)
            output["logits"].square().mean().backward()
            optimizers[index].step()
    assert all(carry.halted.all() for carry in carries)
    evaluate(left, pool(1), {"batch_size": 2}, "cross", "step_16", lambda: None)
    results = []
    for index, instance in enumerate((left, right)):
        optimizers[index].zero_grad(set_to_none=True)
        _, output = instance(carries[index], batch)
        results.append(output["logits"].detach().clone())
        output["logits"].square().mean().backward()
        optimizers[index].step()
    assert torch.equal(*results)
    assert all(torch.equal(value, right.state_dict()[key]) for key, value in left.state_dict().items())
