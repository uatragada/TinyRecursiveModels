"""Matched control capacity is active and preserves common initialization."""
import pytest
import torch

from models.recursive_reasoning.trm_comparison import LargerDenseB
from models.recursive_reasoning.trm_dense import DensePlasticTRM
from models.recursive_reasoning.trm_stacked import StackedDenseTRM


@pytest.fixture(autouse=True)
def one_thread():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def config():
    return dict(batch_size=2, seq_len=81, vocab_size=11, num_puzzle_identifiers=1,
        puzzle_emb_ndim=32, puzzle_emb_len=1, H_cycles=3, L_cycles=2, H_layers=0,
        L_layers=2, hidden_size=32, expansion=2., num_heads=4, pos_encodings="rope",
        halt_max_steps=16, halt_exploration_prob=.1, forward_dtype="float32",
        no_ACT_continue=True, dense_controller_width=64, dense_controller_heads=4,
        dense_controller_layers=2, dense_tile_size=32, dense_chunk_tiles=128,
        dense_decoder_width=32, dense_relative_cap=.03, meta_controller_width=32,
        meta_controller_heads=4, meta_controller_layers=1, meta_history=4,
        meta_cross_attention=True, meta_cross_heads=4, meta_read_queries=4,
        meta_tile_size=16, meta_chunk_tiles=128, meta_decoder_width=16, meta_relative_cap=.1,
        comparison_expanded_width=224)


def test_larger_b_matches_all_hierarchy_parameters_exactly():
    base, larger, cross = DensePlasticTRM(config()), LargerDenseB(config()), StackedDenseTRM(config())
    size = lambda parameters: sum(p.numel() for p in parameters)
    assert size(larger.b_parameters) == size(cross.b_parameters + cross.c_parameters)
    assert size(larger.b_parameters) - size(base.b_parameters) == 29168


def test_extra_b_capacity_receives_gradients_while_a_stays_frozen():
    torch.manual_seed(17)
    model = LargerDenseB(config()).eval()
    batch = dict(inputs=torch.ones(2, 81, dtype=torch.long),
        labels=torch.ones(2, 81, dtype=torch.long) * 2,
        puzzle_identifiers=torch.zeros(2, dtype=torch.long))
    before = model.a_state()
    optimizer = torch.optim.Adam(model.b_parameters, lr=1e-3)
    _, output = model(model.initial_carry(batch), batch)
    output["logits"].square().mean().backward()
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in model.extra_context.parameters())
    assert model.extra_phase.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in model.inner.parameters())
    optimizer.step()
    assert all(torch.equal(value, model.a_state()[key]) for key, value in before.items())


def test_common_ab_load_preserves_extra_parameters_and_validates_atomically():
    source, larger = DensePlasticTRM(config()), LargerDenseB(config())
    before = {k: v.clone() for k, v in larger.state_dict().items() if k.startswith("extra_")}
    larger.load_dense_state(source.state_dict())
    assert all(torch.equal(value, larger.state_dict()[key]) for key, value in source.state_dict().items())
    assert all(torch.equal(value, larger.state_dict()[key]) for key, value in before.items())
    invalid = dict(source.state_dict())
    invalid["generator.decoder.2.weight"] = torch.zeros(1)
    with pytest.raises(ValueError):
        larger.load_dense_state(invalid)
    assert all(torch.equal(value, larger.state_dict()[key]) for key, value in source.state_dict().items())
