"""Parity against the pinned upstream reference, loaded without CUDA backend."""
from __future__ import annotations

import ast
import copy
import hashlib
import io
import math
from pathlib import Path
from typing import Tuple

import pytest
import torch

from local_lab.adam_atan2 import AdamATan2

REFERENCE = Path(__file__).resolve().parents[1] / "third_party/adam_atan2_0_0_3/adam_atan2/adam_atan2_test.py"
REFERENCE_SHA256 = "723acc9321e4949553367104272314030dffc441821d1361d5ce876af9594fbf"


def upstream_reference():
    raw = REFERENCE.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == REFERENCE_SHA256
    tree = ast.parse(raw.decode())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "_adam_atan2_reference_impl")
    namespace = {"torch": torch, "Tuple": Tuple}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[])),
                 str(REFERENCE), "exec"), namespace)
    return namespace[function.name]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("weight_decay", [0., 1.])
def test_multi_update_matches_actual_pinned_upstream_reference(dtype, weight_decay):
    reference = upstream_reference()
    generator = torch.Generator().manual_seed(17)
    parameter = torch.nn.Parameter(torch.randn((5, 9), generator=generator, dtype=dtype))
    expected = parameter.detach().clone()
    average, squared = torch.zeros_like(expected), torch.zeros_like(expected)
    optimizer = AdamATan2([parameter], lr=1e-4, betas=(.9, .95), weight_decay=weight_decay)
    for step in range(1, 41):
        gradient = torch.randn(parameter.shape, generator=generator, dtype=dtype) * .01
        gradient[0, 0] = 0
        parameter.grad = gradient.clone()
        optimizer.step()
        expected, average, squared = reference(expected, gradient, average, squared,
            step_size=1e-4 / (1 - .9 ** step), wd_step_size=1e-4 * weight_decay,
            bias_correction2_sqrt=math.sqrt(1 - .95 ** step), beta1=.9, beta2=.95)
        torch.testing.assert_close(parameter, expected, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg"], average, rtol=0, atol=0)
        torch.testing.assert_close(optimizer.state[parameter]["exp_avg_sq"], squared, rtol=0, atol=0)
        assert optimizer.state[parameter]["step"] == step


def test_no_gradient_skips_parameters_decay_and_state_step():
    used, unused = torch.nn.Parameter(torch.ones(3)), torch.nn.Parameter(torch.ones(3))
    optimizer = AdamATan2([used, unused], lr=.1, weight_decay=1.)
    used.grad = torch.ones_like(used)
    optimizer.step()
    assert unused not in optimizer.state
    torch.testing.assert_close(unused, torch.ones_like(unused), rtol=0, atol=0)
    previous = used.detach().clone()
    used.grad = None
    optimizer.step()
    assert optimizer.state[used]["step"] == 1
    torch.testing.assert_close(used, previous, rtol=0, atol=0)


def test_zero_gradient_atan2_zero_zero_is_finite_and_only_decays():
    parameter = torch.nn.Parameter(torch.tensor([0., 1., -1.]))
    optimizer = AdamATan2([parameter], lr=.1, betas=(.9, .95), weight_decay=.2)
    for _ in range(3):
        parameter.grad = torch.zeros_like(parameter)
        optimizer.step()
        assert torch.isfinite(parameter).all()
    torch.testing.assert_close(parameter, torch.tensor([0., 1., -1.]) * .98 ** 3)


def test_state_checkpoint_roundtrip_continues_exactly():
    first = torch.nn.Parameter(torch.tensor([.5, -1., 2.]))
    optimizer = AdamATan2([first], lr=1e-4, betas=(.9, .95), weight_decay=1.)
    for _ in range(5):
        first.grad = torch.tensor([.1, -.2, 0.])
        optimizer.step()
    stream = io.BytesIO()
    torch.save({"parameter": first.detach().clone(), "optimizer": optimizer.state_dict()}, stream)
    stream.seek(0)
    saved = torch.load(stream, weights_only=True)
    second = torch.nn.Parameter(saved["parameter"])
    restored = AdamATan2([second], lr=.7)
    restored.load_state_dict(saved["optimizer"])
    for _ in range(3):
        first.grad = torch.tensor([-.3, .2, .01])
        second.grad = first.grad.clone()
        optimizer.step()
        restored.step()
        torch.testing.assert_close(first, second, rtol=0, atol=0)
        assert optimizer.state[first]["step"] == restored.state[second]["step"]


def test_closure_enables_gradients_and_returns_loss():
    parameter = torch.nn.Parameter(torch.tensor([1.]))
    optimizer = AdamATan2([parameter])
    def closure():
        optimizer.zero_grad()
        loss = parameter.square().sum()
        loss.backward()
        return loss
    loss = optimizer.step(closure)
    assert loss.item() == 1.
    assert parameter.item() < 1.


def test_sparse_gradients_rejected_before_any_parameter_mutation():
    dense = torch.nn.Parameter(torch.ones(3))
    sparse = torch.nn.Parameter(torch.ones(3))
    optimizer = AdamATan2([dense, sparse])
    dense.grad = torch.ones_like(dense)
    sparse.grad = torch.sparse_coo_tensor(torch.tensor([[0]]), torch.tensor([1.]), size=(3,))
    with pytest.raises(RuntimeError, match="sparse"):
        optimizer.step()
    torch.testing.assert_close(dense, torch.ones_like(dense), rtol=0, atol=0)
    assert not optimizer.state


@pytest.mark.parametrize("kwargs", [{"lr": -1.}, {"lr": math.nan}, {"betas": (1., .95)},
                                    {"betas": (.9, -1.)}, {"weight_decay": -1.}])
def test_invalid_hyperparameters_rejected(kwargs):
    with pytest.raises(ValueError):
        AdamATan2([torch.nn.Parameter(torch.ones(1))], **kwargs)
