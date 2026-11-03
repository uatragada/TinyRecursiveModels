"""Portable AdamATan2 with the upstream 0.0.3 reference update.

Derived from https://github.com/imoneoi/adam_atan2 (Apache-2.0). The original
source/license are retained in third_party/adam_atan2_0_0_3. PyPI sdist SHA256:
f07faef4a6d7476ad3be6096155a35a5ecf6a3932282367f05d45158b81ff5a8.
Reference adam_atan2/adam_atan2_test.py SHA256:
723acc9321e4949553367104272314030dffc441821d1361d5ce876af9594fbf.

Modification: use ordinary Torch operations instead of the CUDA extension,
with a Python integer step counter and optional closure support. The moments,
bias corrections, atan2 update (without epsilon), and decoupled decay preserve
the reference algorithm. This local port supports dense real FP32/FP64 model
parameters; native TRM keeps its trainable parameters in FP32.
"""
from __future__ import annotations

import math

import torch
from torch.optim import Optimizer


class AdamATan2(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(.9, .999), weight_decay=1e-2):
        if isinstance(lr, torch.Tensor) or not math.isfinite(lr) or lr < 0:
            raise ValueError(f"Invalid scalar learning rate: {lr}")
        if len(betas) != 2 or any(not math.isfinite(beta) or not 0 <= beta < 1 for beta in betas):
            raise ValueError(f"Invalid beta parameters: {betas}")
        if not math.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError(f"Invalid weight decay: {weight_decay}")
        super().__init__(params, dict(lr=lr, betas=betas, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        """Apply exactly one reference update to parameters with gradients."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        # Validate before any mutation, including when a later group contains
        # an unsupported sparse gradient or parameter type.
        for group in self.param_groups:
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("Portable AdamATan2 does not support sparse gradients")
                if parameter.dtype not in (torch.float32, torch.float64):
                    raise TypeError("Portable AdamATan2 requires dense FP32 or FP64 parameters")
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr, weight_decay = group["lr"], group["weight_decay"]
            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                state["step"] += 1
                step = state["step"]
                step_size = lr / (1 - beta1 ** step)
                bias_correction2_sqrt = math.sqrt(1 - beta2 ** step)
                parameter.mul_(1 - lr * weight_decay)
                state["exp_avg"].lerp_(gradient, 1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                denominator = state["exp_avg_sq"].sqrt() / bias_correction2_sqrt
                parameter.add_(torch.atan2(state["exp_avg"], denominator), alpha=-step_size)
        return loss
