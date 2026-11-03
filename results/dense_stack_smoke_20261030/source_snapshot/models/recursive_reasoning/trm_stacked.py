"""Dense hierarchy: C temporarily changes B's generator, and B changes A.

C observes B's current context and its causal history. It directly generates a
dense update to B's generator output matrix. No A/B base tensor is overwritten
by an inference intervention, and no low-rank factors are introduced.
"""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn

from models.recursive_reasoning.trm_dense import (DensePlasticTRM,
    DenseTemporaryProjection, DenseWeightGenerator)


class MetaStateController(nn.Module):
    """Small causal Transformer reading B states, changes and its own state.

    A fixed, zero-padded history makes independent per-puzzle resets explicit.
    Past history and recurrent state are detached between ACT updates.
    """
    def __init__(self, observed_width, max_steps, width=32, heads=4, layers=1, history=4):
        super().__init__()
        values = (observed_width, max_steps, width, heads, layers, history)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in values) or width % heads:
            raise ValueError("Positive dimensions and width divisible by heads are required")
        self.observed_width, self.width, self.history_length = observed_width, width, history
        self.observation = nn.Linear(3 * observed_width + width, width)
        self.position = nn.Parameter(torch.randn(history, width) * .02)
        self.phase = nn.Embedding(max_steps, width)
        nn.init.normal_(self.phase.weight, std=.02)
        block = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.static_context = nn.Parameter(torch.zeros(width))
        self.previous, self.state, self.history = None, None, None

    def reset(self, mask):
        if self.state is None or len(self.state) != len(mask):
            self.previous = self.position.new_zeros(len(mask), self.observed_width)
            self.state = self.position.new_zeros(len(mask), self.width)
            self.history = self.position.new_zeros(len(mask), self.history_length, self.width)
        self.previous = torch.where(mask[:, None], 0., self.previous.detach())
        self.state = torch.where(mask[:, None], 0., self.state.detach())
        self.history = torch.where(mask[:, None, None], 0., self.history.detach())

    def forward(self, current, phase, mode="dynamic"):
        current = current.detach().float()
        if mode == "phase":
            self.previous = current
            return self.norm(self.static_context[None] + self.phase(phase))
        previous, state, history = self.previous, self.state, self.history
        if mode == "reset_memory":
            previous, state, history = (torch.zeros_like(value) for value in (previous, state, history))
        observation = self.observation(torch.cat((current, previous, current - previous, state), -1))
        observation = observation + self.phase(phase)
        self.history = torch.cat((history[:, 1:], observation[:, None]), 1)
        sequence = self.history + self.position[None]
        causal = torch.ones(self.history_length, self.history_length,
                            device=current.device, dtype=torch.bool).triu(1)
        self.state = self.norm(self.transformer(sequence, mask=causal)[:, -1])
        self.previous = current
        return self.state

    def detach_runtime(self):
        if self.state is not None:
            self.previous, self.state, self.history = (
                value.detach() for value in (self.previous, self.state, self.history))

    def clear_runtime(self):
        self.previous, self.state, self.history = None, None, None


class StackedDenseTRM(DensePlasticTRM):
    """A reads its input; B reads A; C changes how B writes A's next weights.

    C's only write port is B's dense generator output weight (1024 by 32 with
    the current B config). C never receives labels or a direct A write port.
    """
    def __init__(self, config_dict):
        super().__init__(config_dict)
        self.generator.decoder[-1] = DenseTemporaryProjection(self.generator.decoder[-1])
        width = config_dict.get("meta_controller_width", 32)
        self.meta_controller = MetaStateController(self.controller.width, self.config.halt_max_steps,
            width=width, heads=config_dict.get("meta_controller_heads", 4),
            layers=config_dict.get("meta_controller_layers", 1), history=config_dict.get("meta_history", 4))
        self.meta_generator = DenseWeightGenerator([tuple(self.b_write_port.base.weight.shape)],
            width=width, tile_size=config_dict.get("meta_tile_size", 16),
            chunk_tiles=config_dict.get("meta_chunk_tiles", 128),
            decoder_width=config_dict.get("meta_decoder_width", 16),
            relative_cap=config_dict.get("meta_relative_cap", .1))
        self.meta_mode, self.last_c_action, self.last_b_context = "dynamic", None, None
        self.generator.register_forward_pre_hook(self._control_b)
        self.generator.register_forward_hook(self._finish_b, always_call=True)

    @property
    def b_write_port(self):
        return self.generator.decoder[-1]

    @property
    def c_parameters(self):
        return list(self.meta_controller.parameters()) + list(self.meta_generator.parameters())

    def set_meta_mode(self, mode):
        if mode not in ("dynamic", "off", "phase", "shuffle", "reset_memory"):
            raise ValueError(f"Unknown meta-controller mode {mode!r}")
        self.meta_mode = mode
        return self

    def configure_hierarchy(self, train_b=True, train_c=True, train_a=False):
        self.configure_training("joint" if train_a else "frozen")
        for parameter in self.b_parameters:
            parameter.requires_grad_(train_b)
        for parameter in self.c_parameters:
            parameter.requires_grad_(train_c)
        return self

    def dense_state(self):
        """A/B state in the original dense-model namespace, excluding C."""
        return {name.replace("generator.decoder.2.base.", "generator.decoder.2."):
            value.detach().cpu().clone() for name, value in self.state_dict().items()
            if not name.startswith(("meta_controller.", "meta_generator."))}

    def b_state(self):
        return {name: value for name, value in self.dense_state().items() if not name.startswith("inner.")}

    def load_dense_state(self, state):
        if not isinstance(state, Mapping):
            raise TypeError("Dense A/B state must be a tensor mapping")
        current = self.state_dict()
        expected = {name.replace("generator.decoder.2.base.", "generator.decoder.2."): name
            for name in current if not name.startswith(("meta_controller.", "meta_generator."))}
        if set(state) != set(expected):
            raise ValueError("Dense A/B checkpoint keys differ")
        for canonical, name in expected.items():
            value = state[canonical]
            if not isinstance(value, torch.Tensor) or value.shape != current[name].shape or value.dtype != current[name].dtype:
                raise ValueError(f"Dense A/B tensor shape/dtype mismatch: {canonical}")
        with torch.no_grad():
            for canonical, name in expected.items():
                current[name].copy_(state[canonical])

    def _control_b(self, _module, args):
        self.last_b_context = args[0].detach()
        self.b_write_port.delta = None
        if self.meta_mode == "off":
            return
        context = self.meta_controller(args[0], self.phase,
            "dynamic" if self.meta_mode == "shuffle" else self.meta_mode)
        delta = self.meta_generator(context, [self.b_write_port.base.weight])[0]
        if self.meta_mode == "shuffle":
            delta = delta.roll(1, 0)
        self.b_write_port.delta = delta
        self.last_c_action = delta.detach()

    def _finish_b(self, _module, _args, _output):
        self.b_write_port.delta = None

    def clear_runtime(self):
        super().clear_runtime()
        self.b_write_port.delta = None
        self.meta_controller.clear_runtime()
        self.last_c_action, self.last_b_context = None, None

    def forward(self, carry, batch):
        self.meta_controller.reset(carry.halted)
        self.last_c_action, self.last_b_context = None, None
        try:
            result = super().forward(carry, batch)
        finally:
            self.b_write_port.delta = None
            self.meta_controller.detach_runtime()
        return result
