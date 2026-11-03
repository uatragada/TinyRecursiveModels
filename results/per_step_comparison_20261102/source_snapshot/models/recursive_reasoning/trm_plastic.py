"""Native TRM with a separate Transformer B controlling temporary adapters.

The upstream inner/ACT forward methods and their gradient truncation are kept
intact. Hooks observe native reasoning calls and install bounded functional
residuals on the two shared MLP output projections; base weights never change
as a consequence of an inference intervention.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn
import torch.nn.functional as F

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


class TrajectoryController(nn.Module):
    """A causal, bounded history of content, changes, and B's previous state."""
    def __init__(self, hidden, width=128, heads=8, layers=3, ff_multiplier=4,
                 history=4, outputs=8):
        super().__init__()
        dimensions = (hidden, width, heads, layers, ff_multiplier, history, outputs)
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in dimensions):
            raise ValueError("Controller dimensions must be positive integers")
        if width % heads:
            raise ValueError("Controller width must be divisible by attention heads")
        self.width, self.history = width, history
        self.observation = nn.Linear(3 * hidden + width, width)
        self.position = nn.Parameter(torch.zeros(history, width))
        layer = nn.TransformerEncoderLayer(width, heads, width * ff_multiplier,
                                           dropout=0., activation="gelu", batch_first=True,
                                           norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.gates = nn.Linear(width, outputs)
        nn.init.normal_(self.gates.weight, std=1e-3)
        nn.init.zeros_(self.gates.bias)

    def forward(self, current, previous, previous_state, history):
        if previous is None:
            previous = torch.zeros_like(current)
        if previous_state is None:
            previous_state = current.new_zeros(current.shape[0], self.width)
        observation = self.observation(torch.cat((current, previous, current - previous, previous_state), -1))
        history = (history + [observation])[-self.history:]
        sequence = torch.stack(history, 1) + self.position[:len(history)]
        causal = torch.ones(len(history), len(history), dtype=torch.bool, device=current.device).triu(1)
        state = self.norm(self.transformer(sequence, mask=causal)[:, -1])
        return torch.tanh(self.gates(state)), state, history


class ConditionalNativeProjection(nn.Module):
    """CastedLinear plus alpha * U diag(g) V^T, using unit-norm columns.

    Each gate is in [-1, 1], so every effective weight delta has Frobenius norm
    at most alpha * rank. Parameters are float32; calculations follow the native
    input dtype, including bfloat16.
    """
    def __init__(self, base, rank=4, alpha=.2):
        super().__init__()
        if isinstance(rank, bool) or not isinstance(rank, int) or rank <= 0:
            raise ValueError("Adapter rank must be a positive integer")
        if not math.isfinite(alpha) or alpha < 0:
            raise ValueError("Adapter alpha must be finite and nonnegative")
        self.base, self.rank, self.alpha = base, rank, float(alpha)
        out_features, in_features = base.weight.shape
        self.u = nn.Parameter(torch.randn(out_features, rank) * .02)
        self.v = nn.Parameter(torch.randn(in_features, rank) * .02)
        self.gate = None

    def forward(self, inputs):
        output = self.base(inputs)
        if self.gate is None:
            return output
        u = F.normalize(self.u, dim=0).to(inputs.dtype)
        v = F.normalize(self.v, dim=0).to(inputs.dtype)
        coefficients = F.linear(inputs, v.T)
        gates = self.gate.to(inputs.dtype)
        while gates.ndim < coefficients.ndim:
            gates = gates.unsqueeze(-2)
        return output + self.alpha * F.linear(coefficients * gates, u)


class PlasticTinyRecursiveReasoningModel_ACTV1(TinyRecursiveReasoningModel_ACTV1):
    """Use the repository's native TRM A and our separate trajectory Transformer B."""
    def __init__(self, config_dict):
        super().__init__(config_dict)
        self.adapter_rank = config_dict.get("adapter_rank", 4)
        self.adapter_alpha = config_dict.get("adapter_alpha", .2)
        self.controller_architecture = {
            "hidden": self.config.hidden_size,
            "width": config_dict.get("controller_width", 128),
            "heads": config_dict.get("controller_heads", 8),
            "layers": config_dict.get("controller_layers", 3),
            "ff_multiplier": config_dict.get("controller_ff_multiplier", 4),
            "history": config_dict.get("controller_history", 4),
            "outputs": self.config.L_layers * self.adapter_rank,
        }
        self.controller = TrajectoryController(**self.controller_architecture)
        self.static_gates = nn.Parameter(torch.full((self.config.L_layers, self.adapter_rank), .01))
        for layer in self.inner.L_level.layers:
            layer.mlp.down_proj = ConditionalNativeProjection(layer.mlp.down_proj,
                                                              self.adapter_rank, self.adapter_alpha)
        self.mode = "dynamic"
        self.last_trace = {}
        self._history, self._previous, self._controller_state, self._fixed_gate = [], None, None, None
        self.inner.register_forward_pre_hook(self._begin_inner)
        self.inner.L_level.register_forward_pre_hook(self._before_reasoning, with_kwargs=True)
        self.inner.register_forward_hook(self._end_inner, always_call=True)

    @property
    def adapters(self):
        return [layer.mlp.down_proj for layer in self.inner.L_level.layers]

    @property
    def controller_parameters(self):
        return sum(parameter.numel() for parameter in self.controller.parameters())

    def set_mode(self, mode):
        if mode not in ("off", "dynamic", "static", "fixed"):
            raise ValueError(f"Unknown intervention mode {mode!r}")
        self.mode = mode
        return self

    def configure_training(self, mode):
        """Choose trainable components without changing upstream recursion."""
        if mode not in ("native", "off", "dynamic", "static", "joint", "static_joint", "probe"):
            raise ValueError(f"Unknown training mode {mode!r}")
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        train_a = mode in ("native", "off", "joint", "static_joint")
        for name, parameter in self.inner.named_parameters():
            is_adapter = name.endswith((".u", ".v"))
            parameter.requires_grad_(train_a if not is_adapter else mode in ("dynamic", "static", "joint", "static_joint"))
        # Native puzzle embeddings are trainable buffers, updated by the native
        # sparse-embedding optimizer rather than a conventional parameter group.
        if hasattr(self.inner, "puzzle_emb"):
            self.inner.puzzle_emb.local_weights.requires_grad_(train_a)
        if mode in ("dynamic", "joint", "probe"):
            for parameter in self.controller.parameters():
                parameter.requires_grad_(True)
        if mode in ("static", "static_joint"):
            self.static_gates.requires_grad_(True)
        self.set_mode("off" if mode in ("native", "off") else "static" if mode in ("static", "static_joint") else "dynamic")
        return self

    def _begin_inner(self, _module, _args):
        # Reset once per native ACT inner call, not once per optimizer step or
        # individual L/H recursion. This also prevents cross-puzzle state leaks.
        self._history, self._previous, self._controller_state, self._fixed_gate = [], None, None, None
        self.last_trace = {"mode": self.mode, "reasoning_calls": 0, "controller_calls": 0,
                           "gradient_enabled_calls": 0, "gradient_disabled_calls": 0,
                           "history_reset": True, "calls": []}
        for adapter in self.adapters:
            adapter.gate = None

    def _before_reasoning(self, _module, args, kwargs):
        hidden = args[0] if args else kwargs["hidden_states"]
        injection = args[1] if len(args) > 1 else kwargs["input_injection"]
        grad_enabled = torch.is_grad_enabled()
        self.last_trace["reasoning_calls"] += 1
        self.last_trace["gradient_enabled_calls" if grad_enabled else "gradient_disabled_calls"] += 1
        if self.mode == "off":
            applied = None
        elif self.mode == "static":
            applied = torch.tanh(self.static_gates)[None].expand(hidden.shape[0], -1, -1)
        else:
            current = (hidden.float() + injection.float()).mean(dim=1)
            gate, state, history = self.controller(current, self._previous, self._controller_state, self._history)
            gate = gate.reshape(hidden.shape[0], len(self.adapters), self.adapter_rank)
            if self._fixed_gate is None:
                self._fixed_gate = gate
            applied = self._fixed_gate if self.mode == "fixed" else gate
            self._previous, self._controller_state, self._history = current, state, history
            self.last_trace["controller_calls"] += 1
        for index, adapter in enumerate(self.adapters):
            adapter.gate = None if applied is None else applied[:, index]
        detached = None if applied is None else applied.detach().clone()
        self.last_trace["calls"].append({"grad_enabled": grad_enabled, "gate": detached,
                                         "gate_rms": None if detached is None else detached.square().mean().sqrt(),
                                         "gate_requires_grad": applied.requires_grad if applied is not None else False,
                                         "history_length": len(self._history)})

    def _end_inner(self, _module, _args, _output):
        for adapter in self.adapters:
            adapter.gate = None
        self._history, self._previous, self._controller_state, self._fixed_gate = [], None, None, None

    def forward(self, carry, batch):
        if self.mode == "fixed" and self.training and torch.is_grad_enabled():
            raise ValueError("Fixed-first-gate mode is evaluation-only: native warmup computes its first gate without gradients")
        return super().forward(carry, batch)

    def load_native_state(self, native_state):
        """Copy an untouched native TRM state, validating everything first."""
        if not isinstance(native_state, Mapping):
            raise TypeError("Native state must be a tensor mapping")
        current = self.state_dict()
        expected = {name for name in current if name.startswith("inner.") and not name.endswith((".u", ".v"))}
        mapped = {}
        for name, value in native_state.items():
            target = name
            for index in range(self.config.L_layers):
                prefix = f"inner.L_level.layers.{index}.mlp.down_proj."
                if target.startswith(prefix):
                    target = prefix + "base." + target[len(prefix):]
                    break
            mapped[target] = value
        if set(mapped) != expected:
            raise ValueError(f"Native state keys differ: missing={sorted(expected-set(mapped))}, unexpected={sorted(set(mapped)-expected)}")
        for name, value in mapped.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"Native state {name} is not a tensor")
            if value.shape != current[name].shape or value.dtype != current[name].dtype:
                raise ValueError(f"Native state shape/dtype mismatch for {name}")
        with torch.no_grad():
            for name, value in mapped.items():
                current[name].copy_(value)
        return {"loaded_keys": sorted(mapped), "loaded_tensors": len(mapped), "controller_unchanged": True}
