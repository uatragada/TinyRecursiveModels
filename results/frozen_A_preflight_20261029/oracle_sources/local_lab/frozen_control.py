"""Frozen TRM substrate with label-free late-window controllers and gate oracle.

All interventions use the same frozen rank-four directions. Native TRM runs
14 warm-up calls unmodified, then applies one action to the seven differentiable
calls. Controller memory persists in value across ACT updates, with gradients
truncated after each optimizer step and state reset for newly admitted puzzles.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
import torch.nn.functional as F

from models.losses import log_stablemax, stablemax_cross_entropy
from .train import state_snapshot


def task_loss(logits, inputs, labels):
    """Per-puzzle blank loss plus a fixed 0.25 clue-preservation loss."""
    losses = stablemax_cross_entropy(logits, labels).float()
    blank = inputs == 1
    given = ~blank
    return ((losses * blank).sum(-1) / blank.sum(-1).clamp_min(1)
            + .25 * (losses * given).sum(-1) / given.sum(-1).clamp_min(1))


def digit_probabilities(logits, inputs):
    probabilities = log_stablemax(logits.float()[..., 2:11]).exp()
    givens = F.one_hot((inputs - 2).clamp(0, 8), 9).float()
    return torch.where((inputs != 1)[..., None], givens, probabilities)


def constraint_features(probabilities):
    """For each cell/digit, row/column/box expected-count residuals; no labels."""
    b = probabilities.shape[0]
    board = probabilities.reshape(b, 9, 9, 9)
    row = board.sum(2, keepdim=True).expand_as(board) - 1
    column = board.sum(1, keepdim=True).expand_as(board) - 1
    boxes = board.reshape(b, 3, 3, 3, 3, 9)
    box = boxes.sum((2, 4), keepdim=True).expand_as(boxes).reshape_as(board) - 1
    return torch.cat((row, column, box), -1).reshape(b, 81, 27)


class ObservationController(nn.Module):
    def __init__(self, hidden=512, width=96, heads=4, layers=2, rank=4, ports=2, pooled=False):
        super().__init__()
        self.width, self.rank, self.ports, self.pooled = width, rank, ports, pooled
        self.project = nn.Linear(3 * hidden + 64, width)
        self.cell_norm = nn.LayerNorm(width)
        self.row_position = nn.Embedding(9, width)
        self.column_position = nn.Embedding(9, width)
        self.box_position = nn.Embedding(9, width)
        self.phase = nn.Embedding(16, width)
        self.queries = nn.Parameter(torch.randn(4, width) * .02)
        self.read = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        block = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.memory_cell = nn.GRUCell(width, width)
        self.output_norm = nn.LayerNorm(width)
        self.head = nn.Linear(width, ports * rank)
        nn.init.normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)
        self.global_logits = nn.Parameter(torch.zeros(ports, rank))
        self.phase_logits = nn.Parameter(torch.zeros(16, ports, rank))
        self.memory = None

    def reset(self, reset_mask):
        if self.memory is None or self.memory.shape[0] != len(reset_mask):
            self.memory = self.queries.new_zeros(len(reset_mask), self.width)
        self.memory = torch.where(reset_mask[:, None], 0., self.memory.detach())

    def forward(self, low, high, previous_high, previous_probs, previous_action, inputs, phase,
                reset_memory=False):
        # A states are observations, not a route for changing frozen A tensors.
        low, high = low.detach().float(), high.detach().float()
        probabilities = self.probabilities
        entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(-1, keepdim=True)
        clues = F.one_hot((inputs - 2).clamp(0, 8), 9).float() * (inputs != 1)[..., None]
        features = torch.cat((low, high, high - previous_high,
            probabilities, probabilities - previous_probs, entropy,
            constraint_features(probabilities), (inputs != 1)[..., None].float(), clues,
            previous_action.flatten(1)[:, None].expand(-1, 81, -1)), -1)
        # 3H + (9+9+1+27+1+9+8) = 3H+64.
        cells = torch.arange(81, device=inputs.device)
        row, col = cells // 9, cells % 9
        tokens = self.cell_norm(self.project(features) + self.row_position(row)[None]
            + self.column_position(col)[None] + self.box_position((row // 3) * 3 + col // 3)[None])
        if self.pooled:
            tokens = tokens.mean(1, keepdim=True).expand_as(tokens)
        memory = torch.zeros_like(self.memory) if reset_memory else self.memory
        query = self.queries[None].expand(len(inputs), -1, -1) + memory[:, None] + self.phase(phase)[:, None]
        read, _ = self.read(query, tokens, tokens, need_weights=False)
        state = self.transformer(query + read).mean(1)
        self.memory = self.memory_cell(state, memory)
        residual = self.head(self.output_norm(self.memory)).reshape(-1, self.ports, self.rank)
        return torch.tanh(self.global_logits[None] + self.phase_logits[phase] + residual)


class PhaseController(nn.Module):
    def __init__(self, ports=2, rank=4):
        super().__init__()
        self.global_logits = nn.Parameter(torch.zeros(ports, rank))
        self.phase_logits = nn.Parameter(torch.zeros(16, ports, rank))

    def reset(self, reset_mask):
        pass

    def forward(self, *args, **kwargs):
        phase = args[-1]
        return torch.tanh(self.global_logits[None] + self.phase_logits[phase])


class FrozenProjection(nn.Module):
    def __init__(self, old):
        super().__init__()
        self.base, self.u, self.v = old.base, old.u, old.v
        self.alpha, self.rank = old.alpha, old.rank
        self.gate = None
        self.actuation = "weight"
        self.register_buffer("activation_scale", torch.ones(self.rank, device=self.u.device), persistent=False)
        self.calibrating, self.recording, self.late = False, False, False
        self.calibration, self.statistics = [], []

    def forward(self, x):
        base = self.base(x)
        if self.gate is None and not self.calibrating:
            return base
        u = F.normalize(self.u, dim=0).to(x.dtype)
        v = F.normalize(self.v, dim=0).to(x.dtype)
        coefficients = F.linear(x, v.T)
        if self.calibrating and self.late:
            self.calibration.append(torch.stack((coefficients.float().square().sum((0, 1)),
                torch.full((self.rank,), coefficients.shape[0] * coefficients.shape[1], device=x.device))).detach())
        if self.gate is None:
            return base
        # Both actions execute the same projection; the activation control uses a
        # fixed train-calibrated RMS instead of token-dependent xV coefficients.
        used = coefficients if self.actuation == "weight" else self.activation_scale.to(x.dtype)[None, None].expand_as(coefficients)
        delta = self.alpha * F.linear(used * self.gate.to(x.dtype)[:, None], u)
        result = base + delta
        if self.recording:
            self.statistics.append(torch.stack((base.float().square().sum(),
                (result.float() - base.float()).square().sum(),
                result.new_tensor(result.numel(), dtype=torch.float32))).detach())
        return result


@dataclass
class LateState:
    low: torch.Tensor
    high: torch.Tensor
    injection: torch.Tensor
    kwargs: dict


class FrozenControllerModel(nn.Module):
    def __init__(self, core, actor, actuation="weight"):
        super().__init__()
        self.core, self.actor = core, actor
        if core.config.seq_len != 81 or core.config.L_layers != 2 or core.adapter_rank != 4 or core.config.L_cycles != 6 or core.config.H_cycles != 3:
            raise ValueError("This experiment requires native Sudoku H3/L6 with two rank-four ports")
        for layer in core.inner.L_level.layers:
            layer.mlp.down_proj = FrozenProjection(layer.mlp.down_proj)
            layer.mlp.down_proj.actuation = actuation
        for parameter in core.parameters():
            parameter.requires_grad_(False)
        core.eval().set_mode("off")
        core.puzzle_emb.local_weights.requires_grad_(False)
        self.mode, self.override = "normal", None
        self.call, self.high, self.action, self.late_state = 0, None, None, None
        self.previous_high, self.previous_probs, self.previous_action = None, None, None
        core.inner.register_forward_pre_hook(self._begin)
        core.inner.L_level.register_forward_pre_hook(self._before, with_kwargs=True)
        core.inner.L_level.register_forward_hook(self._after)

    @property
    def ports(self):
        return self.core.adapters

    def frozen_state(self):
        return state_snapshot(self.core)

    def frozen_buffers(self):
        return {name: value.detach().cpu().clone() for name, value in self.core.named_buffers()}

    def _begin(self, module, args):
        self.call, self.high, self.action, self.late_state = 0, None, None, None
        self.inputs = args[1]["inputs"]

    def _before(self, module, args, kwargs):
        if self.override is not None:
            for index, port in enumerate(self.ports):
                port.gate = self.override[:, index]
            return
        self.call += 1
        for port in self.ports:
            port.late = self.call >= 15
        if self.call == 15:
            low = args[0] if args else kwargs["hidden_states"]
            injection = args[1] if len(args) > 1 else kwargs["input_injection"]
            self.late_state = LateState(low.detach(), self.high.detach(), injection.detach(), kwargs)
            if self.mode != "off":
                prefix = self.core.inner.puzzle_emb_len
                high = self.high[:, prefix:]
                logits = self.core.inner.lm_head(high).detach()
                if isinstance(self.actor, ObservationController):
                    self.actor.probabilities = digit_probabilities(logits, self.inputs)
                self.action = self.actor(low[:, prefix:], high, self.previous_high,
                    self.previous_probs, self.previous_action, self.inputs, self.phase,
                    reset_memory=self.mode == "reset_memory")
                if self.mode == "phase_only":
                    self.action = torch.tanh(self.actor.global_logits[None] + self.actor.phase_logits[self.phase])
                if self.mode == "shuffle":
                    self.action = self.action.roll(1, 0)
        for index, port in enumerate(self.ports):
            port.gate = None if self.call < 15 or self.action is None else self.action[:, index]

    def _after(self, module, args, output):
        if self.override is None and self.call in (7, 14):
            self.high = output.detach()

    def forward(self, carry, batch):
        reset = carry.halted
        self.actor.reset(reset)
        self.phase = torch.where(reset, 0, carry.steps).long()
        b, h = len(reset), self.core.config.hidden_size
        if self.previous_high is None or len(self.previous_high) != b:
            self.previous_high = self.core.inner.H_init.new_zeros(b, 81, h).float()
            self.previous_probs = self.previous_high.new_zeros(b, 81, 9)
            self.previous_action = self.previous_high.new_zeros(b, 2, 4)
        self.previous_high = torch.where(reset[:, None, None], 0., self.previous_high.detach())
        self.previous_probs = torch.where(reset[:, None, None], 0., self.previous_probs.detach())
        self.previous_action = torch.where(reset[:, None, None], 0., self.previous_action.detach())
        new_carry, outputs = self.core(carry, batch)
        prefix = self.core.inner.puzzle_emb_len
        self.previous_high = new_carry.inner_carry.z_H[:, prefix:].float().detach()
        self.previous_probs = digit_probabilities(outputs["logits"].detach(), new_carry.current_data["inputs"])
        self.previous_action = self.previous_action * 0 if self.action is None else self.action.detach()
        if isinstance(self.actor, ObservationController):
            self.actor.memory = self.actor.memory.detach()
        return new_carry, outputs

    def late_logits(self, state, gates):
        # Oracle gates are optimizer-owned Parameters, not model parameters.
        object.__setattr__(self, "override", gates)
        try:
            low, high = state.low, state.high
            for _ in range(6):
                low = self.core.inner.L_level(low, state.injection, **state.kwargs)
            high = self.core.inner.L_level(high, low, **state.kwargs)
            return self.core.inner.lm_head(high)[:, self.core.inner.puzzle_emb_len:]
        finally:
            self.override = None
            for port in self.ports:
                port.gate = None


def gate_oracle(model, state, inputs, labels, initial=None, steps=20, learning_rate=.15):
    """Privileged projected local search, not a global/episode upper bound.

    Each example retains its best measured gate vector, including zero and the
    supplied candidate. This routine is never called in deployed evaluation.
    """
    zero = torch.zeros(len(inputs), 2, 4, device=inputs.device)
    with torch.no_grad():
        baseline_logits = model.late_logits(state, zero)
        best_loss = task_loss(baseline_logits, inputs, labels)
        zero_loss = best_loss.clone()
        best = zero.clone()
        if initial is not None:
            initial_loss = task_loss(model.late_logits(state, initial.detach()), inputs, labels)
            use = initial_loss < best_loss
            best[use], best_loss = initial.detach()[use], torch.minimum(best_loss, initial_loss)
    gates = nn.Parameter(best.clone())
    optimizer = torch.optim.Adam([gates], lr=learning_rate)
    history = [float(best_loss.mean())]
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        losses = task_loss(model.late_logits(state, gates), inputs, labels)
        losses.mean().backward()
        optimizer.step()
        with torch.no_grad():
            gates.clamp_(-1, 1)
            candidate_loss = task_loss(model.late_logits(state, gates), inputs, labels)
            use = candidate_loss < best_loss
            best[use], best_loss = gates.detach()[use], torch.minimum(best_loss, candidate_loss)
            history.append(float(best_loss.mean()))
    with torch.no_grad():
        logits = model.late_logits(state, best)
        verified = task_loss(logits, inputs, labels)
    if not torch.all(verified <= zero_loss + 1e-5) or not torch.isfinite(verified).all():
        raise AssertionError("Oracle failed finite/zero-candidate safeguard")
    return best.detach(), logits.detach(), baseline_logits.detach(), history
