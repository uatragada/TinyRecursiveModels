"""Native TRM with Transformer B generating dense, temporary weight matrices.

Every entry of each selected MLP down-projection receives a directly generated
update. Tiles are output chunks of a shared nonlinear hypernetwork, not matrix
factors. No low-rank parameters or legacy controller are constructed here.
"""
from __future__ import annotations

from collections.abc import Mapping
import math

import torch
from torch import nn
import torch.nn.functional as F

from models.losses import log_stablemax
from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1


def _digit_probabilities(logits, inputs):
    probabilities = log_stablemax(logits.float()[..., 2:11]).exp()
    clues = F.one_hot((inputs - 2).clamp(0, 8), 9).float()
    return torch.where((inputs != 1)[..., None], clues, probabilities)


def _constraint_features(probabilities):
    board = probabilities.reshape(-1, 9, 9, 9)
    row = board.sum(2, keepdim=True).expand_as(board) - 1
    column = board.sum(1, keepdim=True).expand_as(board) - 1
    boxes = board.reshape(-1, 3, 3, 3, 3, 9)
    box = boxes.sum((2, 4), keepdim=True).expand_as(boxes).reshape_as(board) - 1
    return torch.cat((row, column, box), -1).reshape(-1, 81, 27)


class DenseWeightGenerator(nn.Module):
    """Generate all entries; no product of thin matrices represents an update.

    tanh bounds each entry by relative_cap * ||W||_F / sqrt(W.numel()),
    implying ||delta W||_F <= relative_cap * ||W||_F in real arithmetic.
    This numerical bound is not a bound on downstream behavior.
    """
    def __init__(self, shapes, width=64, tile_size=32, chunk_tiles=128,
                 decoder_width=32, relative_cap=.03):
        super().__init__()
        dimensions = (width, tile_size, chunk_tiles, decoder_width)
        if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in dimensions):
            raise ValueError("Generator dimensions must be positive integers")
        if not shapes or any(len(s) != 2 or min(s) <= 0 for s in shapes):
            raise ValueError("Projection shapes must be positive matrix shapes")
        if not math.isfinite(relative_cap) or relative_cap < 0:
            raise ValueError("relative_cap must be finite and nonnegative")
        self.shapes = tuple(tuple(s) for s in shapes)
        self.tile_size, self.chunk_tiles = tile_size, chunk_tiles
        self.relative_cap = float(relative_cap)
        self.grids = [(math.ceil(o / tile_size), math.ceil(i / tile_size)) for o, i in shapes]
        self.tile_codes = nn.Embedding(sum(r * c for r, c in self.grids), width)
        nn.init.normal_(self.tile_codes.weight, std=.2)
        self.decoder = nn.Sequential(nn.Linear(width, decoder_width), nn.GELU(),
                                     nn.Linear(decoder_width, tile_size * tile_size))
        nn.init.normal_(self.decoder[-1].weight, std=.02)
        nn.init.zeros_(self.decoder[-1].bias)

    def forward(self, context, weights):
        if len(weights) != len(self.shapes):
            raise ValueError("One base weight is required per projection")
        matrices, first = [], 0
        for shape, (rows, columns), weight in zip(self.shapes, self.grids, weights):
            if tuple(weight.shape) != shape:
                raise ValueError("Base weight has the wrong projection shape")
            count, pieces = rows * columns, []
            for offset in range(0, count, self.chunk_tiles):
                ids = torch.arange(first + offset, first + min(offset + self.chunk_tiles, count),
                                   device=context.device)
                pieces.append(self.decoder(context[:, None] + self.tile_codes(ids)[None]))
            tiles = torch.cat(pieces, 1).reshape(len(context), rows, columns,
                                               self.tile_size, self.tile_size)
            dense = tiles.permute(0, 1, 3, 2, 4).reshape(
                len(context), rows * self.tile_size, columns * self.tile_size)
            dense = dense[:, :shape[0], :shape[1]].tanh()
            scale = weight.detach().float().norm() * self.relative_cap / math.sqrt(math.prod(shape))
            matrices.append(dense * scale)
            first += count
        return matrices


class DenseStateController(nn.Module):
    """Read cell states and constraint residuals; keep per-puzzle GRU memory."""
    def __init__(self, hidden, ports, max_steps, width=64, heads=4, layers=2):
        super().__init__()
        if min(hidden, ports, max_steps, width, heads, layers) <= 0 or width % heads:
            raise ValueError("Positive dimensions and width divisible by heads are required")
        self.width, self.ports = width, ports
        self.project = nn.Linear(3 * hidden + 56 + ports, width)
        self.cell_norm = nn.LayerNorm(width)
        self.row_position, self.column_position, self.box_position = (
            nn.Embedding(9, width) for _ in range(3))
        self.phase = nn.Embedding(max_steps, width)
        for embedding in (self.row_position, self.column_position, self.box_position, self.phase):
            nn.init.normal_(embedding.weight, std=.02)
        self.queries = nn.Parameter(torch.randn(4, width) * .02)
        self.read = nn.MultiheadAttention(width, heads, dropout=0., batch_first=True)
        block = nn.TransformerEncoderLayer(width, heads, 2 * width, dropout=0.,
            activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.memory_cell = nn.GRUCell(width, width)
        self.output_norm = nn.LayerNorm(width)
        self.static_context = nn.Parameter(torch.zeros(width))
        self.memory = None
        self.reasoning_tokens = None
        stages = torch.arange(layers + 1).repeat_interleave(4)
        self.register_buffer("reasoning_stage_ids", torch.cat((stages,
            torch.tensor([layers + 1, layers + 2]))), persistent=False)
        self.register_buffer("reasoning_slot_ids", torch.cat((torch.arange(4).repeat(layers + 1),
            torch.zeros(2, dtype=torch.long))), persistent=False)

    def reset(self, mask):
        self.reasoning_tokens = None
        if self.memory is None or len(self.memory) != len(mask):
            self.memory = self.queries.new_zeros(len(mask), self.width)
        self.memory = torch.where(mask[:, None], 0., self.memory.detach())

    def encode_with_trace(self, sequence):
        """Expose layer outputs without hooks that disable PyTorch fast paths.

        This encoder has no padding/mask or nested-tensor conversion. Its native
        forward is the same ordered layer loop, followed by optional final norm.
        Captured states are observations and retain no gradient into B.
        """
        observed = [sequence.detach()]
        for layer in self.transformer.layers:
            sequence = layer(sequence)
            observed.append(sequence.detach())
        if self.transformer.norm is not None:
            sequence = self.transformer.norm(sequence)
        return sequence, observed

    def forward(self, low, high, previous_high, probabilities, previous_probs,
                previous_norms, inputs, phase, mode="dynamic"):
        if mode == "phase":
            return self.output_norm(self.static_context[None] + self.phase(phase))
        low, high = low.detach().float(), high.detach().float()
        entropy = -(probabilities.clamp_min(1e-8).log() * probabilities).sum(-1, keepdim=True)
        clues = F.one_hot((inputs - 2).clamp(0, 8), 9).float() * (inputs != 1)[..., None]
        features = torch.cat((low, high, high - previous_high, probabilities,
            probabilities - previous_probs, entropy, _constraint_features(probabilities),
            (inputs != 1)[..., None].float(), clues,
            previous_norms[:, None].expand(-1, 81, -1)), -1)
        cells = torch.arange(81, device=inputs.device)
        row, column = cells // 9, cells % 9
        tokens = self.cell_norm(self.project(features) + self.row_position(row)[None]
            + self.column_position(column)[None]
            + self.box_position((row // 3) * 3 + column // 3)[None])
        memory = torch.zeros_like(self.memory) if mode == "reset_memory" else self.memory
        query = self.queries[None].expand(len(inputs), -1, -1) + memory[:, None] + self.phase(phase)[:, None]
        read, _ = self.read(query, tokens, tokens, need_weights=False)
        encoded, observed = self.encode_with_trace(query + read)
        state = encoded.mean(1)
        self.memory = self.memory_cell(state, memory)
        context = self.output_norm(self.memory)
        self.reasoning_tokens = torch.cat((*observed,
            self.memory.detach()[:, None], context.detach()[:, None]), 1)
        return context


class DenseTemporaryProjection(nn.Module):
    """Evaluate x(W + delta W)^T without overwriting W or storing factors."""
    def __init__(self, base):
        super().__init__()
        self.base = base
        self.delta = None

    def forward(self, inputs):
        result = self.base(inputs)
        if self.delta is None:
            return result
        expected = (len(inputs), *self.base.weight.shape)
        if tuple(self.delta.shape) != expected:
            raise ValueError(f"Expected per-example matrix shape {expected}")
        update = torch.bmm(inputs, self.delta.to(inputs.dtype).transpose(1, 2))
        return result + update


def native_a_from_checkpoint(state):
    """Read only native A tensors, dropping inactive historical experiment extras.

    The legacy native-training arm was saved in a wrapper that also allocated
    unused controller/factor tensors. None are imported into the dense model.
    """
    if not isinstance(state, Mapping):
        raise TypeError("A checkpoint state must be a tensor mapping")
    result = {}
    for name, value in state.items():
        if not name.startswith("inner.") or name.endswith((".u", ".v")):
            continue
        result[name.replace(".mlp.down_proj.base.", ".mlp.down_proj.")] = value
    return result


class DensePlasticTRM(TinyRecursiveReasoningModel_ACTV1):
    """Native recursion with fresh dense B weights before each reasoning call.

    Modes: dynamic, off, phase (input-independent context), shuffle, reset_memory.
    A can be frozen or jointly trainable. B observations are detached, and memory
    credit is truncated between ACT updates, matching the native carry boundary.
    The explicit per_act cadence reproduces the historical final-cycle action.
    """
    def __init__(self, config_dict):
        super().__init__(config_dict)
        if self.config.seq_len != 81 or self.config.vocab_size != 11 or not self.config.no_ACT_continue:
            raise ValueError("The dense pilot requires Sudoku and native no_ACT_continue")
        self.weight_update_cadence = config_dict.get("weight_update_cadence", "per_reasoning_step")
        if self.weight_update_cadence not in ("per_reasoning_step", "per_act"):
            raise ValueError("weight_update_cadence must be per_reasoning_step or per_act")
        for layer in self.inner.L_level.layers:
            layer.mlp.down_proj = DenseTemporaryProjection(layer.mlp.down_proj)
        width = config_dict.get("dense_controller_width", 64)
        self.controller = DenseStateController(self.config.hidden_size, self.config.L_layers,
            self.config.halt_max_steps, width=width, heads=config_dict.get("dense_controller_heads", 4),
            layers=config_dict.get("dense_controller_layers", 2))
        self.generator = DenseWeightGenerator([tuple(p.base.weight.shape) for p in self.ports],
            width=width, tile_size=config_dict.get("dense_tile_size", 32),
            chunk_tiles=config_dict.get("dense_chunk_tiles", 128),
            decoder_width=config_dict.get("dense_decoder_width", 32),
            relative_cap=config_dict.get("dense_relative_cap", .03))
        self.mode, self.call, self.action = "dynamic", 0, None
        self.previous_high, self.previous_probs, self.previous_norms = None, None, None
        self.high, self.phase, self.data = None, None, None
        self.last_action = None
        self.reasoning_calls_per_act = self.config.H_cycles * (self.config.L_cycles + 1)
        self.weight_updates_this_act = 0
        self.warmup_calls = (self.config.H_cycles - 1) * (self.config.L_cycles + 1)
        self.inner.register_forward_pre_hook(self._begin_inner)
        self.inner.L_level.register_forward_pre_hook(self._before_reasoning)
        self.inner.L_level.register_forward_hook(self._after_reasoning)
        self.inner.register_forward_hook(self._end_inner, always_call=True)
        self.configure_training("frozen")

    @property
    def ports(self):
        return [layer.mlp.down_proj for layer in self.inner.L_level.layers]

    @property
    def b_parameters(self):
        return list(self.controller.parameters()) + list(self.generator.parameters())

    def set_mode(self, mode):
        if mode not in ("dynamic", "off", "phase", "shuffle", "reset_memory"):
            raise ValueError(f"Unknown dense-controller mode {mode!r}")
        self.mode = mode
        return self

    def configure_training(self, mode):
        if mode not in ("frozen", "joint"):
            raise ValueError("Training mode must be frozen or joint")
        for parameter in self.inner.parameters():
            parameter.requires_grad_(mode == "joint")
        if hasattr(self.inner, "puzzle_emb"):
            self.inner.puzzle_emb.local_weights.requires_grad_(mode == "joint")
        for parameter in self.b_parameters:
            parameter.requires_grad_(True)
        return self

    def a_state(self):
        return {("inner." + name).replace(".mlp.down_proj.base.", ".mlp.down_proj."):
            value.detach().cpu().clone() for name, value in self.inner.state_dict().items()}

    def load_a_state(self, state):
        current = self.inner.state_dict()
        expected = {("inner." + name).replace(".mlp.down_proj.base.", ".mlp.down_proj."):
                    name for name in current}
        if set(state) != set(expected):
            raise ValueError("Native A state keys do not match the dense model")
        for canonical, name in expected.items():
            value = state[canonical]
            if not isinstance(value, torch.Tensor) or value.shape != current[name].shape or value.dtype != current[name].dtype:
                raise ValueError(f"Native A tensor shape/dtype mismatch: {canonical}")
        with torch.no_grad():
            for canonical, name in expected.items():
                current[name].copy_(state[canonical])

    def clear_runtime(self):
        for port in self.ports:
            port.delta = None
        self.action, self.last_action = None, None
        self.controller.memory = None
        self.controller.reasoning_tokens = None
        self.previous_high, self.previous_probs, self.previous_norms = None, None, None
        self.weight_updates_this_act = 0
        self.high, self.phase, self.data = None, None, None

    def _begin_inner(self, _module, args):
        self.call, self.action = 0, None
        self.weight_updates_this_act = 0
        self.high, self.data = args[0].z_H, args[1]
        for port in self.ports:
            port.delta = None

    def _before_reasoning(self, _module, args):
        self.call += 1
        every_step = self.weight_update_cadence == "per_reasoning_step"
        if self.mode != "off" and (every_step or self.call == self.warmup_calls + 1):
            prefix = self.inner.puzzle_emb_len
            # A high-state update receives (z_H, z_L), whereas a low-state
            # update receives (z_L, z_H + input). Always expose actual z_L.
            high_step = self.call % (self.config.L_cycles + 1) == 0
            low_state = args[1] if high_step else args[0]
            high, low = self.high[:, prefix:], low_state[:, prefix:]
            inputs = self.data["inputs"]
            with torch.no_grad():
                logits = self.inner.lm_head(self.high)[:, prefix:]
                probabilities = _digit_probabilities(logits, inputs)
            context = self.controller(low, high, self.previous_high, probabilities,
                self.previous_probs, self.previous_norms, inputs, self.phase,
                mode="dynamic" if self.mode == "shuffle" else self.mode)
            self.action = self.generator(context, [p.base.weight for p in self.ports])
            if self.mode == "shuffle":
                self.action = [value.roll(1, 0) for value in self.action]
            self.weight_updates_this_act += 1
            if every_step:
                # Differences refer to the preceding observation/action, rather
                # than the preceding ACT. These features carry no gradient.
                self.previous_high = high.detach().float()
                self.previous_probs = probabilities.detach()
                self.previous_norms = torch.stack([
                    value.detach().square().mean((1, 2)).sqrt() for value in self.action], -1)
        for index, port in enumerate(self.ports):
            port.delta = None if self.action is None else self.action[index]

    def _after_reasoning(self, _module, _args, output):
        if self.call % (self.config.L_cycles + 1) == 0:
            self.high = output

    def _end_inner(self, _module, _args, _output):
        for port in self.ports:
            port.delta = None

    def forward(self, carry, batch):
        reset = carry.halted
        count, prefix = len(reset), self.inner.puzzle_emb_len
        self.controller.reset(reset)
        device = self.controller.queries.device
        if self.previous_high is None or len(self.previous_high) != count:
            self.previous_high = torch.zeros(count, 81, self.config.hidden_size, device=device)
            self.previous_probs = torch.zeros(count, 81, 9, device=device)
            self.previous_norms = torch.zeros(count, len(self.ports), device=device)
        self.previous_high = torch.where(reset[:, None, None], 0., self.previous_high.detach())
        self.previous_probs = torch.where(reset[:, None, None], 0., self.previous_probs.detach())
        self.previous_norms = torch.where(reset[:, None], 0., self.previous_norms.detach())
        self.phase = torch.where(reset, 0, carry.steps).long().clamp_max(self.config.halt_max_steps - 1)
        try:
            new_carry, outputs = super().forward(carry, batch)
        except Exception:
            self.clear_runtime()
            raise
        if self.weight_update_cadence == "per_act":
            self.previous_high = new_carry.inner_carry.z_H[:, prefix:].float().detach()
            self.previous_probs = _digit_probabilities(outputs["logits"].detach(), new_carry.current_data["inputs"])
        self.last_action = None if self.action is None else [v.detach() for v in self.action]
        self.previous_norms = (torch.zeros_like(self.previous_norms) if self.action is None else
            torch.stack([v.detach().square().mean((1, 2)).sqrt() for v in self.action], -1))
        self.controller.memory = self.controller.memory.detach()
        return new_carry, outputs
