"""Active parameter-matched B-only control for the C cross-attention study."""
import torch
from torch import nn

from models.recursive_reasoning.trm_dense import DensePlasticTRM


class LargerDenseB(DensePlasticTRM):
    """Spend C's parameter budget on B's context computation instead.

    With width 64, expansion 224 and 16 phases this adds exactly 29,168
    parameters, matching the structured C. Every added component is in the
    computation path; no unused parameter padding is introduced.
    """
    def __init__(self, config_dict):
        super().__init__(config_dict)
        width = self.controller.width
        expanded = config_dict.get("comparison_expanded_width", 224)
        self.extra_context = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, expanded),
            nn.GELU(), nn.Linear(expanded, width), nn.PReLU(width))
        nn.init.normal_(self.extra_context[3].weight, std=.001)
        nn.init.zeros_(self.extra_context[3].bias)
        self.extra_phase = nn.Embedding(self.config.halt_max_steps, 1)
        nn.init.zeros_(self.extra_phase.weight)
        self.generator.register_forward_pre_hook(self._expand_context)

    @property
    def b_parameters(self):
        parameters = super().b_parameters
        if hasattr(self, "extra_context"):
            parameters += list(self.extra_context.parameters()) + list(self.extra_phase.parameters())
        return parameters

    def _expand_context(self, _module, args):
        context, weights = args
        scale = .1 * (1 + self.extra_phase(self.phase).tanh())
        return context + scale * self.extra_context(context), weights

    def load_dense_state(self, state):
        current = self.state_dict()
        expected = {key for key in current if not key.startswith(("extra_context.", "extra_phase."))}
        if set(state) != expected:
            raise ValueError("Common A/B state does not match the larger-B control")
        for key, value in state.items():
            if value.shape != current[key].shape or value.dtype != current[key].dtype:
                raise ValueError(f"Common A/B shape/dtype mismatch: {key}")
        with torch.no_grad():
            for key, value in state.items():
                current[key].copy_(value)
