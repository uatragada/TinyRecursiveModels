"""Runtime audit of weights actually installed before each native reasoning call."""
import math

import torch


class StepWeightTrace:
    def __init__(self, model):
        if model.weight_update_cadence != "per_reasoning_step":
            raise ValueError("The every-step trace requires per_reasoning_step cadence")
        self.model, self.rows = model, []
        self.previous_a, self.previous_c = None, None

    def observe(self, _module, _args):
        model = self.model
        matrices = [port.delta for port in model.ports]
        if any(value is None for value in matrices):
            raise AssertionError("A reasoning call has no generated weight update")
        c_matrix = model.last_c_action
        if c_matrix is None or model.b_write_port.delta is not None:
            raise AssertionError("C must write B before B writes A, then clear its port")
        row = {
            "act_step": int(model.phase[0]) + 1, "reasoning_call": model.call,
            "b_updates_this_act": model.weight_updates_this_act,
            "c_updates_this_act": model.meta_weight_updates_this_act,
            "a_shapes": [list(value.shape) for value in matrices],
            "c_shape": list(c_matrix.shape),
            "a_frobenius_ratios": [float(value.flatten(1).norm(dim=1).max()
                / port.base.weight.detach().float().norm()) for value, port in zip(matrices, model.ports)],
            "c_frobenius_ratio": float(c_matrix.flatten(1).norm(dim=1).max()
                / model.b_write_port.base.weight.detach().float().norm()),
            "a_max_changes": None if self.previous_a is None else [
                float((value - previous).abs().max()) for value, previous in zip(matrices, self.previous_a)],
            "c_max_change": None if self.previous_c is None else float((c_matrix - self.previous_c).abs().max()),
            "c_attention_shape": list(model.meta_controller.last_attention.shape),
        }
        self.rows.append(row)
        self.previous_a = [value.detach().clone() for value in matrices]
        self.previous_c = c_matrix.detach().clone()

    def finish(self):
        model = self.model
        expected = [(act + 1, call + 1) for act in range(model.config.halt_max_steps)
                    for call in range(model.reasoning_calls_per_act)]
        if [(row["act_step"], row["reasoning_call"]) for row in self.rows] != expected:
            raise AssertionError("The trace does not cover every step of one complete puzzle batch")
        for row in self.rows:
            if row["b_updates_this_act"] != row["reasoning_call"] or row["c_updates_this_act"] != row["reasoning_call"]:
                raise AssertionError("A controller skipped a reasoning call")
            if any(not math.isfinite(value) or value > model.generator.relative_cap + 1e-6
                   for value in row["a_frobenius_ratios"]):
                raise AssertionError("A generated A matrix violates its bound")
            if not math.isfinite(row["c_frobenius_ratio"]) or row["c_frobenius_ratio"] > model.meta_generator.relative_cap + 1e-6:
                raise AssertionError("A generated B matrix violates its bound")
            if row["a_max_changes"] is not None and any(value <= 0 for value in row["a_max_changes"]):
                raise AssertionError("A sampled dynamic A matrix did not change between steps")
            if row["c_max_change"] is not None and row["c_max_change"] <= 0:
                raise AssertionError("C's sampled dynamic matrix did not change between steps")
        return {"status": "passed", "cadence": model.weight_update_cadence,
            "reasoning_steps": len(self.rows), "batch_size": self.rows[0]["a_shapes"][0][0],
            "scope": "Fresh dynamic initial-evaluation weights installed before every reasoning call of one full puzzle batch. Training-gradient coverage is tested separately.",
            "rows": self.rows}
