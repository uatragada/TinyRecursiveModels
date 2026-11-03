"""Bounded, read-only development diagnostics for the trained controller.

Uses a small validation prefix, never trains/selects a checkpoint, and changes
only temporary inference modes/scales. This is exploratory mechanism evidence,
not a new benchmark comparison or a test-set hyperparameter search.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from local_lab.settings import seed_all, write_json
import numpy as np
import torch
import torch.nn.functional as F
from local_lab.data import validate_sudoku
from local_lab.train import state_digest, state_snapshot, metrics_from_predictions
from models.losses import stablemax_cross_entropy
from models.recursive_reasoning.trm_plastic import PlasticTinyRecursiveReasoningModel_ACTV1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=32)
    parser.add_argument("--max-seconds", type=float, default=180)
    parser.add_argument("--output", default="results/controller_diagnostic_20261029")
    args = parser.parse_args()
    if not 2 <= args.n <= 64 or args.max_seconds <= 0:
        raise ValueError("Use 2..64 validation puzzles and a positive time bound")
    output = ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    started = time.perf_counter()
    seed_all(17)
    device = "cuda" if torch.cuda.is_available() and torch.cuda.mem_get_info()[0] > 2 * 1024**3 else "cpu"
    checkpoint = ROOT / "checkpoints/longer_sudoku/joint.pt"
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    config = dict(payload["model_config"], batch_size=args.n)
    with torch.device(device):
        core = PlasticTinyRecursiveReasoningModel_ACTV1(config)
    core.load_state_dict(payload["state_dict"], strict=True)
    core.eval()
    before = state_digest(state_snapshot(core))
    data_dir = ROOT / "data/local-expanded-sudoku"
    manifest = json.loads((data_dir / "manifest.json").read_text())
    data_file = data_dir / "val.npz"
    if hashlib.sha256(data_file.read_bytes()).hexdigest() != manifest["file_sha256"]["val.npz"]:
        raise ValueError("Validation cache changed")
    with np.load(data_file) as data:
        all_inputs, all_labels = data["inputs"][:2 * args.n], data["labels"][:2 * args.n]
        all_ids = data["base_ids"][:2 * args.n]
    inputs, labels, ids = all_inputs[:args.n], all_labels[:args.n], all_ids[:args.n]
    for question, answer in zip(all_inputs, all_labels):
        validate_sudoku("".join(str(int(x) - 1) for x in question), "".join(str(int(x) - 1) for x in answer))
    batch = {"inputs": torch.as_tensor(inputs, dtype=torch.long, device=device),
             "labels": torch.as_tensor(labels, dtype=torch.long, device=device),
             "puzzle_identifiers": torch.zeros(args.n, dtype=torch.long, device=device)}
    holdout_batch = {"inputs": torch.as_tensor(all_inputs[args.n:], dtype=torch.long, device=device),
                     "labels": torch.as_tensor(all_labels[args.n:], dtype=torch.long, device=device),
                     "puzzle_identifiers": batch["puzzle_identifiers"]}
    observation = {"active": False, "rows": []}
    bases, hooks = {}, []
    for index, adapter in enumerate(core.adapters):
        def base_hook(module, arguments, value, index=index):
            if observation["active"]:
                bases[index] = value
        def adapter_hook(module, arguments, value, index=index):
            if not observation["active"] or module.gate is None:
                return
            x, base = arguments[0], bases.pop(index)
            u = F.normalize(module.u, dim=0).to(x.dtype)
            v = F.normalize(module.v, dim=0).to(x.dtype)
            gate = module.gate.to(x.dtype).unsqueeze(-2)
            delta = module.alpha * F.linear(F.linear(x, v.T) * gate, u)
            unchanged, nonzero = value == base, delta != 0
            actual_delta = value.float() - base.float()
            observation["rows"].append(torch.stack((delta.float().square().sum(),
                base.float().square().sum(), actual_delta.square().sum(),
                unchanged.sum(), (unchanged & nonzero).sum(), nonzero.sum(),
                value.new_tensor(value.numel(), dtype=torch.float32))).detach())
        hooks.append(adapter.base.register_forward_hook(base_hook))
        hooks.append(adapter.register_forward_hook(adapter_hook))
    original_forward = core.controller.forward
    original_alphas = [adapter.alpha for adapter in core.adapters]
    def shuffled_forward(*arguments, **kwargs):
        gates, state, history = original_forward(*arguments, **kwargs)
        return gates.roll(1, dims=0), state, history
    phase_table, phase_call = None, 0
    def phase_forward(*arguments, **kwargs):
        nonlocal phase_call
        _, state, history = original_forward(*arguments, **kwargs)
        gates = phase_table[phase_call % 21].reshape(1, -1).expand(args.n, -1)
        phase_call += 1
        return gates, state, history
    result = {"scope": "Exploratory read-only validation-prefix mechanism diagnostic; no training or checkpoint selection. Small-panel scores are not evidence of a general accuracy advantage.",
              "checkpoint": str(checkpoint.relative_to(ROOT)),
              "checkpoint_file_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              "data_file_sha256": manifest["file_sha256"]["val.npz"], "source_indices": ids.tolist(),
              "split": "val", "n": args.n, "device": device, "rows": [],
              "phase_replay_holdout_source_indices": all_ids[args.n:].tolist(),
              "controller_parameters": core.controller_parameters,
              "conditions_requested": ["off", "dynamic", "shuffle_gates", "scale_2", "scale_4",
                                       "off_holdout", "dynamic_holdout", "phase_replay_holdout"]}
    try:
        with torch.no_grad():
            for name, mode, scale in (("off", "off", 1), ("dynamic", "dynamic", 1),
                                      ("shuffle_gates", "dynamic", 1),
                                      ("scale_2", "dynamic", 2), ("scale_4", "dynamic", 4),
                                      ("off_holdout", "off", 1), ("dynamic_holdout", "dynamic", 1),
                                      ("phase_replay_holdout", "dynamic", 1)):
                if time.perf_counter() - started >= args.max_seconds:
                    break
                core.set_mode(mode)
                core.controller.forward = phase_forward if name == "phase_replay_holdout" else shuffled_forward if name == "shuffle_gates" else original_forward
                current_batch = holdout_batch if name.endswith("_holdout") else batch
                current_inputs = all_inputs[args.n:] if name.endswith("_holdout") else inputs
                current_labels = all_labels[args.n:] if name.endswith("_holdout") else labels
                for adapter, alpha in zip(core.adapters, original_alphas):
                    adapter.alpha = alpha * scale
                observation["active"] = name == "dynamic"
                gates = []
                condition_started = time.perf_counter()
                with torch.device(device):
                    carry = core.initial_carry(current_batch)
                for _ in range(core.config.halt_max_steps):
                    carry, outputs = core(carry, current_batch)
                    if name == "dynamic":
                        gates.extend(item["gate"] for item in core.last_trace["calls"])
                predictions = outputs["logits"].argmax(-1).cpu().numpy()
                row = {"condition": name, "scale": scale,
                       **metrics_from_predictions(predictions, current_inputs, current_labels),
                       "loss": float(stablemax_cross_entropy(outputs["logits"], current_batch["labels"]).mean()),
                       "seconds": time.perf_counter() - condition_started,
                       "parameter_state_unchanged": before == state_digest(state_snapshot(core)),
                       "runtime_gates_cleared": all(adapter.gate is None for adapter in core.adapters),
                       "predictions": predictions.tolist()}
                if not row["parameter_state_unchanged"] or not row["runtime_gates_cleared"]:
                    raise AssertionError("Inference mutated persistent state or left active gates")
                result["rows"].append(row)
                if gates:
                    g = torch.stack(gates).float()
                    g_by_phase = g.reshape(16, 21, args.n, 2, 4)
                    phase_table = g_by_phase.mean((0, 2))
                    total_variance = (g - g.mean((0, 1), keepdim=True)).square().mean()
                    phase_residual = (g_by_phase - phase_table[None, :, None]).square().mean()
                    stats = torch.stack(observation["rows"]).sum(0).cpu().tolist()
                    result["gate_statistics"] = {
                        "shape_calls_puzzles_layers_rank": list(g.shape),
                        "mean_absolute": float(g.abs().mean()), "rms": float(g.square().mean().sqrt()),
                        "mean_std_between_puzzles": float(g.std(dim=1, unbiased=False).mean()),
                        "mean_std_across_calls": float(g.std(dim=0, unbiased=False).mean()),
                        "fraction_abs_above_0_95": float((g.abs() > .95).float().mean()),
                        "fraction_variance_explained_by_fixed_call_phase": float(1 - phase_residual / total_variance.clamp_min(1e-20)),
                        "between_puzzle_variance_fraction": float((g - g.mean(1, keepdim=True)).square().mean() / total_variance.clamp_min(1e-20)),
                        "rms_by_call_within_act": g.reshape(16, 21, args.n, 2, 4).square().mean((0, 2, 3, 4)).sqrt().cpu().tolist()}
                    result["adapter_statistics"] = {
                        "preaddition_delta_to_base_rms_ratio": (stats[0] / stats[1]) ** .5,
                        "actual_delta_to_base_rms_ratio": (stats[2] / stats[1]) ** .5,
                        "unchanged_output_element_fraction": stats[3] / stats[6],
                        "fraction_nonzero_delta_erased_by_addition_rounding": stats[4] / stats[5],
                        "projection_output_elements_observed": int(stats[6])}
                print(json.dumps({key: value for key, value in row.items() if key != "predictions"}), flush=True)
    finally:
        core.controller.forward = original_forward
        for adapter, alpha in zip(core.adapters, original_alphas):
            adapter.alpha = alpha
        for hook in hooks:
            hook.remove()
    result["status"] = "complete" if len(result["rows"]) == len(result["conditions_requested"]) else "partial_budget"
    result["phase_replay_calls"] = phase_call
    result["elapsed_seconds"] = time.perf_counter() - started
    result["max_seconds"] = args.max_seconds
    result["parameter_state_unchanged"] = before == state_digest(state_snapshot(core))
    output.mkdir(parents=True)
    (output / "source_snapshot.py").write_bytes(Path(__file__).read_bytes())
    write_json(output / "diagnostics.json", result)
    write_json(output / "source_manifest.json", {str(Path(__file__).relative_to(ROOT)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    print(json.dumps({key: value for key, value in result.items() if key not in ("rows", "gate_statistics", "adapter_statistics")}), flush=True)


if __name__ == "__main__":
    main()
