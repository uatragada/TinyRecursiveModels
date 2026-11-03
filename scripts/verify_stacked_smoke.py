"""Audit saved stacked-smoke tensors and metrics without rerunning inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_lab.data import load_data


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(state):
    result = hashlib.sha256()
    for name, value in sorted(state.items()):
        result.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def verify(directory):
    torch.set_num_threads(4)
    cfg = json.loads((directory / "config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    saved = torch.load(directory / "dense_smoke.pt", map_location="cpu", weights_only=True)
    source_path = ROOT / cfg["source_checkpoint"]
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    state, donor = saved["state_dict"], source["state_dict"]
    a = {k: v for k, v in state.items() if k.startswith("inner.")}
    expected_a = {k: v for k, v in donor.items() if k.startswith("inner.")}
    b = {k.replace("generator.decoder.2.base.", "generator.decoder.2."): v
         for k, v in state.items() if k.startswith(("controller.", "generator."))}
    expected_b = {k: v for k, v in donor.items() if k.startswith(("controller.", "generator."))}
    c = {k: v for k, v in state.items() if k.startswith(("meta_controller.", "meta_generator."))}
    initial_c = saved["initial_c_state"]
    checks = {
        "completed_hierarchy_smoke": summary["status"] == "complete" and summary["hierarchy"],
        "source_checkpoint_identity": sha(source_path) == summary["source_checkpoint_sha256"],
        "all_a_tensors_equal_source": set(a) == set(expected_a) and all(torch.equal(v, expected_a[k]) for k, v in a.items()),
        "initial_b_identity": digest(expected_b) == summary["b_before_sha256"],
        "final_b_identity": digest(b) == summary["b_after_sha256"],
        "b_parameters_changed": set(b) == set(expected_b) and any(not torch.equal(v, expected_b[k]) for k, v in b.items()),
        "initial_c_identity": digest(initial_c) == summary["c_before_sha256"],
        "final_c_identity": digest(c) == summary["c_after_sha256"],
        "c_parameters_changed": set(c) == set(initial_c) and any(not torch.equal(v, initial_c[k]) for k, v in c.items()),
        "no_low_rank_factors": all(not k.endswith((".u", ".v")) for k in state),
        "no_runtime_updates_saved": all(not k.endswith(".delta") for k in state),
        "all_saved_tensors_finite": all(torch.isfinite(v).all().item() for v in state.values()),
        "b_parameter_count": sum(v.numel() for v in b.values()) == summary["b_parameters"],
        "c_parameter_count": sum(v.numel() for v in c.values()) == summary["c_parameters"],
        "bounded_runtime": summary["elapsed_seconds"] <= cfg["max_seconds"],
        "all_requested_steps": [r["step"] for r in summary["training"]] == list(range(1, cfg["steps"] + 1)),
        "finite_nonzero_gradients_at_both_levels": all(set(r["gradient_norms"]) == {"reader", "generator", "meta_reader", "meta_generator"}
            and all(np.isfinite(v) and v > 0 for v in r["gradient_norms"].values()) for r in summary["training"]),
        "c_off_runtime_parity_recorded": all(summary["evaluations"][name]["standalone_b_logits_bitwise_equal"]
            for name in ("initial_c_off", "c_off")),
        "native_runtime_parity_recorded": all(summary["evaluations"][name]["native_logits_bitwise_equal"]
            for name in ("off", "off_after_training")),
    }
    for i, matrix in enumerate(torch.load(directory / "initial_dense_matrices.pt", map_location="cpu", weights_only=True)):
        weight = expected_a[f"inner.L_level.layers.{i}.mlp.down_proj.base.weight"].float()
        checks[f"a_update_{i}_full_rank"] = int(torch.linalg.matrix_rank(matrix[0])) == min(weight.shape)
        checks[f"a_update_{i}_norm_bound"] = float(matrix.norm()) <= cfg["architecture"]["dense_relative_cap"] * float(weight.norm()) + 1e-6
    matrix = torch.load(directory / "initial_c_matrix.pt", map_location="cpu", weights_only=True)
    weight = expected_b["generator.decoder.2.weight"].float()
    checks["c_update_full_rank"] = int(torch.linalg.matrix_rank(matrix[0])) == min(weight.shape)
    checks["c_update_norm_bound"] = float(matrix.norm()) <= cfg["architecture"]["meta_relative_cap"] * float(weight.norm()) + 1e-6
    attention = summary.get("inter_model_attention")
    if attention:
        read = torch.load(directory / "initial_cross_attention.pt", map_location="cpu", weights_only=True)
        weights = read["weights"]
        checks["cross_attention_shape"] = tuple(weights.shape) == (1, attention["heads"], attention["queries"], attention["b_tokens"])
        checks["cross_attention_finite_nonnegative"] = torch.isfinite(weights).all().item() and (weights >= 0).all().item()
        checks["cross_attention_normalized_per_head"] = torch.allclose(weights.sum(-1), torch.ones_like(weights[..., 0]), atol=1e-6)
        checks["cross_source_token_shape"] = tuple(read["b_tokens"].shape) == (1, attention["b_tokens"], attention["b_token_width"])
        checks["cross_source_metadata_shape"] = read["stage_ids"].shape == read["slot_ids"].shape == (attention["b_tokens"],)
        checks["cross_source_all_stages_present"] = set(read["stage_ids"].tolist()) == set(range(len(attention["stage_names"])))
        checks["all_cross_heads_received_gradients"] = all(len(r["cross_attention_head_gradient_norms"]) == attention["heads"]
            and all(np.isfinite(v) and v > 0 for v in r["cross_attention_head_gradient_norms"]) for r in summary["training"])
    dataset = load_data(ROOT / cfg["data_dir"])["val"]
    start, count = cfg["dev_offset"], cfg["dev_size"]
    training_ids = set(dataset["base_ids"][cfg["train_offset"]:cfg["train_offset"] + cfg["train_size"]])
    checks["training_ids_in_declared_panel"] = set(summary["admitted_training_base_ids"]) <= training_ids
    with np.load(directory / "predictions.npz") as values:
        checks["evaluation_inputs_match_cache"] = np.array_equal(values["inputs"], dataset["inputs"][start:start + count])
        checks["evaluation_labels_match_cache"] = np.array_equal(values["labels"], dataset["labels"][start:start + count])
        checks["evaluation_ids_match_cache"] = np.array_equal(values["base_ids"], dataset["base_ids"][start:start + count])
        checks["training_eval_disjoint"] = not training_ids & set(values["base_ids"])
        variants = ["baseline", "initial", "final", "c_off", "c_shuffle", "c_phase", "c_reset_memory"]
        if attention:
            variants += ["c_read_pooled", "c_read_shuffle"]
        for variant in variants:
            row = summary["evaluations"]["off" if variant == "baseline" else variant]
            correct = values[variant] == values["labels"]
            blank = values["inputs"] == 1
            for metric, value in (("blank_accuracy", correct[blank].mean()),
                                  ("all_accuracy", correct.mean()),
                                  ("given_agreement", correct[~blank].mean()),
                                  ("exact_accuracy", correct.all(1).mean())):
                checks[f"{variant}:{metric}"] = abs(value - row[metric]) < 1e-12
    for name, expected_hash in json.loads((directory / "source_manifest.json").read_text()).items():
        checks[f"source_snapshot:{name}"] = sha(directory / "source_snapshot" / name) == expected_hash
    if summary.get("weight_update_cadence"):
        steps = summary["architecture"]["H_cycles"] * (summary["architecture"]["L_cycles"] + 1)
        scheduled = steps if summary["weight_update_cadence"] == "per_reasoning_step" else 1
        checks["configured_cadence_recorded"] = cfg["architecture"].get("weight_update_cadence", "per_reasoning_step") == summary["weight_update_cadence"]
        checks["generated_entry_count_matches_cadence"] = summary["generated_weights_per_example_per_act"] == scheduled * summary["generated_weights_per_example_per_update"]
        checks["training_b_c_update_counts"] = all(row["b_weight_updates"] == row["c_weight_updates"] == scheduled for row in summary["training"])
        for name, row in summary["evaluations"].items():
            expected_b = 0 if row["mode"] == "off" else scheduled
            expected_c = 0 if row["meta_mode"] == "off" else expected_b
            count = cfg["dev_size"] // cfg["batch_size"] * row["act_steps"]
            checks[f"{name}:b_update_cadence"] = row["b_weight_updates_per_act"] == [expected_b] * count
            checks[f"{name}:c_update_cadence"] = row["c_weight_updates_per_act"] == [expected_c] * count
    if summary.get("step_trace"):
        trace = json.loads((directory / summary["step_trace"]).read_text())
        rows = trace["rows"]
        expected = [(act + 1, call + 1) for act in range(summary["architecture"]["halt_max_steps"]) for call in range(steps)]
        checks["trace_covers_every_reasoning_call"] = [(r["act_step"], r["reasoning_call"]) for r in rows] == expected
        checks["trace_counts_match"] = trace["reasoning_steps"] == len(rows) == len(expected)
        checks["both_levels_write_before_every_call"] = all(r["b_updates_this_act"] == r["c_updates_this_act"] == r["reasoning_call"] for r in rows)
        checks["sampled_a_changes_every_call"] = all(r["a_max_changes"] is None if i == 0 else all(np.isfinite(v) and v > 0 for v in r["a_max_changes"]) for i, r in enumerate(rows))
        checks["sampled_c_changes_every_call"] = all(r["c_max_change"] is None if i == 0 else np.isfinite(r["c_max_change"]) and r["c_max_change"] > 0 for i, r in enumerate(rows))
        checks["trace_a_updates_bounded"] = all(len(r["a_frobenius_ratios"]) == len(summary["dense_matrices"]) and all(np.isfinite(v) and 0 <= v <= cfg["architecture"]["dense_relative_cap"] + 1e-6 for v in r["a_frobenius_ratios"]) for r in rows)
        checks["trace_c_updates_bounded"] = all(np.isfinite(r["c_frobenius_ratio"]) and 0 <= r["c_frobenius_ratio"] <= cfg["architecture"]["meta_relative_cap"] + 1e-6 for r in rows)
    checks = {k: bool(v) for k, v in checks.items()}
    result = {"status": "passed" if all(checks.values()) else "failed", "passed": sum(checks.values()),
        "total": len(checks), "a_tensors_compared": len(a), "checks": checks,
        "scope": "Saved tensor identity, B/C parameter changes, sampled ranks/norms, panel identity, prediction-derived metrics and source hashes. Runtime parity and gradient records checked for consistency; no optimizer or inference replay."}
    (directory / "artifact_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k != "checks"}, indent=2))
    if result["status"] != "passed":
        raise AssertionError([key for key, value in checks.items() if not value])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/dense_stack_smoke_20261030")
    args = parser.parse_args()
    verify(ROOT / args.run)
