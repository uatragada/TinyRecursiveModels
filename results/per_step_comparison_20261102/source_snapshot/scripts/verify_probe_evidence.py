"""Check saved A-donor probe protocol and metrics without importing Torch.

Tensor equality is established separately by the tensor verifier. This script
checks recorded metadata consistency, source/data/file integrity, exposure,
fixed endpoint coverage and paired gains without reconstructing model tensors.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from verify_local_evidence import ROOT, read, sha


def verify(run_dir):
    directory = Path(run_dir).resolve()
    cfg, summary = read(directory / "config.json"), read(directory / "summary.json")
    source = read(ROOT / cfg["source_run"] / "summary.json")
    source_cfg = read(ROOT / cfg["source_run"] / "config.json")
    checks, checkpoint_files, gains = {}, {}, {}

    def check(name, condition):
        checks[name] = bool(condition)

    variants = ("native", "static_joint", "joint")
    requested_steps = cfg.get("checkpoint_steps", [])
    valid_steps = (isinstance(requested_steps, list) and all(isinstance(step, int) and not isinstance(step, bool) and
                   0 <= step <= cfg["steps"] for step in requested_steps))
    check("checkpoint_steps_predeclared_valid", valid_steps and len(set(requested_steps)) == len(requested_steps))
    if not valid_steps:
        raise ValueError("Checkpoint steps must be a list of integers inside the fixed training range")
    checkpoint_steps = sorted({0, cfg["steps"], *requested_steps})
    fixed_act = summary.get("probe_ACT_training_policy") == "fixed maximum steps; early halting disabled"
    if "fixed_act_training" in cfg:
        check("ACT_policy_matches_configuration", isinstance(cfg["fixed_act_training"], bool) and fixed_act == cfg["fixed_act_training"])
    if "evaluation_checkpoint_steps" in summary:
        check("evaluation_checkpoint_policy_matches_configuration", summary["evaluation_checkpoint_steps"] == checkpoint_steps)
    check("probe_complete", summary["status"] == "complete")
    check("all_probe_training_complete", summary["training_status"] == "complete")
    check("source_run_complete", source["status"] == source["training_status"] == "complete")
    check("configuration_record_consistent", summary["config"] == cfg)
    check("fresh_seed_consistent", summary["seed"] == cfg["seed"])
    check("wall_time_within_bound", summary["elapsed_seconds"] <= cfg["max_seconds"] + 5)
    check("architecture_preserved_except_batch_size", all(value == summary["model_config"].get(key)
          for key, value in source["model_config"].items() if key != "batch_size"))
    check("adaptation_excludes_source_validation_endpoints", cfg["adaptation_val_offset"] >= source_cfg["eval_size"])
    check("same_source_data_collection", cfg["data_dir"] == source_cfg["data_dir"])
    manifest = read(directory / "source_manifest.json")
    for name, expected in manifest.items():
        check(f"source_snapshot:{name}", sha(directory / "source_snapshot" / name) == expected)

    data_root = ROOT / cfg["data_dir"]
    data_manifest = read(data_root / "manifest.json")
    pools = {}
    for split in ("train", "val", "test"):
        check(f"data_checksum:{split}", sha(data_root / f"{split}.npz") == data_manifest["file_sha256"][f"{split}.npz"])
        with np.load(data_root / f"{split}.npz", allow_pickle=False) as arrays:
            pools[split] = {key: arrays[key].copy() for key in arrays.files}
    adaptation = {key: value[cfg["adaptation_val_offset"]:] for key, value in pools["val"].items()}
    endpoint = {key: value[:cfg["eval_size"]] for key, value in pools["test"].items()}
    check("adaptation_pool_record", summary["adaptation_pool"] == {
        "source": "val", "offset": cfg["adaptation_val_offset"], "n": len(adaptation["inputs"]),
        "base_ids_sha256": hashlib.sha256(adaptation["base_ids"].tobytes()).hexdigest()})
    check("endpoint_pool_record", summary["endpoint_pool"] == {
        "source": "test", "offset": 0, "n": len(endpoint["inputs"]),
        "base_ids_sha256": hashlib.sha256(endpoint["base_ids"].tobytes()).hexdigest()})
    check("adaptation_source_base_ids_disjoint", not set(adaptation["base_ids"].tolist()) & set(pools["train"]["base_ids"].tolist()))
    check("adaptation_endpoint_base_ids_disjoint", not set(adaptation["base_ids"].tolist()) & set(endpoint["base_ids"].tolist()))

    stages = summary["stages"]
    check("three_training_arms_present", len(stages) == 3 and {row["variant"] for row in stages} == set(variants))
    check("recorded_common_initial_interface", summary["matched_initial_non_A"] and
          all(row["initial_non_A_sha256"] == row["common_fresh_non_A_sha256"] == summary["common_fresh_non_A_sha256"] for row in stages))
    check("candidate_streams_identical", len({row["candidate_schedule_sha256"] for row in stages}) == 1)
    check("candidate_match_flag_consistent", summary["matched_candidate_streams"] ==
          (len({row["candidate_schedule_sha256"] for row in stages}) == 1))
    check("actual_admission_match_flag_consistent", summary["matched_actual_admission_streams"] ==
          (len({row["admitted_schedule_sha256"] for row in stages}) == 1))
    history = [json.loads(line) for line in (directory / "history.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    paths = [ROOT / summary["common_initial_checkpoint"]]
    for row in stages:
        name, steps, batch = row["variant"], row["steps_executed"], cfg["batch_size"]
        if "seed" in row:
            check(f"{name}:seed_consistent", row["seed"] == cfg["seed"])
        source_stage = next(stage for stage in source["stages"] if stage["variant"] == name)
        check(f"{name}:donor_file_identity", Path(row["donor_checkpoint"]) == Path(source_stage["checkpoint"]) and
              sha(ROOT / row["donor_checkpoint"]) == row["donor_checkpoint_sha256"])
        check(f"{name}:completed_equal_steps", row["complete"] and steps == row["steps_requested"] == cfg["steps"])
        check(f"{name}:B_and_UV_only_parameter_count", row["trainable_parameters"] == 809736 + 16384)
        check(f"{name}:recorded_A_frozen", row["A_unchanged"] and row["A_sha256_before"] == row["A_sha256_after"] == row["donor_A_sha256"])
        check(f"{name}:recorded_native_buffers_frozen", row["native_buffers_unchanged"] and row["puzzle_embedding_unchanged"] and
              row["native_buffers_sha256_before"] == row["native_buffers_sha256_after"])
        check(f"{name}:ACT_slot_accounting", row["act_slot_presentations"] == steps * batch)
        check(f"{name}:ACT_admission_bounds", math.ceil(steps / 16) * batch <= row["new_puzzle_admissions"] <= steps * batch)
        if fixed_act:
            check(f"{name}:fixed_ACT_exact_admissions", row["new_puzzle_admissions"] == math.ceil(steps / 16) * batch)
        check(f"{name}:base_coverage_bounds", 0 < row["unique_base_puzzles_seen"] <= min(row["new_puzzle_admissions"], len(adaptation["inputs"])))
        check(f"{name}:augmented_coverage_bounds", 0 < row["unique_augmented_puzzles_seen"] <= row["new_puzzle_admissions"])
        check(f"{name}:fixed_final_selection", row["selection"] == "fixed final step; no validation checkpoint selection")
        logs = [item for item in history if item["variant"] == name]
        check(f"{name}:logged_seed_consistent", all(item.get("seed", cfg["seed"]) == cfg["seed"] for item in logs))
        check(f"{name}:final_training_log", bool(logs) and logs[-1]["step"] == steps and
              logs[-1]["new_puzzle_admissions"] == row["new_puzzle_admissions"])
        check(f"{name}:finite_training_loss", all(math.isfinite(item["train_loss"]) and item["train_loss"] >= 0 for item in logs))
        checkpoints = row.get("checkpoint_paths")
        if checkpoints is not None:
            check(f"{name}:all_checkpoint_paths_recorded", set(checkpoints) == {str(step) for step in checkpoint_steps})
            check(f"{name}:distinct_checkpoint_paths", len(set(checkpoints.values())) == len(checkpoints))
            check(f"{name}:initial_final_paths_consistent", checkpoints.get("0") == row["initial_checkpoint"] and
                  checkpoints.get(str(cfg["steps"])) == row["checkpoint"])
            paths += [ROOT / path for path in checkpoints.values()]
        else:
            check(f"{name}:intermediate_paths_unneeded", len(checkpoint_steps) == 2)
            paths += [ROOT / row["initial_checkpoint"], ROOT / row["checkpoint"]]

    evaluations = summary["evaluations"]
    actual = [(row["variant"], row["step"]) for row in evaluations]
    check("all_predeclared_endpoints_present", len(actual) == len(variants) * len(checkpoint_steps) and set(actual) ==
          {(variant, step) for variant in variants for step in checkpoint_steps})
    if fixed_act:
        check("fixed_ACT_actual_admission_streams_match", summary["matched_actual_admission_streams"])
    check("saved_evaluation_file_consistent", read(directory / "evaluations.json")["rows"] == evaluations)
    given = endpoint["inputs"] != 1
    n, givens, blanks = len(endpoint["inputs"]), int(given.sum()), int((~given).sum())
    for row in evaluations:
        label = f"{row['variant']}/step{row['step']}"
        if "seed" in row:
            check(f"{label}:seed_consistent", row["seed"] == cfg["seed"])
        check(f"{label}:raw_dynamic_fixed_test_panel", row["weights"] == "raw" and row["mode"] == "dynamic" and row["split"] == "test" and row["n"] == n)
        check(f"{label}:native_ACT16", row["act_steps"] == 16)
        check(f"{label}:given_blank_counts", row["given_cells"] == givens and row["blank_cells"] == blanks)
        check(f"{label}:copy_baseline", math.isclose(row["clue_copy_accuracy"], givens / (n * 81), abs_tol=1e-12))
        weighted = (row["given_agreement"] * givens + row["blank_accuracy"] * blanks) / (n * 81)
        check(f"{label}:weighted_all_cell_identity", math.isclose(row["all_accuracy"], weighted, abs_tol=1e-12))
        check(f"{label}:metric_ranges", all(0 <= row[key] <= 1 for key in (
            "blank_accuracy", "all_accuracy", "given_agreement", "exact_accuracy", "valid_grid_fraction")))
        check(f"{label}:exact_implies_valid_grid", row["exact_accuracy"] <= row["valid_grid_fraction"] + 1e-12)
        check(f"{label}:finite_endpoint_loss", math.isfinite(row["loss"]) and row["loss"] >= 0)
        check(f"{label}:inference_frozen_and_cleared", row["parameter_state_unchanged"] and row["runtime_gates_cleared"])
    for variant in variants:
        initial = next(row for row in evaluations if row["variant"] == variant and row["step"] == 0)
        final = next(row for row in evaluations if row["variant"] == variant and row["step"] == cfg["steps"])
        curve = sorted((row for row in evaluations if row["variant"] == variant), key=lambda row: row["step"])
        duration = cfg["steps"]
        accuracy_auc = sum((right["step"] - left["step"]) *
                           ((left["blank_accuracy"] - initial["blank_accuracy"]) +
                            (right["blank_accuracy"] - initial["blank_accuracy"])) / 2
                           for left, right in zip(curve, curve[1:])) / duration
        loss_auc = sum((right["step"] - left["step"]) *
                       ((initial["loss"] - left["loss"]) + (initial["loss"] - right["loss"])) / 2
                       for left, right in zip(curve, curve[1:])) / duration
        gains[variant] = {"initial_blank_accuracy": initial["blank_accuracy"], "final_blank_accuracy": final["blank_accuracy"],
                          "blank_accuracy_gain_pp": 100 * (final["blank_accuracy"] - initial["blank_accuracy"]),
                          "initial_loss": initial["loss"], "final_loss": final["loss"],
                          "loss_decrease": initial["loss"] - final["loss"],
                          "observed_checkpoint_steps": [row["step"] for row in curve],
                          "blank_gain_auc_pp": 100 * accuracy_auc, "loss_decrease_auc": loss_auc}
    paired = {}
    for comparator in ("native", "static_joint"):
        paired[comparator] = {
            "joint_final_blank_advantage_pp": 100 * (gains["joint"]["final_blank_accuracy"] - gains[comparator]["final_blank_accuracy"]),
            "joint_extra_blank_accuracy_gain_pp": gains["joint"]["blank_accuracy_gain_pp"] - gains[comparator]["blank_accuracy_gain_pp"],
            "joint_extra_loss_decrease": gains["joint"]["loss_decrease"] - gains[comparator]["loss_decrease"],
            "joint_extra_blank_gain_auc_pp": gains["joint"]["blank_gain_auc_pp"] - gains[comparator]["blank_gain_auc_pp"],
            "joint_extra_loss_decrease_auc": gains["joint"]["loss_decrease_auc"] - gains[comparator]["loss_decrease_auc"]}
    for path in dict.fromkeys(paths):
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        with zipfile.ZipFile(path) as archive:
            check(f"checkpoint_ZIP_CRC:{relative}", archive.testzip() is None)
        checkpoint_files[relative] = {"sha256": sha(path), "bytes": path.stat().st_size}
    result = {"passed": all(checks.values()), "checks": checks,
              "scope": "No-Torch source/data/file-integrity, recorded protocol/accounting consistency, metric identities and paired gains. AUC linearly interpolates the predeclared saved checkpoints, with no extrapolation. Actual tensor equality and A-only transplants require the separate tensor verifier; actual admission decisions have no independent per-step ledger.",
              "gains": gains, "paired_joint_comparisons": paired, "checkpoint_file_integrity": checkpoint_files,
              "limitations": "One source seed and one fresh interface seed per run; within-task held-out-puzzle adaptation trains B and U/V. Two endpoints provide only endpoint interpolation; additional predeclared checkpoints provide a sparse adaptation curve."}
    (directory / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/local_sudoku_probe")
    args = parser.parse_args()
    directory = Path(args.run_dir)
    result = verify(directory if directory.is_absolute() else ROOT / directory)
    failures = [name for name, passed in result["checks"].items() if not passed]
    print(json.dumps({"passed": result["passed"], "checks": len(result["checks"]), "failures": failures,
                      "gains": result["gains"], "paired_joint_comparisons": result["paired_joint_comparisons"]}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
