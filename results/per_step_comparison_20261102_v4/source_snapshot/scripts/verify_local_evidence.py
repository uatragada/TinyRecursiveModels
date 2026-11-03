"""Audit saved native-TRM evidence without importing Torch or loading tensors.

This validates source/data provenance, completed protocol coverage, recorded
accounting and metric identities. Checkpoint ZIP CRC/file hashes establish file
integrity, not independently reconstructed model tensor values. Admissions are
checked against ACT bounds and saved hashes; no per-step admission ledger exists.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import subprocess
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def valid_solution(encoded):
    board = encoded.reshape(9, 9)
    wanted = list(range(2, 11))
    return (all(sorted(row.tolist()) == wanted for row in board)
            and all(sorted(col.tolist()) == wanted for col in board.T)
            and all(sorted(board[r:r+3, c:c+3].ravel().tolist()) == wanted
                    for r in (0, 3, 6) for c in (0, 3, 6)))


def verify(run_dir):
    run_dir = Path(run_dir).resolve()
    cfg, summary = read(run_dir / "config.json"), read(run_dir / "summary.json")
    checks = {}
    artifacts = {}

    def check(name, condition):
        checks[name] = bool(condition)

    check("summary_complete", summary["status"] == "complete")
    check("training_complete", summary["training_status"] == "complete")
    check("wall_time_within_bound", summary["elapsed_seconds"] <= cfg["max_seconds"] + 5)
    check("seed_recorded", summary["seed"] == cfg["seed"])
    arch = cfg["model_config"]
    native = dict(hidden_size=512, L_layers=2, H_cycles=3, L_cycles=6,
                  num_heads=8, halt_max_steps=16, puzzle_emb_ndim=512,
                  puzzle_emb_len=16, seq_len=81, vocab_size=11, mlp_t=False)
    for key, value in native.items():
        check(f"native_architecture_{key}", arch[key] == value)
    check("native_A_parameter_count", summary["A_parameters"] == 6828034)
    check("custom_B_parameter_count", summary["B_parameters"] == 809736)
    for key, value in dict(controller_width=128, controller_heads=8,
                           controller_layers=3, controller_ff_multiplier=4,
                           controller_history=4, adapter_rank=4, adapter_alpha=.2).items():
        check(f"controller_architecture_{key}", arch[key] == value)

    manifest = read(run_dir / "source_manifest.json")
    for name, expected in manifest.items():
        check(f"source_snapshot:{name}", sha(run_dir / "source_snapshot" / name) == expected)
    for name in ("models/recursive_reasoning/trm.py", "models/layers.py", "models/losses.py",
                 "models/sparse_embedding.py", "models/ema.py", "models/common.py"):
        original = subprocess.check_output(["git", "show", f"{summary['upstream_commit']}:{name}"], cwd=ROOT)
        saved = (run_dir / "source_snapshot" / name).read_bytes()
        check(f"upstream_native_unchanged:{name}", original.replace(b"\r\n", b"\n") == saved.replace(b"\r\n", b"\n"))

    data_root = ROOT / cfg["data_dir"]
    data_manifest = read(data_root / "manifest.json")
    check("saved_data_audit_matches_manifest", read(run_dir / "data_audit.json") == data_manifest)
    for name, expected in data_manifest["file_sha256"].items():
        check(f"data_file_hash:{name}", sha(data_root / name) == expected)
    puzzle_sets, solution_sets, pools = {}, {}, {}
    for split in ("train", "val", "test"):
        with np.load(data_root / f"{split}.npz", allow_pickle=False) as arrays:
            pools[split] = {key: arrays[key].copy() for key in arrays.files}
        data = pools[split]
        inputs, labels = data["inputs"], data["labels"]
        n = data_manifest["requested_sizes"][split]
        check(f"{split}:shape", inputs.shape == labels.shape == (n, 81))
        check(f"{split}:encoding", np.all((inputs >= 1) & (inputs <= 10)) and np.all((labels >= 2) & (labels <= 10)))
        check(f"{split}:all_solutions_valid", all(valid_solution(row) for row in labels))
        check(f"{split}:all_clues_consistent", np.all((inputs == 1) | (inputs == labels)))
        check(f"{split}:one_shared_puzzle_id", np.all(data["puzzle_identifiers"] == 0))
        puzzle_sets[split] = {hashlib.sha256(row.tobytes()).hexdigest() for row in inputs}
        solution_sets[split] = {hashlib.sha256(row.tobytes()).hexdigest() for row in labels}
        check(f"{split}:unique_puzzles", len(puzzle_sets[split]) == n)
        check(f"{split}:unique_solutions", len(solution_sets[split]) == n)
    for left, right in (("train", "val"), ("train", "test"), ("val", "test")):
        check(f"disjoint_puzzles:{left}/{right}", not puzzle_sets[left] & puzzle_sets[right])
        check(f"disjoint_solutions:{left}/{right}", not solution_sets[left] & solution_sets[right])

    stages = summary["stages"]
    check("all_requested_training_arms_present", len(stages) == len(cfg["variants"]) and
          {row["variant"] for row in stages} == set(cfg["variants"]))
    check("common_initial_state", len({row["initial_state_sha256"] for row in stages}) == 1 and
          all(row["initial_state_sha256"] == summary["initial_state_sha256"] for row in stages))
    check("candidate_streams_match", len({row["candidate_schedule_sha256"] for row in stages}) == 1)
    check("candidate_match_flag_consistent", summary["matched_candidate_streams"] ==
          (len({row["candidate_schedule_sha256"] for row in stages}) == 1))
    check("admission_match_flag_consistent", summary["matched_actual_admission_streams"] ==
          (len({row["admitted_schedule_sha256"] for row in stages}) == 1))
    trainable = {"native": 6828034, "static_joint": 6828034 + 16384 + 8,
                 "joint": 6828034 + 16384 + 809736}
    for row in stages:
        name = row["variant"]
        steps, batch = row["steps_executed"], cfg["batch_size"]
        slots, admissions = steps * batch, row["new_puzzle_admissions"]
        check(f"{name}:fixed_completed_steps", row["complete"] and steps == row["steps_requested"] == cfg["steps"])
        check(f"{name}:slot_presentations", row["act_slot_presentations"] == slots)
        check(f"{name}:ACT_admission_bounds", math.ceil(steps / arch["halt_max_steps"]) * batch <= admissions <= slots)
        check(f"{name}:unique_base_bounds", 0 < row["unique_base_puzzles_seen"] <= min(admissions, len(pools["train"]["inputs"])))
        check(f"{name}:unique_augmented_bounds", 0 < row["unique_augmented_puzzles_seen"] <= admissions)
        check(f"{name}:trainable_parameters", row["trainable_parameters"] == trainable[name])
        check(f"{name}:fixed_final_policy", row["selection"] == "fixed final step; no validation checkpoint selection")
        check(f"{name}:separate_raw_EMA_hashes", len(row["raw_state_sha256"]) == len(row["ema_state_sha256"]) == 64 and
              row["raw_state_sha256"] != row["ema_state_sha256"])
        interval = cfg.get("checkpoint_every", 0)
        expected_intermediate = {str(step) for step in range(interval, cfg["steps"], interval)} if interval else set()
        check(f"{name}:periodic_checkpoint_coverage", set(row.get("intermediate_checkpoints", {})) == expected_intermediate)

    history_path = run_dir / "history.jsonl"
    history = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    check("logged_training_arms_present", {row["variant"] for row in history} == set(cfg["variants"]))
    for variant in cfg["variants"]:
        logged = [row for row in history if row["variant"] == variant]
        check(f"{variant}:final_step_logged", bool(logged) and logged[-1]["step"] == cfg["steps"])
        check(f"{variant}:logged_native_recursion", all(row["reasoning_calls"] == 21 and
              row["gradient_enabled_calls"] == 7 for row in logged))
        check(f"{variant}:logged_controller_schedule", all(row["controller_calls"] ==
              (21 if variant == "joint" else 0) for row in logged))
        check(f"{variant}:logged_admissions_monotonic", all(left["new_puzzle_admissions"] <= right["new_puzzle_admissions"]
              for left, right in zip(logged, logged[1:])))
        stage = next(row for row in stages if row["variant"] == variant)
        check(f"{variant}:final_admissions_match_log", bool(logged) and
              logged[-1]["new_puzzle_admissions"] == stage["new_puzzle_admissions"])
        for field in ("unique_base_puzzles_seen", "unique_augmented_puzzles_seen"):
            if logged and field in logged[-1]:
                check(f"{variant}:final_{field}_matches_log", logged[-1][field] == stage[field])
                check(f"{variant}:{field}_monotonic", all(left[field] <= right[field]
                      for left, right in zip(logged, logged[1:])))
        check(f"{variant}:logged_training_loss_finite", all(math.isfinite(row["train_loss"]) and row["train_loss"] >= 0 for row in logged))

    expected = set()
    for variant in cfg["variants"]:
        normal = "off" if variant == "native" else "static" if variant == "static_joint" else "dynamic"
        options = [("raw", normal), ("ema", normal)] + ([("raw", "off"), ("raw", "fixed")] if variant == "joint" else [])
        expected.update((variant, weights, mode, split) for weights, mode in options for split in ("val", "test"))
    rows = summary["evaluations"]
    actual = [(row["variant"], row["weights"], row["mode"], row["split"]) for row in rows]
    check("all_raw_EMA_counterfactual_evaluations_present", len(actual) == len(expected) and set(actual) == expected)
    check("evaluation_file_matches_summary", read(run_dir / "evaluations.json")["rows"] == rows)
    for row in rows:
        key = "/".join((row["variant"], row["weights"], row["mode"], row["split"]))
        selected = pools[row["split"]]["inputs"][:cfg["eval_size"]]
        given = selected != 1
        n, given_count, blank_count = len(selected), int(given.sum()), int((~given).sum())
        check(f"{key}:full_fixed_panel", row["n"] == n)
        check(f"{key}:cell_counts", row["given_cells"] == given_count and row["blank_cells"] == blank_count)
        check(f"{key}:copy_baseline", math.isclose(row["clue_copy_accuracy"], given_count / (81 * n), abs_tol=1e-12))
        weighted = (row["given_agreement"] * given_count + row["blank_accuracy"] * blank_count) / (81 * n)
        check(f"{key}:all_cell_metric_identity", math.isclose(row["all_accuracy"], weighted, abs_tol=1e-12))
        check(f"{key}:accuracy_ranges", all(0 <= row[name] <= 1 for name in (
            "all_accuracy", "blank_accuracy", "given_agreement", "exact_accuracy", "valid_grid_fraction")))
        check(f"{key}:exact_implies_valid_grid", row["exact_accuracy"] <= row["valid_grid_fraction"] + 1e-12)
        check(f"{key}:loss_finite_nonnegative", math.isfinite(row["loss"]) and row["loss"] >= 0)
        check(f"{key}:native_ACT16", row["act_steps"] == 16)
        check(f"{key}:inference_state_unchanged", row["parameter_state_unchanged"])
        check(f"{key}:runtime_gate_cleanup", row["runtime_gates_cleared"])

    paths = [ROOT / summary["initial_checkpoint"]] + [ROOT / row["checkpoint"] for row in stages]
    paths += [ROOT / path for row in stages for path in row.get("intermediate_checkpoints", {}).values()]
    for path in dict.fromkeys(paths):
        relative = str(path.relative_to(ROOT)).replace("\\", "/")
        with zipfile.ZipFile(path) as archive:
            check(f"checkpoint_zip_CRC:{relative}", archive.testzip() is None)
            check(f"checkpoint_contains_tensor_metadata:{relative}", any(name.endswith("/data.pkl") for name in archive.namelist()))
        artifacts[relative] = {"sha256": sha(path), "bytes": path.stat().st_size}
    result = {"passed": all(checks.values()), "checks": checks,
              "scope": "Independent source/data/hash/protocol/accounting and saved metric identity checks. Upstream source comparison permits Windows CRLF normalization. No Torch import; no checkpoint tensor reconstruction or independent per-step ACT admission ledger.",
              "checkpoint_file_integrity": artifacts}
    (run_dir / "verification.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/local_sudoku")
    args = parser.parse_args()
    directory = Path(args.run_dir)
    result = verify(directory if directory.is_absolute() else ROOT / directory)
    failures = [name for name, passed in result["checks"].items() if not passed]
    print(json.dumps({"passed": result["passed"], "checks": len(result["checks"]), "failures": failures}))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
