"""Compare verified, matched per-step and once-per-ACT training runs."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def report(step_directory, act_directory):
    def read(directory, name):
        return json.loads((directory / name).read_text(encoding="utf-8-sig"))
    step_cfg, act_cfg = (read(directory, "config.json") for directory in (step_directory, act_directory))
    step, act = (read(directory, "summary.json") for directory in (step_directory, act_directory))
    for directory, summary in ((step_directory, step), (act_directory, act)):
        if summary["status"] != "complete" or read(directory, "verification.json")["status"] != "passed":
            raise ValueError("Both runs must be complete and independently verified")
    checks = {}
    for key in ("seeds", "batch_size", "eval_batch_size", "learning_rate", "warmup_steps", "weight_decay",
                "data_dir", "train_offset", "train_size", "dev_offset", "dev_size", "test_offset", "test_size"):
        checks["matched:" + key] = step_cfg[key] == act_cfg[key]
    checks["matched_steps"] = step["steps_per_arm"] == act["steps_per_arm"]
    checks["same_frozen_a"] = step["frozen_a_sha256"] == act["frozen_a_sha256"]
    checks["same_source_checkpoint"] = step["source_checkpoint_sha256"] == act["source_checkpoint_sha256"]
    checks["only_cadence_differs_in_architecture"] = ({k: v for k, v in step_cfg["architecture"].items() if k != "weight_update_cadence"}
        == {k: v for k, v in act_cfg["architecture"].items() if k != "weight_update_cadence"})
    checks["intended_cadences"] = step_cfg["architecture"]["weight_update_cadence"] == "per_reasoning_step" and act_cfg["architecture"]["weight_update_cadence"] == "per_act"
    differences, per_seed, step_scores, act_scores = [], [], [], []
    def locate(summary, seed):
        return next(i for i, row in enumerate(summary["evaluations"])
                    if (row["kind"], row["seed"], row["variant"], row["split"]) == ("cross", seed, "normal", "test"))
    with np.load(step_directory / "per_puzzle_metrics.npz") as step_arrays, np.load(act_directory / "per_puzzle_metrics.npz") as act_arrays:
        for seed in step_cfg["seeds"]:
            first = next(row for row in step["training"] if row["kind"] == "cross" and row["seed"] == seed)
            second = next(row for row in act["training"] if row["kind"] == "cross" and row["seed"] == seed)
            for key in ("initial_actor_sha256", "stream_sha256", "admitted_base_ids", "act_slots", "a_calls_per_act", "parameter_count"):
                checks[f"{seed}:matched_{key}"] = first[key] == second[key]
            i, j = locate(step, seed), locate(act, seed)
            correct, control = step_arrays[f"correct_{i}"], act_arrays[f"correct_{j}"]
            blanks = step_arrays[f"blanks_{i}"]
            checks[f"{seed}:paired_blanks"] = np.array_equal(blanks, act_arrays[f"blanks_{j}"])
            differences.append(correct.astype(float) - control)
            per_seed.append(float(100 * (correct - control).sum() / blanks.sum()))
            step_scores.append(step["evaluations"][i]["blank_accuracy"])
            act_scores.append(act["evaluations"][j]["blank_accuracy"])
        numerator = np.mean(differences, axis=0)
        rng = np.random.default_rng(20261102)
        indices = rng.integers(0, len(blanks), (step_cfg["bootstrap_resamples"], len(blanks)))
        samples = 100 * numerator[indices].sum(1) / blanks[indices].sum(1)
    if not all(checks.values()):
        raise AssertionError([name for name, passed in checks.items() if not passed])
    result = {"status": "passed", "comparison": "per-reasoning-step minus once-per-ACT, same structured B+C",
        "step_accuracy": float(np.mean(step_scores)), "act_accuracy": float(np.mean(act_scores)),
        "gain_pp": float(100 * numerator.sum() / blanks.sum()),
        "conditional_95_interval_pp": np.quantile(samples, [.025, .975]).tolist(),
        "per_seed_gain_pp": per_seed, "checks": checks,
        "reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "Exploratory follow-up specified after primary evaluation began. Same initial actor tensors, training examples, updates, batch, frozen A, and A calls; controller calls and total FLOPs differ. Paired-puzzle interval is conditional on these three runs and one donor. Not a joint-A adaptability test."}
    (step_directory / "cadence_comparison.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    low, high = result["conditional_95_interval_pp"]
    text = f"""# Matched control for update cadence

This exploratory follow-up trains the same structured B+C with one late weight
generation per ACT, held over its final seven reasoning calls. The main run
generates weights before all 21 reasoning calls. All three seeds start from
identical actor tensors and receive identical augmented puzzles, optimizer
updates, batch size and native A calls. Both keep A fixed. Controller computation
and total FLOPs differ.

| Cadence | Mean held-out blank accuracy |
|---|---:|
| Every reasoning call | {100*result['step_accuracy']:.3f}% |
| Once per ACT | {100*result['act_accuracy']:.3f}% |

Per-step minus per-ACT: **{result['gain_pp']:+.3f} percentage points**, with
conditional paired-puzzle 95% interval **[{low:+.3f}, {high:+.3f}]**.
Per-seed differences: {', '.join(f'{v:+.3f}' for v in per_seed)} percentage points.

This follow-up was specified after the primary comparison started. It does not
replace the predeclared structured-C versus larger-B comparison. The interval
resamples this fixed test panel after averaging these three controller seeds;
it does not include uncertainty from retraining A or the full population of
training outcomes. Matching initialization/data/call-budget checks all passed;
both underlying saved-artifact audits passed before this comparison.
"""
    (step_directory / "cadence_comparison.md").write_text(text, encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step-run", default="results/per_step_comparison_20261102_v4")
    parser.add_argument("--act-run", default="results/per_act_control_20261102")
    args = parser.parse_args()
    report(ROOT / args.step_run, ROOT / args.act_run)
