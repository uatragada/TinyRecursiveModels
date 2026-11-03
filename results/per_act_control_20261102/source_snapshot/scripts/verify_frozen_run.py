"""Independent CPU evidence audit; run after the GPU experiment exits."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def digest(state):
    result = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        result.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def metrics(predictions, inputs, labels):
    p, x, y = [np.asarray(value) for value in (predictions, inputs, labels)]
    correct, given = p == y, x != 1
    grids = p.reshape(-1, 9, 9)
    valid = []
    expected = set(range(2, 11))
    for grid, clues, matches in zip(grids, given, correct):
        valid.append(all(set(row) == expected for row in grid)
            and all(set(column) == expected for column in grid.T)
            and all(set(grid[r:r+3, c:c+3].flatten()) == expected for r in (0, 3, 6) for c in (0, 3, 6))
            and np.logical_or(~clues, matches).all())
    return dict(blank_accuracy=float(correct[~given].mean()), all_accuracy=float(correct.mean()),
        given_agreement=float(correct[given].mean()), exact_accuracy=float(correct.all(1).mean()),
        valid_grid_fraction=float(np.mean(valid)), blank_cells=int((~given).sum()), given_cells=int(given.sum()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="results/frozen_A_pilot_20261029")
    args = parser.parse_args()
    directory = ROOT / args.run_dir
    read = lambda name: json.loads((directory / name).read_text())
    cfg, summary, oracle, audit = [read(name) for name in ("config.json", "summary.json", "oracle.json", "data_audit.json")]
    checks = []
    def check(name, passed):
        checks.append({"name": name, "passed": bool(passed)})
    def load(path):
        return torch.load(path, map_location="cpu", weights_only=True)
    common = load(directory / "common_substrate.pt")
    source = load(ROOT / cfg["source_checkpoint"])["state_dict"]
    check("run complete", summary["status"] == "complete")
    check("budget respected", summary["elapsed_seconds"] <= cfg["max_seconds"])
    check("source checkpoint hash", hashlib.sha256((ROOT / cfg["source_checkpoint"]).read_bytes()).hexdigest() == oracle["source_checkpoint_sha256"])
    check("common substrate equals complete source checkpoint", digest(source) == digest(common["state_dict"]) == summary["frozen_state_sha256"])
    check("common buffers digest", digest(common["buffers"]) == summary["frozen_buffers_sha256"])
    for stage_name in ("oracle_sources", "controller_sources"):
        manifest = json.loads((directory / stage_name / "manifest.json").read_text())
        for name, expected in manifest.items():
            check(f"{stage_name}:{name}", hashlib.sha256((directory / stage_name / name).read_bytes()).hexdigest() == expected)
    panel_ids = {name: set(panel["base_ids"]) for name, panel in audit["panels"].items()}
    for left in panel_ids:
        for right in panel_ids:
            if left < right:
                check(f"disjoint:{left}/{right}", not panel_ids[left] & panel_ids[right])
    # Independently reload the exact panel bytes to bind saved labels/inputs to the dataset.
    manifest = json.loads((ROOT / cfg["data_dir"] / "manifest.json").read_text())
    check("dataset manifest hash", hashlib.sha256((ROOT / cfg["data_dir"] / "manifest.json").read_bytes()).hexdigest() == audit["dataset_manifest_sha256"])
    panels = {}
    for split in ("val", "test"):
        path = ROOT / cfg["data_dir"] / f"{split}.npz"
        check(f"cache hash:{split}", hashlib.sha256(path.read_bytes()).hexdigest() == manifest["file_sha256"][f"{split}.npz"])
        with np.load(path) as arrays:
            for name in (("train", "dev", "oracle") if split == "val" else ("test",)):
                offset = cfg[name + "_offset"] if split == "val" else 0
                stop = offset + cfg[name + "_size"]
                panels[name] = {key: arrays[key][offset:stop] for key in ("inputs", "labels", "base_ids")}
                check(f"panel IDs:{name}", panels[name]["base_ids"].tolist() == audit["panels"][name]["base_ids"])
                for key in ("inputs", "labels"):
                    check(f"panel bytes:{name}/{key}", hashlib.sha256(panels[name][key].tobytes()).hexdigest() == audit["panels"][name][key + "_sha256"])
    for row in oracle["rows"]:
        tag = "oracle:" + row["actuation"]
        before, after, gates = [np.asarray(row[name]) for name in ("before_task_loss", "after_task_loss", "gates")]
        check(tag + ":privileged", row["privileged"])
        check(tag + ":finite", np.isfinite(before).all() and np.isfinite(after).all())
        check(tag + ":bounded gates", np.abs(gates).max() <= 1)
        check(tag + ":nonworsening objective per example", (after <= before + 1e-5).all())
        check(tag + ":objective arithmetic", np.isclose((before - after).mean(), row["mean_objective_reduction"], atol=1e-6))
        history = row["best_loss_history"]
        check(tag + ":search budget", len(history) == cfg["oracle_steps"] + 1)
        check(tag + ":monotone incumbent", all(a >= b - 1e-7 for a, b in zip(history, history[1:])))
        for key, prediction_key in (("baseline", "baseline_predictions"), ("optimized", "optimized_predictions")):
            calculated = metrics(row[prediction_key], panels["oracle"]["inputs"], panels["oracle"]["labels"])
            for metric, expected in calculated.items():
                check(tag + f":{key}/{metric}", np.isclose(row[key][metric], expected))
    expected_kinds = {"phase", "pooled", "revised", "activation"}
    check("all arms", len(summary["stages"]) == 4 * len(cfg["seeds"]))
    for seed in cfg["seeds"]:
        stages = [stage for stage in summary["stages"] if stage["seed"] == seed]
        check(f"{seed}:four controls", {stage["kind"] for stage in stages} == expected_kinds)
        check(f"{seed}:candidate stream matched", len({stage["candidate_schedule_sha256"] for stage in stages}) == 1)
        check(f"{seed}:admitted stream matched", len({stage["admitted_schedule_sha256"] for stage in stages}) == 1)
        richer = [stage for stage in stages if stage["kind"] != "phase"]
        check(f"{seed}:controller initial tensors matched", len({stage["initial_actor_sha256"] for stage in richer}) == 1)
        check(f"{seed}:controller parameter counts matched", len({stage["trainable_parameters"] for stage in richer}) == 1)
        for stage in stages:
            tag = f"{seed}/{stage['kind']}"
            path = ROOT / stage["checkpoint"]
            with zipfile.ZipFile(path) as archive:
                check(tag + ":checkpoint ZIP CRC", archive.testzip() is None)
            payload = load(path)
            check(tag + ":metadata exact", payload["metadata"] == stage)
            check(tag + ":fixed training budget", stage["steps_executed"] == cfg["steps"] and stage["complete"])
            check(tag + ":admissions fixed ACT16", stage["new_puzzle_admissions"] == ((cfg["steps"] + 15) // 16) * cfg["batch_size"])
            check(tag + ":unique coverage bounded", 0 < stage["unique_base_puzzles_seen"] <= cfg["train_size"])
            check(tag + ":full frozen state", digest(payload["state_dict"]) == digest(common["state_dict"]))
            check(tag + ":all frozen buffers", digest(payload["buffers"]) == digest(common["buffers"]))
            check(tag + ":actor initial digest", digest(payload["initial_actor_state_dict"]) == stage["initial_actor_sha256"])
            check(tag + ":actor final digest", digest(payload["actor_state_dict"]) == stage["final_actor_sha256"])
            check(tag + ":actor learned", stage["initial_actor_sha256"] != stage["final_actor_sha256"])
            check(tag + ":actor parameter count", sum(value.numel() for value in payload["actor_state_dict"].values()) == stage["trainable_parameters"])
            expected_teacher = 1 + cfg["steps"] // cfg["teacher_every"] - int(cfg["teacher_every"] == 1)
            check(tag + ":teacher budget", len(payload["teacher_records"]) == stage["teacher_calls"] == expected_teacher)
            for record in payload["teacher_records"]:
                step_tag = tag + f":teacher{record['step']}"
                check(step_tag + ":training-only IDs", set(record["base_ids"].tolist()) <= panel_ids["train"] and record["privileged_training_only"])
                check(step_tag + ":bounded target gates", record["target_gates"].abs().max() <= 1)
                check(step_tag + ":not worse than proposal", torch.all(record["target_loss"] <= record["proposed_loss"] + 1e-5))
                check(step_tag + ":zero candidate available", torch.all(record["target_loss"] <= record["zero_loss"] + 1e-5))
            for name, value in payload["actor_state_dict"].items():
                check(tag + ":finite " + name, torch.isfinite(value).all())
    expected_rows = 2 + len(cfg["seeds"]) * (8 + 3)
    check("evaluation coverage", len(summary["evaluations"]) == expected_rows)
    correct_counts = {}
    for row in summary["evaluations"]:
        name = f"{row['kind']}_{row['seed']}_{row['mode']}_{row['split']}"
        evidence = load(directory / f"predictions_{name}.pt")
        split = panels[row["split"]]
        check(name + ":nonprivileged evaluation", row["privileged"] is False)
        check(name + ":same evaluation IDs", evidence["base_ids"].tolist() == split["base_ids"].tolist())
        check(name + ":same evaluation inputs", np.array_equal(evidence["inputs"].numpy(), split["inputs"]))
        check(name + ":same evaluation labels", np.array_equal(evidence["labels"].numpy(), split["labels"]))
        for key, expected in metrics(evidence["predictions"].numpy(), split["inputs"], split["labels"]).items():
            check(name + ":recomputed " + key, np.isclose(row[key], expected))
        for key in ("frozen_state_unchanged", "frozen_buffers_unchanged", "actor_parameters_unchanged", "runtime_gates_cleared"):
            check(name + ":recorded " + key, row[key])
        check(name + ":16ACT", row["act_steps"] == 16)
        blank = split["inputs"] == 1
        correct_counts[(row["seed"], row["kind"], row["mode"], row["split"])] = ((evidence["predictions"].numpy() == split["labels"]) & blank).sum(1)
    # Fixed-controller, paired puzzle bootstrap: average seeds first, then resample
    # puzzles as clusters. This does not estimate A/controller-training uncertainty.
    denominator = (panels["test"]["inputs"] == 1).sum(1)
    rng = np.random.default_rng(1743)
    sample_indices = rng.integers(0, len(denominator), size=(2000, len(denominator)))
    effects = []
    for control in ("off", "phase", "pooled", "activation", "shuffle", "phase_only", "reset_memory"):
        differences, by_seed = [], []
        for seed in cfg["seeds"]:
            target = correct_counts[(seed, "revised", "normal", "test")]
            key = (cfg["seeds"][0], "off", "off", "test") if control == "off" else (seed, "revised", control, "test") if control in ("shuffle", "phase_only", "reset_memory") else (seed, control, "normal", "test")
            difference = target - correct_counts[key]
            differences.append(difference)
            by_seed.append({"seed": seed, "gain_pp": float(100 * difference.sum() / denominator.sum())})
        average = np.mean(differences, axis=0)
        bootstrap = 100 * average[sample_indices].sum(1) / denominator[sample_indices].sum(1)
        effects.append({"comparison": "revised-minus-" + control, "primary": control == "phase",
            "gain_pp": float(100 * average.sum() / denominator.sum()), "per_seed": by_seed,
            "conditional_paired_puzzle_interval_95_pp": np.quantile(bootstrap, [.025, .975]).tolist()})
    (directory / "paired_effects.json").write_text(json.dumps({"rows": effects,
        "scope": "Descriptive paired bootstrap over test puzzles, averaging these fixed controller seeds before resampling. No independent A-training replicates or controller-training uncertainty; secondary intervals have no multiple-comparison adjustment."}, indent=2))
    result = {"passed": all(row["passed"] for row in checks), "checks_passed": sum(row["passed"] for row in checks),
        "checks_total": len(checks), "scope": "Independent saved-tensor equality, frozen buffers, actor learning, oracle accounting, training-only teacher IDs/budgets, source/data hashes, ZIP CRCs and metrics recomputed from saved predictions. Does not independently rerun model inference or reconstruct every optimizer step.", "checks": checks}
    (directory / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}))
    if not result["passed"]:
        print(json.dumps([row for row in checks if not row["passed"]], indent=2))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
