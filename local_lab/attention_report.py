"""Summarize the verified local comparison with paired puzzle uncertainty."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
NAMES = {"b": "B only", "larger_b": "Larger B, parameter matched",
         "pooled": "B+C, pooled reads", "cross": "B+C, structured attention"}


def report(directory):
    cfg = json.loads((directory / "config.json").read_text())
    summary = json.loads((directory / "summary.json").read_text())
    verification = json.loads((directory / "verification.json").read_text())
    if summary["status"] != "complete" or verification["status"] != "passed":
        raise ValueError("A completed and verified comparison is required")
    rows, seeds = summary["evaluations"], cfg["seeds"]
    unit_path = directory / "unit_checks.json"
    units = json.loads(unit_path.read_text()) if unit_path.exists() else {"passed": "unrecorded"}
    def locate(kind, seed, variant="normal", split="test"):
        return next(i for i, row in enumerate(rows)
            if (row["kind"], row["seed"], row["variant"], row["split"]) == (kind, seed, variant, split))
    baseline = rows[locate("a", 0, "a_only")]
    means = {}
    for kind, name in NAMES.items():
        scores = [rows[locate(kind, seed)] for seed in seeds]
        means[kind] = {"name": name, "accuracy": float(np.mean([r["blank_accuracy"] for r in scores])),
            "task_loss": float(np.mean([r["task_loss"] for r in scores])),
            "exact_accuracy": float(np.mean([r["exact_accuracy"] for r in scores])),
            "per_seed_accuracy": [r["blank_accuracy"] for r in scores]}
    paired = []
    rng = np.random.default_rng(20261030)
    with np.load(directory / "per_puzzle_metrics.npz") as arrays:
        def counts(kind, seed, variant="normal"):
            index = locate(kind, seed, variant)
            return arrays[f"correct_{index}"], arrays[f"blanks_{index}"]
        comparisons = [("larger_b", "normal", "cross-minus-larger_b"),
            ("pooled", "normal", "cross-minus-pooled"), ("b", "normal", "cross-minus-b"),
            ("cross", "c_off", "cross-minus-c_off"),
            ("cross", "c_write_shuffle", "cross-minus-c_write_shuffle"),
            ("cross", "c_read_shuffle", "cross-minus-c_read_shuffle"),
            ("cross", "c_read_pooled", "cross-minus-c_read_pooled"),
            ("cross", "c_phase", "cross-minus-c_phase"),
            ("cross", "c_reset_memory", "cross-minus-c_reset_memory")]
        comparisons = [(kind, variant, name) for kind, variant, name in comparisons
                       if any(row["kind"] == kind and row["variant"] == variant and row["split"] == "test" for row in rows)]
        for kind, variant, name in comparisons:
            differences, per_seed = [], []
            for seed in seeds:
                correct, blanks = counts("cross", seed)
                control, control_blanks = counts(kind, seed, variant)
                if not np.array_equal(blanks, control_blanks):
                    raise AssertionError("Paired panels have different blank counts")
                differences.append(correct.astype(float) - control)
                per_seed.append(float(100 * (correct - control).sum() / blanks.sum()))
            numerator = np.mean(differences, axis=0)
            indices = rng.integers(0, len(blanks), (cfg["bootstrap_resamples"], len(blanks)))
            samples = 100 * numerator[indices].sum(1) / blanks[indices].sum(1)
            paired.append({"comparison": name, "primary": name == cfg["primary_comparison"],
                "gain_pp": float(100 * numerator.sum() / blanks.sum()),
                "per_seed_gain_pp": per_seed, "conditional_95_interval_pp": np.quantile(samples, [.025, .975]).tolist()})
    primary, pooled = paired[0], paired[1]
    if primary["conditional_95_interval_pp"][0] > 0:
        verdict = "Structured attention improved accuracy over the parameter-matched larger-B control in this bounded run."
        if pooled["conditional_95_interval_pp"][0] <= 0:
            verdict += " The comparison with trained pooled reads does not establish that the additional state structure caused that improvement."
    else:
        verdict = "This run did not establish an accuracy advantage for structured inter-model attention over the parameter-matched larger-B control."
    effects = {"rows": paired, "scope": "Paired bootstrap over held-out puzzles, averaging these three fixed controller seeds before resampling. Intervals are conditional on this A donor and these controller runs, not estimates of A-training or full controller-training uncertainty. One predeclared primary comparison; secondary intervals are descriptive and unadjusted."}
    (directory / "paired_effects.json").write_text(json.dumps(effects, indent=2), encoding="utf-8")
    aggregate = {"verdict": verdict, "baseline": baseline["blank_accuracy"], "means": means,
        "primary": primary, "paired_effects": effects, "test_puzzles": cfg["test_size"],
        "controller_seeds": seeds, "elapsed_seconds": summary["elapsed_seconds"]}
    (directory / "aggregate.json").write_text(json.dumps(aggregate, indent=2), encoding="utf-8")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1.3, 1]})
    colors = ["#567993", "#8a6884", "#c58a3b", "#23846b"]
    for i, (kind, values) in enumerate(means.items()):
        scores = np.array(values["per_seed_accuracy"]) * 100
        axes[0].scatter(i + np.linspace(-.09, .09, len(scores)), scores, color=colors[i], s=35, alpha=.75)
        axes[0].scatter(i, values["accuracy"] * 100, color=colors[i], marker="D", s=70, edgecolors="white", zorder=4)
    axes[0].axhline(baseline["blank_accuracy"] * 100, color="#777777", ls="--", lw=1, label="Frozen A")
    axes[0].set_xticks(range(4), ["B", "Larger B", "Pooled B+C", "Structured B+C"])
    axes[0].set_ylabel("Held-out blank-cell accuracy (%, zoomed)")
    axes[0].set_title("Three training seeds; diamonds are means")
    axes[0].legend(frameon=False)
    axes[0].grid(axis="y", alpha=.2)
    for i, effect in enumerate(paired[:3]):
        low, high = effect["conditional_95_interval_pp"]
        axes[1].plot([low, high], [i, i], color=colors[3], lw=2)
        axes[1].scatter(effect["gain_pp"], i, color=colors[3], s=45)
    axes[1].axvline(0, color="#777777", lw=1)
    axes[1].set_yticks(range(3), ["vs larger B (primary)", "vs trained pooled reads", "vs B only"])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Structured B+C difference (percentage points)")
    axes[1].set_title("Conditional paired-puzzle 95% intervals")
    axes[1].grid(axis="x", alpha=.2)
    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    (directory / "plots").mkdir(exist_ok=True)
    fig.savefig(directory / "plots/comparison.png", dpi=170)
    plt.close(fig)
    interval = cfg.get("dev_every", 0)
    checkpoint_steps = [0] + (list(range(interval, cfg["locked_steps"], interval)) if interval else []) + [cfg["locked_steps"]]
    curves = {}
    fig, curve_axes = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for color, (kind, name) in zip(colors, NAMES.items()):
        curve = []
        for step in checkpoint_steps:
            variant = "initial" if step == 0 else "normal" if step == cfg["locked_steps"] else f"step_{step}"
            measurements = [rows[locate(kind, seed, variant, "dev")] for seed in seeds]
            curve.append({"step": step,
                "mean_blank_accuracy": float(np.mean([r["blank_accuracy"] for r in measurements])),
                "mean_task_loss": float(np.mean([r["task_loss"] for r in measurements])),
                "per_seed_blank_accuracy": [r["blank_accuracy"] for r in measurements],
                "per_seed_task_loss": [r["task_loss"] for r in measurements]})
        curves[kind] = curve
        for seed_index in range(len(seeds)):
            curve_axes[0].plot(checkpoint_steps, [100*r["per_seed_blank_accuracy"][seed_index] for r in curve], color=color, alpha=.17, lw=.8)
            curve_axes[1].plot(checkpoint_steps, [r["per_seed_task_loss"][seed_index] for r in curve], color=color, alpha=.17, lw=.8)
        curve_axes[0].plot(checkpoint_steps, [100*r["mean_blank_accuracy"] for r in curve], color=color, label=name, marker="o", ms=3)
        curve_axes[1].plot(checkpoint_steps, [r["mean_task_loss"] for r in curve], color=color, label=name, marker="o", ms=3)
    for ax in curve_axes:
        ax.set_xlabel("Optimizer updates")
        ax.grid(alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    curve_axes[0].set_ylabel("Development blank-cell accuracy (%)")
    curve_axes[1].set_ylabel("Development task loss")
    curve_axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Development curves: thin lines are seeds; bold lines are means")
    fig.tight_layout()
    fig.savefig(directory / "plots/learning_curves.png", dpi=170)
    plt.close(fig)
    (directory / "learning_curves.json").write_text(json.dumps(curves, indent=2), encoding="utf-8")
    training = summary["training"]
    table = "\n".join(f"| {v['name']} | {next(r['parameter_count'] for r in training if r['kind'] == k):,} | {100*v['accuracy']:.3f}% | {v['task_loss']:.6f} |"
        for k, v in means.items())
    seed_table = "\n".join("| " + NAMES[k] + " | " + " | ".join(f"{100*v:.3f}%" for v in means[k]["per_seed_accuracy"]) + " |" for k in NAMES)
    effect_table = "\n".join(f"| {row['comparison']} | {row['gain_pp']:+.3f} | [{row['conditional_95_interval_pp'][0]:+.3f}, {row['conditional_95_interval_pp'][1]:+.3f}] |" for row in paired)
    admissions = training[0]["augmented_admissions"]
    exact_note = ("No evaluated variant produced a complete valid puzzle solution."
        if all(row["exact_accuracy"] == 0 and row["valid_grid_fraction"] == 0 for row in rows if row["split"] == "test")
        else "Exact-puzzle and valid-grid fractions for every condition are retained in summary.json.")
    cadence = ("Fresh weights were generated before each of the 21 reasoning calls per ACT, "
        "including all 14 native no-gradient warm-up calls: 336 generations per complete puzzle. "
        "Each of the final seven calls has its own differentiable B/C write."
        if cfg["architecture"].get("weight_update_cadence") == "per_reasoning_step" else
        "The historical per_act cadence generated one action held over the final seven native reasoning calls.")
    freshness = ("Exact puzzle and solution hashes of the final test panel were checked against "
        "every split of the previous cache; the panel starts beyond that cache's test prefix. "
        "The frozen A donor's original training pool was verified unchanged."
        if cfg.get("previous_data_dir") else "")
    config_path = cfg.get("config_path", "local_lab/config_attention_test.json")
    text = f"""# Controlled test of inter-model attention

**{verdict}**

The predeclared primary comparison is structured B+C versus an active,
parameter-matched larger B. Its mean difference was **{primary['gain_pp']:+.3f}
percentage points**, with conditional paired-puzzle 95% interval
**[{primary['conditional_95_interval_pp'][0]:+.3f}, {primary['conditional_95_interval_pp'][1]:+.3f}]**.

## Final held-out results

Blank-cell accuracy is evaluated on {cfg['test_size']} official test puzzles at
offset {cfg['test_offset']}, using all 16 ACT updates per puzzle. Frozen A alone
scored **{100*baseline['blank_accuracy']:.3f}%**. Controller results average seeds {seeds}.

| Trained system | Trainable parameters | Blank-cell accuracy | Task loss |
|---|---:|---:|---:|
{table}

| System | Seed {seeds[0]} | Seed {seeds[1]} | Seed {seeds[2]} |
|---|---:|---:|---:|
{seed_table}

{exact_note}

![Controlled comparison](plots/comparison.png)

## Protocol and executed budget

A was fixed at the same native-training checkpoint for every run. The experiment
does not train A or test whether joint A training makes A more adaptable. B writes
complete dense temporary changes to A's two MLP down-projections. C cross-attends
to 14 internal B tokens and writes a dense temporary change to B's generation
head. There are no low-rank factors in the tested models.

{cadence}

Each of the 12 training runs received **{cfg['locked_steps']} updates**, batch
**{cfg['batch_size']}**, identical data/augmentation and native A-call budgets
within each seed. This admitted **{admissions} augmented puzzles per arm** from
a {cfg['train_size']}-puzzle pool. Pooled and structured C had exactly identical
initial parameters. The larger-B control spent C's 29,168-parameter budget on
an active context network, rather than unused parameter padding. Its total
trainable count exactly matches B+C. Total FLOPs and wall-clock costs are not
identical; the matched compute boundary is the number of A calls.

The {cfg['dev_size']}-puzzle development panel and held-out test panel are disjoint
from controller training. {freshness} Hyperparameters and the primary comparison were fixed
before study evaluation. A training-only throughput profile selected
{cfg['locked_steps']} updates from a requested {cfg['requested_steps']} to respect
the **{cfg['max_seconds']/60:.0f}-minute cap**, with a 50% timing margin. Profile
models were discarded. Every reported checkpoint is the final training step;
development scores were not used to choose checkpoints. No oracle was used.

The comparison completed in **{summary['elapsed_seconds']/60:.2f} minutes**.
All {len(training)} arms completed. The local suite passed {units['passed']} tests before the
run. The independent saved-artifact audit passed **{verification['passed']}/{verification['total']}**
checks, including all native A tensors unchanged, common initialization,
regenerated augmented streams, exact parameter matching, snapshot hashes,
prediction-derived accuracy and independently calculated Stablemax task loss.
The auditor does not replay optimizer steps or model inference.

## Development learning curves

All arms were measured at updates {checkpoint_steps}. Intermediate evaluations
follow complete 16-ACT episodes; the next training episode resets all puzzle
slots, preserving the matched training stream. Every final test score uses the
last checkpoint, with no development-based selection or early stopping.

![Development curves](plots/learning_curves.png)

Structured B+C development loss changed from {curves['cross'][0]['mean_task_loss']:.6f}
to {curves['cross'][-1]['mean_task_loss']:.6f}; its blank accuracy changed from
{100*curves['cross'][0]['mean_blank_accuracy']:.3f}% to
{100*curves['cross'][-1]['mean_blank_accuracy']:.3f}%. These measurements describe
the executed budget and do not establish convergence or predict a longer run.

## Paired effects and inference ablations

Positive values favor normal structured B+C. C ablations use the same trained
checkpoint and do not replace separately trained controls. The pooled trained
arm is the matched experiment for whether the structured read interface helps.

| Comparison | Mean difference (pp) | Conditional 95% interval (pp) |
|---|---:|---:|
{effect_table}

The intervals resample held-out puzzles after averaging these three fixed seeds.
They do not capture uncertainty from retraining A or the full distribution of
controller training outcomes. Secondary intervals are descriptive and are not
adjusted for multiple comparisons. Only one A donor, a bounded controller budget
and one Sudoku test panel were evaluated. Nonzero attention gradients or distinct
attention maps do not demonstrate useful specialization, reasoning transparency
or safety. Exact-puzzle results and each inference condition remain in summary.json.

## Reproduction

Copy `{config_path}` with fresh `results_dir` and
`profile_file` paths; existing runs are never overwritten. Then run:

```powershell
& '..\\plastic-latent-lab\\.venv\\Scripts\\python.exe' -m local_lab.attention_test --config {config_path} --stage all
& '..\\plastic-latent-lab\\.venv\\Scripts\\python.exe' scripts/verify_attention_test.py --run {directory.relative_to(ROOT).as_posix()}
& '..\\plastic-latent-lab\\.venv\\Scripts\\python.exe' -m local_lab.attention_report --run {directory.relative_to(ROOT).as_posix()}
```
"""
    (directory / "report.md").write_text(text, encoding="utf-8")
    print(json.dumps({"verdict": verdict, "means": {k: v["accuracy"] for k, v in means.items()}, "primary": primary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/cross_attention_test_20261030")
    report(ROOT / parser.parse_args().run)
