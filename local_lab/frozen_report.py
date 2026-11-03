"""Saved-evidence report for the frozen-A pilot; no Torch/model imports."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import statistics

ROOT = Path(__file__).resolve().parents[1]
KINDS = ("phase", "pooled", "revised", "activation")


def generate(directory):
    read = lambda name: json.loads((directory / name).read_text())
    cfg, summary, oracle = [read(name) for name in ("config.json", "summary.json", "oracle.json")]
    verification, paired = read("verification.json"), read("paired_effects.json")
    names = {"off": "Frozen A, no adjustments", "phase": "Learned ACT-step schedule", "pooled": "Matched pooled-reader B",
             "revised": "Cell-aware B with persistent memory", "activation": "Matched activation controller"}
    rows = summary["evaluations"]
    normal_test = [row for row in rows if row["split"] == "test" and row["mode"] in ("normal", "off")]
    means = []
    for kind in names:
        selected = [row for row in normal_test if row["kind"] == kind]
        if not selected:
            continue
        means.append({"kind": kind, "name": names[kind], "seeds": sorted({row["seed"] for row in selected}),
            **{key: statistics.mean(row.get(key, 0.) for row in selected) for key in ("blank_accuracy", "exact_accuracy", "loss", "task_loss", "given_agreement", "valid_grid_fraction", "gate_rms", "between_puzzle_gate_variance_fraction", "delta_to_base_rms_ratio")}})
    primary = next(row for row in paired["rows"] if row["primary"])
    low, high = primary["conditional_paired_puzzle_interval_95_pp"]
    outcome = ("The conditional paired interval favors revised B on these fixed controllers/puzzles." if low > 0 else
        "The conditional paired interval favors the learned schedule on these fixed controllers/puzzles." if high < 0 else
        "The conditional paired interval includes zero; this pilot does not demonstrate an accuracy advantage for revised B over the learned schedule.")
    unit_checks = read("unit_checks.json") if (directory / "unit_checks.json").exists() else None
    lines = ["# Frozen-A gate oracle and revised-controller experiment", "",
        f"Saved status: **{summary['status']}**. Measured oracle/training/evaluation execution: **{summary['elapsed_seconds']:.1f} seconds**, bound {cfg['max_seconds']} seconds. Frozen substrate: `{cfg['source_checkpoint']}`; source A and learned U/V directions remain unchanged.", "",
        f"Controller-stage peak CUDA tensor allocation: {summary['peak_cuda_allocated_bytes'] / 1024**3:.3f} GiB. This excludes driver/other-process memory and is not a dedicated latency benchmark.", "",
        "This is an intervention/interface experiment on one previously joint-trained A checkpoint. It does not train A, introduce a new task family, perform meta-learning, or establish increased A-weight adaptability. Two fresh controller seeds are conditional on the same A and puzzle panels.", "",
        f"Primary revised-B minus learned-schedule test contrast: **{primary['gain_pp']:+.3f} percentage points**, conditional paired puzzle interval [{low:+.3f}, {high:+.3f}]. {outcome}", "",
        "## Privileged gate oracle", "",
        "The oracle sees answers to 32 separately reserved diagnostic puzzles (or the configured smoke size). It optimizes eight bounded gates in the terminal seven-call reasoning window, starting from a common untouched A trajectory. Zero is always an eligible candidate, and each puzzle retains its best measured task-loss action. This is finite local search, not a proved global or full-trajectory upper bound. These scores are not deployed-controller scores.", "",
        "| Action interface | Baseline blank accuracy | Local oracle blank accuracy | Mean task-loss reduction | Nonworsening loss per example |",
        "| --- | ---: | ---: | ---: | --- |"]
    for row in oracle["rows"]:
        lines.append(f"| {row['actuation']} | {100*row['baseline']['blank_accuracy']:.3f}% | {100*row['optimized']['blank_accuracy']:.3f}% | {row['mean_objective_reduction']:.6f} | {row['per_example_nonworsening_objective']} |")
    lines += ["", "The oracle optimizes blank Stablemax cross-entropy plus 0.25 times given-clue cross-entropy. Lower cross-entropy need not improve argmax accuracy. A small or negative accuracy change limits claims about the usefulness of this action space; it does not prove that every possible gate sequence is ineffective.", "",
        "## Nonprivileged test results", "",
        f"Training uses {cfg['train_size']} source-A-untrained puzzles; development uses {cfg['dev_size']}; oracle diagnostics use {cfg['oracle_size']} separate puzzles. The final fixed test panel contains {cfg['test_size']} official held-out puzzles. All panels have disjoint puzzle IDs and hashes. Test and development evaluation never calls the oracle.", "",
        "| Controller | Blank accuracy | Exact puzzle | Given agreement | Stablemax per cell | Task objective |",
        "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for row in means:
        lines.append(f"| {row['name']} | {100*row['blank_accuracy']:.3f}% | {100*row['exact_accuracy']:.3f}% | {100*row['given_agreement']:.3f}% | {row['loss']:.6f} | {row['task_loss']:.6f} |")
    lines += ["", "Trainable-arm values are equal-weight means across the configured fresh controller seeds; the no-adjustment computation is deterministic and evaluated once. Fixed final checkpoints are used, with no development-best or test-best selection. Development scores and every per-seed endpoint remain in summary.json.", "",
        "| Paired test contrast | Mean revised-B gain (pp) | Per-seed gains (pp) | Conditional puzzle interval (95%, pp) |",
        "| --- | ---: | --- | --- |"]
    for row in paired["rows"]:
        interval = row["conditional_paired_puzzle_interval_95_pp"]
        seeds = ", ".join(f"{item['seed']}: {item['gain_pp']:+.3f}" for item in row["per_seed"])
        lines.append(f"| {row['comparison']}{' (primary)' if row['primary'] else ''} | {row['gain_pp']:+.3f} | {seeds} | [{interval[0]:+.3f}, {interval[1]:+.3f}] |")
    lines += ["", paired["scope"], "",
        "Shuffle rolls actions between the examples of each evaluation batch. Phase-only suppresses the revised model's content residual while still computing it; this is a same-checkpoint intervention, distinct from the independently trained schedule. Reset-memory zeros only the GRU state at each ACT update; previous latent/prediction/action features remain available.", "",
        "## Architecture and matching", "",
        "Each ACT update retains native H3/L6: 14 untouched no-gradient calls followed by one action held over seven differentiable calls. All training and evaluation use fixed ACT16; there is no learned-halting exposure confound. Native A weights, U/V, puzzle embeddings, constants and buffers are frozen. The controller sees no targets: it reads cell-level latent states, their changes, candidate probabilities, entropy, clues, soft row/column/box count residuals, the previous action and the ACT-step embedding. Four attention queries read the 81-cell tokens, two width-96 Transformer blocks process them, and a width-96 GRU retains within-puzzle memory. Memory values persist across ACT updates but gradients truncate between optimizer steps; fresh puzzle slots reset all runtime history.", "",
        "The pooled-reader arm uses identical neural parameters/memory but averages the token representations. The activation arm uses the same controller, output dimensions and U directions; a label-free training-activation RMS calibration supplies constant coefficients instead of token-dependent xV. Both projection forms execute xV, but calibration does not guarantee identical realized perturbation norms or identical function classes. The schedule uses a global eight-value action plus sixteen eight-value ACT-step residuals.", "",
        "All trainable arms use the same task loss and periodic training-only oracle-imitation targets, with equal local-search budgets. Thus the comparison does not give revised B additional answer access. The teacher searches only training examples; separate diagnostic-oracle and evaluation labels never enter actor training. The oracle itself is a privileged optimization routine, not a deployable controller.", "",
        "| Seed | Arm | Completed updates | Trainable parameters | Puzzle admissions | Unique training puzzles | Teacher calls | Training seconds |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for stage in summary["stages"]:
        lines.append(f"| {stage['seed']} | {stage['kind']} | {stage['steps_executed']}/{stage['steps_requested']} | {stage['trainable_parameters']:,} | {stage['new_puzzle_admissions']:,} | {stage['unique_base_puzzles_seen']:,} | {stage['teacher_calls']} | {stage['seconds']:.1f} |")
    lines += ["", "| Controller | Gate RMS | Between-puzzle gate variance fraction | Realized adapter/base output RMS ratio |",
        "| --- | ---: | ---: | ---: |"]
    for row in means:
        lines.append(f"| {row['name']} | {row['gate_rms']:.5f} | {row['between_puzzle_gate_variance_fraction']:.5f} | {row['delta_to_base_rms_ratio']:.6f} |")
    lines += ["", "Variation alone is not a benefit: compare the held-out controls and the same-checkpoint interventions. Unique puzzle counts are shared across matched arms and cannot be summed as independent training examples. Equal actor/A-call/search budgets do not equalize all wall-time or every FLOP with the much smaller phase/no-adjustment controls.", "",
        "## Verification and practical limits", "",
        f"Independent verification: **{verification['checks_passed']}/{verification['checks_total']} passed**. {verification['scope']}", "",
        f"Local unit tests: **{unit_checks['passed']} passed** ({unit_checks['elapsed_seconds']:.2f} seconds), with a completed full-width integration preflight." if unit_checks else "No local unit-test evidence file was provided for this run.", "",
        "The source is still an early-learning Sudoku model, the fixed rank-four directions may offer little useful control, and there is only one A donor and two fresh actor seeds. The actor search, teacher budget, shallow temporal credit and small validation/test panels limit general conclusions. This pilot tests a revised controller at a shared frozen interface; it cannot adjudicate whether larger learned interfaces or explicit episodic meta-training would work.", "",
        "![Frozen-A controller comparison](plots/frozen_controls.png)", "",
        "## Reproduce locally", "", "```powershell",
        r"$python = 'G:\Projects\TransformerLab\plastic-latent-lab\.venv\Scripts\python.exe'",
        "& $python -m local_lab.frozen_run --config local_lab/config_frozen.json --stage oracle",
        "& $python -m local_lab.frozen_run --config local_lab/config_frozen.json --stage controllers",
        f"& $python scripts/verify_frozen_run.py --run-dir {cfg['results_dir']}",
        f"& $python -m local_lab.frozen_report --run-dir {cfg['results_dir']}",
        "```", "", "Run outputs are protected against overwrites. Use a fresh results_dir in a copied configuration. Runtime sources are preserved separately for the oracle and controller stages; reports do not load Torch or rerun inference."]
    # Scientific plot, saved as a normal local artifact. No model runtime needed.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    colors = ["#667988", "#bd821f", "#7b65ae", "#2469a5", "#36886a"]
    labels = ["A off", "Phase", "Pooled B", "Revised B", "Activation"][:len(means)]
    values = [100 * row["blank_accuracy"] for row in means]
    axes[0].bar(labels, values, color=colors[:len(means)])
    axes[0].set_ylim(0, max(values) * 1.2)
    axes[0].set_ylabel("Blank-cell accuracy (%)")
    axes[0].set_title("Fixed test panel: mean across fresh actor seeds")
    for index, value in enumerate(values):
        axes[0].text(index, value + .5, f"{value:.2f}%", ha="center")
    for kind in KINDS:
        selected = [row for row in normal_test if row["kind"] == kind]
        axes[1].plot([row["seed"] for row in selected], [100 * row["blank_accuracy"] for row in selected], marker="o", label=kind, color=colors[list(names).index(kind)])
    baseline = next(row for row in means if row["kind"] == "off")["blank_accuracy"] * 100
    axes[1].axhline(baseline, color=colors[0], linestyle="--", label="A off")
    axes[1].set_xticks(cfg["seeds"])
    axes[1].set_xlabel("Fresh controller seed")
    axes[1].set_ylabel("Blank-cell accuracy (%)")
    axes[1].set_title("Individual trained controllers")
    axes[1].legend()
    (directory / "plots").mkdir(exist_ok=True)
    fig.savefig(directory / "plots/frozen_controls.png", dpi=150)
    plt.close(fig)
    (directory / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (directory / "report_summary.json").write_text(json.dumps({"means": means, "paired": paired, "status": summary["status"]}, indent=2))
    inputs = ["config.json", "summary.json", "oracle.json", "calibration.json", "data_audit.json", "verification.json", "paired_effects.json"]
    if unit_checks:
        inputs.append("unit_checks.json")
    manifest = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in inputs}
    manifest["reporter_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    (directory / "report_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(json.dumps({"report": str(directory / "report.md"), "means": means, "verification_passed": verification["passed"]}))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="results/frozen_A_pilot_20261029")
    args = parser.parse_args()
    generate(ROOT / args.run_dir)


if __name__ == "__main__":
    main()
