"""Report a bounded native-TRM/controller experiment from saved evidence only.

No Torch imports or model execution. Interrupted and absent measurements remain
unavailable, and from-scratch Sudoku learning is not an isolated A-donor probe.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LABELS = {"native": "Native A", "static_joint": "Static joint", "joint": "Dynamic joint"}
COLORS = {"native": "#637583", "static_joint": "#B57A20", "joint": "#246AAA"}
METRICS = {
    "blank_accuracy": ("blank_accuracy", "blank_cell_accuracy"),
    "exact_accuracy": ("exact_accuracy", "exact_puzzle_accuracy"),
    "given_agreement": ("given_agreement", "given_clue_agreement", "clue_agreement"),
    "valid_grid_fraction": ("valid_grid_fraction", "valid_grid_accuracy", "valid_grids_fraction"),
    "all_accuracy": ("all_accuracy", "all_cell_accuracy", "accuracy"),
    "clue_copy_accuracy": ("clue_copy_accuracy", "clue_copy_baseline"),
    "loss": ("loss", "stablemax_loss", "lm_loss"),
}


def read(path, default=None):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def finite(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def number(value, digits=4):
    return f"{value:.{digits}f}" if finite(value) else "not recorded"


def percent(value):
    return f"{100 * value:.2f}%" if finite(value) else "not recorded"


def count(value):
    return f"{value:,}" if isinstance(value, int) and not isinstance(value, bool) else "not recorded"


def table(headers, rows):
    def cell(value):
        return str(value).replace("|", "\\|").replace("\n", " ")
    result = ["| " + " | ".join(map(cell, headers)) + " |",
              "| " + " | ".join("---" for _ in headers) + " |"]
    result.extend("| " + " | ".join(map(cell, row)) + " |" for row in rows)
    return "\n".join(result)


def first(mapping, keys, default=None):
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def records(value):
    if isinstance(value, list):
        return value
    if isinstance(value, dict):
        return value.get("rows", [])
    return []


def metric_rows(rows):
    """Keep saved units; do not convert unknown summed metrics to accuracies."""
    result = []
    for row in rows:
        raw = {**row, **row.get("metrics", {})}
        normalized = {**raw, **{key: first(raw, aliases) for key, aliases in METRICS.items()}}
        for key in METRICS:
            if key != "loss" and finite(normalized[key]) and not 0 <= normalized[key] <= 1:
                raise ValueError(f"Saved {key} must be a normalized fraction, not a summed count: {normalized[key]}")
        result.append(normalized)
    return result


def history_rows(directory):
    path = directory / "history.jsonl"
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def _plots(directory, evaluations, history, smoke=False):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = directory / "plots"
    output.mkdir(exist_ok=True)
    pictures = []
    chosen_split = "test" if any(row.get("split") == "test" for row in evaluations) else "val"
    # Raw and EMA are separate checkpoints; never pool them or duplicate bars.
    endpoints = [row for row in evaluations if row.get("split") == chosen_split and row.get("weights") == "raw"]
    if endpoints and any(finite(row.get("blank_accuracy")) for row in endpoints):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), constrained_layout=True)
        labels = [f"{LABELS.get(row.get('variant'), row.get('variant', '?'))}\n{row.get('mode', 'native')}" for row in endpoints]
        colors = [COLORS.get(row.get("variant"), "#8299A4") for row in endpoints]
        for ax, key, title in zip(axes, ("blank_accuracy", "exact_accuracy"), ("Unknown-cell accuracy", "Exact whole-puzzle accuracy")):
            values = [row.get(key) if finite(row.get(key)) else float("nan") for row in endpoints]
            bars = ax.bar(range(len(endpoints)), values, color=colors, width=.7)
            ax.bar_label(bars, labels=[percent(value) if finite(value) else "unavailable" for value in values], padding=3, fontsize=8)
            ax.set(title=title, ylabel="Accuracy", xticks=range(len(endpoints)), xticklabels=labels, ylim=(0, 1.04))
            ax.tick_params(axis="x", labelsize=8, rotation=25)
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
            ax.grid(axis="y", alpha=.16)
            if key == "blank_accuracy":
                ax.axhline(1 / 9, color="#777777", linestyle=":", linewidth=1, label="Uniform digit reference, 11.11%")
                ax.legend(fontsize=7)
        fig.suptitle("Integration smoke only: raw final weights, not hypothesis evidence" if smoke else
                     f"Bounded from-scratch TRM: {chosen_split}, raw final weights", fontsize=13)
        name = "sudoku_endpoints.png"
        fig.savefig(output / name, dpi=170, facecolor="white")
        plt.close(fig)
        pictures.append(f"plots/{name}")
    grouped = defaultdict(list)
    for row in history:
        grouped[row.get("variant", "unknown")].append(row)
    if grouped:
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), constrained_layout=True)
        panels = (("loss", ("train_loss",), "Training Stablemax + 0.5 halt BCE", "Per-puzzle normalized total loss"),
                  ("blank_accuracy", ("blank_accuracy", "blank_cell_accuracy"), "Logged training unknown-cell accuracy", "Accuracy"))
        plotted = False
        for ax, (key, aliases, title, ylabel) in zip(axes, panels):
            for variant, rows in grouped.items():
                points = [(first(row, ("step", "optimizer_step", "update")), first({**row, **row.get("metrics", {})}, aliases)) for row in rows]
                points = sorted((x, y) for x, y in points if finite(x) and finite(y))
                if points:
                    plotted = True
                    ax.plot([x for x, _ in points], [y for _, y in points], marker="o", markersize=3, linewidth=1.7,
                            color=COLORS.get(variant, "#8299A4"), label=LABELS.get(variant, variant))
            ax.set(title=title, xlabel="Recorded optimization step", ylabel=ylabel)
            ax.grid(axis="y", alpha=.17)
            if key == "blank_accuracy":
                ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
                ax.axhline(1 / 9, color="#777777", linestyle=":", linewidth=1)
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                ax.legend(fontsize=8)
            else:
                ax.text(.5, .5, "Metric not recorded", ha="center", va="center", transform=ax.transAxes)
        if plotted:
            fig.suptitle("Integration smoke only: training batches" if smoke else
                         "Local optimization traces; sampled training batches", fontsize=13)
            name = "sudoku_training.png"
            fig.savefig(output / name, dpi=170, facecolor="white")
            pictures.append(f"plots/{name}")
        plt.close(fig)
    return pictures


def matching_checks(summary, stages, evaluations, cfg, history=None):
    history = history or []
    expected = cfg.get("variants", ["native", "static_joint", "joint"])
    checks = {"all_configured_arms_recorded": {row.get("variant") for row in stages} == set(expected),
              "same_initial_full_state": bool(summary.get("initial_state_sha256")) and bool(stages) and
              all(row.get("initial_state_sha256") == summary["initial_state_sha256"] for row in stages),
              "all_stages_completed_requested_steps": bool(stages) and all(
                  row.get("complete") is True and row.get("steps_executed") == row.get("steps_requested") == cfg.get("steps") for row in stages),
              "equal_completed_step_budgets": bool(stages) and len({row.get("steps_executed") for row in stages}) == 1,
              "same_proposed_candidate_stream": bool(stages) and all(row.get("candidate_schedule_sha256") for row in stages) and len({row.get("candidate_schedule_sha256") for row in stages}) == 1,
              "all_evaluations_use_16_ACT_steps": bool(evaluations) and all(row.get("act_steps") == 16 for row in evaluations),
              "evaluation_parameter_states_unchanged": bool(evaluations) and all(row.get("parameter_state_unchanged") is True for row in evaluations),
              "evaluation_runtime_gates_cleared": bool(evaluations) and all(row.get("runtime_gates_cleared") is True for row in evaluations),
              "logged_training_reasoning_calls_21": bool(history) and all(row.get("reasoning_calls") == 21 for row in history),
              "logged_training_gradient_enabled_calls_7": bool(history) and all(row.get("gradient_enabled_calls") == 7 for row in history),
              "logged_training_controller_calls_match_mode": bool(history) and all(row.get("controller_calls") == (21 if row.get("variant") == "joint" else 0) for row in history)}
    admitted_match = bool(stages) and all(row.get("admitted_schedule_sha256") for row in stages) and len({row.get("admitted_schedule_sha256") for row in stages}) == 1
    return {"checks": checks, "all_checks_passed": all(checks.values()),
            "actual_admission_streams_equal": admitted_match,
            "actual_admission_note": "Actual admissions are model-dependent under native ACT; equal proposed batches do not guarantee equal admitted puzzles."}


def build_report(run_dir, plots=True):
    directory = Path(run_dir).resolve()
    summary = read(directory / "summary.json")
    summary_file = "summary.json"
    if summary is None:
        summary = read(directory / "training_summary.json")
        summary_file = "training_summary.json"
    if not isinstance(summary, dict):
        raise FileNotFoundError(f"No saved runner summary in {directory}")
    cfg = read(directory / "config.json", {})
    audit = read(directory / "data_audit.json", read(directory / "dataset_audit.json", {}))
    evaluation_file = read(directory / "evaluations.json")
    evaluations = metric_rows(records(evaluation_file if evaluation_file is not None else summary.get("evaluations", [])))
    stages = records(summary.get("stages", summary.get("variants", [])))
    history = history_rows(directory)
    verification = read(directory / "verification.json", read(directory / "independent_verification.json", {}))
    status = summary.get("status", "not recorded")
    matching = matching_checks(summary, stages, evaluations, cfg, history)
    normal_modes = {"native": "off", "static_joint": "static", "joint": "dynamic"}
    primary = [row for row in evaluations if row.get("split") == "test" and row.get("weights") == "raw"
               and row.get("mode") == normal_modes.get(row.get("variant"))]
    primary_lookup = {row["variant"]: row for row in primary}
    effects = []
    for split in ("val", "test"):
        rows = {row["mode"]: row for row in evaluations if row.get("variant") == "joint" and row.get("weights") == "raw" and row.get("split") == split}
        if "dynamic" in rows:
            dynamic = rows["dynamic"]
            for mode in ("off", "fixed"):
                other = rows.get(mode)
                if other is not None:
                    effects.append({"split": split, "comparator_mode": mode,
                                    "blank_accuracy_gain": dynamic["blank_accuracy"] - other["blank_accuracy"],
                                    "exact_accuracy_gain": dynamic["exact_accuracy"] - other["exact_accuracy"],
                                    "loss_reduction": other["loss"] - dynamic["loss"]})
    conclusion = ("This run measures early from-scratch Sudoku learning with native TRM and our plasticity controller. "
                  "It does not measure isolated A-weight adaptability, and it does not reproduce the paper's final benchmark accuracy.")
    if set(primary_lookup) == set(normal_modes):
        outcomes = "; ".join(f"{LABELS[variant]} {percent(primary_lookup[variant]['blank_accuracy'])} unknown-cell / {percent(primary_lookup[variant]['exact_accuracy'])} exact-puzzle" for variant in normal_modes)
        conclusion = "Raw final test-panel scores: " + outcomes + ". " + conclusion
    is_smoke = cfg.get("steps", 0) <= 2
    if is_smoke:
        conclusion = "**Integration smoke only; its short scores are not hypothesis evidence.** " + conclusion
    if status != "complete":
        conclusion = f"**Saved run status: `{status}`; incomplete and absent cells are unavailable evidence.** " + conclusion
    pictures = _plots(directory, evaluations, history, smoke=is_smoke) if plots else []
    lines = ["# Local native TRM and controller B", "", conclusion, "",
             f"Saved status: `{status}`. Seed: `{first(summary, ('seed', 'seeds'), first(cfg, ('seed', 'seeds'), 'not recorded'))}`. Wall time: {number(first(summary, ('elapsed_seconds', 'wall_seconds')), 1)} seconds. Configured bound: {number(first(summary, ('max_seconds',), cfg.get('max_seconds')), 0)} seconds.", "",
             "A retains native width 512 and two shared reasoning layers. Each ACT inner update uses H=3 outer cycles and L=6 latent updates, giving 21 shared-module calls: 14 without gradients and seven with gradients. Carries detach between ACT updates, and evaluation retains the native maximum of 16 ACT updates. Our B uses width 128, three causal Transformer layers, eight heads, FF width 512, and history four; it controls rank-four temporary updates on the two native MLP down-projections at scale 0.2. B's history resets per ACT inner update. The preserved executed configuration and sources establish the actual settings.", "",
             "Native A-only, static joint, and dynamic joint arms start from the same full state and receive the same proposed candidate stream. Native ACT decides which slots admit new puzzles, so actual admissions are checked separately. Stablemax, halt BCE, AdamATan2, sparse puzzle-embedding optimization, and EMA follow the local runner's native setup. Dynamic-checkpoint gates-off/fixed evaluations are counterfactual inference modes, not independently trained controls.", "",
             f"Recorded native A dense parameters: **{count(summary.get('A_parameters'))}**. Our B parameters: **{count(summary.get('B_parameters'))}**. The separately optimized native puzzle-embedding table has {count(summary.get('native_puzzle_embedding_table_elements'))} elements; its batch-local gradient buffer has {count(summary.get('native_puzzle_gradient_buffer_elements'))} elements, which are not additional logical puzzle-embedding degrees of freedom. Dense optimizer: {summary.get('optimizer', 'not recorded')}. Native sparse optimizer: {summary.get('native_sparse_optimizer', 'not recorded')}.", "",
             f"The saved pool contains 512 validation and 512 test puzzles. Endpoint evaluations use the first **{count(cfg.get('eval_size'))}** puzzles of each pool, fixed across arms, modes, and weight versions. These are prefix panels, not full-pool scores. Evaluation batch size: {count(cfg.get('eval_batch_size'))}.", "",
             "## Saved evaluations", "",
             "Unknown-cell accuracy excludes given clues. Exact accuracy requires the whole saved target solution. Given agreement measures copied clues; valid-grid fraction requires a completed legal Sudoku that also preserves all givens. All-cell accuracy includes copied clues and can look strong when unknown cells remain wrong. The clue-copy baseline exposes that inflation. A uniform guess over nine digits gives an 11.11% unknown-cell reference, rather than a trained solver.", ""]
    endpoint_rows = []
    for row in evaluations:
        endpoint_rows.append([LABELS.get(row.get("variant"), row.get("variant", "not recorded")), row.get("weights", "not recorded"), row.get("mode", "native"), row.get("split", "not recorded"),
                              count(first(row, ("n", "puzzles", "n_puzzles"))), percent(row.get("blank_accuracy")), percent(row.get("exact_accuracy")),
                              percent(row.get("given_agreement")), percent(row.get("valid_grid_fraction")), percent(row.get("all_accuracy")), percent(row.get("clue_copy_accuracy")), number(row.get("loss"))])
    if endpoint_rows:
        lines.extend([table(["Arm", "Weights", "Mode", "Split", "Puzzles", "Unknown-cell acc", "Exact puzzle acc", "Given agreement", "Valid grids", "All-cell acc", "Clue-copy baseline", "Stablemax per cell"], endpoint_rows), ""])
    else:
        lines.extend(["No endpoint evaluation was saved. Missing measurements are not zero-accuracy results.", ""])
    lines.extend(["Evaluation checkpoint policy: " + str(first(summary, ("primary_weights", "evaluation_checkpoint_policy", "checkpoint_policy", "selection"), "not recorded; inspect the saved runner source and checkpoint metadata")) + ".", "",
                  "Raw final weights are primary. EMA is secondary and is kept separate in every table; its native rate of 0.999 can lag substantially in a short from-scratch run. No raw/EMA averaging occurs. Endpoint loss is normalized per-cell Stablemax loss. Training curves log per-puzzle normalized Stablemax task loss plus 0.5 × halting BCE on the sampled training batch. No held-out validation curve was measured during training, and no validation-best checkpoint was selected.", "",
                  "Raw dynamic-checkpoint interventions compare identical A/B weights and test examples. Positive gains favor evolving dynamic gates over the indicated inference mode:", "",
                  table(["Panel", "Comparator mode", "Dynamic unknown-cell gain (pp)", "Dynamic exact gain (pp)", "Dynamic Stablemax decrease"],
                        [[row["split"], row["comparator_mode"], f"{100 * row['blank_accuracy_gain']:+.3f}", f"{100 * row['exact_accuracy_gain']:+.3f}", f"{row['loss_reduction']:+.5f}"] for row in effects]), "",
                  "## Exposure and local resources", "",
                  "Loader advancement, ACT batch-slot presentations, and newly admitted source puzzles are different quantities. An unhalted carry can continue an earlier puzzle. Training augmentations create new symmetry views of existing puzzles, not independent new puzzles. Use the admission and unique-source counters below to assess actual data coverage.", ""])
    stage_rows = []
    for row in stages:
        unique_base = first(row, ("unique_base_puzzles_seen", "unique_source_puzzles", "unique_training_puzzles", "unique_puzzles_seen"))
        pool_size = audit.get("requested_sizes", {}).get("train")
        stage_rows.append([LABELS.get(row.get("variant"), row.get("variant", row.get("stage", "not recorded"))),
                           f"{row.get('steps_executed', row.get('steps', '?'))}/{row.get('steps_requested', '?')}",
                           count(first(row, ("new_puzzle_admissions", "puzzle_admissions", "new_puzzles"))),
                           count(unique_base), percent(unique_base / pool_size if finite(unique_base) and finite(pool_size) and pool_size > 0 else None),
                           count(row.get("unique_augmented_puzzles_seen")),
                           count(first(row, ("act_slot_presentations", "slot_presentations", "examples_seen"))),
                           count(first(row, ("trainable_parameters", "parameters"))),
                           number(first(row, ("training_seconds", "elapsed_seconds")), 1), row.get("complete", "not recorded")])
    if stage_rows:
        lines.extend([table(["Arm", "Executed/requested steps", "New puzzle admissions", "Unique source puzzles", "Train-pool coverage", "Distinct admitted augmentation views", "ACT-slot presentations", "Trainable parameters", "Seconds", "Complete"], stage_rows), ""])
    else:
        lines.extend(["No per-arm exposure summary was saved.", ""])
    peak = first(summary, ("peak_cuda_allocated_bytes", "peak_cuda_bytes"))
    lines.extend([f"Peak CUDA tensor allocation: {number(peak / 2**30 if finite(peak) else None, 2)} GiB. GPU: `{summary.get('gpu', 'not recorded')}`. Tensor allocation excludes driver and other-process memory. Timing includes whatever the runner records as stage work and is not a dedicated hardware benchmark.", "",
                  table(["Saved matching / inference check", "Passed"], [[key.replace("_", " "), value] for key, value in matching["checks"].items()]), "",
                  f"All recorded matching/inference checks passed: **{matching['all_checks_passed']}**. Actual admitted puzzle streams equal: **{matching['actual_admission_streams_equal']}**. ACT admissions can differ as models halt at different times. An admission difference is a consequence of native ACT and must be considered when attributing gains; equal candidate batches alone do not establish equal realized training data.", "",
                  "## Data and integrity", ""])
    if audit:
        source = {key: audit[key] for key in ("repository", "source", "source_repo", "revision", "source_revision", "stream_prefix", "selection", "requested_sizes", "splits", "counts", "encoding", "checks", "passed", "limitations") if key in audit}
        lines.extend(["The saved data evidence records pinned source identity, stream selection, split sizes, hashes, and integrity checks. Data labels are training/evaluation targets; B receives no solutions at inference.", "",
                      "```json", json.dumps(source, indent=2), "```", ""])
    else:
        lines.extend(["Data audit not recorded in the expected evidence file.", ""])
    if verification:
        lines.extend([f"Saved independent verification passed: **{verification.get('passed', 'not recorded')}**. Recorded check count: {count(len(verification.get('checks', [])))}. Scope: {verification.get('scope', 'not recorded')}", ""])
    lines.extend(["## Figures", ""])
    for picture in pictures:
        lines.extend([f"![Local TRM evidence]({picture})", ""])
    if not pictures:
        lines.extend(["No quantitative figure could be generated from the saved metrics.", ""])
    lines.extend(["## Interpretation and limits", "",
                  "A begins from scratch. An early dynamic/static/native difference can reflect ordinary task learning, controller dependence, initialization, optimization noise, or unequal effective work if any matching check fails. It is not an isolated test of whether A learned to become easier to adapt. This single-seed diagnostic shares its puzzle collection across arms; examples and augmented views must not be treated as independent model-training replicates.", "",
                  "The native README's smallest full Sudoku experiment takes about 18 hours on a 48 GB L40S. The short local 16 GB run is setup and early-learnability evidence. Paper headline accuracy, broad retention, and H3 remain unestablished. A future adaptability experiment needs a learned A, matched fresh-B A-only donor swaps, step-0 and learning-speed curves, static-joint controls, and independent A-training seeds.", "",
                  "Upstream references: [official code and native training protocol](https://github.com/SamsungSAILMontreal/TinyRecursiveModels), [primary paper](https://arxiv.org/html/2510.04871v1). These links describe the upstream research; the local saved evidence establishes this run's behavior.", "",
                  "The reporter imports no Torch, loads no model weights, performs no training, and publishes nothing. `report_summary.json` preserves normalized saved metric rows; `report_manifest.json` hashes evidence and the reporter. Original configuration, logs, data audit, checkpoints, and source snapshots remain the owning evidence.", ""])
    (directory / "report.md").write_text("\n".join(lines), encoding="utf-8")
    result = {"status": status, "conclusion": conclusion, "evaluations": evaluations, "stages": stages,
              "matching": matching,
              "primary_raw_test_endpoints": primary, "raw_dynamic_intervention_effects": effects,
              "integration_smoke_only": is_smoke,
              "H3_isolated_adaptability_measured": False, "paper_benchmark_reproduced": False,
              "aggregation_unit": "one optimization run; shared source puzzles across matched arms"}
    (directory / "report_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    filenames = ("config.json", summary_file, "evaluations.json", "history.jsonl", "data_audit.json", "dataset_audit.json", "verification.json", "independent_verification.json", "source_manifest.json")
    manifest = {"reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "inputs_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in filenames if (directory / name).exists()}, "plots": pictures}
    (directory / "report_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/local_sudoku")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    directory = Path(args.run_dir)
    directory = directory if directory.is_absolute() else ROOT / directory
    result = build_report(directory, plots=not args.no_plots)
    print(f"Wrote {directory / 'report.md'}; saved status: {result['status']}")


if __name__ == "__main__":
    main()
