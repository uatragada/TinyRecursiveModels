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
import os
from pathlib import Path
from statistics import fmean


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


def saved_pool_size(audit, split):
    return audit.get("requested_sizes", {}).get(split, audit.get("splits", {}).get(split, {}).get("n"))


def validation_evidence(directory, cfg):
    """A new run must not inherit the earlier run's validation claims silently."""
    candidates = [directory / "local_checks.json"]
    if cfg.get("validation_file"):
        path = Path(cfg["validation_file"])
        candidates.append(path if path.is_absolute() else ROOT / path)
    if directory.name in ("local_sudoku", "local_sudoku_smoke", "local_sudoku_probe"):
        candidates.append(ROOT / "results" / "local_checks.json")
    path = next((path for path in candidates if path.exists()), None)
    return (read(path, {}), path) if path else ({}, None)


def report_link(directory, target):
    return os.path.relpath(target, directory).replace("\\", "/")


def normalized_gain_auc(points):
    """Trapezoidal observed gain AUC, normalized by its step span.

    Two endpoints alone are insufficient for the multi-checkpoint AUC claim.
    """
    if len(points) < 3 or points[-1][0] <= points[0][0]:
        return None
    return sum((x1 - x0) * (y0 + y1) / 2 for (x0, y0), (x1, y1) in zip(points, points[1:])) / (points[-1][0] - points[0][0])


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
    tensor_verification = read(directory / "tensor_verification.json", {})
    local_checks, local_checks_path = validation_evidence(directory, cfg)
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
             "Identical saved initial parameters do not imply identical initial functions: native uses gates off, static uses tanh(0.01), and dynamic uses B's near-zero predicted gates. Equal completed optimizer steps and A reasoning passes also do not equalize total compute or wall time; B adds its own computation and about 810,000 parameters.", "",
             f"Recorded native A dense parameters: **{count(summary.get('A_parameters'))}**. Our B parameters: **{count(summary.get('B_parameters'))}**. The separately optimized native puzzle-embedding table has {count(summary.get('native_puzzle_embedding_table_elements'))} elements; its batch-local gradient buffer has {count(summary.get('native_puzzle_gradient_buffer_elements'))} elements, which are not additional logical puzzle-embedding degrees of freedom. Dense optimizer: {summary.get('optimizer', 'not recorded')}. Native sparse optimizer: {summary.get('native_sparse_optimizer', 'not recorded')}.", "",
             f"Saved validation pool: {count(saved_pool_size(audit, 'val'))} puzzles; saved test pool: {count(saved_pool_size(audit, 'test'))}. Configured endpoint size: {count(cfg.get('eval_size'))}; evaluation batch size: {count(cfg.get('eval_batch_size'))}. Every arm, mode, and weight version uses the same saved examples. A panel equal to a pool's size evaluates that full saved pool; a smaller panel uses its prefix. The saved pools themselves are bounded sequential subsets of the official dataset, not the complete published benchmark.", "",
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
                  f"Raw final weights are primary. EMA is secondary and is kept separate in every table; its configured rate is {number(cfg.get('ema_rate'), 4)}. EMA can lag during learning, so raw and EMA results are never averaged. Endpoint loss is normalized per-cell Stablemax loss. Training curves log per-puzzle normalized Stablemax task loss plus 0.5 × halting BCE on the sampled training batch. The plotted history contains sampled training measurements; it does not invent held-out validation curves or a validation-best checkpoint selection.", "",
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
    if tensor_verification:
        lines.extend([f"Separate saved-tensor verification passed: **{tensor_verification.get('passed', 'not recorded')}**, with {count(tensor_verification.get('checks_passed'))}/{count(tensor_verification.get('checks_total'))} checks. Scope: {tensor_verification.get('scope', 'not recorded')}", "",
                      table(["Arm", "Changed A tensors", "Changed B tensors", "Changed adapter tensors", "Changed static gates"],
                            [[LABELS.get(variant, variant), row.get("A", "?"), row.get("B", "?"), row.get("adapters", "?"), row.get("static_gates", "?")] for variant, row in tensor_verification.get("changed_tensor_counts", {}).items()]), ""])
    unit = local_checks.get("final_validation", {}).get("unit_tests", {})
    if unit:
        lines.extend([f"Shared final local validation records **{count(unit.get('passed'))} tests passed** in {number(unit.get('elapsed_seconds'), 2)} seconds using `{unit.get('command', 'not recorded')}`. Discovery: {unit.get('discovery', 'not recorded')}. This reporter does not rerun those tests.", ""])
    related_probe = cfg.get("probe_results_dir", "results/local_sudoku_probe" if directory.name == "local_sudoku" else None)
    probe_report = ROOT / related_probe / "probe_report.md" if related_probe else None
    if probe_report and probe_report.exists():
        lines.extend([f"The [isolated A-weight probe]({report_link(directory, probe_report)}) is reported separately. It freezes donor A and adapts common fresh B plus U/V; source-learning scores in this report are not substituted for those adaptation results.", ""])
    lines.extend(["## Figures", ""])
    for picture in pictures:
        lines.extend([f"![Local TRM evidence]({picture})", ""])
    if not pictures:
        lines.extend(["No quantitative figure could be generated from the saved metrics.", ""])
    lines.extend(["## Interpretation and limits", "",
                  "A begins from scratch. An early dynamic/static/native difference can reflect ordinary task learning, controller dependence, initialization, optimization noise, or unequal effective work if any matching check fails. It is not an isolated test of whether A learned to become easier to adapt. This single-seed diagnostic shares its puzzle collection across arms; examples and augmented views must not be treated as independent model-training replicates.", "",
                  "The native README's smallest full Sudoku experiment takes about 18 hours on a 48 GB L40S. This bounded local run uses its own recorded data and optimization protocol. It does not reproduce the complete published benchmark. Source-task accuracy alone establishes neither broad retention nor H3; the separate probe must compare donor initial states, learning changes, static-joint controls, and independent A-training seeds.", "",
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
    filenames = ("config.json", summary_file, "evaluations.json", "history.jsonl", "data_audit.json", "dataset_audit.json", "verification.json", "independent_verification.json", "tensor_verification.json", "source_manifest.json")
    manifest = {"reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "inputs_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in filenames if (directory / name).exists()}, "plots": pictures}
    if local_checks_path:
        manifest["local_checks_path"] = str(local_checks_path)
        manifest["shared_local_checks_sha256"] = hashlib.sha256(local_checks_path.read_bytes()).hexdigest()
    (directory / "report_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return result


def build_probe_report(run_dir, plots=True):
    """Keep the frozen-A/fresh-B comparison distinct from source training."""
    directory = Path(run_dir).resolve()
    summary = read(directory / "summary.json")
    if not isinstance(summary, dict):
        raise FileNotFoundError(f"No saved probe summary in {directory}")
    cfg = read(directory / "config.json", summary.get("config", {}))
    stages = records(summary.get("stages", []))
    saved = read(directory / "evaluations.json")
    evaluations = metric_rows(records(saved if saved is not None else summary.get("evaluations", [])))
    history = history_rows(directory)
    verification = read(directory / "verification.json", {})
    tensor_verification = read(directory / "tensor_verification.json", {})
    local_checks = read(ROOT / "results" / "local_checks.json", {})
    variants = ("native", "static_joint", "joint")
    donor_labels = {"native": "Native-trained A donor", "static_joint": "Static-joint A donor", "joint": "Dynamic-joint A donor"}
    rows_by_variant = {variant: sorted([row for row in evaluations if row.get("variant") == variant and row.get("mode") == "dynamic" and row.get("weights") == "raw" and row.get("split") == "test"], key=lambda row: row["step"]) for variant in variants}
    learning = []
    for variant, rows in rows_by_variant.items():
        if len({row["step"] for row in rows}) != len(rows):
            raise ValueError(f"Duplicate dynamic probe checkpoint for {variant}")
        if not rows or rows[0]["step"] != 0:
            continue
        if len({row["n"] for row in rows}) != 1:
            raise ValueError("Probe checkpoint panel sizes must match")
        initial, final = rows[0], rows[-1]
        learning.append({"variant": variant, "n": initial["n"], "steps": [row["step"] for row in rows],
                         "initial_blank_accuracy": initial["blank_accuracy"], "final_blank_accuracy": final["blank_accuracy"],
                         "blank_accuracy_gain": final["blank_accuracy"] - initial["blank_accuracy"],
                         "initial_exact_accuracy": initial["exact_accuracy"], "final_exact_accuracy": final["exact_accuracy"],
                         "initial_loss": initial["loss"], "final_loss": final["loss"],
                         "loss_reduction": initial["loss"] - final["loss"]})
    checks = {
        "all_three_donor_arms_recorded": {row.get("variant") for row in stages} == set(variants),
        "all_probe_stages_complete": bool(stages) and all(row.get("complete") is True and row.get("steps_executed") == row.get("steps_requested") for row in stages),
        "equal_executed_steps": bool(stages) and len({row.get("steps_executed") for row in stages}) == 1,
        "common_fresh_non_A_initialization": bool(summary.get("common_fresh_non_A_sha256")) and bool(stages) and all(row.get("initial_non_A_sha256") == summary["common_fresh_non_A_sha256"] for row in stages),
        "A_hashes_unchanged": bool(stages) and all(row.get("A_unchanged") is True and bool(row.get("A_sha256_before")) and row.get("A_sha256_before") == row.get("A_sha256_after") for row in stages),
        "puzzle_embeddings_unchanged": bool(stages) and all(row.get("puzzle_embedding_unchanged") is True for row in stages),
        "all_native_buffers_unchanged": bool(stages) and all(row.get("native_buffers_unchanged") is True and bool(row.get("native_buffers_sha256_before")) and row.get("native_buffers_sha256_before") == row.get("native_buffers_sha256_after") for row in stages),
        "same_candidate_streams": bool(stages) and all(row.get("candidate_schedule_sha256") for row in stages) and len({row.get("candidate_schedule_sha256") for row in stages}) == 1,
        "all_test_endpoints_use_16_ACT_steps": bool(evaluations) and all(row.get("act_steps") == 16 for row in evaluations),
        "evaluation_state_unchanged": bool(evaluations) and all(row.get("parameter_state_unchanged") is True for row in evaluations),
        "evaluation_gates_cleared": bool(evaluations) and all(row.get("runtime_gates_cleared") is True for row in evaluations),
        "three_matched_step_0_final_panels": len(learning) == 3 and len({tuple(row["steps"]) for row in learning}) == 1 and all(len(row["steps"]) >= 2 for row in learning),
    }
    matched_admissions = bool(stages) and all(row.get("admitted_schedule_sha256") for row in stages) and len({row.get("admitted_schedule_sha256") for row in stages}) == 1
    lookup = {row["variant"]: row for row in learning}
    contrasts = []
    if all(checks.values()):
        for comparator in ("native", "static_joint"):
            joint, other = lookup["joint"], lookup[comparator]
            contrasts.append({"comparator": comparator,
                              "initial_blank_accuracy_advantage": joint["initial_blank_accuracy"] - other["initial_blank_accuracy"],
                              "final_blank_accuracy_advantage": joint["final_blank_accuracy"] - other["final_blank_accuracy"],
                              "extra_blank_accuracy_gain": joint["blank_accuracy_gain"] - other["blank_accuracy_gain"],
                              "final_loss_advantage": other["final_loss"] - joint["final_loss"],
                              "extra_loss_reduction": joint["loss_reduction"] - other["loss_reduction"]})
    status = summary.get("status", "not recorded")
    conclusion = "The saved probe does not yet provide a complete matched three-donor comparison."
    if len(learning) == 3:
        outcomes = "; ".join(f"{donor_labels[variant]} {percent(lookup[variant]['initial_blank_accuracy'])} → {percent(lookup[variant]['final_blank_accuracy'])}" for variant in variants)
        conclusion = "Fresh-B test-panel unknown-cell accuracy, step 0 → final: " + outcomes + ". "
        conclusion += "This is same-task readaptation on new Sudoku instances, conditional on one source A-training seed and one fresh-B seed."
        static_contrast = next((row for row in contrasts if row["comparator"] == "static_joint"), None)
        if static_contrast and static_contrast["extra_blank_accuracy_gain"] <= 0 and static_contrast["final_loss_advantage"] <= 0:
            conclusion += " The dynamic-joint A donor shows no consistent adaptability advantage over the static-joint donor in this bounded run."
    if status != "complete":
        conclusion = f"**Saved probe status: `{status}`; incomplete cells are unavailable evidence.** " + conclusion
    lines = ["# Isolated A-weight probe with fresh B and adapter factors", "", conclusion, "",
             "The source models and their completed early-learning comparison are documented in the separate [source-training report](../local_sudoku/report.md).", "",
             "This report is separate from the source-training comparison. It transplants native A's inner state from each raw source donor while excluding the low-rank adapter factors. A's weights, constants, and puzzle-embedding table are frozen. Each arm receives the same fresh B, U/V factors, static gates, and other non-A initial state. B and the fresh U/V factors train; all native A tensors remain frozen. Thus the donor intervention changes only A while the complete trainable small interface starts identically. This is broader than training B alone.", "",
             "Adaptation uses the 384 previously unused validation-pool puzzles after the first 128 source-evaluation puzzles. Training-only legal Sudoku augmentations are new views of those puzzles. The test panel remains the same first 128 official-test-prefix puzzles. This tests readaptation within Sudoku, not an unfamiliar task family or a held-out graph-composition/path-length shift.", "",
             f"Saved status: `{status}`. Fresh-B seed: `{cfg.get('seed', summary.get('seed', 'not recorded'))}`. Wall time: {number(summary.get('elapsed_seconds'), 1)} seconds of {number(first(summary, ('max_seconds',), cfg.get('max_seconds')), 0)}. All recorded probe evaluations use raw weights; EMA is not mixed into this probe.", "",
             f"Peak CUDA tensor allocation: {number(summary.get('peak_cuda_allocated_bytes', 0) / 2**30, 2)} GiB. This excludes driver and other-process allocation.", "",
             "## Step-0 state and learning changes", "",
             "Unknown-cell gains subtract each donor arm's own initial score. Stablemax reduction is initial minus final loss. Different initial A capabilities can create different headroom, so final scores, initial scores, and changes must be interpreted together. Only step 0 and the final probe checkpoint are evaluated; this does not identify a detailed learning-speed curve or an observed multi-checkpoint gain AUC.", ""]
    lines.extend([table(["A donor", "Test puzzles", "Recorded steps", "Initial unknown-cell acc", "Final unknown-cell acc", "Gain (pp)", "Initial Stablemax", "Final Stablemax", "Stablemax decrease", "Final exact-puzzle acc"],
                        [[donor_labels[row["variant"]], count(row["n"]), str(row["steps"]), percent(row["initial_blank_accuracy"]), percent(row["final_blank_accuracy"]), f"{100 * row['blank_accuracy_gain']:+.3f}", number(row["initial_loss"]), number(row["final_loss"]), f"{row['loss_reduction']:+.5f}", percent(row["final_exact_accuracy"])] for row in learning]), "",
                  "Positive paired values below favor the dynamic-joint A donor over the indicated source donor:", "",
                  table(["Comparator donor", "Initial unknown-cell advantage (pp)", "Final unknown-cell advantage (pp)", "Extra unknown-cell gain (pp)", "Final Stablemax advantage", "Extra Stablemax decrease"],
                        [[donor_labels[row["comparator"]], f"{100 * row['initial_blank_accuracy_advantage']:+.3f}", f"{100 * row['final_blank_accuracy_advantage']:+.3f}", f"{100 * row['extra_blank_accuracy_gain']:+.3f}", f"{row['final_loss_advantage']:+.5f}", f"{row['extra_loss_reduction']:+.5f}"] for row in contrasts]), "",
                  "An endpoint advantage inherited entirely from step 0 is not an improvement in adaptation. A larger reduction from a worse initial loss is also not sufficient without a useful final result. The source models remain early in learning and have not demonstrated complete Sudoku solving; this limits conclusions about learned adaptability.", "",
                  "## Isolation and exposure checks", "",
                  table(["Recorded check", "Passed"], [[key.replace("_", " "), value] for key, value in checks.items()]), "",
                  f"All recorded isolation/matching checks passed: **{all(checks.values())}**. Actual admitted augmentation streams equal: **{matched_admissions}**. Native ACT can make actual admissions model-dependent even when proposed candidates match. Checkpoint hashes are saved evidence; independent verification scope is stated separately below.", ""])
    lines.extend([table(["A donor", "Steps", "New puzzle admissions", "Unique adaptation puzzles", "Adaptation-pool coverage", "Distinct augmentation views", "ACT-slot presentations", "Trainable B + U/V parameters", "Seconds"],
                        [[donor_labels.get(row.get("variant"), row.get("variant", "?")), f"{row.get('steps_executed', '?')}/{row.get('steps_requested', '?')}", count(row.get("new_puzzle_admissions")), count(row.get("unique_base_puzzles_seen")), percent(row["unique_base_puzzles_seen"] / 384 if finite(row.get("unique_base_puzzles_seen")) else None), count(row.get("unique_augmented_puzzles_seen")), count(row.get("act_slot_presentations")), count(row.get("trainable_parameters")), number(row.get("training_seconds"), 1)] for row in stages]), "",
                  "Equal probe steps match native A reasoning work, not total source-training history or all model FLOPs. Source and probe exposure are separate; repeated ACT slots and augmented views do not create independent puzzles or optimization seeds.", ""])
    pictures = []
    if plots and learning:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        (directory / "plots").mkdir(exist_ok=True)
        fig, axes = plt.subplots(1, 2, figsize=(11.7, 4.8), constrained_layout=True)
        for ax, field, title in zip(axes, ("blank_accuracy", "loss"), ("Unknown-cell accuracy", "Per-cell Stablemax loss")):
            for variant in variants:
                rows = rows_by_variant[variant]
                if rows:
                    ax.plot([row["step"] for row in rows], [row[field] for row in rows], marker="o", linewidth=1.8,
                            color=COLORS[variant], label=donor_labels[variant])
            ax.set(title=title, xlabel="Fresh B + U/V optimization steps")
            ax.grid(axis="y", alpha=.17)
            if field == "blank_accuracy":
                ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1))
            ax.legend(fontsize=8)
        fig.suptitle("A frozen; fresh B + U/V adapted: only step 0 and final measured", fontsize=13)
        name = "isolated_A_probe.png"
        fig.savefig(directory / "plots" / name, dpi=170, facecolor="white")
        plt.close(fig)
        pictures.append(f"plots/{name}")
        lines.extend([f"![Isolated A-donor probe]({pictures[0]})", "",
                      "Lines connect the two measured endpoints for readability; intermediate test scores were not measured.", ""])
    if verification:
        lines.extend([f"Saved independent verification passed: **{verification.get('passed', 'not recorded')}** across {count(len(verification.get('checks', [])))} checks. Scope: {verification.get('scope', 'not recorded')}", ""])
    if tensor_verification:
        lines.extend([f"Separate actual-tensor verification passed: **{tensor_verification.get('passed', 'not recorded')}**, with {count(tensor_verification.get('checks_passed'))}/{count(tensor_verification.get('checks_total'))} checks. Scope: {tensor_verification.get('scope', 'not recorded')}", "",
                      table(["A donor", "Changed A tensors", "Changed B tensors", "Changed U/V tensors", "Changed static gates"],
                            [[donor_labels.get(variant, variant), row.get("A", "?"), row.get("B", "?"), row.get("adapters", "?"), row.get("static_gates", "?")] for variant, row in tensor_verification.get("changed_tensor_counts", {}).items()]), ""])
    unit = local_checks.get("final_validation", {}).get("unit_tests", {})
    if unit:
        lines.extend([f"Shared final local validation records **{count(unit.get('passed'))} tests passed** in {number(unit.get('elapsed_seconds'), 2)} seconds. Portable optimizer parity and probe isolation tests are included; the reporter does not rerun them.", ""])
    lines.extend(["## Limits", "",
                  "One source-training seed and one fresh-controller seed provide descriptive evidence. Donor differences can reflect ordinary source capability as well as a representation's compatibility with the common fresh low-rank interface. The probes do not establish broad plasticity, robust retention, task-family transfer, or statistical significance. No explicit counterfactual-action distillation or persistent memory is introduced. A strong learning-to-adapt conclusion would require more learned A donors, independent training seeds, intermediate probe checkpoints, and controlled new-task shifts.", "",
                  "`probe_report_summary.json` retains per-donor endpoint changes and paired contrasts. `probe_report_manifest.json` hashes saved evidence and reporter code. This report imports no Torch and runs no model.", ""])
    (directory / "probe_report.md").write_text("\n".join(lines), encoding="utf-8")
    result = {"status": status, "conclusion": conclusion, "learning": learning, "joint_donor_contrasts": contrasts,
              "checks": checks, "all_checks_passed": all(checks.values()), "actual_admission_streams_equal": matched_admissions,
              "isolated_A_probe_executed": bool(stages), "H3_established": False,
              "scope": "One-seed same-task Sudoku readaptation with A frozen and common fresh B plus U/V trained"}
    (directory / "probe_report_summary.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    filenames = ("config.json", "summary.json", "evaluations.json", "history.jsonl", "verification.json", "tensor_verification.json", "source_manifest.json")
    manifest = {"reporter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "inputs_sha256": {name: hashlib.sha256((directory / name).read_bytes()).hexdigest() for name in filenames if (directory / name).exists()}, "plots": pictures}
    if (ROOT / "results" / "local_checks.json").exists():
        manifest["shared_local_checks_sha256"] = hashlib.sha256((ROOT / "results" / "local_checks.json").read_bytes()).hexdigest()
    (directory / "probe_report_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", default="results/local_sudoku")
    parser.add_argument("--probe-run-dir", help="Generate a separate frozen-A/fresh-B probe report instead")
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    directory = Path(args.probe_run_dir or args.run_dir)
    directory = directory if directory.is_absolute() else ROOT / directory
    result = (build_probe_report if args.probe_run_dir else build_report)(directory, plots=not args.no_plots)
    filename = "probe_report.md" if args.probe_run_dir else "report.md"
    print(f"Wrote {directory / filename}; saved status: {result['status']}")


if __name__ == "__main__":
    main()
