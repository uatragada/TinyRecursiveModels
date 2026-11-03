"""Fresh-controller adaptation with isolated native A donors; no source B reuse."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from models.losses import ACTLossHead
from models.recursive_reasoning.trm_plastic import PlasticTinyRecursiveReasoningModel_ACTV1
from .adam_atan2 import AdamATan2
from .data import load_data
from .settings import ROOT, model_config, seed_all, write_json
from .train import CandidateStream, evaluate, metrics_from_predictions, state_digest, state_snapshot

VARIANTS = ("native", "static_joint", "joint")
DEFAULT_CONFIG = dict(name="local_sudoku_probe", results_dir="results/local_sudoku_probe",
                      source_run="results/local_sudoku", data_dir="data/local-bounded-sudoku",
                      seed=29, batch_size=64, steps=60, warmup_steps=10,
                      learning_rate=1e-4, betas=[.9, .95], weight_decay=1.,
                      train_log_steps=[1, 20, 40, 60], adaptation_val_offset=128,
                      checkpoint_steps=[], fixed_act_training=False,
                      eval_size=128, eval_batch_size=64, max_seconds=300,
                      training_cutoff_seconds=180)


def native_a_state(state):
    """All persistent native A tensors, including H/L constants and puzzle table."""
    return {name: value for name, value in state.items()
            if name.startswith("inner.") and not name.endswith((".u", ".v"))}


def non_a_state(state):
    selected = native_a_state(state)
    return {name: value for name, value in state.items() if name not in selected}


def transplant_native_a(common, donor):
    """Copy native A only; validate the entire transplant before cloning it."""
    selected, expected = native_a_state(donor), native_a_state(common)
    if not selected or set(selected) != set(expected):
        raise ValueError("Donor native A key set differs from common recipient")
    for name, value in selected.items():
        target = expected[name]
        if not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype:
            raise ValueError(f"Donor native A shape/dtype mismatch: {name}")
    return {name: (selected[name] if name in selected else value).detach().cpu().clone()
            for name, value in common.items()}


def shared_native_initial_constants(donors):
    reference = next(iter(donors.values()))
    keys = ("inner.H_init", "inner.L_init")
    if any(key not in state for state in donors.values() for key in keys):
        raise ValueError("Native H/L initial-state constants missing")
    if any(not torch.equal(state[key], reference[key]) for state in donors.values() for key in keys):
        raise ValueError("Source donors have different native H/L initial-state constants")
    return state_digest({key: reference[key] for key in keys})


def native_buffer_snapshot(core):
    # Include nonpersistent sparse buffers too: adaptation uses the embedding's
    # frozen eval lookup, so neither its table nor scratch buffers can change.
    return {"inner." + name: value.detach().cpu().clone() for name, value in core.inner.named_buffers()}


def configure_probe(core, fixed_act_training=False):
    core.configure_training("dynamic").train()
    if fixed_act_training:
        # Evaluation mode disables only native training-time early halting and
        # exploration. It does not disable autograd or the final H-cycle graph.
        core.eval()
    if hasattr(core.inner, "puzzle_emb"):
        core.inner.puzzle_emb.eval()
        core.inner.puzzle_emb.local_weights.requires_grad_(False)
    active = []
    for name, parameter in core.named_parameters():
        allowed = name.startswith("controller.") or name.endswith((".u", ".v"))
        if parameter.requires_grad != allowed:
            raise AssertionError(f"Unexpected probe trainability: {name}")
        if allowed:
            active.append(parameter)
    return active


def _slice(data, start, stop=None):
    return {name: value[start:stop] for name, value in data.items()}


def validate_adaptation_partition(cfg, source_cfg):
    offset = cfg["adaptation_val_offset"]
    if isinstance(offset, bool) or not isinstance(offset, int) or offset < source_cfg["eval_size"]:
        raise ValueError("Adaptation validation offset overlaps the source validation endpoint pool")
    if (ROOT / cfg["data_dir"]).resolve() != (ROOT / source_cfg["data_dir"]).resolve():
        raise ValueError("Probe must reuse the source dataset cache to preserve endpoint and partition identity")


def checkpoint_steps(cfg):
    requested = cfg.get("checkpoint_steps", [])
    if (not isinstance(requested, list)
            or any(isinstance(step, bool) or not isinstance(step, int) or not 0 < step <= cfg["steps"] for step in requested)
            or len(set(requested)) != len(requested)):
        raise ValueError("Checkpoint steps must be unique positive integers within the optimizer budget")
    return sorted(set([0, cfg["steps"], *requested]))


def run(config=None):
    cfg = dict(DEFAULT_CONFIG, **(config or {}))
    if not 0 < cfg["training_cutoff_seconds"] < cfg["max_seconds"]:
        raise ValueError("Training cutoff must reserve endpoint evaluation time")
    if cfg["steps"] <= 0 or cfg["batch_size"] <= 0 or cfg["warmup_steps"] <= 0:
        raise ValueError("Probe steps, batch size and warmup must be positive")
    evaluation_steps = checkpoint_steps(cfg)
    if not isinstance(cfg["fixed_act_training"], bool):
        raise ValueError("fixed_act_training must be boolean")
    run_dir, checkpoint_dir = ROOT / cfg["results_dir"], ROOT / "checkpoints" / cfg["name"]
    for directory in (run_dir, checkpoint_dir):
        if directory.exists() and any(directory.iterdir()):
            raise FileExistsError(f"Refusing to overwrite {directory}")
    source_summary = json.loads((ROOT / cfg["source_run"] / "summary.json").read_text())
    if source_summary["status"] != "complete" or source_summary["training_status"] != "complete":
        raise ValueError("All source donors must come from a completed fixed-step source run")
    source_cfg = json.loads((ROOT / cfg["source_run"] / "config.json").read_text())
    validate_adaptation_partition(cfg, source_cfg)
    source_stages = {row["variant"]: row for row in source_summary["stages"]}
    if set(source_stages) != set(VARIANTS) or not all(row["complete"] for row in source_stages.values()):
        raise ValueError("Three completed source donors are required")
    started = time.perf_counter()
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    seed_all(cfg["seed"])
    arch = model_config(cfg["batch_size"])
    for key, value in source_summary["model_config"].items():
        if key != "batch_size" and arch.get(key) != value:
            raise ValueError(f"Probe changes native/controller architecture: {key}")
    cfg["model_config"] = arch
    write_json(run_dir / "config.json", cfg)
    source_manifest = {}
    for folder in (ROOT / "models", ROOT / "local_lab"):
        for source in folder.rglob("*.py"):
            relative = source.relative_to(ROOT)
            raw = source.read_bytes()
            destination = run_dir / "source_snapshot" / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(raw)
            source_manifest[str(relative).replace("\\", "/")] = hashlib.sha256(raw).hexdigest()
    write_json(run_dir / "source_manifest.json", source_manifest)
    data = load_data(ROOT / cfg["data_dir"])
    adaptation = _slice(data["val"], cfg["adaptation_val_offset"])
    test = _slice(data["test"], 0, cfg["eval_size"])
    if not len(adaptation["inputs"]) or len(test["inputs"]) != cfg["eval_size"]:
        raise ValueError("Probe adaptation or fixed endpoint pool has the wrong size")
    donors, donor_files = {}, {}
    for variant in VARIANTS:
        path = ROOT / source_stages[variant]["checkpoint"]
        payload = torch.load(path, map_location="cpu", weights_only=True)
        donors[variant] = payload["state_dict"]
        if state_digest(donors[variant]) != source_stages[variant]["raw_state_sha256"]:
            raise ValueError(f"Actual raw donor checkpoint differs from source metadata: {variant}")
        donor_files[variant] = {"donor_checkpoint": str(path.relative_to(ROOT)),
                                "donor_checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                                "donor_A_sha256": state_digest(native_a_state(donors[variant]))}
    constants_sha = shared_native_initial_constants(donors)
    with torch.device("cuda"):
        core = PlasticTinyRecursiveReasoningModel_ACTV1(arch)
    common = state_snapshot(core)
    common_non_a = state_digest(non_a_state(common))
    common_path = checkpoint_dir / "common_initial_full.pt"
    torch.save({"state_dict": common, "model_config": arch, "seed": cfg["seed"]}, common_path)
    del core
    gc.collect()
    summary = {"status": "running", "config": cfg, "model_config": arch, "seed": cfg["seed"],
               "stages": [], "evaluations": [], "common_initial_checkpoint": str(common_path.relative_to(ROOT)),
               "common_fresh_state_sha256": state_digest(common), "common_fresh_non_A_sha256": common_non_a,
               "shared_donor_initial_constants_sha256": constants_sha,
               "A_policy": "All persistent native inner tensors copied from raw source donor, including learned puzzle table and H/L constants; only adapter U/V excluded",
               "fresh_policy": "One seed-29 common B, U/V and static gates; source B/adapters never transplanted; runtime ACT carry reset for every arm and endpoint batch",
               "training_policy": "Only B and fresh U/V trained; native A and all native buffers frozen; no sparse optimizer and no EMA",
               "probe_ACT_training_policy": "fixed maximum steps; early halting disabled" if cfg["fixed_act_training"] else "native training ACT with learned halting and exploration",
               "evaluation_checkpoint_steps": evaluation_steps,
               "adaptation_pool": {"source": "val", "offset": cfg["adaptation_val_offset"], "n": len(adaptation["inputs"]),
                                   "base_ids_sha256": hashlib.sha256(adaptation["base_ids"].tobytes()).hexdigest()},
               "endpoint_pool": {"source": "test", "offset": 0, "n": len(test["inputs"]),
                                 "base_ids_sha256": hashlib.sha256(test["base_ids"].tobytes()).hexdigest()},
               "limitations": ["One fresh-controller seed and one from-scratch source seed; this is a preliminary within-task adaptation diagnostic.",
                               "Native ACT can admit different numbers of puzzles for different A donors despite identical candidate streams; report actual admissions separately.",
                               "Any donor advantage can reflect source skill, headroom or compatibility with the shared fresh interface; compare initial and final endpoints."]}

    def elapsed():
        return time.perf_counter() - started

    def save_summary():
        summary.update(elapsed_seconds=elapsed(), max_seconds=cfg["max_seconds"],
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(run_dir / "summary.json", summary)

    save_summary()
    for variant in VARIANTS:
        if elapsed() >= cfg["training_cutoff_seconds"]:
            break
        seed_all(cfg["seed"])
        initial = transplant_native_a(common, donors[variant])
        if state_digest(non_a_state(initial)) != common_non_a:
            raise AssertionError("Fresh non-A interface differs across probe arms")
        initial_path = checkpoint_dir / f"initial_{variant}.pt"
        initial_meta = {"variant": variant, "seed": cfg["seed"], "step": 0, **donor_files[variant],
                        "common_fresh_non_A_sha256": common_non_a,
                        "initial_non_A_sha256": state_digest(non_a_state(initial)),
                        "A_sha256_before": state_digest(native_a_state(initial)),
                        "initial_checkpoint": str(initial_path.relative_to(ROOT))}
        torch.save({"state_dict": initial, "model_config": arch, "metadata": initial_meta}, initial_path)
        with torch.device("cuda"):
            core = PlasticTinyRecursiveReasoningModel_ACTV1(arch)
        core.load_state_dict(initial, strict=True)
        active = configure_probe(core, cfg["fixed_act_training"])
        frozen_buffers = native_buffer_snapshot(core)
        before_buffers = state_digest(frozen_buffers)
        loss_model = ACTLossHead(core, "stablemax_cross_entropy")
        optimizer = AdamATan2(active, lr=cfg["learning_rate"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
        stream = CandidateStream(adaptation, cfg["batch_size"], cfg["seed"])
        carry, done, admissions = None, 0, 0
        seen_base, seen_augmented = set(), set()
        candidates_hash, admitted_hash = hashlib.sha256(), hashlib.sha256()
        saved_paths = {"0": str(initial_path.relative_to(ROOT))}
        stage_started = time.perf_counter()
        for step in range(1, cfg["steps"] + 1):
            if elapsed() >= cfg["training_cutoff_seconds"]:
                break
            proposed = stream.next()
            candidates_hash.update(proposed["base_ids"].tobytes() + proposed["inputs"].tobytes())
            accepted = np.ones(cfg["batch_size"], dtype=bool) if carry is None else carry.halted.cpu().numpy()
            admitted_hash.update(proposed["sample_ids"][accepted].tobytes() + proposed["inputs"][accepted].tobytes())
            admissions += int(accepted.sum())
            seen_base.update(proposed["base_ids"][accepted].tolist())
            seen_augmented.update(hashlib.sha256(row.tobytes()).hexdigest() for row in proposed["inputs"][accepted])
            batch = {key: torch.as_tensor(value, device="cuda", dtype=torch.long) for key, value in proposed.items()}
            if carry is None:
                with torch.device("cuda"):
                    carry = core.initial_carry(batch)
            optimizer.zero_grad(set_to_none=True)
            carry, loss, _, outputs, _ = loss_model(carry=carry, batch=batch, return_keys=["preds"])
            normalized = loss / cfg["batch_size"]
            if not torch.isfinite(normalized):
                raise FloatingPointError(f"Nonfinite adaptation loss: {variant}/{step}")
            normalized.backward()
            for group in optimizer.param_groups:
                group["lr"] = cfg["learning_rate"] * min(1., step / cfg["warmup_steps"])
            optimizer.step()
            done = step
            if step in evaluation_steps and step != cfg["steps"]:
                intermediate = state_snapshot(core)
                intermediate_a = state_digest(native_a_state(intermediate))
                intermediate_buffers = state_digest(native_buffer_snapshot(core))
                if intermediate_a != initial_meta["A_sha256_before"] or intermediate_buffers != before_buffers:
                    raise AssertionError("Intermediate probe checkpoint changed native A or buffers")
                intermediate_path = checkpoint_dir / f"{variant}_step{step}.pt"
                intermediate_meta = {**initial_meta, "step": step,
                    "A_sha256_after": intermediate_a, "A_unchanged": True,
                    "native_buffers_sha256_before": before_buffers,
                    "native_buffers_sha256_after": intermediate_buffers,
                    "native_buffers_unchanged": True, "puzzle_embedding_unchanged": True,
                    "raw_state_sha256": state_digest(intermediate)}
                torch.save({"state_dict": intermediate, "model_config": arch, "metadata": intermediate_meta}, intermediate_path)
                saved_paths[str(step)] = str(intermediate_path.relative_to(ROOT))
                del intermediate
            if step in cfg["train_log_steps"] or step == cfg["steps"]:
                metrics = metrics_from_predictions(outputs["preds"].cpu().numpy(),
                          carry.current_data["inputs"].cpu().numpy(), carry.current_data["labels"].cpu().numpy())
                row = {"variant": variant, "seed": cfg["seed"], "step": step, "train_loss": normalized.item(), **metrics,
                       "new_puzzle_admissions": admissions, "elapsed_s": time.perf_counter() - stage_started}
                with (run_dir / "history.jsonl").open("a", encoding="utf-8") as history:
                    history.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
        torch.cuda.synchronize()
        final = state_snapshot(core)
        after_a = state_digest(native_a_state(final))
        after_buffers = state_digest(native_buffer_snapshot(core))
        if initial_meta["A_sha256_before"] != after_a or before_buffers != after_buffers:
            raise AssertionError("Probe changed native A or native buffers")
        path = checkpoint_dir / f"{variant}.pt"
        if done:
            saved_paths[str(done)] = str(path.relative_to(ROOT))
        meta = {**initial_meta, "step": done, "steps_requested": cfg["steps"], "steps_executed": done,
                "complete": done == cfg["steps"], "A_sha256_after": after_a, "A_unchanged": True,
                "native_buffers_sha256_before": before_buffers, "native_buffers_sha256_after": after_buffers,
                "native_buffers_unchanged": True, "puzzle_embedding_unchanged": True,
                "trainable_parameters": sum(parameter.numel() for parameter in active),
                "act_slot_presentations": done * cfg["batch_size"], "new_puzzle_admissions": admissions,
                "unique_base_puzzles_seen": len(seen_base), "unique_augmented_puzzles_seen": len(seen_augmented),
                "candidate_schedule_sha256": candidates_hash.hexdigest(), "admitted_schedule_sha256": admitted_hash.hexdigest(),
                "training_seconds": time.perf_counter() - stage_started, "checkpoint": str(path.relative_to(ROOT)),
                "checkpoint_paths": saved_paths,
                "raw_state_sha256": state_digest(final), "selection": "fixed final step; no validation checkpoint selection"}
        torch.save({"state_dict": final, "model_config": arch, "metadata": meta}, path)
        summary["stages"].append(meta)
        save_summary()
        del core, loss_model, optimizer, carry, final, initial
        gc.collect()
        torch.cuda.empty_cache()
    summary["training_status"] = "complete" if len(summary["stages"]) == len(VARIANTS) and all(row["complete"] for row in summary["stages"]) else "partial"
    summary["training_elapsed_seconds"] = elapsed()
    save_summary()
    # All arms receive their training budget before any held-out evaluation.
    for stage in summary["stages"]:
        if not stage["complete"]:
            continue
        for step in evaluation_steps:
            path = stage["checkpoint_paths"][str(step)]
            # Reserve enough slack for one bounded 128-puzzle endpoint and file
            # writes; a skipped endpoint is explicitly incomplete evidence.
            if elapsed() >= cfg["max_seconds"] - 20:
                break
            payload = torch.load(ROOT / path, map_location="cpu", weights_only=True)
            with torch.device("cuda"):
                core = PlasticTinyRecursiveReasoningModel_ACTV1(arch)
            core.load_state_dict(payload["state_dict"], strict=True)
            metrics = evaluate(core, test, "dynamic", cfg["eval_size"], cfg["eval_batch_size"])
            summary["evaluations"].append({"variant": stage["variant"], "seed": cfg["seed"], "step": step,
                                            "split": "test", "weights": "raw", "mode": "dynamic", **metrics})
            write_json(run_dir / "evaluations.json", {"rows": summary["evaluations"]})
            save_summary()
            del core, payload
            gc.collect()
            torch.cuda.empty_cache()
    summary["matched_initial_non_A"] = all(row["initial_non_A_sha256"] == common_non_a for row in summary["stages"])
    summary["matched_candidate_streams"] = len({row["candidate_schedule_sha256"] for row in summary["stages"]}) == 1
    summary["matched_actual_admission_streams"] = len({row["admitted_schedule_sha256"] for row in summary["stages"]}) == 1
    summary["status"] = "complete" if summary["training_status"] == "complete" and len(summary["evaluations"]) == len(VARIANTS) * len(evaluation_steps) and elapsed() <= cfg["max_seconds"] else "partial"
    save_summary()
    print(json.dumps({"run_dir": str(run_dir), "status": summary["status"], "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config")
    args = parser.parse_args()
    run(json.loads((ROOT / args.config).read_text()) if args.config else None)


if __name__ == "__main__":
    main()
