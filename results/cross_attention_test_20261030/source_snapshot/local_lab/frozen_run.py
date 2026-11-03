"""Bounded frozen-A oracle, controller training, controls and held-out evaluation."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
from pathlib import Path
import time

from .settings import ROOT, seed_all, write_json
import numpy as np
import torch
from .adam_atan2 import AdamATan2
from .data import load_data
from .frozen_control import (FrozenControllerModel, ObservationController, PhaseController,
                             gate_oracle, task_loss)
from .train import CandidateStream, state_digest, state_snapshot, metrics_from_predictions
from models.recursive_reasoning.trm_plastic import PlasticTinyRecursiveReasoningModel_ACTV1
from models.losses import stablemax_cross_entropy

KINDS = ("phase", "pooled", "revised", "activation")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot_sources(directory):
    hashes = {}
    for folder in ("models", "local_lab"):
        for path in (ROOT / folder).rglob("*.py"):
            relative = path.relative_to(ROOT)
            target = directory / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            hashes[str(relative).replace("\\", "/")] = sha(path)
    write_json(directory / "manifest.json", hashes)


def pools(cfg):
    data = load_data(ROOT / cfg["data_dir"])
    def take(pool, start, n):
        return {name: value[start:start+n] for name, value in pool.items()}
    result = {"train": take(data["val"], cfg["train_offset"], cfg["train_size"]),
              "dev": take(data["val"], cfg["dev_offset"], cfg["dev_size"]),
              "oracle": take(data["val"], cfg["oracle_offset"], cfg["oracle_size"]),
              "test": take(data["test"], 0, cfg["test_size"])}
    for left in result:
        if len(result[left]["inputs"]) != cfg[left + "_size"]:
            raise ValueError(f"Incorrect {left} size")
        for right in result:
            if left != right and set(result[left]["base_ids"]) & set(result[right]["base_ids"]):
                raise ValueError("Overlapping training/development/oracle/test IDs")
    return result


def batch_on_device(pool, offset=0, n=None):
    stop = None if n is None else offset + n
    return {key: torch.as_tensor(pool[key][offset:stop], device="cuda", dtype=torch.long)
            for key in ("inputs", "labels", "puzzle_identifiers")}


def build(cfg, kind, seed, batch_size, calibration=None):
    seed_all(seed)
    payload = torch.load(ROOT / cfg["source_checkpoint"], map_location="cpu", weights_only=True)
    arch = dict(payload["model_config"], batch_size=batch_size)
    with torch.device("cuda"):
        core = PlasticTinyRecursiveReasoningModel_ACTV1(arch)
    core.load_state_dict(payload["state_dict"], strict=True)
    with torch.device("cuda"):
        actor = PhaseController() if kind == "phase" else ObservationController(
            hidden=arch["hidden_size"], width=cfg["controller_width"],
            heads=cfg["controller_heads"], layers=cfg["controller_layers"], pooled=kind == "pooled")
    model = FrozenControllerModel(core, actor, "activation" if kind == "activation" else "weight")
    if calibration is not None:
        for index, port in enumerate(model.ports):
            port.activation_scale.copy_(torch.tensor(calibration[index], device="cuda"))
    return model, arch


@torch.no_grad()
def calibrate(model, training, n):
    model.mode = "off"
    for port in model.ports:
        port.calibrating = True
        port.calibration = []
    batch = batch_on_device(training, n=n)
    with torch.device("cuda"):
        carry = model.core.initial_carry(batch)
    for _ in range(16):
        carry, _ = model(carry, batch)
    values = []
    for port in model.ports:
        totals = torch.stack(port.calibration).sum(0)
        values.append((totals[0] / totals[1]).sqrt().cpu().tolist())
        port.calibrating = False
        port.calibration = []
    return values


def run_oracle(cfg):
    directory = ROOT / cfg["results_dir"]
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    started = time.perf_counter()
    write_json(directory / "config.json", cfg)
    snapshot_sources(directory / "oracle_sources")
    data = pools(cfg)
    write_json(directory / "data_audit.json", {
        "dataset_manifest_sha256": sha(ROOT / cfg["data_dir"] / "manifest.json"),
        "panels": {name: {"n": len(pool["inputs"]), "base_ids": pool["base_ids"].tolist(),
            "inputs_sha256": hashlib.sha256(pool["inputs"].tobytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(pool["labels"].tobytes()).hexdigest()} for name, pool in data.items()},
        "exact_base_ids_disjoint": True,
        "provenance": "All controller training and development/oracle panels are from source-A-untrained validation data; official test is separate. Whole expanded cache is hash-validated and puzzle/solution disjoint."})
    model, arch = build(cfg, "phase", cfg["seeds"][0], cfg["batch_size"])
    before = state_digest(model.frozen_state())
    calibration = calibrate(model, data["train"], min(cfg["calibration_size"], cfg["train_size"]))
    for port, values in zip(model.ports, calibration):
        port.activation_scale.copy_(torch.tensor(values, device="cuda"))
    write_json(directory / "calibration.json", {"channel_rms": calibration,
        "training_examples": min(cfg["calibration_size"], cfg["train_size"]),
        "scope": "Activation-control direction amplitudes calibrated once on label-free training activations; frozen for all arms. This is a calibration match, not guaranteed identical realized delta norms."})
    # Save the exact common frozen substrate and all buffers including the calibrated port scales.
    torch.save({"state_dict": model.frozen_state(), "buffers": model.frozen_buffers(), "model_config": arch}, directory / "common_substrate.pt")
    frozen_buffers_sha = state_digest(model.frozen_buffers())
    batch = batch_on_device(data["oracle"])
    model.mode = "off"
    with torch.no_grad(), torch.device("cuda"):
        carry = model.core.initial_carry(batch)
        for _ in range(16):
            carry, outputs = model(carry, batch)
    state = model.late_state
    rows = []
    for kind in ("weight", "activation"):
        for port in model.ports:
            port.actuation = kind
        gates, best, zero, history = gate_oracle(model, state, batch["inputs"], batch["labels"],
            steps=cfg["oracle_steps"], learning_rate=cfg["oracle_lr"])
        if not torch.equal(zero, outputs["logits"]):
            raise AssertionError("Zero-gate local replay differs from the untouched terminal window")
        before_loss = task_loss(zero, batch["inputs"], batch["labels"])
        after_loss = task_loss(best, batch["inputs"], batch["labels"])
        row = {"actuation": kind, "privileged": True,
            "scope": "Finite per-example label-assisted gate search in a common terminal seven-call window after 15 untouched ACT updates. Not a global optimum, full-trajectory oracle, or deployed accuracy result.",
            "baseline": metrics_from_predictions(zero.argmax(-1).cpu().numpy(), data["oracle"]["inputs"], data["oracle"]["labels"]),
            "optimized": metrics_from_predictions(best.argmax(-1).cpu().numpy(), data["oracle"]["inputs"], data["oracle"]["labels"]),
            "before_task_loss": before_loss.cpu().tolist(), "after_task_loss": after_loss.cpu().tolist(),
            "mean_objective_reduction": float((before_loss - after_loss).mean()),
            "per_example_nonworsening_objective": bool((after_loss <= before_loss + 1e-5).all()),
            "gates": gates.cpu().tolist(), "best_loss_history": history,
            "baseline_predictions": zero.argmax(-1).cpu().tolist(),
            "optimized_predictions": best.argmax(-1).cpu().tolist()}
        rows.append(row)
        print(json.dumps({"stage": "oracle", "actuation": kind, "baseline_blank_accuracy": row["baseline"]["blank_accuracy"],
                          "oracle_blank_accuracy": row["optimized"]["blank_accuracy"], "objective_reduction": row["mean_objective_reduction"]}), flush=True)
    if before != state_digest(model.frozen_state()) or frozen_buffers_sha != state_digest(model.frozen_buffers()):
        raise AssertionError("Oracle changed the frozen substrate or buffers")
    write_json(directory / "oracle.json", {"status": "complete", "rows": rows, "elapsed_seconds": time.perf_counter() - started,
        "frozen_state_sha256": before, "frozen_buffers_sha256": frozen_buffers_sha,
        "source_checkpoint_sha256": sha(ROOT / cfg["source_checkpoint"]), "frozen_state_unchanged": True})
    del model
    gc.collect()
    torch.cuda.empty_cache()


@torch.no_grad()
def evaluate(model, pool, batch_size, mode="normal"):
    model.mode = mode
    model.actor.eval()
    before = state_digest(model.frozen_state())
    buffers_before = state_digest(model.frozen_buffers())
    actor_before = state_digest(state_snapshot(model.actor))
    predictions, all_gates, losses, objectives, statistics = [], [], [], [], []
    started = time.perf_counter()
    for port in model.ports:
        port.recording, port.statistics = True, []
    for offset in range(0, len(pool["inputs"]), batch_size):
        batch = batch_on_device(pool, offset, batch_size)
        with torch.device("cuda"):
            carry = model.core.initial_carry(batch)
        gates = []
        for _ in range(16):
            carry, outputs = model(carry, batch)
            if model.action is not None:
                gates.append(model.action.cpu())
        predictions.append(outputs["logits"].argmax(-1).cpu().numpy())
        losses.append(stablemax_cross_entropy(outputs["logits"], batch["labels"]).cpu())
        objectives.append(task_loss(outputs["logits"], batch["inputs"], batch["labels"]).cpu())
        if gates:
            all_gates.append(torch.stack(gates))
    for port in model.ports:
        if port.statistics:
            statistics.append(torch.stack(port.statistics).sum(0).cpu())
        port.recording, port.statistics = False, []
    preds = np.concatenate(predictions)
    row = {**metrics_from_predictions(preds, pool["inputs"], pool["labels"]),
        "loss": float(torch.cat(losses).mean()), "task_loss": float(torch.cat(objectives).mean()),
        "seconds": time.perf_counter() - started, "act_steps": 16, "actions_per_act": 0 if mode == "off" else 1,
        "frozen_state_unchanged": before == state_digest(model.frozen_state()),
        "frozen_buffers_unchanged": buffers_before == state_digest(model.frozen_buffers()),
        "actor_parameters_unchanged": actor_before == state_digest(state_snapshot(model.actor)),
        "runtime_gates_cleared": all(port.gate is None for port in model.ports)}
    if statistics:
        totals = torch.stack(statistics).sum(0)
        row["delta_to_base_rms_ratio"] = float((totals[1] / totals[0]).sqrt())
    gates = torch.cat(all_gates, dim=1) if all_gates else None
    if gates is not None:
        variance = (gates - gates.mean((0, 1), keepdim=True)).square().mean()
        row["gate_rms"] = float(gates.square().mean().sqrt())
        row["between_puzzle_gate_variance_fraction"] = float((gates - gates.mean(1, keepdim=True)).square().mean() / variance.clamp_min(1e-20))
    if not all(row[name] for name in ("frozen_state_unchanged", "frozen_buffers_unchanged", "actor_parameters_unchanged", "runtime_gates_cleared")):
        raise AssertionError("Evaluation mutated frozen/actor state or leaked active gates")
    return row, {"predictions": torch.as_tensor(preds), "gates": gates,
                 "base_ids": torch.as_tensor(pool["base_ids"]), "inputs": torch.as_tensor(pool["inputs"]),
                 "labels": torch.as_tensor(pool["labels"])}


def train_controllers(cfg):
    directory = ROOT / cfg["results_dir"]
    if json.loads((directory / "config.json").read_text()) != cfg:
        raise ValueError("Configuration differs from the completed oracle stage")
    if (directory / "summary.json").exists():
        raise FileExistsError("Controller training already started; choose a fresh run")
    oracle = json.loads((directory / "oracle.json").read_text())
    if oracle["status"] != "complete":
        raise ValueError("Run the gate oracle first")
    started = time.perf_counter()
    base_elapsed = oracle["elapsed_seconds"]
    def elapsed():
        return base_elapsed + time.perf_counter() - started
    snapshot_sources(directory / "controller_sources")
    data = pools(cfg)
    calibration = json.loads((directory / "calibration.json").read_text())["channel_rms"]
    common = torch.load(directory / "common_substrate.pt", map_location="cpu", weights_only=True)
    expected_state, expected_buffers = state_digest(common["state_dict"]), state_digest(common["buffers"])
    summary = {"status": "running", "stages": [], "evaluations": [], "seeds": cfg["seeds"],
        "source_checkpoint": cfg["source_checkpoint"], "frozen_state_sha256": expected_state,
        "frozen_buffers_sha256": expected_buffers,
        "scope": "Frozen joint-trained A AND its learned U/V, same task family. No meta-learning or A-weight adaptation claim. All trained controls receive equal support data and privileged training-only teacher budgets.",
        "gradient_policy": "Native14 unmodified no-grad calls; one action before final7 differentiable calls, held fixed for those7. Native carry and controller memory detach after each optimizer step, with memory values retained for the same puzzle.",
        "training_feedback": "Blank CE +0.25 given-clue CE each ACT update; periodic training-label-only local gate-oracle targets, equal search budgets per trainable arm. Evaluation never calls the oracle.",
        "oracle_elapsed_seconds": base_elapsed}
    def save():
        summary.update(elapsed_seconds=elapsed(), max_seconds=cfg["max_seconds"],
                       peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
        write_json(directory / "summary.json", summary)
    save()
    matched_initial = {}
    for seed in cfg["seeds"]:
        for kind in KINDS:
            model, arch = build(cfg, kind, seed, cfg["batch_size"], calibration)
            if state_digest(model.frozen_state()) != expected_state or state_digest(model.frozen_buffers()) != expected_buffers:
                raise AssertionError("Different frozen substrate between controls")
            initial_actor = state_snapshot(model.actor)
            initial_hash = state_digest(initial_actor)
            if kind != "phase":
                matched_initial.setdefault(seed, initial_hash)
                if matched_initial[seed] != initial_hash:
                    raise AssertionError("Pooled/revised/activation controller initial tensors differ")
            model.actor.train()
            model.mode = "normal"
            optimizer = AdamATan2(model.actor.parameters(), lr=cfg["learning_rate"], betas=(.9, .95), weight_decay=.01)
            stream = CandidateStream(data["train"], cfg["batch_size"], seed)
            carry, done, admissions, teacher_count = None, 0, 0, 0
            teacher_records = []
            candidates, accepted_digest = hashlib.sha256(), hashlib.sha256()
            seen = set()
            stage_started = time.perf_counter()
            for step in range(1, cfg["steps"] + 1):
                if elapsed() >= cfg["training_cutoff_seconds"]:
                    break
                proposed = stream.next()
                accepted = np.ones(cfg["batch_size"], dtype=bool) if carry is None else carry.halted.cpu().numpy()
                candidates.update(proposed["inputs"].tobytes() + proposed["base_ids"].tobytes())
                accepted_digest.update(proposed["sample_ids"][accepted].tobytes() + proposed["inputs"][accepted].tobytes())
                admissions += int(accepted.sum())
                seen.update(proposed["base_ids"][accepted].tolist())
                batch = {name: torch.as_tensor(value, device="cuda", dtype=torch.long) for name, value in proposed.items()}
                if carry is None:
                    with torch.device("cuda"):
                        carry = model.core.initial_carry(batch)
                optimizer.zero_grad(set_to_none=True)
                carry, outputs = model(carry, batch)
                current = carry.current_data
                loss = task_loss(outputs["logits"], current["inputs"], current["labels"]).mean()
                imitation = loss.new_zeros(())
                if step == 1 or step % cfg["teacher_every"] == 0:
                    target, teacher_logits, zero_logits, _ = gate_oracle(model, model.late_state, current["inputs"], current["labels"],
                        initial=model.action, steps=cfg["teacher_steps"], learning_rate=cfg["oracle_lr"])
                    imitation = (model.action - target).square().mean()
                    teacher_count += 1
                    teacher_records.append({"step": step, "base_ids": current["base_ids"].detach().cpu(),
                        "proposed_gates": model.action.detach().cpu(), "target_gates": target.cpu(),
                        "proposed_loss": task_loss(outputs["logits"].detach(), current["inputs"], current["labels"]).cpu(),
                        "zero_loss": task_loss(zero_logits, current["inputs"], current["labels"]).cpu(),
                        "target_loss": task_loss(teacher_logits, current["inputs"], current["labels"]).cpu(),
                        "privileged_training_only": True})
                total = loss + cfg["teacher_weight"] * imitation
                if not torch.isfinite(total):
                    raise FloatingPointError(f"Nonfinite {kind}/{seed}/{step}")
                total.backward()
                if any(parameter.grad is not None for parameter in model.core.parameters()):
                    raise AssertionError("Frozen substrate received gradients")
                for group in optimizer.param_groups:
                    group["lr"] = cfg["learning_rate"] * min(1., step / cfg["warmup_steps"])
                optimizer.step()
                done = step
                if step == 1 or step % cfg["log_every"] == 0 or step == cfg["steps"]:
                    row = {"seed": seed, "kind": kind, "step": step, "task_loss": float(loss.detach()),
                           "imitation_loss": float(imitation.detach()), "new_puzzle_admissions": admissions,
                           "unique_base_puzzles_seen": len(seen), "seconds": time.perf_counter() - stage_started,
                           "gate_rms": float(model.action.detach().square().mean().sqrt())}
                    with (directory / "history.jsonl").open("a") as history:
                        history.write(json.dumps(row) + "\n")
                    print(json.dumps(row), flush=True)
            final_actor = state_snapshot(model.actor)
            if state_digest(model.frozen_state()) != expected_state or state_digest(model.frozen_buffers()) != expected_buffers:
                raise AssertionError("Training changed A, U/V or frozen buffers")
            checkpoint = directory / f"{kind}_seed{seed}.pt"
            metadata = {"seed": seed, "kind": kind, "steps_requested": cfg["steps"], "steps_executed": done,
                "complete": done == cfg["steps"], "new_puzzle_admissions": admissions,
                "unique_base_puzzles_seen": len(seen), "act_slot_presentations": done * cfg["batch_size"],
                "candidate_schedule_sha256": candidates.hexdigest(), "admitted_schedule_sha256": accepted_digest.hexdigest(),
                "initial_actor_sha256": initial_hash, "final_actor_sha256": state_digest(final_actor),
                "trainable_parameters": sum(p.numel() for p in model.actor.parameters()),
                "teacher_calls": teacher_count, "seconds": time.perf_counter() - stage_started,
                "frozen_state_unchanged": True, "frozen_buffers_unchanged": True,
                "checkpoint": str(checkpoint.relative_to(ROOT))}
            torch.save({"metadata": metadata, "model_config": arch, "state_dict": model.frozen_state(),
                "buffers": model.frozen_buffers(), "initial_actor_state_dict": initial_actor,
                "actor_state_dict": final_actor, "teacher_records": teacher_records}, checkpoint)
            summary["stages"].append(metadata)
            save()
            del model, optimizer, carry, outputs, final_actor, initial_actor
            gc.collect()
            torch.cuda.empty_cache()
    summary["training_status"] = "complete" if len(summary["stages"]) == len(KINDS) * len(cfg["seeds"]) and all(stage["complete"] for stage in summary["stages"]) else "partial"
    summary["training_elapsed_seconds"] = elapsed()
    save()
    # Evaluate all fixed-final checkpoints only after all training arms finish.
    evaluation_settings = []
    for seed in cfg["seeds"]:
        for kind in KINDS:
            for split in ("dev", "test"):
                evaluation_settings.append((seed, kind, "normal", split))
        for mode in ("shuffle", "phase_only", "reset_memory"):
            evaluation_settings.append((seed, "revised", mode, "test"))
    evaluation_settings = [(cfg["seeds"][0], "off", "off", split) for split in ("dev", "test")] + evaluation_settings
    for seed, kind, mode, split in evaluation_settings:
        if elapsed() >= cfg["max_seconds"] - 25:
            break
        actual_kind = "phase" if kind == "off" else kind
        model, _ = build(cfg, actual_kind, seed, cfg["eval_batch_size"], calibration)
        if kind != "off":
            stage = next((stage for stage in summary["stages"] if stage["seed"] == seed and stage["kind"] == kind and stage["complete"]), None)
            if stage is None:
                del model
                continue
            payload = torch.load(ROOT / stage["checkpoint"], map_location="cpu", weights_only=True)
            model.actor.load_state_dict(payload["actor_state_dict"], strict=True)
        row, evidence = evaluate(model, data[split], cfg["eval_batch_size"], mode)
        row.update(seed=seed, kind=kind, mode=mode, split=split, privileged=False)
        summary["evaluations"].append(row)
        torch.save(evidence, directory / f"predictions_{kind}_{seed}_{mode}_{split}.pt")
        save()
        print(json.dumps(row), flush=True)
        del model, evidence
        gc.collect()
        torch.cuda.empty_cache()
    summary["status"] = "complete" if summary["training_status"] == "complete" and len(summary["evaluations"]) == len(evaluation_settings) else "partial_budget"
    summary["matched_streams_by_seed"] = {str(seed): len({stage["admitted_schedule_sha256"] for stage in summary["stages"] if stage["seed"] == seed}) == 1 for seed in cfg["seeds"]}
    save()
    print(json.dumps({"status": summary["status"], "elapsed_seconds": summary["elapsed_seconds"], "results_dir": cfg["results_dir"]}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local_lab/config_frozen.json")
    parser.add_argument("--stage", choices=("oracle", "controllers", "all"), default="all")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    if args.stage in ("oracle", "all"):
        run_oracle(cfg)
    if args.stage in ("controllers", "all"):
        train_controllers(cfg)


if __name__ == "__main__":
    main()
