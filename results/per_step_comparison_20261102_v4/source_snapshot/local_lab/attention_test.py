"""Matched, bounded training test of inter-model attention, with frozen A."""
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

from models.recursive_reasoning.trm import TinyRecursiveReasoningModel_ACTV1Config
from models.recursive_reasoning.trm_dense import DensePlasticTRM, native_a_from_checkpoint
from models.recursive_reasoning.trm_stacked import StackedDenseTRM
from models.recursive_reasoning.trm_comparison import LargerDenseB
from .adam_atan2 import AdamATan2
from .data import load_data
from .dense_smoke import loss
from .train import CandidateStream, metrics_from_predictions, state_digest, state_snapshot

KINDS = ("b", "larger_b", "pooled", "cross")
ABLATIONS = ("c_off", "c_write_shuffle", "c_read_shuffle", "c_read_pooled", "c_phase", "c_reset_memory")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def config_digest(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def source_a(cfg):
    payload = torch.load(ROOT / cfg["source_checkpoint"], map_location="cpu", weights_only=True)
    arch = {key: value for key, value in payload["model_config"].items()
            if key in TinyRecursiveReasoningModel_ACTV1Config.model_fields}
    arch.update(cfg["architecture"], batch_size=cfg["batch_size"])
    return arch, native_a_from_checkpoint(payload["state_dict"])


def panels(cfg):
    data = load_data(ROOT / cfg["data_dir"])
    result = {}
    for name, pool_name in (("train", "val"), ("dev", "val"), ("test", "test")):
        start, count = cfg[name + "_offset"], cfg[name + "_size"]
        result[name] = {key: value[start:start + count] for key, value in data[pool_name].items()}
        if len(result[name]["inputs"]) != count:
            raise ValueError("Panel is shorter than configured")
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        if set(result[left]["base_ids"]) & set(result[right]["base_ids"]):
            raise ValueError("Overlapping panel IDs")
        left_hashes = {hashlib.sha256(row.tobytes()).hexdigest() for row in result[left]["inputs"]}
        if any(hashlib.sha256(row.tobytes()).hexdigest() in left_hashes for row in result[right]["inputs"]):
            raise ValueError("Overlapping puzzle bytes")
    if cfg.get("previous_data_dir"):
        previous = load_data(ROOT / cfg["previous_data_dir"])
        if not np.array_equal(previous["train"]["inputs"], data["train"]["inputs"]):
            raise ValueError("The new cache changed the frozen A donor's training pool")
        for key in ("inputs", "labels"):
            seen = {hashlib.sha256(row.tobytes()).hexdigest()
                    for pool in previous.values() for row in pool[key]}
            if any(hashlib.sha256(row.tobytes()).hexdigest() in seen for row in result["test"][key]):
                raise ValueError("The final test panel overlaps a previous cached puzzle or solution")
    return result


def device_batch(pool, offset=0, size=None, device="cuda"):
    stop = None if size is None else offset + size
    return {key: torch.as_tensor(pool[key][offset:stop], device=device, dtype=torch.long)
            for key in ("inputs", "labels", "puzzle_identifiers")}


def build(arch, a_state, seed, kind="b", common=None):
    seed_all(seed if common is None else seed + 10000)
    with torch.device("cuda"):
        model = (DensePlasticTRM(arch) if kind == "b" else LargerDenseB(arch) if kind == "larger_b"
                 else StackedDenseTRM(arch)).eval()
    if common is None:
        model.load_a_state(a_state)
    elif kind == "b":
        model.load_state_dict(common, strict=True)
    else:
        model.load_dense_state(common)
    model.configure_training("frozen")
    if isinstance(model, StackedDenseTRM):
        model.configure_hierarchy(train_b=True, train_c=True, train_a=False)
        model.set_meta_read_mode("pooled" if kind == "pooled" else "tokens")
    return model


def actors(model):
    return model.b_parameters + (model.c_parameters if isinstance(model, StackedDenseTRM) else [])


def actor_state(model):
    return {key: value for key, value in state_snapshot(model).items() if not key.startswith("inner.")}


def release(model):
    model.clear_runtime()
    del model
    gc.collect()
    torch.cuda.empty_cache()


def checkpoint_steps(cfg, steps):
    interval = cfg.get("dev_every", 0)
    return list(range(interval, steps, interval)) if interval else []


def evaluation_calls(cfg, steps):
    batch = cfg.get("eval_batch_size", cfg["batch_size"])
    dev = int(np.ceil(cfg["dev_size"] / batch)) * 16
    test = int(np.ceil(cfg["test_size"] / batch)) * 16
    seeds = len(cfg["seeds"])
    ablations = cfg.get("ablations", ABLATIONS)
    return (seeds * len(KINDS) * ((2 + len(checkpoint_steps(cfg, steps))) * dev + test)
            + seeds * len(ablations) * test + dev + test)


def run_profile(cfg):
    if not torch.cuda.is_available():
        raise RuntimeError("The bounded full-width comparison requires the local CUDA device")
    target = ROOT / cfg["profile_file"]
    if target.exists():
        raise FileExistsError(target)
    arch, a_state = source_a(cfg)
    data = panels(cfg)["train"]
    model = build(arch, a_state, cfg["seeds"][0], "cross")
    optimizer = AdamATan2(actors(model), lr=cfg["learning_rate"], betas=(.9, .95), weight_decay=cfg["weight_decay"])
    current = device_batch(data, size=cfg["batch_size"])
    with torch.device("cuda"):
        carry = model.initial_carry(current)
    torch.cuda.reset_peak_memory_stats()
    training_times, eval_times = [], []
    for index in range(8):
        torch.cuda.synchronize()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        carry, outputs = model(carry, current)
        objective = loss(outputs["logits"], current["inputs"], current["labels"])
        objective.backward()
        optimizer.step()
        torch.cuda.synchronize()
        if index >= 2:
            training_times.append(time.perf_counter() - started)
    evaluation_current = device_batch(data, size=cfg.get("eval_batch_size", cfg["batch_size"]))
    with torch.device("cuda"):
        carry = model.initial_carry(evaluation_current)
    with torch.no_grad():
        for _ in range(8):
            torch.cuda.synchronize()
            started = time.perf_counter()
            carry, _ = model(carry, evaluation_current)
            torch.cuda.synchronize()
            eval_times.append(time.perf_counter() - started)
    train_s, eval_s = float(np.median(training_times)), float(np.median(eval_times))
    # Counts include initial dev, final dev/test for all arms and every planned
    # held-out ablation. A-only is conservatively charged the same evaluation rate.
    n = len(cfg["seeds"])
    options = range(cfg["requested_steps"], cfg["minimum_steps"] - 1, -16)
    locked_steps = next((steps for steps in options
        if 1.5 * (n * len(KINDS) * steps * train_s + evaluation_calls(cfg, steps) * eval_s) + 120 <= cfg["max_seconds"]), None)
    eval_calls = None if locked_steps is None else evaluation_calls(cfg, locked_steps)
    result = {"config_sha256": config_digest(cfg), "training_step_median_seconds": train_s,
        "evaluation_act_median_seconds": eval_s, "profile_training_times": training_times,
        "profile_eval_times": eval_times, "evaluation_act_calls": eval_calls,
        "locked_steps": locked_steps, "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "scope": "Training-panel-only timing calibration. Profile models are discarded; all study arms are initialized fresh. No held-out evaluation or tuning.",
        "estimated_study_seconds_with_50_percent_margin": None if locked_steps is None else
            1.5 * (n * len(KINDS) * locked_steps * train_s + eval_calls * eval_s) + 120}
    write_json(target, result)
    print(json.dumps({"stage": "profile", **result}), flush=True)
    release(model)
    if locked_steps is None:
        raise RuntimeError("The configured comparison cannot fit its budget; revise the protocol before running it")
    return result


def modes(model, kind, variant):
    model.set_mode("off" if variant == "a_only" else "dynamic")
    if isinstance(model, StackedDenseTRM):
        model.set_meta_mode({"c_off": "off", "c_write_shuffle": "shuffle", "c_phase": "phase",
            "c_reset_memory": "reset_memory"}.get(variant, "dynamic"))
        model.set_meta_read_mode("pooled" if kind == "pooled" or variant == "c_read_pooled"
            else "shuffle" if variant == "c_read_shuffle" else "tokens")


@torch.no_grad()
def evaluate(model, pool, cfg, kind, variant, check_budget):
    model.clear_runtime()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    modes(model, kind, variant)
    before, frozen = state_digest(actor_state(model)), state_digest(model.a_state())
    logits, cross_weights = [], None
    device = next(model.parameters()).device
    eval_batch = cfg.get("eval_batch_size", cfg["batch_size"])
    b_updates, c_updates = [], []
    started = time.perf_counter()
    for offset in range(0, len(pool["inputs"]), eval_batch):
        current = device_batch(pool, offset, eval_batch, device=device)
        with torch.device(device):
            carry = model.initial_carry(current)
        for _ in range(16):
            check_budget()
            carry, outputs = model(carry, current)
            expected = (0 if variant == "a_only" else model.reasoning_calls_per_act
                        if model.weight_update_cadence == "per_reasoning_step" else 1)
            if model.weight_updates_this_act != expected:
                raise AssertionError("Evaluation skipped a scheduled B weight generation")
            b_updates.append(model.weight_updates_this_act)
            if isinstance(model, StackedDenseTRM):
                expected_c = 0 if variant == "c_off" else expected
                if model.meta_weight_updates_this_act != expected_c:
                    raise AssertionError("Evaluation skipped a scheduled C weight generation")
                c_updates.append(model.meta_weight_updates_this_act)
            if any(port.delta is not None for port in model.ports):
                raise AssertionError("A runtime matrix leaked")
            if isinstance(model, StackedDenseTRM) and model.b_write_port.delta is not None:
                raise AssertionError("C runtime matrix leaked")
        logits.append(outputs["logits"].float().cpu())
        if cross_weights is None and isinstance(model, StackedDenseTRM) and model.meta_controller.last_attention is not None:
            cross_weights = model.meta_controller.last_attention[:1].cpu().clone()
    values = torch.cat(logits)
    prediction = values.argmax(-1).numpy()
    row = {**metrics_from_predictions(prediction, pool["inputs"], pool["labels"]),
        "task_loss": float(loss(values, torch.as_tensor(pool["inputs"]), torch.as_tensor(pool["labels"]))),
        "seconds": time.perf_counter() - started, "act_steps": 16,
        "weight_update_cadence": model.weight_update_cadence,
        "b_weight_updates_per_act": b_updates, "c_weight_updates_per_act": c_updates,
        "actor_unchanged": before == state_digest(actor_state(model)),
        "a_unchanged": frozen == state_digest(model.a_state())}
    if not row["actor_unchanged"] or not row["a_unchanged"]:
        raise AssertionError("Inference changed stored weights")
    model.clear_runtime()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return row, {"logits": values, "predictions": torch.as_tensor(prediction),
        "inputs": torch.as_tensor(pool["inputs"]), "labels": torch.as_tensor(pool["labels"]),
        "base_ids": torch.as_tensor(pool["base_ids"]), "sample_attention": cross_weights}


def run(cfg, profile):
    if profile["config_sha256"] != config_digest(cfg) or profile["locked_steps"] is None:
        raise ValueError("Timing profile does not match the locked protocol")
    directory = ROOT / cfg["results_dir"]
    if directory.exists():
        raise FileExistsError(directory)
    directory.mkdir(parents=True)
    started = time.perf_counter()
    def check_budget():
        if time.perf_counter() - started > cfg["max_seconds"]:
            raise TimeoutError("The local comparison reached its explicit wall-clock cap")
    steps = profile["locked_steps"]
    write_json(directory / "config.json", dict(cfg, locked_steps=steps))
    write_json(directory / "profile.json", profile)
    snapshot_hashes = {}
    for folder in ("models", "local_lab", "scripts", "tests"):
        for path in (ROOT / folder).rglob("*.py"):
            relative = path.relative_to(ROOT)
            target = directory / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
            snapshot_hashes[str(relative).replace("\\", "/")] = sha(target)
    write_json(directory / "source_manifest.json", snapshot_hashes)
    data = panels(cfg)
    write_json(directory / "data_audit.json", {"dataset_manifest_sha256": sha(ROOT / cfg["data_dir"] / "manifest.json"),
        "panels": {name: {"n": len(pool["inputs"]), "base_ids": pool["base_ids"].tolist(),
            "inputs_sha256": hashlib.sha256(pool["inputs"].tobytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(pool["labels"].tobytes()).hexdigest()} for name, pool in data.items()},
        "id_and_exact_puzzle_disjoint": True, "oracle_used": False,
        "fresh_test_vs_previous_cache": bool(cfg.get("previous_data_dir")),
        "previous_cache_manifest_sha256": sha(ROOT / cfg["previous_data_dir"] / "manifest.json") if cfg.get("previous_data_dir") else None,
        "source": "Controller train/dev from A-untrained validation split; held-out official test offset specified before any study evaluation."})
    arch, a_state = source_a(cfg)
    a_digest, source_digest = state_digest(a_state), sha(ROOT / cfg["source_checkpoint"])
    torch.save({"a_state": a_state, "model_config": arch, "source_sha256": source_digest}, directory / "frozen_a.pt")
    rows, training_rows = [], []
    def record_eval(model, kind, seed, variant, split):
        row, saved = evaluate(model, data[split], cfg, kind, variant, check_budget)
        name = f"predictions_{kind}_{seed}_{variant}_{split}.pt"
        torch.save(saved, directory / name)
        row.update(kind=kind, seed=seed, variant=variant, split=split, artifact=name)
        rows.append(row)
        write_json(directory / "evaluation_progress.json", rows)
        print(json.dumps({"stage": "evaluation", "kind": kind, "seed": seed, "variant": variant,
            "split": split, "blank_accuracy": row["blank_accuracy"], "seconds": row["seconds"]}), flush=True)
        return row
    for seed in cfg["seeds"]:
        common_model = build(arch, a_state, seed)
        common = state_snapshot(common_model)
        torch.save({"state_dict": common, "model_config": arch}, directory / f"common_seed{seed}.pt")
        if seed == cfg["seeds"][0]:
            record_eval(common_model, "a", 0, "a_only", "dev")
            record_eval(common_model, "a", 0, "a_only", "test")
        release(common_model)
        del common_model
        initial_c_hash, stream_hash, parameter_match = None, None, None
        for kind in KINDS:
            check_budget()
            model = build(arch, a_state, seed, kind, common)
            parameters = actors(model)
            n_parameters = sum(p.numel() for p in parameters)
            if kind != "b":
                if parameter_match is None:
                    parameter_match = n_parameters
                if n_parameters != parameter_match:
                    raise AssertionError("The larger-B and B/C arms are not parameter matched")
            initial = actor_state(model)
            if isinstance(model, StackedDenseTRM):
                c_hash = state_digest({k: v for k, v in initial.items() if k.startswith("meta_")})
                if initial_c_hash is None:
                    initial_c_hash = c_hash
                elif c_hash != initial_c_hash:
                    raise AssertionError("Pooled and structured C did not start from identical parameters")
            record_eval(model, kind, seed, "initial", "dev")
            modes(model, kind, "normal")
            optimizer = AdamATan2(parameters, lr=cfg["learning_rate"], betas=(.9, .95), weight_decay=cfg["weight_decay"])
            stream = CandidateStream(data["train"], cfg["batch_size"], seed)
            digest, admitted, history, carry = hashlib.sha256(), [], [], None
            torch.cuda.reset_peak_memory_stats()
            arm_started, dev_seconds = time.perf_counter(), 0.
            for step in range(1, steps + 1):
                check_budget()
                if (step - 1) % 16 == 0:
                    candidate = stream.next()
                    for key in ("inputs", "labels", "base_ids", "sample_ids"):
                        digest.update(candidate[key].tobytes())
                    admitted.extend(candidate["base_ids"].tolist())
                    current = device_batch(candidate)
                    with torch.device("cuda"):
                        carry = model.initial_carry(current)
                optimizer.zero_grad(set_to_none=True)
                factor = min(1., step / cfg["warmup_steps"])
                for group in optimizer.param_groups:
                    group["lr"] = cfg["learning_rate"] * factor
                carry, outputs = model(carry, current)
                expected = model.reasoning_calls_per_act if model.weight_update_cadence == "per_reasoning_step" else 1
                if model.weight_updates_this_act != expected or (isinstance(model, StackedDenseTRM)
                        and model.meta_weight_updates_this_act != expected):
                    raise AssertionError("Training skipped a scheduled weight generation")
                objective = loss(outputs["logits"], current["inputs"], current["labels"])
                if not torch.isfinite(objective):
                    raise FloatingPointError("Nonfinite training loss")
                objective.backward()
                if any(p.grad is not None for p in model.inner.parameters()):
                    raise AssertionError("Frozen A received gradients")
                grad_norm = torch.nn.utils.clip_grad_norm_(parameters, 1., error_if_nonfinite=True)
                if not grad_norm > 0:
                    raise AssertionError("No gradient reached trainable controllers")
                read_norms = None
                if isinstance(model, StackedDenseTRM):
                    read = model.meta_controller.cross_read
                    g = read.in_proj_weight.grad.reshape(3, read.num_heads, read.embed_dim // read.num_heads, read.embed_dim)
                    read_norms = g.square().sum((0, 2, 3)).sqrt().detach().cpu().tolist()
                    if not all(np.isfinite(v) and v > 0 for v in read_norms):
                        raise AssertionError("An inter-model head lost gradient flow")
                optimizer.step()
                if step == 1 or step % 32 == 0 or step == steps:
                    history.append({"step": step, "task_loss": float(objective.detach()),
                        "gradient_norm": float(grad_norm), "cross_head_gradient_norms": read_norms,
                        "b_weight_updates": model.weight_updates_this_act,
                        "c_weight_updates": model.meta_weight_updates_this_act if isinstance(model, StackedDenseTRM) else 0,
                        "cuda_allocated_mib": torch.cuda.memory_allocated() / 2**20,
                        "cuda_reserved_mib": torch.cuda.memory_reserved() / 2**20})
                    print(json.dumps({"stage": "training", "seed": seed, "kind": kind,
                        **history[-1], "elapsed_seconds": time.perf_counter() - started}), flush=True)
                    write_json(directory / "live_progress.json", {"seed": seed, "kind": kind,
                        **history[-1], "elapsed_seconds": time.perf_counter() - started})
                if step % 16 == 0:
                    if not bool(carry.halted.all()):
                        raise AssertionError("Runtime cleanup requires all puzzle slots to have completed")
                    del outputs, objective
                    model.clear_runtime()
                    gc.collect()
                    torch.cuda.empty_cache()
                if step in checkpoint_steps(cfg, steps):
                    if not bool(carry.halted.all()):
                        raise AssertionError("Development evaluation must follow a complete puzzle episode")
                    tick = time.perf_counter()
                    record_eval(model, kind, seed, f"step_{step}", "dev")
                    dev_seconds += time.perf_counter() - tick
                    modes(model, kind, "normal")
            training_seconds = time.perf_counter() - arm_started - dev_seconds
            a_unchanged = state_digest(model.a_state()) == a_digest
            if not a_unchanged or any(not torch.isfinite(p).all() for p in parameters):
                raise AssertionError("A changed or trained parameters are nonfinite")
            if stream_hash is None:
                stream_hash = digest.hexdigest()
            elif stream_hash != digest.hexdigest():
                raise AssertionError("Training streams differ between matched arms")
            final_state = state_snapshot(model)
            torch.save({"state_dict": final_state, "initial_actor_state": initial,
                "model_config": arch, "kind": kind, "seed": seed,
                "steps": steps, "a_digest": a_digest}, directory / f"{kind}_seed{seed}.pt")
            info = {"kind": kind, "seed": seed, "steps": steps, "act_slots": steps * cfg["batch_size"],
                "a_calls_per_act": 21, "parameter_count": n_parameters,
                "initial_actor_sha256": state_digest(initial), "final_actor_sha256": state_digest(actor_state(model)),
                "stream_sha256": digest.hexdigest(), "admitted_base_ids": admitted,
                "unique_base_puzzles": len(set(admitted)), "augmented_admissions": len(admitted),
                "a_unchanged": a_unchanged, "training_seconds": training_seconds,
                "development_evaluation_seconds": dev_seconds,
                "weight_update_cadence": model.weight_update_cadence,
                "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                "peak_cuda_reserved_mib": torch.cuda.max_memory_reserved() / 2**20, "history": history}
            training_rows.append(info)
            write_json(directory / "training_progress.json", training_rows)
            record_eval(model, kind, seed, "normal", "dev")
            record_eval(model, kind, seed, "normal", "test")
            if kind == "cross":
                for variant in cfg.get("ablations", ABLATIONS):
                    record_eval(model, kind, seed, variant, "test")
            release(model)
            del model, optimizer, parameters, final_state
            gc.collect()
            torch.cuda.empty_cache()
    result = {"status": "complete", "source_checkpoint_sha256": source_digest,
        "frozen_a_sha256": a_digest, "steps_per_arm": steps, "training": training_rows,
        "evaluations": rows, "elapsed_seconds": time.perf_counter() - started,
        "scope": "One frozen native A donor; three controller seeds; matched updates/data/A calls; exact active parameter match for larger B versus B+C. No oracle, no joint A training, no safety or mechanistic interpretability claims."}
    write_json(directory / "summary.json", result)
    print(json.dumps({"stage": "complete", "elapsed_seconds": result["elapsed_seconds"]}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="local_lab/config_attention_test.json")
    parser.add_argument("--stage", choices=("profile", "run", "all"), default="all")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    if cfg["requested_steps"] % 16 or cfg["minimum_steps"] % 16:
        raise ValueError("Training budgets must consist of complete 16-ACT episodes")
    cap = 3600 if cfg["architecture"].get("weight_update_cadence") == "per_reasoning_step" else 1800
    if not 0 < cfg["max_seconds"] <= cap or not 0 < cfg["batch_size"] <= 32 or not 0 < cfg.get("eval_batch_size", cfg["batch_size"]) <= 32:
        raise ValueError("Study caps: batch 32; per-step studies one hour, historical studies 30 minutes")
    if cfg.get("dev_every", 0) < 0 or cfg.get("dev_every", 0) % 16:
        raise ValueError("Development checkpoints must follow complete 16-ACT episodes")
    if any(variant not in ABLATIONS for variant in cfg.get("ablations", ABLATIONS)):
        raise ValueError("Unsupported inference ablation")
    if args.stage in ("profile", "all"):
        profile = run_profile(cfg)
    else:
        profile = json.loads((ROOT / cfg["profile_file"]).read_text())
    if args.stage in ("run", "all"):
        run(cfg, profile)


if __name__ == "__main__":
    main()
