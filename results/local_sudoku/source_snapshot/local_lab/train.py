"""Bounded native-TRM Sudoku comparison with our trajectory controller B.

The native model, ACT carry, 14/7 gradient truncation, stablemax/halt loss,
sparse-embedding update and EMA are used directly. The dense optimizer is the
upstream AdamATan2 reference formula in portable PyTorch on Windows.
"""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import platform
import subprocess
import time

from .settings import ROOT, model_config, seed_all, write_json
import numpy as np
import torch
from models.ema import EMAHelper
from models.losses import ACTLossHead, stablemax_cross_entropy
from models.recursive_reasoning.trm_plastic import PlasticTinyRecursiveReasoningModel_ACTV1
from models.sparse_embedding import CastedSparseEmbeddingSignSGD_Distributed
from .adam_atan2 import AdamATan2
from .data import load_data, augment_batch


def state_snapshot(model):
    return {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}


def state_digest(state):
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


class CandidateStream:
    """Fixed candidate stream; ACT admits only candidates for halted slots."""
    def __init__(self, data, batch_size, seed):
        self.data, self.batch_size = data, batch_size
        self.rng = np.random.default_rng(seed + 101)
        self.order, self.offset, self.serial = np.array([], dtype=int), 0, 0

    def next(self):
        indices = []
        while len(indices) < self.batch_size:
            if self.offset == len(self.order):
                self.order, self.offset = self.rng.permutation(len(self.data["inputs"])), 0
            count = min(self.batch_size - len(indices), len(self.order) - self.offset)
            indices.extend(self.order[self.offset:self.offset + count].tolist())
            self.offset += count
        inputs, labels = augment_batch(self.data["inputs"][indices], self.data["labels"][indices], self.rng)
        serials = np.arange(self.serial, self.serial + self.batch_size, dtype=np.int64)
        self.serial += self.batch_size
        return {"inputs": inputs, "labels": labels,
                "puzzle_identifiers": np.zeros(self.batch_size, dtype=np.int64),
                "base_ids": self.data["base_ids"][indices], "sample_ids": serials}


def metrics_from_predictions(predictions, inputs, labels):
    predictions, inputs, labels = [np.asarray(value) for value in (predictions, inputs, labels)]
    given = inputs != 1
    correct = predictions == labels
    n = len(labels)
    grids = predictions.reshape(-1, 9, 9)
    expected = np.arange(2, 11)
    rows_ok = (np.sort(grids, axis=2) == expected).all(axis=(1, 2))
    cols_ok = (np.sort(grids, axis=1) == expected[None, :, None]).all(axis=(1, 2))
    boxes = grids.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    boxes_ok = (np.sort(boxes, axis=2) == expected).all(axis=(1, 2))
    givens_ok = np.logical_or(~given, correct).all(axis=1)
    return {"n": n, "all_accuracy": float(correct.mean()),
            "blank_accuracy": float(correct[~given].mean()),
            "given_agreement": float(correct[given].mean()),
            "exact_accuracy": float(correct.all(axis=1).mean()),
            "valid_grid_fraction": float((rows_ok & cols_ok & boxes_ok & givens_ok).mean()),
            "clue_copy_accuracy": float(given.mean()), "blank_cells": int((~given).sum()),
            "given_cells": int(given.sum())}


@torch.no_grad()
def evaluate(core, data, mode, size, batch_size):
    core.eval().set_mode(mode)
    predictions, inputs, labels = [], [], []
    losses = []
    torch.cuda.synchronize()
    started = time.perf_counter()
    before = state_digest(state_snapshot(core))
    for offset in range(0, min(size, len(data["inputs"])), batch_size):
        end = min(offset + batch_size, size, len(data["inputs"]))
        batch = {key: torch.as_tensor(data[key][offset:end], device="cuda", dtype=torch.long)
                 for key in ("inputs", "labels", "puzzle_identifiers")}
        with torch.device("cuda"):
            carry = core.initial_carry(batch)
        outputs = None
        for _ in range(core.config.halt_max_steps):
            carry, outputs = core(carry, batch)
        assert carry.halted.all() and int(carry.steps.min()) == core.config.halt_max_steps
        predictions.append(outputs["logits"].argmax(-1).cpu().numpy())
        inputs.append(batch["inputs"].cpu().numpy())
        labels.append(batch["labels"].cpu().numpy())
        losses.append(float(stablemax_cross_entropy(outputs["logits"], batch["labels"]).sum()))
    torch.cuda.synchronize()
    count = sum(len(value) for value in labels)
    result = metrics_from_predictions(np.concatenate(predictions), np.concatenate(inputs), np.concatenate(labels))
    result.update(loss=sum(losses) / (count * 81), act_steps=core.config.halt_max_steps,
                  inference_seconds=time.perf_counter() - started,
                  parameter_state_unchanged=before == state_digest(state_snapshot(core)),
                  runtime_gates_cleared=all(adapter.gate is None for adapter in core.adapters))
    if not result["parameter_state_unchanged"] or not result["runtime_gates_cleared"]:
        raise AssertionError("Evaluation changed model state or left runtime gates")
    return result


def run(cfg):
    run_dir = ROOT / cfg["results_dir"]
    if run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Refusing to overwrite {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    seed_all(cfg["seed"])
    arch = model_config(cfg["batch_size"])
    cfg = dict(cfg, model_config=arch)
    write_json(run_dir / "config.json", cfg)
    source_manifest = {}
    for folder in (ROOT / "models", ROOT / "local_lab"):
        for source in folder.rglob("*.py"):
            relative = source.relative_to(ROOT)
            raw = source.read_bytes()
            target = run_dir / "source_snapshot" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(raw)
            source_manifest[str(relative).replace("\\", "/")] = hashlib.sha256(raw).hexdigest()
    write_json(run_dir / "source_manifest.json", source_manifest)
    data_root = ROOT / cfg["data_dir"]
    data = load_data(data_root)
    write_json(run_dir / "data_audit.json", json.loads((data_root / "manifest.json").read_text()))
    with torch.device("cuda"):
        initial_model = PlasticTinyRecursiveReasoningModel_ACTV1(arch)
    initial = state_snapshot(initial_model)
    initial_sha = state_digest(initial)
    checkpoint_dir = ROOT / "checkpoints" / cfg["name"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    initial_path = checkpoint_dir / "initial_full.pt"
    if initial_path.exists():
        raise FileExistsError(initial_path)
    torch.save({"state_dict": initial, "model_config": arch}, initial_path)
    summary = {"status": "running", "seed": cfg["seed"], "stages": [], "evaluations": [],
               "upstream_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
               "model_config": arch, "A_parameters": sum(p.numel() for n,p in initial_model.inner.named_parameters() if not n.endswith((".u", ".v"))),
               "B_parameters": initial_model.controller_parameters, "controller_architecture": initial_model.controller_architecture,
               "native_puzzle_embedding_table_elements": initial_model.puzzle_emb.weights.numel(),
               "native_puzzle_gradient_buffer_elements": initial_model.puzzle_emb.local_weights.numel(),
               "initial_state_sha256": initial_sha, "initial_checkpoint": str(initial_path.relative_to(ROOT)),
               "torch": torch.__version__, "python": platform.python_version(), "gpu": torch.cuda.get_device_name(),
               "optimizer": "Portable PyTorch implementation of adam-atan2 0.0.3 upstream reference formula; fused CUDA build unavailable on Windows",
               "native_sparse_optimizer": "CastedSparseEmbeddingSignSGD_Distributed, world_size=1",
               "primary_weights": "raw fixed final; EMA also evaluated separately with mu=0.999",
               "gradient_policy": "Native H3/L6: 21 shared reasoning calls, first 14 no_grad, last 7 differentiable; carry detached after every ACT update",
               "B_history_policy": "Reset per native ACT update; evolve over all 21 reasoning calls",
               "scope": "One-seed from-scratch bounded early-learning comparison. No isolated-A adaptation or published-score reproduction claim."}
    del initial_model
    gc.collect()

    def save():
        summary["elapsed_seconds"] = time.perf_counter() - started
        summary["max_seconds"] = cfg["max_seconds"]
        summary["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
        write_json(run_dir / "summary.json", summary)

    for variant in cfg["variants"]:
        seed_all(cfg["seed"])
        with torch.device("cuda"):
            core = PlasticTinyRecursiveReasoningModel_ACTV1(arch)
        core.load_state_dict(initial, strict=True)
        assert state_digest(state_snapshot(core)) == initial_sha
        core.configure_training(variant).train()
        loss_model = ACTLossHead(core, "stablemax_cross_entropy")
        active = [p for p in core.parameters() if p.requires_grad]
        dense = AdamATan2(active, lr=cfg["learning_rate"], betas=tuple(cfg["betas"]), weight_decay=cfg["weight_decay"])
        sparse = CastedSparseEmbeddingSignSGD_Distributed(
            [core.puzzle_emb.weights, core.puzzle_emb.local_weights, core.puzzle_emb.local_ids],
            world_size=1, lr=cfg["puzzle_embedding_learning_rate"], weight_decay=cfg["weight_decay"])
        ema = EMAHelper(mu=cfg["ema_rate"])
        ema.register(core)
        stream = CandidateStream(data["train"], cfg["batch_size"], cfg["seed"])
        carry, done, admissions = None, 0, 0
        seen_base, seen_augmented = set(), set()
        candidates_hash, admitted_hash = hashlib.sha256(), hashlib.sha256()
        stage_started = time.perf_counter()
        for step in range(1, cfg["steps"] + 1):
            if time.perf_counter() - started >= cfg["training_cutoff_seconds"]:
                break
            proposed = stream.next()
            candidates_hash.update(proposed["base_ids"].tobytes() + proposed["inputs"].tobytes())
            accepted = np.ones(cfg["batch_size"], dtype=bool) if carry is None else carry.halted.cpu().numpy()
            admitted_hash.update(proposed["sample_ids"][accepted].tobytes() + proposed["inputs"][accepted].tobytes())
            admissions += int(accepted.sum())
            seen_base.update(proposed["base_ids"][accepted].tolist())
            seen_augmented.update(hashlib.sha256(row.tobytes()).hexdigest() for row in proposed["inputs"][accepted])
            batch = {key: torch.as_tensor(value, device="cuda", dtype=torch.long) for key,value in proposed.items()}
            if carry is None:
                with torch.device("cuda"):
                    carry = core.initial_carry(batch)
            dense.zero_grad(set_to_none=True)
            sparse.zero_grad(set_to_none=True)
            carry, loss, native_metrics, outputs, _ = loss_model(carry=carry, batch=batch, return_keys=["preds"])
            normalized = loss / cfg["batch_size"]
            if not torch.isfinite(normalized):
                raise FloatingPointError(f"Nonfinite {variant} loss, step {step}")
            normalized.backward()
            warmup = min(1., step / cfg["warmup_steps"])
            for group in dense.param_groups:
                group["lr"] = cfg["learning_rate"] * warmup
            for group in sparse.param_groups:
                group["lr"] = cfg["puzzle_embedding_learning_rate"] * warmup
            dense.step()
            sparse.step()
            ema.update(core)
            done = step
            if step == 1 or step % cfg["log_every"] == 0 or step == cfg["steps"]:
                current = metrics_from_predictions(outputs["preds"].cpu().numpy(),
                           carry.current_data["inputs"].cpu().numpy(), carry.current_data["labels"].cpu().numpy())
                row = {"variant": variant, "step": step, "train_loss": normalized.item(), **current,
                       "new_puzzle_admissions": admissions, "elapsed_s": time.perf_counter() - stage_started,
                       "reasoning_calls": core.last_trace["reasoning_calls"],
                       "gradient_enabled_calls": core.last_trace["gradient_enabled_calls"],
                       "controller_calls": core.last_trace["controller_calls"]}
                with (run_dir / "history.jsonl").open("a", encoding="utf-8") as history:
                    history.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
        torch.cuda.synchronize()
        raw = state_snapshot(core)
        ema_state = {name: value.clone() for name,value in raw.items()}
        for name, value in ema.state_dict().items():
            ema_state[name] = value.detach().cpu().clone()
        path = checkpoint_dir / f"{variant}.pt"
        if path.exists():
            raise FileExistsError(path)
        meta = {"variant": variant, "steps_requested": cfg["steps"], "steps_executed": done,
                "complete": done == cfg["steps"], "initial_state_sha256": initial_sha,
                "trainable_parameters": sum(p.numel() for p in active),
                "act_slot_presentations": done * cfg["batch_size"], "new_puzzle_admissions": admissions,
                "unique_base_puzzles_seen": len(seen_base), "unique_augmented_puzzles_seen": len(seen_augmented),
                "candidate_schedule_sha256": candidates_hash.hexdigest(), "admitted_schedule_sha256": admitted_hash.hexdigest(),
                "training_seconds": time.perf_counter() - stage_started, "checkpoint": str(path.relative_to(ROOT)),
                "selection": "fixed final step; no validation checkpoint selection",
                "raw_state_sha256": state_digest(raw), "ema_state_sha256": state_digest(ema_state)}
        torch.save({"state_dict": raw, "ema_state_dict": ema_state, "model_config": arch, "metadata": meta}, path)
        summary["stages"].append(meta)
        save()
        del loss_model, core, dense, sparse, ema, carry, raw, ema_state
        gc.collect()
        torch.cuda.empty_cache()
    summary["training_status"] = "complete" if len(summary["stages"]) == len(cfg["variants"]) and all(row["complete"] for row in summary["stages"]) else "partial"
    summary["training_elapsed_seconds"] = time.perf_counter() - started
    save()
    # No endpoint or counterfactual evaluation steals a later arm's training budget.
    for stage in summary["stages"]:
        if not stage["complete"]:
            continue
        payload = torch.load(ROOT / stage["checkpoint"], map_location="cpu", weights_only=True)
        with torch.device("cuda"):
            core = PlasticTinyRecursiveReasoningModel_ACTV1(arch)
        normal = "off" if stage["variant"] == "native" else "static" if stage["variant"] == "static_joint" else "dynamic"
        settings = [("raw", normal), ("ema", normal)]
        if stage["variant"] == "joint":
            settings += [("raw", "off"), ("raw", "fixed")]
        for weights, mode in settings:
            for split in ("val", "test"):
                if time.perf_counter() - started >= cfg["max_seconds"]:
                    break
                core.load_state_dict(payload["state_dict" if weights == "raw" else "ema_state_dict"], strict=True)
                metrics = evaluate(core, data[split], mode, cfg["eval_size"], cfg["eval_batch_size"])
                row = {"variant": stage["variant"], "weights": weights, "mode": mode, "split": split, **metrics}
                summary["evaluations"].append(row)
                write_json(run_dir / "evaluations.json", {"rows": summary["evaluations"]})
                save()
                print(json.dumps(row), flush=True)
        del core, payload
        gc.collect()
        torch.cuda.empty_cache()
    expected = 16 if cfg["variants"] == ["native", "static_joint", "joint"] else sum(8 if variant == "joint" else 4 for variant in cfg["variants"])
    summary["status"] = "complete" if summary["training_status"] == "complete" and len(summary["evaluations"]) == expected else "partial"
    summary["matched_candidate_streams"] = len({row["candidate_schedule_sha256"] for row in summary["stages"]}) == 1
    summary["matched_actual_admission_streams"] = len({row["admitted_schedule_sha256"] for row in summary["stages"]}) == 1
    save()
    print(json.dumps({"run_dir": str(run_dir), "status": summary["status"], "elapsed_seconds": summary["elapsed_seconds"]}), flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="local_lab/config.json")
    args = parser.parse_args()
    run(json.loads((ROOT / args.config).read_text()))


if __name__ == "__main__":
    main()
