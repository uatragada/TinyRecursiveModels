"""Bounded, local engineering smoke for direct dense weight generation.

This checks native-checkpoint reuse, real-data optimization and evaluation. It
does not establish an accuracy, adaptability, interpretability or safety gain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

from .settings import ROOT, seed_all, write_json
import numpy as np
import torch

from models.losses import stablemax_cross_entropy
from models.recursive_reasoning.trm import (TinyRecursiveReasoningModel_ACTV1,
    TinyRecursiveReasoningModel_ACTV1Config)
from models.recursive_reasoning.trm_dense import DensePlasticTRM, native_a_from_checkpoint
from .adam_atan2 import AdamATan2
from .data import load_data
from .train import metrics_from_predictions, state_digest, state_snapshot


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def loss(logits, inputs, labels):
    values = stablemax_cross_entropy(logits, labels).float()
    blank, given = inputs == 1, inputs != 1
    return ((values * blank).sum(-1) / blank.sum(-1).clamp_min(1)
        + .25 * (values * given).sum(-1) / given.sum(-1).clamp_min(1)).mean()


def run(cfg):
    started = time.perf_counter()
    directory = (ROOT / cfg["results_dir"]).resolve()
    if not directory.is_relative_to(ROOT.resolve() / "results"):
        raise ValueError("Smoke artifacts must stay under this checkout's results directory")
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite {directory}")
    directory.mkdir(parents=True)
    write_json(directory / "config.json", cfg)
    sources = ("models/recursive_reasoning/trm.py", "models/recursive_reasoning/trm_dense.py",
        "models/layers.py", "models/losses.py", "models/sparse_embedding.py",
        "local_lab/dense_smoke.py", "local_lab/adam_atan2.py", "local_lab/settings.py",
        "local_lab/data.py", "local_lab/train.py")
    source_manifest = {}
    for name in sources:
        target = directory / "source_snapshot" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT / name).read_bytes())
        source_manifest[name] = file_sha(target)
    write_json(directory / "source_manifest.json", source_manifest)
    seed_all(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    def check_budget():
        if time.perf_counter() - started > cfg["max_seconds"]:
            raise TimeoutError("Dense smoke reached its wall-clock budget")
    checkpoint = ROOT / cfg["source_checkpoint"]
    source_sha = file_sha(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    a_state = native_a_from_checkpoint(payload["state_dict"])
    # Retain native A settings only. Historical factor/controller settings are
    # discarded along with their tensors, rather than silently reused.
    arch = {key: value for key, value in payload["model_config"].items()
            if key in TinyRecursiveReasoningModel_ACTV1Config.model_fields}
    arch.update(batch_size=cfg["batch_size"], **cfg["architecture"])
    with torch.device(device):
        model = DensePlasticTRM(arch).eval()
        native = TinyRecursiveReasoningModel_ACTV1(arch).eval()
    model.load_a_state(a_state)
    native.load_state_dict(a_state, strict=True)
    expected_a = state_digest(a_state)
    if state_digest(model.a_state()) != expected_a:
        raise AssertionError("Dense model did not load exactly the native A tensors")
    model_state_keys = list(model.state_dict())
    if any(key.endswith((".u", ".v")) or "static_gates" in key for key in model_state_keys):
        raise AssertionError("A historical low-rank component entered the dense model")
    local_weights_before = model.puzzle_emb.local_weights.detach().cpu().clone()
    b_before = state_digest({k: v for k, v in state_snapshot(model).items() if not k.startswith("inner.")})
    dataset = load_data(ROOT / cfg["data_dir"])
    def panel(start, count):
        return {key: value[start:start + count] for key, value in dataset["val"].items()}
    training = panel(cfg["train_offset"], cfg["train_size"])
    development = panel(cfg["dev_offset"], cfg["dev_size"])
    if set(training["base_ids"]) & set(development["base_ids"]):
        raise AssertionError("Training and smoke evaluation panels overlap")
    write_json(directory / "data_audit.json", {
        "dataset_manifest_sha256": file_sha(ROOT / cfg["data_dir"] / "manifest.json"),
        "panels": {name: {"base_ids": panel["base_ids"].tolist(),
            "inputs_sha256": hashlib.sha256(panel["inputs"].tobytes()).hexdigest(),
            "labels_sha256": hashlib.sha256(panel["labels"].tobytes()).hexdigest()}
            for name, panel in (("train", training), ("dev", development))},
        "official_test_used": False, "oracle_used": False})
    def batch(panel, offset, count):
        return {key: torch.as_tensor(panel[key][offset:offset + count], device=device, dtype=torch.long)
                for key in ("inputs", "labels", "puzzle_identifiers")}

    @torch.no_grad()
    def evaluate(mode, check_native=False):
        model.set_mode(mode)
        predictions, labels, inputs, objectives = [], [], [], []
        initial_matrices = None
        for offset in range(0, cfg["dev_size"], cfg["batch_size"]):
            current = batch(development, offset, cfg["batch_size"])
            with torch.device(device):
                carry = model.initial_carry(current)
                native_carry = native.initial_carry(current) if check_native else None
            for _ in range(model.config.halt_max_steps):
                check_budget()
                carry, outputs = model(carry, current)
                if check_native:
                    native_carry, untouched = native(native_carry, current)
                    if not torch.equal(outputs["logits"], untouched["logits"]):
                        raise AssertionError("Disabled dense controller differs from native A")
                if initial_matrices is None and model.last_action is not None:
                    initial_matrices = [value[:1].cpu().clone() for value in model.last_action]
                if any(port.delta is not None for port in model.ports):
                    raise AssertionError("A temporary matrix leaked beyond its ACT call")
            predictions.append(outputs["logits"].argmax(-1).cpu().numpy())
            labels.append(current["labels"].cpu().numpy())
            inputs.append(current["inputs"].cpu().numpy())
            objectives.append(float(loss(outputs["logits"], current["inputs"], current["labels"])) * len(current["inputs"]))
        pred, inp, lab = map(np.concatenate, (predictions, inputs, labels))
        return {"mode": mode, **metrics_from_predictions(pred, inp, lab),
            "task_loss": sum(objectives) / len(pred), "act_steps": model.config.halt_max_steps,
            "native_logits_bitwise_equal": True if check_native else None}, pred, initial_matrices

    baseline, baseline_predictions, _ = evaluate("off", check_native=True)
    initial, initial_predictions, matrices = evaluate("dynamic")
    matrix_diagnostics = []
    for index, (matrix, port) in enumerate(zip(matrices, model.ports)):
        matrix_diagnostics.append({"projection": index, "shape": list(matrix.shape[1:]),
            "measured_initial_rank": int(torch.linalg.matrix_rank(matrix[0])),
            "maximum_possible_rank": min(matrix.shape[1:]),
            "initial_frobenius_ratio": float(matrix.norm() / port.base.weight.detach().float().cpu().norm())})
    torch.save(matrices, directory / "initial_dense_matrices.pt")
    optimizer = AdamATan2(model.b_parameters, lr=cfg["learning_rate"], betas=(.9, .95), weight_decay=.01)
    model.set_mode("dynamic")
    carry, history, admitted_ids, offset = None, [], [], 0
    training_started = time.perf_counter()
    for step in range(1, cfg["steps"] + 1):
        check_budget()
        if carry is None or bool(carry.halted.all()):
            offset = (len(admitted_ids) // cfg["batch_size"] * cfg["batch_size"]) % cfg["train_size"]
            current = batch(training, offset, cfg["batch_size"])
            admitted_ids.extend(training["base_ids"][offset:offset + cfg["batch_size"]].tolist())
            with torch.device(device):
                carry = model.initial_carry(current)
        optimizer.zero_grad(set_to_none=True)
        carry, outputs = model(carry, current)
        objective = loss(outputs["logits"], current["inputs"], current["labels"])
        if not torch.isfinite(objective):
            raise FloatingPointError("Dense training loss is not finite")
        objective.backward()
        if any(p.grad is not None for p in model.inner.parameters()):
            raise AssertionError("Frozen A received gradients")
        component_norms = {}
        for name, parameters in (("reader", list(model.controller.parameters())),
                                 ("generator", list(model.generator.parameters()))):
            gradients = [p.grad for p in parameters if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError(f"Missing or nonfinite {name} gradients")
            component_norms[name] = float(torch.stack([g.square().sum() for g in gradients]).sum().sqrt())
            if component_norms[name] <= 0:
                raise AssertionError(f"No gradient reached the {name}")
        torch.nn.utils.clip_grad_norm_(model.b_parameters, 1., error_if_nonfinite=True)
        optimizer.step()
        history.append({"step": step, "task_loss": float(objective.detach()),
            "gradient_norms": component_norms})
        print(json.dumps({"stage": "dense_smoke", **history[-1]}), flush=True)
    training_seconds = time.perf_counter() - training_started
    after, final_predictions, _ = evaluate("dynamic")
    disabled_after, disabled_predictions, _ = evaluate("off", check_native=True)
    if not np.array_equal(baseline_predictions, disabled_predictions):
        raise AssertionError("A predictions changed after B-only training")
    if expected_a != state_digest(model.a_state()):
        raise AssertionError("Frozen A tensors changed")
    if not torch.equal(local_weights_before, model.puzzle_emb.local_weights.detach().cpu()):
        raise AssertionError("Frozen sparse embedding scratch weights changed")
    b_after = state_digest({k: v for k, v in state_snapshot(model).items() if not k.startswith("inner.")})
    if b_before == b_after or file_sha(checkpoint) != source_sha:
        raise AssertionError("B failed to update or the source checkpoint changed")
    check_budget()
    torch.save({"state_dict": state_snapshot(model), "model_config": arch,
        "source_checkpoint_sha256": source_sha, "a_sha256": expected_a,
        "steps": cfg["steps"], "b_sha256": b_after}, directory / "dense_smoke.pt")
    np.savez_compressed(directory / "predictions.npz", inputs=development["inputs"],
        labels=development["labels"], base_ids=development["base_ids"],
        baseline=baseline_predictions, initial=initial_predictions, final=final_predictions)
    result = {"status": "complete", "scope": "Engineering smoke only; not a matched performance experiment",
        "device": device, "seed": cfg["seed"], "source_checkpoint_sha256": source_sha,
        "source_a_arm": "native, with historical inactive wrapper/controller/factor tensors discarded",
        "architecture": arch, "b_parameters": sum(p.numel() for p in model.b_parameters),
        "a_parameters": sum(p.numel() for p in model.inner.parameters()),
        "generated_weights_per_example_per_act": sum(p.base.weight.numel() for p in model.ports),
        "dense_matrices": matrix_diagnostics, "low_rank_factors_present": False,
        "temporary_weights_cleared": all(p.delta is None for p in model.ports),
        "frozen_a_unchanged": True, "frozen_sparse_scratch_unchanged": True,
        "a_sha256": expected_a, "b_before_sha256": b_before, "b_after_sha256": b_after,
        "admitted_training_base_ids": admitted_ids, "training": history,
        "evaluations": {"off": baseline, "initial": initial, "final": after,
                        "off_after_training": disabled_after},
        "training_seconds": training_seconds,
        "peak_cuda_allocated_mib": torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None,
        "elapsed_seconds": time.perf_counter() - started}
    write_json(directory / "summary.json", result)
    report = f"""# Direct dense weight generation: local smoke

B now directly generates complete temporary matrices for both native TRM MLP
down-projections. Each is 512 by 1536 in this checkpoint. A shared nonlinear
decoder emits 32 by 32 tiles and assembles every entry. There are no U/V factors,
learned low-rank directions, or rank-four gates in this model. The construction is
a [hypernetwork](https://arxiv.org/abs/1609.09106), with an explicit per-matrix
Frobenius limit of {cfg['architecture']['dense_relative_cap']:.0%} of its base weight norm.
That numerical limit does not guarantee safe outputs or interpretable changes.

The original attachment specified low-rank changes. The user's subsequent
instruction rejects that architecture and supersedes the earlier requirement.
Historical experiments and their checkpoints remain preserved as baselines.

A comes from the native-training arm, with all historical unused controller and
factor tensors discarded. It is frozen throughout this smoke. B reads per-cell
latent states, prediction changes and Sudoku constraint residuals; a Transformer
and per-puzzle GRU produce its context. Memory resets on new puzzles. One set of
dense matrices is used in the final seven native calls of each 21-call ACT update.
Observations and memory credit are detached at the ACT boundary. Joint A/B
gradients are unit-tested; joint training has not been run for this architecture.

## Executed checks

- B has {result['b_parameters']:,} parameters and generates {result['generated_weights_per_example_per_act']:,} temporary weight entries per example per ACT.
- First sampled generated matrices have measured ranks {[v['measured_initial_rank'] for v in matrix_diagnostics]}, each out of 512 possible.
- Completed {cfg['steps']} B-only updates on {len(admitted_ids)} admitted puzzles, with finite nonzero reader and generator gradients.
- Full frozen A and its sparse scratch weights remained identical; disabling B reproduced native logits bit for bit before and after training.
- No oracle or official test data was used. This smoke used {cfg['dev_size']} separate validation puzzles.
- Runtime: {result['elapsed_seconds']:.2f} seconds, cap {cfg['max_seconds']} seconds. Peak Torch CUDA allocation: {result['peak_cuda_allocated_mib']} MiB.

The initial/final blank-cell accuracies were {100*initial['blank_accuracy']:.3f}%
and {100*after['blank_accuracy']:.3f}%; untouched A was {100*baseline['blank_accuracy']:.3f}%.
These tiny-panel smoke measurements are integration evidence only. No matched
training controls, new isolated-A comparison, interpretability evaluation or
safety evaluation has been completed for the dense architecture.

## Reproduce

From this checkout, using its existing shared local Python environment:

```powershell
& '..\\plastic-latent-lab\\.venv\\Scripts\\python.exe' -m local_lab.dense_smoke --config local_lab/config_dense_smoke.json
```

Choose a fresh results directory to rerun; existing artifacts are never overwritten.
"""
    (directory / "report.md").write_text(report, encoding="utf-8")
    print(json.dumps({"status": "complete", "results_dir": str(directory),
        "b_parameters": result["b_parameters"], "elapsed_seconds": result["elapsed_seconds"]}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="local_lab/config_dense_smoke.json")
    args = parser.parse_args()
    cfg = json.loads((ROOT / args.config).read_text())
    if not 1 <= cfg["steps"] <= 32 or not 1 <= cfg["batch_size"] <= 8 or not 1 <= cfg["max_seconds"] <= 300:
        raise ValueError("This entrypoint is restricted to a bounded engineering smoke")
    if cfg["train_size"] % cfg["batch_size"] or cfg["dev_size"] % cfg["batch_size"]:
        raise ValueError("Smoke panel sizes must be multiples of batch size")
    run(cfg)


if __name__ == "__main__":
    main()
