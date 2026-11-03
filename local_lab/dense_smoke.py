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
from models.recursive_reasoning.trm_stacked import StackedDenseTRM
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
    hierarchy = cfg.get("hierarchy", False)
    directory = (ROOT / cfg["results_dir"]).resolve()
    if not directory.is_relative_to(ROOT.resolve() / "results"):
        raise ValueError("Smoke artifacts must stay under this checkout's results directory")
    if directory.exists():
        raise FileExistsError(f"Refusing to overwrite {directory}")
    directory.mkdir(parents=True)
    write_json(directory / "config.json", cfg)
    sources = sorted(str(path.relative_to(ROOT)).replace("\\", "/")
        for folder in ("models", "local_lab") for path in (ROOT / folder).rglob("*.py"))
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
        model = (StackedDenseTRM(arch) if hierarchy else DensePlasticTRM(arch)).eval()
        native = TinyRecursiveReasoningModel_ACTV1(arch).eval()
        reference_b = DensePlasticTRM(arch).eval() if hierarchy else None
    if hierarchy:
        model.load_dense_state(payload["state_dict"])
        model.configure_hierarchy(train_b=True, train_c=True, train_a=False)
    else:
        model.load_a_state(a_state)
    native.load_state_dict(a_state, strict=True)
    expected_a = state_digest(a_state)
    if state_digest(model.a_state()) != expected_a:
        raise AssertionError("Dense model did not load exactly the native A tensors")
    model_state_keys = list(model.state_dict())
    if any(key.endswith((".u", ".v")) or "static_gates" in key for key in model_state_keys):
        raise AssertionError("A historical low-rank component entered the dense model")
    local_weights_before = model.puzzle_emb.local_weights.detach().cpu().clone()
    def b_state():
        if hierarchy:
            return model.b_state()
        return {k: v for k, v in state_snapshot(model).items() if not k.startswith("inner.")}
    def c_state():
        return {k: v for k, v in state_snapshot(model).items()
                if k.startswith(("meta_controller.", "meta_generator."))}
    b_before = state_digest(b_state())
    initial_c_state = c_state() if hierarchy else None
    c_before = state_digest(initial_c_state) if hierarchy else None
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
    def evaluate(mode, check_native=False, meta_mode="dynamic", check_b=False, meta_read_mode="tokens"):
        model.set_mode(mode)
        if hierarchy:
            model.set_meta_mode(meta_mode)
            model.set_meta_read_mode(meta_read_mode)
        if check_b:
            reference_b.load_state_dict(model.dense_state(), strict=True)
        predictions, labels, inputs, objectives = [], [], [], []
        initial_matrices = None
        for offset in range(0, cfg["dev_size"], cfg["batch_size"]):
            current = batch(development, offset, cfg["batch_size"])
            with torch.device(device):
                carry = model.initial_carry(current)
                native_carry = native.initial_carry(current) if check_native else None
                b_carry = reference_b.initial_carry(current) if check_b else None
            for _ in range(model.config.halt_max_steps):
                check_budget()
                carry, outputs = model(carry, current)
                if check_native:
                    native_carry, untouched = native(native_carry, current)
                    if not torch.equal(outputs["logits"], untouched["logits"]):
                        raise AssertionError("Disabled dense controller differs from native A")
                if check_b:
                    b_carry, expected_b = reference_b(b_carry, current)
                    if not torch.equal(outputs["logits"], expected_b["logits"]):
                        raise AssertionError("C-off differs from the corresponding standalone A/B model")
                if initial_matrices is None and model.last_action is not None:
                    initial_matrices = [value[:1].cpu().clone() for value in model.last_action]
                if any(port.delta is not None for port in model.ports):
                    raise AssertionError("A temporary matrix leaked beyond its ACT call")
                if hierarchy and model.b_write_port.delta is not None:
                    raise AssertionError("C left a temporary matrix installed in B")
            predictions.append(outputs["logits"].argmax(-1).cpu().numpy())
            labels.append(current["labels"].cpu().numpy())
            inputs.append(current["inputs"].cpu().numpy())
            objectives.append(float(loss(outputs["logits"], current["inputs"], current["labels"])) * len(current["inputs"]))
        pred, inp, lab = map(np.concatenate, (predictions, inputs, labels))
        return {"mode": mode, **metrics_from_predictions(pred, inp, lab),
            "task_loss": sum(objectives) / len(pred), "act_steps": model.config.halt_max_steps,
            "native_logits_bitwise_equal": True if check_native else None,
            "meta_mode": meta_mode if hierarchy else None,
            "meta_read_mode": meta_read_mode if hierarchy else None,
            "standalone_b_logits_bitwise_equal": True if check_b else None}, pred, initial_matrices

    baseline, baseline_predictions, _ = evaluate("off", check_native=True)
    initial, initial_predictions, matrices = evaluate("dynamic")
    matrix_diagnostics = []
    for index, (matrix, port) in enumerate(zip(matrices, model.ports)):
        matrix_diagnostics.append({"projection": index, "shape": list(matrix.shape[1:]),
            "measured_initial_rank": int(torch.linalg.matrix_rank(matrix[0])),
            "maximum_possible_rank": min(matrix.shape[1:]),
            "initial_frobenius_ratio": float(matrix.norm() / port.base.weight.detach().float().cpu().norm())})
    torch.save(matrices, directory / "initial_dense_matrices.pt")
    meta_diagnostics, initial_b_only, inter_model_attention = None, None, None
    if hierarchy:
        if model.meta_controller.cross_attention:
            weights = model.meta_controller.last_attention[:1].cpu().clone()
            torch.save({"weights": weights,
                "b_tokens": model.controller.reasoning_tokens[:1].cpu().clone(),
                "stage_ids": model.controller.reasoning_stage_ids.cpu().clone(),
                "slot_ids": model.controller.reasoning_slot_ids.cpu().clone()},
                directory / "initial_cross_attention.pt")
            inter_model_attention = {"enabled": True, "heads": int(weights.shape[1]),
                "queries": int(weights.shape[2]), "b_tokens": int(weights.shape[3]),
                "b_token_width": model.controller.width,
                "stage_names": ["reader_output"] + [f"encoder_{i+1}" for i in range(len(model.controller.transformer.layers))]
                    + ["memory", "readout"],
                "scope": "Per-head C-to-B attention at the last initial-evaluation ACT. Attention weights are read diagnostics, not causal explanations."}
        meta_matrix = model.last_c_action[:1].cpu().clone()
        torch.save(meta_matrix, directory / "initial_c_matrix.pt")
        base = model.b_write_port.base.weight.detach().float().cpu()
        meta_diagnostics = {"shape": list(meta_matrix.shape[1:]),
            "measured_initial_rank": int(torch.linalg.matrix_rank(meta_matrix[0])),
            "maximum_possible_rank": min(base.shape),
            "initial_frobenius_ratio": float(meta_matrix.norm() / base.norm()),
            "scope": "Last sampled initial-evaluation ACT; C modifies only B's generator output weight"}
        initial_b_only, _, _ = evaluate("dynamic", meta_mode="off", check_b=True)
    trainable = model.b_parameters + (model.c_parameters if hierarchy else [])
    optimizer = AdamATan2(trainable, lr=cfg["learning_rate"], betas=(.9, .95), weight_decay=.01)
    model.set_mode("dynamic")
    if hierarchy:
        model.set_meta_mode("dynamic")
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
        components = [("reader", list(model.controller.parameters())),
                      ("generator", list(model.generator.parameters()))]
        if hierarchy:
            components += [("meta_reader", list(model.meta_controller.parameters())),
                           ("meta_generator", list(model.meta_generator.parameters()))]
        for name, parameters in components:
            gradients = [p.grad for p in parameters if p.grad is not None]
            if not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise FloatingPointError(f"Missing or nonfinite {name} gradients")
            component_norms[name] = float(torch.stack([g.square().sum() for g in gradients]).sum().sqrt())
            if component_norms[name] <= 0:
                raise AssertionError(f"No gradient reached the {name}")
        torch.nn.utils.clip_grad_norm_(trainable, 1., error_if_nonfinite=True)
        optimizer.step()
        history.append({"step": step, "task_loss": float(objective.detach()),
            "gradient_norms": component_norms})
        if hierarchy and model.meta_controller.cross_attention:
            read = model.meta_controller.cross_read
            gradient = read.in_proj_weight.grad.reshape(3, read.num_heads, read.embed_dim // read.num_heads, read.embed_dim)
            norms = gradient.square().sum((0, 2, 3)).sqrt()
            if not torch.isfinite(norms).all() or not (norms > 0).all():
                raise AssertionError("An inter-model attention head received no finite gradient")
            history[-1]["cross_attention_head_gradient_norms"] = norms.detach().cpu().tolist()
        print(json.dumps({"stage": "dense_smoke", **history[-1]}), flush=True)
    training_seconds = time.perf_counter() - training_started
    after, final_predictions, _ = evaluate("dynamic")
    ablations, ablation_predictions = {}, {}
    if hierarchy:
        for name in ("off", "shuffle", "phase", "reset_memory"):
            row, predictions, _ = evaluate("dynamic", meta_mode=name, check_b=name == "off")
            ablations["c_" + name] = row
            ablation_predictions["c_" + name] = predictions
        if model.meta_controller.cross_attention:
            for read_mode in ("pooled", "shuffle"):
                row, predictions, _ = evaluate("dynamic", meta_read_mode=read_mode)
                ablations["c_read_" + read_mode] = row
                ablation_predictions["c_read_" + read_mode] = predictions
    disabled_after, disabled_predictions, _ = evaluate("off", check_native=True)
    if not np.array_equal(baseline_predictions, disabled_predictions):
        raise AssertionError("A predictions changed after B-only training")
    if expected_a != state_digest(model.a_state()):
        raise AssertionError("Frozen A tensors changed")
    if not torch.equal(local_weights_before, model.puzzle_emb.local_weights.detach().cpu()):
        raise AssertionError("Frozen sparse embedding scratch weights changed")
    b_after = state_digest(b_state())
    if b_before == b_after or file_sha(checkpoint) != source_sha:
        raise AssertionError("B failed to update or the source checkpoint changed")
    c_after = state_digest(c_state()) if hierarchy else None
    if hierarchy and c_after == c_before:
        raise AssertionError("C failed to update")
    check_budget()
    torch.save({"state_dict": state_snapshot(model), "model_config": arch,
        "source_checkpoint_sha256": source_sha, "a_sha256": expected_a,
        "steps": cfg["steps"], "b_sha256": b_after, "c_sha256": c_after,
        "initial_c_state": initial_c_state}, directory / "dense_smoke.pt")
    np.savez_compressed(directory / "predictions.npz", inputs=development["inputs"],
        labels=development["labels"], base_ids=development["base_ids"],
        baseline=baseline_predictions, initial=initial_predictions, final=final_predictions,
        **ablation_predictions)
    result = {"status": "complete", "scope": "Engineering smoke only; not a matched performance experiment",
        "device": device, "seed": cfg["seed"], "source_checkpoint_sha256": source_sha,
        "source_a_arm": "native, with historical inactive wrapper/controller/factor tensors discarded",
        "architecture": arch, "b_parameters": sum(p.numel() for p in model.b_parameters),
        "hierarchy": hierarchy, "c_parameters": sum(p.numel() for p in model.c_parameters) if hierarchy else 0,
        "a_parameters": sum(p.numel() for p in model.inner.parameters()),
        "generated_weights_per_example_per_act": sum(p.base.weight.numel() for p in model.ports),
        "dense_matrices": matrix_diagnostics, "low_rank_factors_present": False,
        "c_matrix": meta_diagnostics,
        "inter_model_attention": inter_model_attention,
        "temporary_weights_cleared": all(p.delta is None for p in model.ports),
        "c_temporary_weights_cleared": model.b_write_port.delta is None if hierarchy else None,
        "frozen_a_unchanged": True, "frozen_sparse_scratch_unchanged": True,
        "a_sha256": expected_a, "b_before_sha256": b_before, "b_after_sha256": b_after,
        "c_before_sha256": c_before, "c_after_sha256": c_after,
        "admitted_training_base_ids": admitted_ids, "training": history,
        "evaluations": {"off": baseline, "initial": initial, "final": after,
                        "off_after_training": disabled_after,
                        **({"initial_c_off": initial_b_only} if hierarchy else {}), **ablations},
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
    if hierarchy:
        structured = model.meta_controller.cross_attention
        config_example = ("local_lab/config_cross_attention_smoke.json" if structured
                          else "local_lab/config_stack_smoke.json")
        c_read = ("C cross-attends to B's reader output, each encoder block, memory and readout, using stage/slot identifiers. "
            "A causal Transformer then processes a four-step history of those attended observations."
            if structured else "C has a causal Transformer, its own state and four-step B-context history.")
        rows = "\n".join(f"| {name} | {100*row['blank_accuracy']:.3f}% | {row['task_loss']:.6f} |"
            for name, row in result["evaluations"].items())
        report = f"""# Stacked dense controllers: C modifies B, B modifies A

The implemented chain is:

`A states -> B -> C reads B -> dense delta_W_B -> B uses W_B + delta_W_B -> dense delta_W_A -> A uses W_A + delta_W_A`

C writes the complete {meta_diagnostics['shape'][0]} by {meta_diagnostics['shape'][1]}
output weight of B's nonlinear weight generator. It does not replace all of B.
B continues to write both complete 512 by 1536 A projections. There are no
low-rank factors at either level. {c_read} Both levels use dense chunked generation.

C's update has a {cfg['architecture']['meta_relative_cap']:.0%} relative Frobenius
limit; B's A updates retain their {cfg['architecture']['dense_relative_cap']:.0%}
limit. These numerical bounds do not establish behavioral safety or interpretability.
C observes detached B states and past C state without labels. C modifies B
before B generates A's next updates, avoiding a same-step circular dependency.
Memory values persist across ACT updates and reset independently for new puzzles;
temporal gradient credit is truncated at ACT boundaries.

This run loaded the earlier dense A/B smoke checkpoint, then jointly trained B
and fresh C while keeping A frozen. B has {result['b_parameters']:,} parameters;
C adds {result['c_parameters']:,}. The sampled initial C matrix had measured rank
{meta_diagnostics['measured_initial_rank']} out of {meta_diagnostics['maximum_possible_rank']}.

## Executed smoke

- {cfg['steps']} B/C updates, {len(admitted_ids)} admitted training puzzles, {cfg['dev_size']} separate validation puzzles.
- Finite nonzero gradients reached both readers and both weight generators on every update.
- All A tensors and sparse scratch weights stayed identical. Disabling both controllers reproduced native logits bit for bit.
- Disabling C reproduced a standalone A/B model with the corresponding current B weights bit for bit, before and after training.
- Temporary matrices at both levels were cleared after use.
- Runtime {result['elapsed_seconds']:.2f} seconds against a {cfg['max_seconds']}-second cap; peak Torch CUDA allocation {result['peak_cuda_allocated_mib']:.2f} MiB.
- No oracle or official test split was used.

| Smoke condition | Blank-cell accuracy | Task loss |
|---|---:|---:|
{rows}

The C-off/shuffled/phase/history-reset rows are inference ablations of the same
jointly trained checkpoint. They are not separately trained matched controls.
Eight updates and eight validation puzzles cannot establish an advantage from
stacking. A matched B-only training arm, an equally larger B, multiple seeds and
held-out adaptation probes remain necessary before a performance claim.

## Reproduce

```powershell
& '..\\plastic-latent-lab\\.venv\\Scripts\\python.exe' -m local_lab.dense_smoke --config {config_example}
```

Choose a fresh results directory before rerunning; existing artifacts are preserved.
"""
        if structured:
            report += f"""
## Attention between models

B has four cross-attention heads reading A's per-cell features. C now has
{inter_model_attention['heads']} additional cross-attention heads with
{inter_model_attention['queries']} learned queries reading
{inter_model_attention['b_tokens']} B tokens, each {inter_model_attention['b_token_width']}-dimensional.
The tokens retain B's reader/encoder stages, query slots, memory and readout;
they are not collapsed to a single B context before C attends. C's separate
causal self-attention remains in place. The original summary-reader variant is
preserved by leaving `meta_cross_attention` false.

All four inter-model heads received nonzero answer-loss gradients on every
training step. `initial_cross_attention.pt` saves per-head read weights and
source metadata. The `c_read_pooled` and `c_read_shuffle` rows are inference
ablations with the same C parameters. They test the read interface, without
establishing that any head has a human-readable reasoning role. The core
attention mechanism follows [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
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
