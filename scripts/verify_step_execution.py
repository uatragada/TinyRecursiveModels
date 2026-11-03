"""Replay one puzzle batch and check weight changes after projection dtype casts."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models.recursive_reasoning.trm_stacked import StackedDenseTRM


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify(directory):
    started = time.perf_counter()
    receipt = directory / "step_execution_verification.json"
    if receipt.exists() or (directory / "applied_step_trace.json").exists():
        raise FileExistsError("Refusing to overwrite a prior execution check")
    torch.set_num_threads(4)
    cfg = json.loads((directory / "config.json").read_text())
    payload = torch.load(directory / "dense_smoke.pt", map_location="cpu", weights_only=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    with torch.device(device):
        model = StackedDenseTRM(payload["model_config"]).eval()
    model.load_state_dict(payload["state_dict"], strict=True)
    if model.weight_update_cadence != "per_reasoning_step":
        raise ValueError("This execution check requires per_reasoning_step cadence")
    with np.load(directory / "predictions.npz") as values:
        inputs = values["inputs"][:cfg["batch_size"]].copy()
    current = {"inputs": torch.as_tensor(inputs, device=device, dtype=torch.long),
               "puzzle_identifiers": torch.zeros(len(inputs), device=device, dtype=torch.long)}
    previous, rows = {}, {}
    def observe(name):
        def hook(module, args):
            key = (int(model.phase[0]) + 1, model.call)
            row = rows.setdefault(key, {"act_step": key[0], "reasoning_call": key[1], "ports": {}})
            if name in row["ports"]:  # B's same output weight is used across decoder chunks.
                return
            if module.delta is None:
                raise AssertionError(f"{name} did not receive its next weight matrix")
            applied = module.delta.detach().to(args[0].dtype)
            if not torch.isfinite(applied).all():
                raise AssertionError("A projection received a nonfinite update")
            changed = None if name not in previous else int((applied != previous[name]).sum())
            change = None if name not in previous else float((applied - previous[name]).abs().max())
            if changed is not None and changed == 0:
                raise AssertionError(f"{name} reused its applied matrix at {key}")
            row["ports"][name] = {"dtype": str(applied.dtype), "shape": list(applied.shape),
                "changed_entries": changed, "maximum_change": change}
            previous[name] = applied.clone()
        return hook
    handles = [port.register_forward_pre_hook(observe(f"A_projection_{i}")) for i, port in enumerate(model.ports)]
    handles.append(model.b_write_port.register_forward_pre_hook(observe("B_generation_head")))
    with torch.no_grad():
        with torch.device(device):
            carry = model.initial_carry(current)
        for _ in range(model.config.halt_max_steps):
            if time.perf_counter() - started > 120:
                raise TimeoutError("Applied-weight verification exceeded its two-minute cap")
            carry, outputs = model(carry, current)
    for handle in handles:
        handle.remove()
    expected = [(act + 1, call + 1) for act in range(model.config.halt_max_steps)
                for call in range(model.reasoning_calls_per_act)]
    if list(rows) != expected or any(len(row["ports"]) != len(model.ports) + 1 for row in rows.values()):
        raise AssertionError("Applied weights were not observed at every projection and step")
    if not carry.halted.all() or not torch.isfinite(outputs["logits"]).all():
        raise AssertionError("The replay did not finish a finite full puzzle trajectory")
    trace = directory / "applied_step_trace.json"
    trace.write_text(json.dumps(list(rows.values()), indent=2), encoding="utf-8")
    result = {"status": "passed", "device": device, "reasoning_steps": len(rows),
        "projections_per_step": len(model.ports) + 1,
        "all_consecutive_applied_matrices_changed": True,
        "elapsed_seconds": time.perf_counter() - started,
        "checkpoint_sha256": sha(directory / "dense_smoke.pt"),
        "trace_sha256": sha(trace), "verifier_sha256": sha(Path(__file__)),
        "source_sha256": {str(path.relative_to(ROOT)): sha(path) for path in [
            ROOT / "models/recursive_reasoning/trm_dense.py", ROOT / "models/recursive_reasoning/trm_stacked.py"]},
        "scope": "Inference replay of one complete validation batch using the final smoke checkpoint. Updates are inspected after casting to each projection's actual input dtype; no optimizer replay or performance comparison."}
    receipt.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/per_step_weights_smoke_20261102")
    verify(ROOT / parser.parse_args().run)
