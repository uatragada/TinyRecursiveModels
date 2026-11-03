"""Read saved checkpoint tensors independently of the training/model code.

Run after training has exited. This imports Torch on CPU; it never constructs a
model, changes checkpoints, or trusts recorded digests without recomputing them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch


def fingerprint(state):
    digest = hashlib.sha256()
    for name in sorted(state):
        value = state[name].detach().cpu().contiguous()
        digest.update(f"{name}:{value.dtype}:{tuple(value.shape)}".encode())
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def partition(name):
    if name.startswith("controller."):
        return "B"
    if name.endswith((".u", ".v")):
        return "adapters"
    if name == "static_gates":
        return "static_gates"
    if name.startswith("inner."):
        return "A"
    raise ValueError(f"Unclassified persistent tensor: {name}")


def verify_main(root, run_dir):
    summary = json.loads((run_dir / "summary.json").read_text())
    initial = torch.load(root / summary["initial_checkpoint"], map_location="cpu", weights_only=True)["state_dict"]
    checks = []

    def check(name, result):
        checks.append({"name": name, "passed": bool(result)})

    check("initial checkpoint digest", fingerprint(initial) == summary["initial_state_sha256"])
    changes = {}
    for stage in summary["stages"]:
        variant = stage["variant"]
        payload = torch.load(root / stage["checkpoint"], map_location="cpu", weights_only=True)
        raw, ema = payload["state_dict"], payload["ema_state_dict"]
        check(f"{variant}: complete tensor keys", raw.keys() == initial.keys() == ema.keys())
        check(f"{variant}: raw digest", fingerprint(raw) == stage["raw_state_sha256"])
        check(f"{variant}: EMA digest", fingerprint(ema) == stage["ema_state_sha256"])
        check(f"{variant}: embedded metadata", payload["metadata"] == stage)
        changed = {key: [] for key in ("A", "B", "adapters", "static_gates")}
        for name, value in raw.items():
            check(f"{variant}: finite {name}", torch.isfinite(value).all())
            if not torch.equal(initial[name], value):
                changed[partition(name)].append(name)
        changes[variant] = {key: len(names) for key, names in changed.items()}
        check(f"{variant}: A learned", bool(changed["A"]))
        if variant == "native":
            check("native: B and adapters and static gates unchanged", not any(changed[key] for key in ("B", "adapters", "static_gates")))
        elif variant == "static_joint":
            check("static_joint: B unchanged", not changed["B"])
            check("static_joint: factors and global gates learned", bool(changed["adapters"]) and bool(changed["static_gates"]))
        elif variant == "joint":
            check("joint: B and factors learned", bool(changed["B"]) and bool(changed["adapters"]))
            check("joint: static gates unchanged", not changed["static_gates"])
        for step, path in stage.get("intermediate_checkpoints", {}).items():
            saved = torch.load(root / path, map_location="cpu", weights_only=True)
            state = saved["state_dict"]
            check(f"{variant}/{step}: checkpoint step", saved["step"] == int(step))
            check(f"{variant}/{step}: complete keys", state.keys() == initial.keys())
            frozen_groups = ("B", "adapters", "static_gates") if variant == "native" else ("B",) if variant == "static_joint" else ("static_gates",)
            for name in state:
                check(f"{variant}/{step}: finite {name}", torch.isfinite(state[name]).all())
                if partition(name) in frozen_groups:
                    check(f"{variant}/{step}: frozen {name}", torch.equal(state[name], initial[name]))
    return {"scope": "Independent saved-tensor hashes, finite values and expected trained/frozen components; CPU Torch only. Does not replay training or independently recompute predictions.",
            "passed": all(item["passed"] for item in checks), "checks_passed": sum(item["passed"] for item in checks),
            "checks_total": len(checks), "changed_tensor_counts": changes, "checks": checks}


def verify_probe(root, run_dir):
    summary = json.loads((run_dir / "summary.json").read_text())
    common = torch.load(root / summary["common_initial_checkpoint"], map_location="cpu", weights_only=True)["state_dict"]
    common_interface = {name: value for name, value in common.items() if partition(name) != "A"}
    checks, changes, constants = [], {}, []

    def check(name, result):
        checks.append({"name": name, "passed": bool(result)})

    check("common full tensor digest", fingerprint(common) == summary["common_fresh_state_sha256"])
    check("common interface digest", fingerprint(common_interface) == summary["common_fresh_non_A_sha256"])
    for stage in summary["stages"]:
        variant = stage["variant"]
        donor_path = root / stage["donor_checkpoint"]
        donor = torch.load(donor_path, map_location="cpu", weights_only=True)["state_dict"]
        initial = torch.load(root / stage["initial_checkpoint"], map_location="cpu", weights_only=True)["state_dict"]
        payload = torch.load(root / stage["checkpoint"], map_location="cpu", weights_only=True)
        final = payload["state_dict"]
        check(f"{variant}: all checkpoint keys equal", donor.keys() == initial.keys() == final.keys() == common.keys())
        check(f"{variant}: donor file hash", hashlib.sha256(donor_path.read_bytes()).hexdigest() == stage["donor_checkpoint_sha256"])
        check(f"{variant}: final tensor digest", fingerprint(final) == stage["raw_state_sha256"])
        check(f"{variant}: embedded metadata", payload["metadata"] == stage)
        changed = {key: 0 for key in ("A", "B", "adapters", "static_gates")}
        for name in initial:
            group = partition(name)
            check(f"{variant}: finite final {name}", torch.isfinite(final[name]).all())
            if group == "A":
                check(f"{variant}: donor A copied {name}", torch.equal(initial[name], donor[name]))
                check(f"{variant}: A frozen {name}", torch.equal(final[name], initial[name]))
            else:
                check(f"{variant}: fresh common interface {name}", torch.equal(initial[name], common[name]))
            changed[group] += int(not torch.equal(initial[name], final[name]))
        changes[variant] = changed
        initial_a = {name: value for name, value in initial.items() if partition(name) == "A"}
        final_a = {name: value for name, value in final.items() if partition(name) == "A"}
        donor_a = {name: value for name, value in donor.items() if partition(name) == "A"}
        interface = {name: value for name, value in initial.items() if partition(name) != "A"}
        check(f"{variant}: A before/after/donor digests", fingerprint(initial_a) == stage["A_sha256_before"] == fingerprint(final_a) == stage["A_sha256_after"] == fingerprint(donor_a) == stage["donor_A_sha256"])
        check(f"{variant}: initial interface digest", fingerprint(interface) == stage["initial_non_A_sha256"] == summary["common_fresh_non_A_sha256"])
        check(f"{variant}: only B and factors learned", changed["A"] == changed["static_gates"] == 0 and changed["B"] > 0 and changed["adapters"] > 0)
        for step, path in stage.get("checkpoint_paths", {}).items():
            if int(step) in (0, stage["steps_executed"]):
                continue
            intermediate = torch.load(root / path, map_location="cpu", weights_only=True)
            state, metadata = intermediate["state_dict"], intermediate["metadata"]
            check(f"{variant}/{step}: checkpoint step", metadata["step"] == int(step))
            check(f"{variant}/{step}: tensor digest", fingerprint(state) == metadata["raw_state_sha256"])
            check(f"{variant}/{step}: complete keys", state.keys() == initial.keys())
            for name in state:
                check(f"{variant}/{step}: finite {name}", torch.isfinite(state[name]).all())
                if partition(name) in ("A", "static_gates"):
                    check(f"{variant}/{step}: frozen {name}", torch.equal(state[name], initial[name]))
        constants.append({name: donor[name] for name in ("inner.H_init", "inner.L_init")})
    check("three complete frozen donors", len(summary["stages"]) == 3 and all(stage["complete"] for stage in summary["stages"]))
    check("native initial constants common across donors", bool(constants) and len({fingerprint(state) for state in constants}) == 1)
    return {"scope": "Independent checkpoint tensor comparisons: complete native A copied from donors and frozen, common fresh B/U/V without donor interface reuse, actual tensor hashes and expected learning. Nonpersistent runtime buffers and predictions are not reconstructed from checkpoints.",
            "passed": all(item["passed"] for item in checks), "checks_passed": sum(item["passed"] for item in checks),
            "checks_total": len(checks), "changed_tensor_counts": changes, "checks": checks}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="results/local_sudoku")
    parser.add_argument("--kind", choices=("main", "probe"), default="main")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    run_dir = root / args.run_dir
    result = (verify_probe if args.kind == "probe" else verify_main)(root, run_dir)
    (run_dir / "tensor_verification.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "checks"}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
