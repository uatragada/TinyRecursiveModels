"""Independent saved-artifact audit for the bounded attention comparison."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from local_lab.data import load_data
from local_lab.train import CandidateStream

KINDS = ("b", "larger_b", "pooled", "cross")
ABLATIONS = ("c_off", "c_write_shuffle", "c_read_shuffle", "c_read_pooled", "c_phase", "c_reset_memory")


def digest(state):
    result = hashlib.sha256()
    for key, value in sorted(state.items()):
        result.update(f"{key}:{value.dtype}:{tuple(value.shape)}".encode())
        result.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return result.hexdigest()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def a_state(state):
    return {key.replace(".mlp.down_proj.base.", ".mlp.down_proj."): value
        for key, value in state.items() if key.startswith("inner.") and not key.endswith((".u", ".v"))}


def base_b(state):
    return {key.replace("generator.decoder.2.base.", "generator.decoder.2."): value
        for key, value in state.items() if key.startswith(("controller.", "generator."))}


def metrics(prediction, inputs, labels, logits):
    correct, blank = prediction == labels, inputs == 1
    grids = prediction.reshape(-1, 9, 9)
    allowed = np.arange(2, 11)
    rows = (np.sort(grids, axis=2) == allowed).all((1, 2))
    columns = (np.sort(grids, axis=1) == allowed[None, :, None]).all((1, 2))
    boxes = grids.reshape(-1, 3, 3, 3, 3).transpose(0, 1, 3, 2, 4).reshape(-1, 9, 9)
    boxes = (np.sort(boxes, axis=2) == allowed).all((1, 2))
    clues = (correct | blank).all(1)
    values = logits.astype(np.float64)
    scores = np.where(values < 0, 1 / (1 - np.minimum(values, 0)), np.maximum(values, 0) + 1)
    ce = np.log(scores.sum(-1)) - np.log(np.take_along_axis(scores, labels[..., None], -1)[..., 0])
    objective = ((ce * blank).sum(1) / blank.sum(1) + .25 * (ce * ~blank).sum(1) / (~blank).sum(1)).mean()
    return {"blank_accuracy": float(correct[blank].mean()), "all_accuracy": float(correct.mean()),
        "given_agreement": float(correct[~blank].mean()), "exact_accuracy": float(correct.all(1).mean()),
        "valid_grid_fraction": float((rows & columns & boxes & clues).mean()),
        "task_loss": float(objective)}, (correct & blank).sum(1), blank.sum(1)


def verify(directory):
    torch.set_num_threads(4)
    cfg = json.loads((directory / "config.json").read_text())
    result = json.loads((directory / "summary.json").read_text())
    checks = {}
    def check(name, condition):
        checks[name] = bool(condition)
    check("completed", result["status"] == "complete")
    check("within_explicit_budget", result["elapsed_seconds"] <= cfg["max_seconds"] <= 1800)
    check("locked_steps", result["steps_per_arm"] == cfg["locked_steps"] and cfg["locked_steps"] % 16 == 0)
    check("predeclared_primary", cfg["primary_comparison"] == "cross-minus-larger_b")
    profile = json.loads((directory / "profile.json").read_text())
    unmodified_cfg = {key: value for key, value in cfg.items() if key != "locked_steps"}
    check("profile_matches_config", hashlib.sha256(json.dumps(unmodified_cfg, sort_keys=True).encode()).hexdigest() == profile["config_sha256"])
    check("profile_lock_matches_run", profile["locked_steps"] == cfg["locked_steps"])
    source_path = ROOT / cfg["source_checkpoint"]
    source = torch.load(source_path, map_location="cpu", weights_only=True)
    donor = a_state(source["state_dict"])
    check("source_checkpoint_identity", sha(source_path) == result["source_checkpoint_sha256"])
    frozen = torch.load(directory / "frozen_a.pt", map_location="cpu", weights_only=True)
    check("saved_a_is_native_donor", set(frozen["a_state"]) == set(donor)
        and all(torch.equal(value, donor[key]) for key, value in frozen["a_state"].items()))
    check("a_digest", digest(donor) == result["frozen_a_sha256"])
    dataset = load_data(ROOT / cfg["data_dir"])
    pools = {}
    audit = json.loads((directory / "data_audit.json").read_text())
    check("dataset_manifest", sha(ROOT / cfg["data_dir"] / "manifest.json") == audit["dataset_manifest_sha256"])
    for name, source_name in (("train", "val"), ("dev", "val"), ("test", "test")):
        offset, size = cfg[name + "_offset"], cfg[name + "_size"]
        pools[name] = {key: value[offset:offset + size] for key, value in dataset[source_name].items()}
        for key in ("inputs", "labels"):
            check(f"panel:{name}:{key}", hashlib.sha256(pools[name][key].tobytes()).hexdigest() == audit["panels"][name][key + "_sha256"])
        check(f"panel:{name}:ids", pools[name]["base_ids"].tolist() == audit["panels"][name]["base_ids"])
    for left, right in (("train", "dev"), ("train", "test"), ("dev", "test")):
        check(f"disjoint:{left}:{right}", not set(pools[left]["base_ids"]) & set(pools[right]["base_ids"]))
    expected_arms = {(seed, kind) for seed in cfg["seeds"] for kind in KINDS}
    check("all_training_arms", {(row["seed"], row["kind"]) for row in result["training"]} == expected_arms and len(result["training"]) == len(expected_arms))
    c_initials = {}
    for seed in cfg["seeds"]:
        common = torch.load(directory / f"common_seed{seed}.pt", map_location="cpu", weights_only=True)["state_dict"]
        check(f"common:{seed}:a", all(torch.equal(value, donor[key]) for key, value in a_state(common).items()))
        stream = CandidateStream(pools["train"], cfg["batch_size"], seed)
        stream_hash, admitted = hashlib.sha256(), []
        for _ in range(cfg["locked_steps"] // 16):
            batch = stream.next()
            for key in ("inputs", "labels", "base_ids", "sample_ids"):
                stream_hash.update(batch[key].tobytes())
            admitted.extend(batch["base_ids"].tolist())
        matched_counts = []
        for kind in KINDS:
            prefix = f"{seed}:{kind}"
            row = next(row for row in result["training"] if row["seed"] == seed and row["kind"] == kind)
            saved = torch.load(directory / f"{kind}_seed{seed}.pt", map_location="cpu", weights_only=True)
            state, initial = saved["state_dict"], saved["initial_actor_state"]
            actors = {key: value for key, value in state.items() if not key.startswith("inner.")}
            check(prefix + ":steps", saved["steps"] == row["steps"] == cfg["locked_steps"])
            check(prefix + ":actor_hashes", digest(initial) == row["initial_actor_sha256"] and digest(actors) == row["final_actor_sha256"])
            check(prefix + ":initial_b_common", set(base_b(initial)) == set(base_b(common)) and all(torch.equal(v, base_b(common)[k]) for k, v in base_b(initial).items()))
            actual_a = a_state(state)
            check(prefix + ":all_a_tensors_unchanged", set(actual_a) == set(donor) and all(torch.equal(v, donor[k]) for k, v in actual_a.items()))
            check(prefix + ":finite", all(torch.isfinite(v).all().item() for v in state.values()))
            check(prefix + ":no_factors_or_runtime_weights", all(not key.endswith((".u", ".v", ".delta")) for key in state))
            check(prefix + ":parameters_updated", digest(initial) != digest(actors))
            check(prefix + ":parameter_count", sum(v.numel() for v in actors.values()) == row["parameter_count"])
            check(prefix + ":data_stream", row["stream_sha256"] == stream_hash.hexdigest() and row["admitted_base_ids"] == admitted)
            check(prefix + ":exposure_counts", row["augmented_admissions"] == len(admitted) and row["act_slots"] == cfg["locked_steps"] * cfg["batch_size"])
            check(prefix + ":native_calls", row["a_calls_per_act"] == 21)
            check(prefix + ":gradient_records", row["history"][-1]["step"] == cfg["locked_steps"] and all(np.isfinite(v["task_loss"]) and v["gradient_norm"] > 0 for v in row["history"]))
            if kind != "b":
                matched_counts.append(row["parameter_count"])
            if kind in ("pooled", "cross"):
                c_initials[(seed, kind)] = digest({key: value for key, value in initial.items() if key.startswith("meta_")})
                check(prefix + ":cross_head_gradients", all(len(v["cross_head_gradient_norms"]) == 4 and all(np.isfinite(n) and n > 0 for n in v["cross_head_gradient_norms"]) for v in row["history"]))
            if kind == "larger_b":
                check(prefix + ":extra_capacity_updated", all(not torch.equal(value, initial[key]) for key, value in actors.items() if key.startswith("extra_")))
            del saved, state, actors, initial
        check(f"{seed}:active_parameter_match", len(set(matched_counts)) == 1)
        check(f"{seed}:identical_c_initialization", c_initials[(seed, "pooled")] == c_initials[(seed, "cross")])
    expected_evals = {("a", 0, "a_only", split) for split in ("dev", "test")}
    for seed, kind in expected_arms:
        expected_evals |= {(kind, seed, "initial", "dev"), (kind, seed, "normal", "dev"), (kind, seed, "normal", "test")}
        if kind == "cross":
            expected_evals |= {(kind, seed, variant, "test") for variant in ABLATIONS}
    check("evaluation_coverage", {(r["kind"], r["seed"], r["variant"], r["split"]) for r in result["evaluations"]} == expected_evals and len(result["evaluations"]) == len(expected_evals))
    arrays = {}
    for index, row in enumerate(result["evaluations"]):
        prefix = f"evaluation:{index}"
        saved = torch.load(directory / row["artifact"], map_location="cpu", weights_only=True)
        pool = pools[row["split"]]
        for key in ("inputs", "labels", "base_ids"):
            check(prefix + ":" + key, np.array_equal(saved[key].numpy(), pool[key]))
        logits, prediction = saved["logits"].numpy(), saved["predictions"].numpy()
        check(prefix + ":argmax_and_finite", np.isfinite(logits).all() and np.array_equal(logits.argmax(-1), prediction))
        calculated, correct, blanks = metrics(prediction, pool["inputs"], pool["labels"], logits)
        for name, value in calculated.items():
            check(prefix + ":" + name, abs(value - row[name]) <= (2e-6 if name == "task_loss" else 1e-12))
        check(prefix + ":runtime_flags", row["actor_unchanged"] and row["a_unchanged"] and row["act_steps"] == 16)
        if saved["sample_attention"] is not None:
            attention = saved["sample_attention"]
            check(prefix + ":attention", tuple(attention.shape) == (1, 4, 4, 14) and torch.isfinite(attention).all()
                and torch.allclose(attention.sum(-1), torch.ones_like(attention[..., 0]), atol=1e-6))
        arrays[f"correct_{index}"] = correct
        arrays[f"blanks_{index}"] = blanks
    for name, expected in json.loads((directory / "source_manifest.json").read_text()).items():
        check("snapshot:" + name, sha(directory / "source_snapshot" / name) == expected)
    np.savez_compressed(directory / "per_puzzle_metrics.npz", **arrays)
    receipt = {"status": "passed" if all(checks.values()) else "failed", "passed": sum(checks.values()),
        "total": len(checks), "checks": checks, "validator_sha256": sha(Path(__file__)),
        "scope": "Independent saved tensor identities and finite values, regenerated augmented data streams, parameter matching, source hashes, prediction and Stablemax-loss calculations. Runtime gradient/inference flags checked for consistency; no optimizer or model-inference replay."}
    (directory / "verification.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in receipt.items() if k != "checks"}, indent=2))
    if receipt["status"] != "passed":
        raise AssertionError([key for key, value in checks.items() if not value])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", default="results/cross_attention_test_20261030")
    verify(ROOT / parser.parse_args().run)
