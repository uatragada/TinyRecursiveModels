"""Small GPU forward/backward measurement of the unchanged native core."""
import gc
import time
import torch
from models.losses import ACTLossHead
from models.recursive_reasoning.trm_plastic import PlasticTinyRecursiveReasoningModel_ACTV1
from .data import load_data
from .settings import ROOT, model_config, seed_all, write_json


def main():
    seed_all(17)
    data = load_data(ROOT / "data/local-bounded-sudoku")["train"]
    rows = []
    for size in (32, 64, 128):
        free, _ = torch.cuda.mem_get_info()
        if rows and rows[-1]["peak_allocated_bytes"] * 2 > free * .8:
            break
        with torch.device("cuda"):
            core = PlasticTinyRecursiveReasoningModel_ACTV1(model_config(size))
        core.configure_training("joint").train()
        model = ACTLossHead(core, "stablemax_cross_entropy")
        batch = {key: torch.as_tensor(data[key][:size], device="cuda", dtype=torch.long)
                 for key in ("inputs", "labels", "puzzle_identifiers")}
        torch.cuda.reset_peak_memory_stats()
        samples = []
        for _ in range(2):
            with torch.device("cuda"):
                carry = model.initial_carry(batch)
            torch.cuda.synchronize()
            started = time.perf_counter()
            _, loss, _, _, _ = model(carry=carry, batch=batch, return_keys=[])
            (loss / size).backward()
            torch.cuda.synchronize()
            samples.append(time.perf_counter() - started)
            model.zero_grad(set_to_none=True)
            core.puzzle_emb.local_weights.grad = None
        row = {"batch_size": size, "forward_backward_seconds": samples,
               "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
               "native_A_parameters": sum(p.numel() for n,p in core.inner.named_parameters() if not n.endswith((".u", ".v"))),
               "B_parameters": core.controller_parameters}
        rows.append(row)
        print(row, flush=True)
        del model, core, carry, loss, batch
        gc.collect()
        torch.cuda.empty_cache()
    write_json(ROOT / "results/native_benchmark.json", {"rows": rows})


if __name__ == "__main__":
    main()
