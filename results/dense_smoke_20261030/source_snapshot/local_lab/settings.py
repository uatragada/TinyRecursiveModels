"""Explicit native architecture and local deterministic execution settings."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("DISABLE_COMPILE", "1")
from pathlib import Path
import random
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def model_config(batch_size):
    return dict(batch_size=batch_size, seq_len=81, vocab_size=11,
                num_puzzle_identifiers=1, puzzle_emb_ndim=512, puzzle_emb_len=16,
                H_cycles=3, L_cycles=6, H_layers=0, L_layers=2,
                hidden_size=512, expansion=4., num_heads=8, pos_encodings="rope",
                halt_max_steps=16, halt_exploration_prob=.1, forward_dtype="bfloat16",
                mlp_t=False, no_ACT_continue=True, controller_width=128,
                controller_heads=8, controller_layers=3, controller_ff_multiplier=4,
                controller_history=4, adapter_rank=4, adapter_alpha=.2)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.enable_flash_sdp(False)
    torch.backends.cuda.enable_mem_efficient_sdp(False)
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.matmul.allow_tf32 = False


def write_json(path, value):
    import json
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
