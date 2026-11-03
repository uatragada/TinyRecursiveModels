"""Two-step full-width/native-recursion integration, not a learning result."""
import json
from .settings import ROOT
from .train import run

cfg = json.loads((ROOT / "local_lab/config.json").read_text())
cfg.update(name="local_sudoku_smoke", results_dir="results/local_sudoku_smoke",
           batch_size=8, steps=2, eval_size=8, eval_batch_size=8,
           max_seconds=240, training_cutoff_seconds=120)
run(cfg)
