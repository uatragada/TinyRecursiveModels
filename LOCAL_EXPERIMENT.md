# Local TRM and plasticity-controller experiment

Completed locally: native TRM with our 809,736-parameter B, matched native/static/
dynamic source arms, and the separate isolated A-weight probe. Source test-panel
unknown-cell accuracy was 16.69% native, 18.46% static joint, and 18.47% dynamic
joint. No arm solved a complete test puzzle. In the fresh B+U/V probe, static-joint
A improved from 18.35% to 18.85%, while dynamic-joint A moved from 18.32% to
18.18%. This bounded run demonstrated setup integrity and early learning, with
no consistent dynamic-joint adaptability advantage. See the
[source report](results/local_sudoku/report.md) and
[separate probe report](results/local_sudoku_probe/probe_report.md).

Final local validation recorded 49 tests passed. Main protocol checks passed
305/305 and independent actual-tensor checks 210/210; probe protocol checks
passed 139/139 and actual-tensor checks 448/448. These scopes remain distinct in
the reports and [local validation evidence](results/local_checks.json).

This workspace preserves the full native architecture from
[Samsung SAIL Montreal's Tiny Recursive Models repository](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)
and adds our own controller B in
[trm_plastic.py](models/recursive_reasoning/trm_plastic.py), with local training,
data preparation, and reporting entry points in `local_lab/`. The primary reference is
[Less is More: Recursive Reasoning with Tiny Networks](https://arxiv.org/html/2510.04871v1).
The local experiment is a bounded from-scratch learning diagnostic on native
9×9 Sudoku-Extreme data. It is not a reproduction of the paper's final accuracy
and does not by itself establish that joint training makes A easier to adapt.

Native A uses hidden width 512, two shared reasoning layers, eight attention
heads, and SwiGLU feed-forward blocks. Each ACT inner update retains three
outer cycles, each containing six latent updates and one answer update. This
executes the shared reasoning module 21 times. The first 14 calls run without
gradient tracking, and the final seven remain differentiable. The answer and
latent carries detach between ACT updates. Evaluation retains the native maximum
of 16 ACT updates. These recurrence and gradient rules remain the native rules;
the controller does not replace A or shorten the reasoning network.

Our B has width 128, three causal Transformer layers, eight heads, feed-forward
width 512, and at most four recent observations. It predicts eight bounded
low-rank gates for rank-four temporary weight changes on the two native MLP
`down_proj` layers, with adapter scale 0.2. B observes latent information and
its own state; it receives no solution labels or correctness feedback at
inference. Its within-update history resets for each ACT inner update.
Interventions in the initial no-gradient cycles do not receive direct gradient
credit through those cycles. Training retains the native loss and optimizer
structure: Stablemax task loss, halting binary cross-entropy, AdamATan2, the
sparse puzzle-embedding optimizer, and EMA. Native A has 6,828,034 dense parameters
plus its separately optimized 512-element puzzle-embedding table; our B has
809,736 parameters. The native batch-local puzzle-embedding gradient buffer does
not add logical puzzle-identity degrees of freedom. On Windows the dense optimizer uses a portable PyTorch
implementation of the upstream AdamATan2 reference formula because the fused
CUDA extension did not build; this is recorded explicitly in run metadata.

The matched arms start from the same full initial state and receive the same
ordered candidate batches. Native ACT admits candidates only into halted slots,
so actual admitted training streams can diverge and are separately hashed:

- **Native:** A learns without a plasticity controller.
- **Static joint:** A and low-rank factors learn with global gates.
- **Dynamic joint:** A, low-rank factors, and our B learn together.

The common saved parameter state does not make the initial functions identical:
native starts with gates off, static uses `tanh(0.01)`, and B predicts near-zero
dynamic gates. Completed arms match optimizer updates and native A reasoning
passes. They do not match total compute or wall time, because B adds its own
computation and approximately 810,000 parameters.

The dynamic checkpoint also receives gates-off and fixed-gate evaluations. These
are inference interventions on that checkpoint, not independently trained arms.
Static gates can act throughout the recursion, while dynamic actions follow
their recorded observation/action schedule; the saved source is authoritative
for that timing.

The bounded run uses one optimization seed and the pinned Sudoku-Extreme stream
prefix selected for this workspace: 8,192 training puzzles, 512 validation
puzzles, and 512 test puzzles. Source identity, revision, file hashes, prefix
selection, and actual processed counts are recorded in the data evidence.
Training-only on-the-fly Sudoku symmetries augment existing puzzles; augmented
presentations must not be counted as additional independent puzzles. Solution
digits are tokenized as 2–10 and blanks as token 1. Sequential source prefixes
can be biased in source and difficulty. Exact puzzle and solution hashes are
separated across splits; local validation does not independently prove solution
uniqueness or mathematical inequivalence of all puzzles.

The main configuration requests 200 optimizer steps per arm at batch size 128.
Raw fixed-final weights are the primary comparison; EMA at rate 0.999 is evaluated
separately as secondary evidence and may lag in such a short run. Each endpoint
uses the same first 128 puzzles from its 512-puzzle validation or test pool;
these prefix panels are not full-pool evaluations. No held-out validation curve
or validation-based checkpoint selection occurs during training. The report
keeps raw and EMA rows separate.

Report blank-cell accuracy, exact whole-puzzle accuracy, given-clue agreement,
and valid completed grids. All-cell accuracy includes easy copied givens and is
shown alongside the clue-copy baseline. A model can achieve substantial
all-cell accuracy while failing to solve unknown cells. A valid grid must be a
complete legal Sudoku consistent with every given clue; an exact solution must
also match the saved target.

Training coverage has several distinct units. Unique newly admitted source
puzzles count data coverage; batch slots and repeated ACT updates count compute
and optimization exposure. An ACT carry can continue processing an earlier
puzzle while a batch loader advances. The report uses the runner's actual carry
and admission counters, rather than equating loader batches with new puzzles.
Native ACT training can halt early; evaluation executes the fixed maximum.

The local time bound is 900 seconds, with a 650-second training cutoff and the
remaining budget reserved for endpoint evaluations.
Full native training in the upstream README takes about 18 hours for its
smallest Sudoku experiment on a 48 GB L40S. A short run on the local 16 GB GPU
therefore measures setup integrity, resource use, and early learnability. It
cannot support claims of paper-level reproduction, general adaptability, broad
retention, or a capacity-scaled isolated A-donor result. The completed source
run is documented in [its report](results/local_sudoku/report.md); the separate
probe below keeps its adaptation conclusions distinct from source learning.

The isolated A-weight probe copies all persistent native A tensors from each
raw source donor, including its learned puzzle-embedding table and H/L initial
constants. It excludes the source adapters and controller. A stays frozen while
the common fresh seed-29 B **and fresh U/V factors** train for 60 updates at
batch size 64. This is not a B-only probe. The static gates and all native
buffers stay fixed, and no sparse-embedding optimizer or EMA is used. All donor
arms begin with the same complete non-A state.

Probe training uses the remaining 384 validation-pool puzzles after the source
run's first 128 evaluation puzzles. Its test panel is the same first 128 test
puzzles, evaluated at fresh step 0 and final step 60 after all arms receive
their training budget. This is readaptation on new Sudoku instances within the
same task family. It is not an unfamiliar task or a held-out pointer-composition
shift. Initial capabilities and learning headroom remain visible; two test
endpoints cannot establish a detailed learning-speed curve. The additional probe
has a 300-second bound and a 180-second training cutoff.

Strong adaptability claims still require successful source-task learning,
intermediate probe checkpoints, controlled new-task shifts, and independent
A-training seeds. The local probe is a preliminary conditional test of the
selected donors and the common fresh B/U/V interface.

The local runtime uses Python 3.12.2 with Torch 2.10.0+cu128, NumPy 1.26.4,
einops 0.8.2, and Pydantic 2.12.5. Reporting uses Matplotlib 3.8.4; local tests
use pytest 8.4.2. These are the installed versions used here, rather than the
upstream paper's pinned Linux/CUDA environment. Native attention uses PyTorch
SDPA; local settings select its deterministic math implementation, disable TF32
and compilation, and retain bfloat16 forward calculations with float32 learned
weights. The local runner does not require the unavailable fused AdamATan2
extension, Triton, a FlashAttention extension, or a Weights & Biases login.
The exact observed local distribution list is saved in
[requirements-local.txt](local_lab/requirements-local.txt); this file records
the working environment rather than implying a separate clean installation was
performed.

Run these entry points serially from
`G:\Projects\TransformerLab\TinyRecursiveModels` using the existing interpreter:

```powershell
$trmPython = 'G:\Projects\TransformerLab\plastic-latent-lab\.venv\Scripts\python.exe'
& $trmPython -c "from local_lab.data import prepare_data; prepare_data('data/local-bounded-sudoku', 8192, 512, 512)"
& $trmPython -m pytest -q tests
& $trmPython -m local_lab.smoke
& $trmPython -m local_lab.train --config local_lab/config.json
& $trmPython scripts/verify_local_evidence.py --run-dir results/local_sudoku
& $trmPython scripts/verify_local_tensors.py --run-dir results/local_sudoku --kind main
& $trmPython -m local_lab.report --run-dir results/local_sudoku
```

After the completed source run, the separate frozen-A probe and report use:

```powershell
& $trmPython -m local_lab.probe
& $trmPython scripts/verify_probe_evidence.py --run-dir results/local_sudoku_probe
& $trmPython scripts/verify_local_tensors.py --run-dir results/local_sudoku_probe --kind probe
& $trmPython -m local_lab.report --probe-run-dir results/local_sudoku_probe
```

Data preparation fetches bounded sequential CSV prefixes at pinned revision
`58942f96baeb572ca3127e2a9e9c70f330783d6b` of
`sapientinc/sudoku-extreme`, then reuses the verified local cache. It does not
fetch pretrained model weights. The smoke/main entry points refuse to overwrite
existing run and checkpoint files. To repeat either run, copy its configuration
and set a fresh `name` and `results_dir`; pass that file to `local_lab.train`.
`summary.json` records the upstream Git commit, and `source_snapshot/` plus
`source_manifest.json` preserve the executed local code. Use those sources and
the recorded configuration to reproduce this run rather than assuming a future
checkout of upstream `main` is identical.

`local_lab/report.py` reads saved measurements only. It imports no Torch,
does not load checkpoints, and performs no training. The evidence verifier also
avoids Torch: it checks source/data hashes, saved protocol/accounting identities,
and checkpoint ZIP integrity. It does not reconstruct checkpoint tensors or
independently replay every ACT admission. Native-off tensor parity, gradients,
and the portable optimizer formula are covered by the separate local tests.
`verify_local_tensors.py` is a separate CPU-Torch verification step that loads
saved tensors to check actual donor copies, shared fresh interfaces, frozen
weights, finite values, hashes, and trained-component changes. It does not replay
training or independently recompute predictions. The final default pytest run
uses `pytest.ini` to discover our `tests/` suite, including portable optimizer
parity, rather than the vendored package's fused-CUDA extension tests.

The report preserves completion status and labels interrupted or unrun cells as
unavailable evidence rather than zero measurements. Local report manifests hash
the evidence and report source. No model weights, media, or paid services are
downloaded by the reporter, and it does not publish anything externally.
