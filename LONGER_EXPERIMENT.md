# Longer local TRM joint-training experiment

This experiment extends the completed [short local study](LOCAL_EXPERIMENT.md)
with ten times as many source-training updates, a larger native Sudoku-Extreme
pool, and two fresh-interface probe seeds with intermediate checkpoints. The
earlier reports and checkpoints remain preserved. The longer source run
completed all 2,000 updates per arm in 2,506.18 seconds. Raw unknown-cell
accuracy on the full saved 1,024-puzzle test pool was 40.65% native, 40.13%
static joint, and 39.19% dynamic joint; all exact-puzzle scores were zero.
The two frozen-A probe runs also completed. Their mean final unknown-cell
accuracy was 41.17% with native-trained A, 40.71% with static-joint A, and
40.03% with dynamic-joint A. Mean gains from each donor's own starting point
were +0.698, +1.201, and +1.018 percentage points respectively. Dynamic-joint A
gained more than native A, while gains relative to static-joint A depended on
the fresh seed. It finished below both controls in both seeds. These results do
not show a consistent adaptability benefit over static joint training.

The recorded source/probe wall times total 3,446.32 seconds, or 57.44 minutes,
within the combined configured bound. Setup, data preparation, preflight tests,
and reporting are outside that total. No training process remains running.
See the [source report](results/longer_sudoku/report.md),
[paired two-seed probe report](results/longer_probe_comparison/aggregate_report.md),
and individual probe reports for [seed 29](results/longer_probe_seed29/probe_report.md)
and [seed 43](results/longer_probe_seed43/probe_report.md).

Native A remains the full width-512, two-layer TRM, with H=3, L=6, and maximum
ACT=16. Each ACT update executes 21 shared reasoning-module calls: the first 14
without gradients and the final seven with gradients. A has 6,828,034 dense
parameters plus its separately optimized 512-element puzzle-embedding table.
Our [Transformer B](models/recursive_reasoning/trm_plastic.py) remains width 128,
three layers, eight heads, FF width 512, history four, and 809,736 parameters.
It controls rank-four temporary updates on the two native MLP down-projections
at scale 0.2. Architecture and adapter capacity are unchanged from the short
study.

## Source training and actual coverage

The matched source arms are native A, static joint A+U/V+global gates, and dynamic
joint A+U/V+B. Each starts from the same full seed-17 state and receives the same
proposed candidate stream. Initial functions still differ because the native
arm has gates off, static uses `tanh(0.01)`, and B predicts near-zero gates.
Source training retains native adaptive ACT halting and exploration. Actual
admitted puzzle streams can therefore diverge between models and are checked
separately from proposed candidates.

| Setting | Short source run | Longer source run |
| --- | ---: | ---: |
| Requested updates per arm | 200 | 2,000 |
| Batch size | 128 | 128 |
| Training pool | 8,192 | 65,536 |
| Validation pool | 512 | 2,048 |
| Test pool | 512 | 1,024 |
| Validation endpoint | First 128 | First 1,024 |
| Test endpoint | First 128 | Full saved 1,024-puzzle pool |
| Maximum wall time | 900 seconds | 3,000 seconds |

The 2,000 × 128 = 256,000 requested ACT-slot presentations per arm are not
256,000 newly seen puzzles. Unhalted slots keep their earlier puzzle while the
candidate stream advances. The report records actual new-puzzle admissions,
unique base puzzles, distinct admitted augmentation views, and pool coverage.
Legal Sudoku symmetries add views of existing puzzles, not independent source
puzzles. Counts must not be summed across matched arms as distinct data.
The completed source arms each admitted 16,000 puzzles and saw 14,574 distinct
base puzzles, covering 22.24% of the configured training pool. Proposed and
actual admitted streams matched across all three arms.

The pinned dataset revision remains
`58942f96baeb572ca3127e2a9e9c70f330783d6b` of
`sapientinc/sudoku-extreme`. The expanded cache is separate from the short-run
cache. All 68,608 selected puzzles have locally validated solutions and clue
agreement, with exact puzzle and solution hashes separated across splits.
Preparation consumed 12,845,056 application bytes of official CSV prefixes,
approximately 12.25 MiB, rather than downloading the complete source files.
The sequential prefixes can be biased in source and difficulty. Local checks do
not independently prove solution uniqueness or mathematical inequivalence of
all puzzles.

Learning rates, betas, weight decay, the 20-update warmup, Stablemax/halting loss,
portable AdamATan2 formula, native sparse optimizer, and EMA rate 0.999 retain
the short-run values. Recoverable source checkpoints are saved every 500
updates. The fixed final checkpoint is used for comparison; intermediate
checkpoints do not select a validation-best model. Raw and EMA results remain
separate, with raw final weights primary. Unknown-cell accuracy, exact puzzle
accuracy, clue agreement, legal clue-consistent grids, and the clue-copy baseline
are reported. All-cell accuracy includes givens and cannot substitute for
unknown-cell or puzzle-solving accuracy.

Longer training, expanded data, and larger evaluation panels all change together.
Consequently, differences from the earlier short report are not a controlled
ablation of update count alone. Equal source updates and A reasoning passes do
not equalize total computation or wall time; B adds its own work.

## Frozen-A probes and paired fresh-interface seeds

Two probe runs use fresh seeds 29 and 43. Each copies only persistent native A
from the three raw source donors, including its puzzle-embedding table and H/L
initial constants. Source B and adapters are excluded. Within each seed, all
donors receive the same fresh B, U/V factors, and static gates. Only B and fresh
U/V factors train; all native A weights and buffers remain frozen. This is an
A-only donor intervention with a trainable B/U/V interface, rather than a B-only
optimization.

Each donor receives 300 updates at batch size 64 on the remaining 1,024
validation-pool puzzles after offset 1,024. These puzzles were excluded from the
source validation endpoint. Every probe checkpoint evaluates the same first
512 saved test puzzles. Test checkpoints are predeclared at steps 0, 100, 200,
and 300; all arms complete their training budget before held-out endpoints are
evaluated. No probe EMA or sparse-embedding update is used.

Unlike the earlier adaptive probe, the longer probe forces 16 ACT updates per
puzzle during training. This disables early halting and exploration while
leaving the native differentiable final cycle enabled. The choice makes actual
admission schedules comparable across A donors within a seed. It does not change
the source run's native adaptive halting. Saved admission hashes and counters
verify the realized exposure rather than assuming that equal candidate batches
are enough.

Interpret final capability, step-0 headroom, within-arm gains, and static-joint
controls together. Normalized gain AUC integrates the measured relative-gain
curve by trapezoids and divides by its observed update span, without
extrapolation. Paired joint-donor contrasts are computed within each seed before
their mean across seeds. Repeated test puzzles are not pooled as independent
observations. The two fresh-interface seeds vary initialization, augmentations,
and optimization order; they still share one source A-training seed and the
same A donor checkpoints.

The completed mean probe measurements on the fixed 512-puzzle test panel were:

| Frozen A donor | Initial unknown-cell accuracy | Final unknown-cell accuracy | Gain (pp) | Normalized gain AUC (pp) | Final Stablemax loss |
| --- | ---: | ---: | ---: | ---: | ---: |
| Native-trained | 40.47% | 41.17% | +0.698 | 0.321 | 0.91533 |
| Static joint | 39.51% | 40.71% | +1.201 | 0.517 | 0.92394 |
| Dynamic joint | 39.01% | 40.03% | +1.018 | 0.436 | 0.93122 |

Dynamic-joint A's extra accuracy gain over static-joint A was −0.524 points in
seed 29 and +0.157 points in seed 43, averaging −0.183 points. Over native A its
mean extra gain was +0.319 points, and its mean gain AUC was +0.115 points higher.
These positive changes do not erase its lower starting and final capability:
its mean final accuracy was 1.141 points below native A and 0.684 points below
static-joint A. All three donors' losses decreased; dynamic-joint A's final loss
remained higher than both controls. Exact-puzzle accuracy stayed zero throughout
the saved source and probe evaluations.

Within each seed, all three probe arms admitted 1,216 augmented puzzles across
19,200 ACT-slot presentations. Seed 29 saw 729 unique adaptation puzzles and
seed 43 saw 737, out of the same 1,024-puzzle pool. Candidate and actual
admission hashes matched across donor arms within each seed. Neither the
admission counts nor distinct views should be summed across arms as independent
data.

This remains readaptation to new Sudoku instances in the same task family. It
does not establish unfamiliar-task transfer, broad plasticity, retained prior
capabilities, or a complete reproduction of the published benchmark. More source
A-training seeds would be needed for a strong general adaptability conclusion.

## Local bounds, evidence, and reproduction

The source has a 3,000-second bound and 2,500-second training cutoff. Each probe
has a 600-second bound and 300-second training cutoff, reserving time for the
four endpoint checkpoints. These bounds total at most 70 minutes of configured
source/probe execution. All configured stages and checkpoints completed. The
source took 2,506.18 seconds, seed 29 took 443.89 seconds, and seed 43 took 496.26
seconds. Training stayed local, and Torch processes ran serially.

Final validation recorded 56 tests passed. Verification scopes remain separate:

| Evidence | Protocol/accounting checks | Actual-tensor checks |
| --- | ---: | ---: |
| Longer source | 339/339 | 1,071/1,071 |
| Probe seed 29 | 233/233 | 940/940 |
| Probe seed 43 | 233/233 | 940/940 |

The independent [cross-seed consistency/arithmetic audit](results/longer_probe_comparison/verification.json)
also passed 40/40 checks. Actual-tensor verification confirmed that donor A
remained unchanged at every saved probe checkpoint, while B and U/V learned from
their common fresh initial state within each seed. The protocol checks and
tensor checks do not independently replay training or recompute predictions.

Use the existing Windows runtime and dependency versions documented in
[LOCAL_EXPERIMENT.md](LOCAL_EXPERIMENT.md) and
[requirements-local.txt](local_lab/requirements-local.txt). From the repository
root, the configured entry points are:

```powershell
$trmPython = 'G:\Projects\TransformerLab\plastic-latent-lab\.venv\Scripts\python.exe'
& $trmPython -c "from local_lab.data import prepare_data; prepare_data('data/local-expanded-sudoku', 65536, 2048, 1024)"
& $trmPython -m local_lab.train --config local_lab/config_longer.json
& $trmPython -m local_lab.probe --config local_lab/config_longer_probe_seed29.json
& $trmPython -m local_lab.probe --config local_lab/config_longer_probe_seed43.json
```

Run directories and checkpoints are protected against overwrites. Repeating a
run requires a copied configuration with a fresh `name` and `results_dir`.
The recorded configuration, Git revision, and executed source snapshots are the
reproduction inputs; a future upstream `main` checkout need not be identical.

Protocol/accounting verifiers and CPU-Torch tensor verifiers have separate
scopes. Run them serially with the saved run paths;
neither replaces independently recomputing predictions or replaying every ACT
admission. The reporter itself uses saved JSON and lazy Matplotlib, imports no
Torch, and never trains or publishes.

```powershell
& $trmPython scripts/verify_local_evidence.py --run-dir results/longer_sudoku
& $trmPython scripts/verify_local_tensors.py --run-dir results/longer_sudoku --kind main
& $trmPython scripts/verify_probe_evidence.py --run-dir results/longer_probe_seed29
& $trmPython scripts/verify_local_tensors.py --run-dir results/longer_probe_seed29 --kind probe
& $trmPython scripts/verify_probe_evidence.py --run-dir results/longer_probe_seed43
& $trmPython scripts/verify_local_tensors.py --run-dir results/longer_probe_seed43 --kind probe
& $trmPython -m local_lab.report --probe-run-dir results/longer_probe_seed29
& $trmPython -m local_lab.report --probe-run-dir results/longer_probe_seed43
& $trmPython -m local_lab.report --aggregate-probe-runs results/longer_probe_seed29 results/longer_probe_seed43 --aggregate-output results/longer_probe_comparison
& $trmPython -m local_lab.report --run-dir results/longer_sudoku --related-probe-reports results/longer_probe_seed29/probe_report.md results/longer_probe_seed43/probe_report.md results/longer_probe_comparison/aggregate_report.md
```

Completed source, per-seed probe, and aggregate reports live in those result
directories. Their manifests hash the owning evidence and reporter code. The
earlier short-run reports remain separate.
