# Fresh weights at every reasoning step

The controlled local test is now complete: per-step structured B+C scored
**43.080%**, versus **42.797%** for the same B+C trained with the old timing in a
matched exploratory follow-up. The +0.283 percentage-point difference had a
conditional paired-puzzle 95% interval of [+0.019, +0.563]. C still did not
establish an advantage over parameter-matched larger B (43.061%) or B only
(43.132%). [Full report](results/per_step_comparison_20261102_v4/report.md).

The November 2 correction makes `weight_update_cadence=per_reasoning_step` the
default for dense A/B and stacked A/B/C models. Before each recurrent call:

1. B attends to A's current low/high states, available predictions, constraint
   residuals, the previous observation/update statistics, and its own memory.
2. C attends to B's current internal tokens and its own recent history, then
   generates a temporary dense change to B's generation-head weight.
3. B uses the changed head to generate complete dense changes to A's two MLP
   down-projections. A performs that reasoning call with those effective weights.
4. The resulting A states and B/C memories feed the next reasoning call.

Both latent updates and high/answer-state updates receive fresh weights. The
high-state call reads the actual current low state, rather than treating its
first argument (the high state) as the low state. No labels enter the read path.
The numerical update bounds and dense generation without low-rank factors are
unchanged. Temporary matrices replace the previous intervention relative to the
stored base; controller memory carries history across calls. Stored base tensors
are not overwritten during inference.

For native H3/L6, there are `3 * (6 + 1) = 21` reasoning calls per ACT update.
A complete 16-ACT puzzle therefore receives **336 C writes to B and 336 B
generations for A**, with two A matrices generated each time. The generated A
entry count is 1,572,864 per generation and 33,030,144 per ACT, per example.
These are sequentially generated entries, not all simultaneously stored or
additional learned parameters. The controller parameter counts are unchanged.

Native truncated backpropagation is retained: the first 14 calls of each ACT run
without gradient recording, and the final seven have distinct differentiable
weight generations. B/C memory resets only for new puzzle slots and detaches at
ACT boundaries. This correction does not introduce gradient credit through all
336 steps or jointly train A in the engineering smoke.

## Historical results

The October 30 controlled study generated a single C/B action per ACT and held
it over the final seven calls, leaving 14 warm-up calls unchanged. Its null
result applies to that cadence. The old configuration files now explicitly set
`weight_update_cadence=per_act` to reproduce that protocol; result artifacts and
their source snapshots are preserved.

Changing the cadence and extending training are separate changes. A new
matched comparison is reported below; it does not establish convergence or
joint-A adaptability.

## Controlled local study, November 2

All four per-step variants used three seeds, 304 updates, batch 16, and the same
304 augmented training puzzles per seed from a 3,584-puzzle pool. Development
measurements at updates 0, 128, 256 and 304 used 32 separate puzzles. The 128
official test puzzles were beyond the earlier cache's test prefix; exact puzzle
and solution hashes were checked against all earlier cached splits. A was fixed
at the same native-training donor in every run.

The primary structured-C minus larger-B effect was +0.019 percentage points,
conditional 95% interval [-0.071, +0.118]. Shuffling C's observations changed its
mean accuracy by -0.033 points, interval spanning zero. The run therefore does
not demonstrate a useful C advantage or reliable C content specialization.

The additional timing control kept all initial actor tensors, augmented example
streams, optimizer settings/updates and native A calls identical. Only the
cadence differed. Every seed favored the per-step model, but this follow-up was
specified after primary evaluation began and is exploratory. Per-step control
also uses more controller computation; total FLOPs are not matched.

The main comparison completed in 30.49 minutes and the cadence follow-up in
2.95 minutes. All 15 training runs completed. The suite passed 136 tests; saved
artifact audits passed 1,192 and 359 checks respectively, followed by cross-run
initialization/data/budget identity checks. An earlier batch-32 attempt stopped
at its first development checkpoint due to host memory pressure, before any
trained-arm test scores. Its files remain preserved. Batch 16, completed-episode
cleanup and larger decoder chunks allowed the successful run.

No variant solved a complete puzzle. These are small cell-accuracy gains on one
frozen A donor, not evidence that jointly trained A is more adaptable. See
[the main report and learning curves](results/per_step_comparison_20261102_v4/report.md)
and [the matched cadence report](results/per_step_comparison_20261102_v4/cadence_comparison.md).

The new configs are `local_lab/config_per_step_test.json` and
`local_lab/config_cadence_control.json`. Prepare their larger data cache with:

```powershell
python -m local_lab.data --root data/local-per-step-sudoku-20261102 --train-size 65536 --val-size 4096 --test-size 1280
```

Copy the configs with fresh result/profile paths. To reproduce the executed
budget, set both `requested_steps` and `minimum_steps` to 304. Point the timing
control's `paired_run` at the new main run. The configs reference the original
local native donor; other checkouts must regenerate it or provide an equivalent
saved native donor. Then run each config through `local_lab.attention_test`,
audit each with `scripts/verify_attention_test.py`, generate the main report with
`local_lab.attention_report`, and compare them with `local_lab.cadence_report`.

## Local verification

The full local suite passed **125 tests**. New checks cover every low/high call,
all 336 calls of a complete puzzle and the next puzzle reset, per-write gradient
flow in the final seven calls, previous-step feature timing, labels and batch
isolation, and cleanup after a failure in the middle of an ACT update.

The full-width real-data smoke is configured by
`local_lab/config_per_step_smoke.json`. It has a 300-second cap, two examples
per batch, four B/C optimization updates, a separate two-puzzle validation panel,
and no oracle or official test data. Its trace measures the matrices actually
installed before every call of one complete validation batch, including changes
between consecutive calls and relative norm bounds. This is engineering
verification, not a performance comparison.

```powershell
python -m pytest -q
python -m local_lab.dense_smoke --config local_lab/config_per_step_smoke.json
python scripts/verify_stacked_smoke.py --run results/per_step_weights_smoke_20261102
python scripts/verify_step_execution.py --run results/per_step_weights_smoke_20261102
```

Use a fresh results directory for a rerun. The original local donor is
`results/dense_smoke_20261030/dense_smoke.pt`; model tensors are excluded from
Git, so another checkout must first regenerate that dense A/B checkpoint, then
update the source path in a copy of the smoke configuration.

## Executed November 2 checks

- **125 unit tests passed** in 10.50 seconds, including 17 new timing and
  causality cases.
- The full-width smoke completed four B/C optimization updates in **64.67
  seconds** against its 300-second cap. Both readers, both generators and all
  four C cross-attention heads had finite nonzero gradients on each update.
- `step_trace.json` records all **336** C/B generations over one complete
  validation batch, with a new matrix before every low/high reasoning call.
- A separate GPU replay of the final checkpoint inspected the actual projection
  updates **after dtype conversion**. All three inspected ports changed between
  every consecutive call: A's two projections used BF16 updates and B's generation
  head used FP32. The replay passed in **20.53 seconds**. The trace and its
  source/checkpoint hashes are saved in `applied_step_trace.json` and
  `step_execution_verification.json`.
- The saved-artifact audit passed **134/134 checks**, including all 15 A tensors
  unchanged, actual B/C parameter changes, update schedules, source hashes,
  norm/rank bounds and prediction-derived metrics. Disabling both controllers
  reproduced native A logits bit for bit; disabling C reproduced standalone B.

All tiny-panel blank accuracies were 47.273%, including untouched A. Two puzzles
and four optimization updates supply no performance evidence. Earlier controlled
results still apply only to the original per-ACT cadence.

The complete smoke report is
[results/per_step_weights_smoke_20261102/report.md](results/per_step_weights_smoke_20261102/report.md).
