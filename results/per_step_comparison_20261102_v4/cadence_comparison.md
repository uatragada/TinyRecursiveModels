# Matched control for update cadence

This exploratory follow-up trains the same structured B+C with one late weight
generation per ACT, held over its final seven reasoning calls. The main run
generates weights before all 21 reasoning calls. All three seeds start from
identical actor tensors and receive identical augmented puzzles, optimizer
updates, batch size and native A calls. Both keep A fixed. Controller computation
and total FLOPs differ.

| Cadence | Mean held-out blank accuracy |
|---|---:|
| Every reasoning call | 43.080% |
| Once per ACT | 42.797% |

Per-step minus per-ACT: **+0.283 percentage points**, with
conditional paired-puzzle 95% interval **[+0.019, +0.563]**.
Per-seed differences: +0.425, +0.340, +0.085 percentage points.

This follow-up was specified after the primary comparison started. It does not
replace the predeclared structured-C versus larger-B comparison. The interval
resamples this fixed test panel after averaging these three controller seeds;
it does not include uncertainty from retraining A or the full population of
training outcomes. Matching initialization/data/call-budget checks all passed;
both underlying saved-artifact audits passed before this comparison.
