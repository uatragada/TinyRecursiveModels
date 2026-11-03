# Direct dense weight generation: local smoke

B now directly generates complete temporary matrices for both native TRM MLP
down-projections. Each is 512 by 1536 in this checkpoint. A shared nonlinear
decoder emits 32 by 32 tiles and assembles every entry. There are no U/V factors,
learned low-rank directions, or rank-four gates in this model. The construction is
a [hypernetwork](https://arxiv.org/abs/1609.09106), with an explicit per-matrix
Frobenius limit of 3% of its base weight norm.
That numerical limit does not guarantee safe outputs or interpretable changes.

The original attachment specified low-rank changes. The user's subsequent
instruction rejects that architecture and supersedes the earlier requirement.
Historical experiments and their checkpoints remain preserved as baselines.

A comes from the native-training arm, with all historical unused controller and
factor tensors discarded. It is frozen throughout this smoke. B reads per-cell
latent states, prediction changes and Sudoku constraint residuals; a Transformer
and per-puzzle GRU produce its context. Memory resets on new puzzles. One set of
dense matrices is used in the final seven native calls of each 21-call ACT update.
Observations and memory credit are detached at the ACT boundary. Joint A/B
gradients are unit-tested; joint training has not been run for this architecture.

## Executed checks

- B has 348,128 parameters and generates 1,572,864 temporary weight entries per example per ACT.
- First sampled generated matrices have measured ranks [512, 512], each out of 512 possible.
- Completed 8 B-only updates on 4 admitted puzzles, with finite nonzero reader and generator gradients.
- Full frozen A and its sparse scratch weights remained identical; disabling B reproduced native logits bit for bit before and after training.
- No oracle or official test data was used. This smoke used 8 separate validation puzzles.
- Runtime: 55.30 seconds, cap 180 seconds. Peak Torch CUDA allocation: 497.6875 MiB.

The initial/final blank-cell accuracies were 42.478%
and 42.478%; untouched A was 42.478%.
These tiny-panel smoke measurements are integration evidence only. No matched
training controls, new isolated-A comparison, interpretability evaluation or
safety evaluation has been completed for the dense architecture.

## Reproduce

From this checkout, using its existing shared local Python environment:

```powershell
& '..\plastic-latent-lab\.venv\Scripts\python.exe' -m local_lab.dense_smoke --config local_lab/config_dense_smoke.json
```

Choose a fresh results directory to rerun; existing artifacts are never overwritten.
