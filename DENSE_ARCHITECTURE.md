# Current direction: directly generated dense weights

The user's instruction on October 30, 2026 rejects a LoRA-style architecture.
It supersedes the original attachment's low-rank-update requirement. Earlier
experiments remain preserved; their results describe that older mechanism.

The optional third controller C is now implemented: it modifies B's dense
weight-generation head before B generates A's changes. See
[STACKED_ARCHITECTURE.md](STACKED_ARCHITECTURE.md) for its architecture and the
separate bounded smoke evidence.

The new entrypoint is `local_lab.dense_smoke` and its model is
`models/recursive_reasoning/trm_dense.py`. It constructs native TRM A directly,
plus our Transformer B and a shared nonlinear weight generator. It does not
construct the earlier plastic model or load its controller/factors.

B reads A's per-cell latent trajectory, prediction changes and Sudoku constraint
residuals. Its Transformer and per-puzzle GRU produce a context. A nonlinear
decoder combines that context with learned tile identifiers and **directly emits
all entries** of two 512 by 1536 temporary matrices. Output chunks are 32 by 32
tiles; the assembled matrices are dense, with no prescribed low rank and no
factorized update. The generator's own linear layers are ordinary neural-network
layers, not factors of A's effective update.

For a selected projection, the forward calculation is:

`output = linear(input, W_A) + batch_matmul(input, delta_W_B.transpose(-1, -2))`

There is one complete `delta_W_B` per example and projection, freshly generated
before **every reasoning call**. Native H3/L6 has 21 calls per ACT update and 16
ACT updates per full puzzle, so B generates weights 336 times per puzzle. Each
call uses the latest low/high states and carried controller memory. When C is
enabled, C first changes B's generator at that same cadence. Updates replace the
previous temporary matrix; they are relative to the stored base, with history
carried by B/C state rather than an accumulating matrix sum.

The native first 14 calls still run without gradient recording, but their
effective weights also change. Each of the final seven calls has its own
differentiable B/C generation. Memory credit is truncated between ACT updates,
and each puzzle slot resets independently. See [the cadence correction and
local checks](PER_STEP_WEIGHTS.md).

`weight_update_cadence=per_reasoning_step` is the model default. The original
October 30 configurations explicitly select `per_act` to reproduce their one
late action held over seven calls. Those results do not evaluate the new cadence.

The generator bounds each entry so that each update's Frobenius norm is at most
3% of its base weight's Frobenius norm, up to floating-point rounding. This is a
numerical control, not a safety or interpretability guarantee. These are
[hypernetwork-generated weights](https://arxiv.org/abs/1609.09106).

This trades more generated-weight memory and batched multiplication for a much
wider intervention space. The generator remains shared and compact; generated
weight count is different from the number of learned B parameters. The first
local run is deliberately only a short engineering smoke, not an accuracy study.

The smoke imports A from the **native-training arm**, whose old wrapper happened
to allocate inactive low-rank/controller tensors. The conversion discards all of
those extras and strictly validates every imported A tensor. Future training can
start from a clean native checkpoint without this compatibility conversion.

```powershell
& '..\plastic-latent-lab\.venv\Scripts\python.exe' -m pytest tests/test_trm_dense.py -q
& '..\plastic-latent-lab\.venv\Scripts\python.exe' -m local_lab.dense_smoke --config local_lab/config_dense_smoke.json
```

The smoke has an eight-update, four-example-batch, 180-second cap. It checks real
data gradient flow, frozen A and sparse weights, full matrix rank on sampled
updates, and exact native logits with B disabled. It writes a new results folder
and refuses to overwrite an existing run. The model supports joint A/B gradients;
that mode requires the native sparse-embedding optimizer as well as dense
optimizer groups in a future training runner. It is not executed by this smoke.

Before claiming improvement, run matched training controls, controller-disabled
and input-independent controls, and fresh-B isolated-A probes. An update with
full matrix rank is not itself evidence of useful adaptation.

## Verified local smoke, October 30

- Full local suite: **83 passed** in 11.88 seconds.
- B: **348,128 learned parameters**, generating 1,572,864 temporary weight entries
  per example per ACT. Both sampled 512 by 1536 updates had measured rank 512.
- Eight B-only optimization steps completed in a 55.30-second smoke; peak Torch
  CUDA allocation was 497.69 MiB. A stayed frozen and disabling B reproduced
  untouched native logits bit for bit.
- Separate saved-artifact checks passed **23/23**, including equality of all 15
  native A tensors, absence of factors, matrix ranks/norms, checkpoint identity,
  source snapshot hashes and accuracy recomputed from saved predictions.
- All three tiny-panel blank-cell accuracies (native A, initial B, final B) were
  42.478%. This is an integration check, with four training puzzles admitted and
  eight separate validation puzzles. It supplies no evidence of a performance gain.

See `results/dense_smoke_20261030/report.md`, `summary.json`,
`artifact_verification.json`, and `unit_checks.json` for the saved evidence.
