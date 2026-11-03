# Stacked dense controllers: C modifies B, B modifies A

The implemented chain is:

`A states -> B -> C reads B -> dense delta_W_B -> B uses W_B + delta_W_B -> dense delta_W_A -> A uses W_A + delta_W_A`

C writes the complete 1024 by 32
output weight of B's nonlinear weight generator. It does not replace all of B.
B continues to write both complete 512 by 1536 A projections. There are no
low-rank factors at either level. C cross-attends to B's reader output, each encoder block, memory and readout, using stage/slot identifiers. A causal Transformer then processes a four-step history of those attended observations. Both levels use dense chunked generation.

C's update has a 10% relative Frobenius
limit; B's A updates retain their 3%
limit. These numerical bounds do not establish behavioral safety or interpretability.
C observes detached B states and past C state without labels. C modifies B
before B generates A's next updates, avoiding a same-step circular dependency.
Memory values persist across ACT updates and reset independently for new puzzles;
temporal gradient credit is truncated at ACT boundaries.

Fresh matrices are generated before every one of the 21 reasoning calls per ACT, including the native no-gradient warm-up. A full 16-ACT puzzle therefore uses 336 weight-generation steps. C, when enabled, writes B before each of those B generations.
The first 14 native calls still run without gradient recording; each of the final 7 calls can train its own B/C write.

This run loaded the earlier dense A/B smoke checkpoint, then jointly trained B
and fresh C while keeping A frozen. B has 348,128 parameters;
C adds 29,168. The sampled initial C matrix had measured rank
32 out of 32.

## Executed smoke

- 4 B/C updates, 2 admitted training puzzles, 2 separate validation puzzles.
- Finite nonzero gradients reached both readers and both weight generators on every update.
- All A tensors and sparse scratch weights stayed identical. Disabling both controllers reproduced native logits bit for bit.
- Disabling C reproduced a standalone A/B model with the corresponding current B weights bit for bit, before and after training.
- Temporary matrices at both levels were cleared after use.
- Runtime 64.67 seconds against a 300-second cap; peak Torch CUDA allocation 495.49 MiB.
- No oracle or official test split was used.

| Smoke condition | Blank-cell accuracy | Task loss |
|---|---:|---:|
| off | 47.273% | 1.309241 |
| initial | 47.273% | 1.309024 |
| final | 47.273% | 1.307540 |
| off_after_training | 47.273% | 1.309241 |
| initial_c_off | 47.273% | 1.310339 |
| c_off | 47.273% | 1.309578 |
| c_shuffle | 47.273% | 1.307962 |
| c_phase | 47.273% | 1.308157 |
| c_reset_memory | 47.273% | 1.307869 |
| c_read_pooled | 47.273% | 1.309018 |
| c_read_shuffle | 47.273% | 1.307962 |

The C-off/shuffled/phase/history-reset rows are inference ablations of the same
jointly trained checkpoint. They are not separately trained matched controls.
4 updates and 2 validation puzzles cannot establish an advantage from
stacking. A matched B-only training arm, an equally larger B, multiple seeds and
held-out adaptation probes remain necessary before a performance claim.

## Reproduce

```powershell
& '..\plastic-latent-lab\.venv\Scripts\python.exe' -m local_lab.dense_smoke --config local_lab/config_per_step_smoke.json
```

Choose a fresh results directory before rerunning; existing artifacts are preserved.

## Attention between models

B has four cross-attention heads reading A's per-cell features. C now has
4 additional cross-attention heads with
4 learned queries reading
14 B tokens, each 64-dimensional.
The tokens retain B's reader/encoder stages, query slots, memory and readout;
they are not collapsed to a single B context before C attends. C's separate
causal self-attention remains in place. The original summary-reader variant is
preserved by leaving `meta_cross_attention` false.

All four inter-model heads received nonzero answer-loss gradients on every
training step. `initial_cross_attention.pt` saves per-head read weights and
source metadata. The `c_read_pooled` and `c_read_shuffle` rows are inference
ablations with the same C parameters. They test the read interface, without
establishing that any head has a human-readable reasoning role. The core
attention mechanism follows [Attention Is All You Need](https://arxiv.org/abs/1706.03762).
