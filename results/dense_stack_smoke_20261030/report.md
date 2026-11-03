# Stacked dense controllers: C modifies B, B modifies A

The implemented chain is:

`A states -> B context -> C reads B context/history -> dense delta_W_B -> B uses W_B + delta_W_B -> dense delta_W_A -> A uses W_A + delta_W_A`

C writes the complete 1024 by 32
output weight of B's nonlinear weight generator. It does not replace all of B.
B continues to write both complete 512 by 1536 A projections. There are no
low-rank factors at either level. C has a causal Transformer, its own state and
four-step B-context history. Both levels use dense chunked generation.

C's update has a 10% relative Frobenius
limit; B's A updates retain their 3%
limit. These numerical bounds do not establish behavioral safety or interpretability.
C observes detached B context and past C state without labels. C modifies B
before B generates A's next updates, avoiding a same-step circular dependency.
Memory values persist across ACT updates and reset independently for new puzzles;
temporal gradient credit is truncated at ACT boundaries.

This run loaded the earlier dense A/B smoke checkpoint, then jointly trained B
and fresh C while keeping A frozen. B has 348,128 parameters;
C adds 25,456. The sampled initial C matrix had measured rank
32 out of 32.

## Executed smoke

- 8 B/C updates, 4 admitted training puzzles, 8 separate validation puzzles.
- Finite nonzero gradients reached both readers and both weight generators on every update.
- All A tensors and sparse scratch weights stayed identical. Disabling both controllers reproduced native logits bit for bit.
- Disabling C reproduced a standalone A/B model with the corresponding current B weights bit for bit, before and after training.
- Temporary matrices at both levels were cleared after use.
- Runtime 85.46 seconds against a 240-second cap; peak Torch CUDA allocation 552.95 MiB.
- No oracle or official test split was used.

| Smoke condition | Blank-cell accuracy | Task loss |
|---|---:|---:|
| off | 42.478% | 1.391086 |
| initial | 42.257% | 1.390618 |
| final | 42.478% | 1.388622 |
| off_after_training | 42.478% | 1.391086 |
| initial_c_off | 42.478% | 1.389416 |
| c_off | 42.257% | 1.389014 |
| c_shuffle | 42.478% | 1.389465 |
| c_phase | 42.257% | 1.389757 |
| c_reset_memory | 42.257% | 1.389228 |

The C-off/shuffled/phase/history-reset rows are inference ablations of the same
jointly trained checkpoint. They are not separately trained matched controls.
Eight updates and eight validation puzzles cannot establish an advantage from
stacking. A matched B-only training arm, an equally larger B, multiple seeds and
held-out adaptation probes remain necessary before a performance claim.

## Reproduce

```powershell
& '..\plastic-latent-lab\.venv\Scripts\python.exe' -m local_lab.dense_smoke --config local_lab/config_stack_smoke.json
```

Choose a fresh results directory before rerunning; existing artifacts are preserved.
