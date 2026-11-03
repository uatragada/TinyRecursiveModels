# Local native TRM and controller B

Raw final test-panel scores: Native A 40.65% unknown-cell / 0.00% exact-puzzle; Static joint 40.13% unknown-cell / 0.00% exact-puzzle; Dynamic joint 39.19% unknown-cell / 0.00% exact-puzzle. This run measures from-scratch Sudoku learning with native TRM and our plasticity controller. It does not measure isolated A-weight adaptability, and it does not reproduce the paper's final benchmark accuracy.

Saved status: `complete`. Seed: `17`. Wall time: 2506.2 seconds. Configured bound: 3000 seconds.

A retains native width 512 and two shared reasoning layers. Each ACT inner update uses H=3 outer cycles and L=6 latent updates, giving 21 shared-module calls: 14 without gradients and seven with gradients. Carries detach between ACT updates, and evaluation retains the native maximum of 16 ACT updates. Our B uses width 128, three causal Transformer layers, eight heads, FF width 512, and history four; it controls rank-four temporary updates on the two native MLP down-projections at scale 0.2. B's history resets per ACT inner update. The preserved executed configuration and sources establish the actual settings.

Native A-only, static joint, and dynamic joint arms start from the same full state and receive the same proposed candidate stream. Native ACT decides which slots admit new puzzles, so actual admissions are checked separately. Stablemax, halt BCE, AdamATan2, sparse puzzle-embedding optimization, and EMA follow the local runner's native setup. Dynamic-checkpoint gates-off/fixed evaluations are counterfactual inference modes, not independently trained controls.

Identical saved initial parameters do not imply identical initial functions: native uses gates off, static uses tanh(0.01), and dynamic uses B's near-zero predicted gates. Equal completed optimizer steps and A reasoning passes also do not equalize total compute or wall time; B adds its own computation and about 810,000 parameters.

Recorded native A dense parameters: **6,828,034**. Our B parameters: **809,736**. The separately optimized native puzzle-embedding table has 512 elements; its batch-local gradient buffer has 65,536 elements, which are not additional logical puzzle-embedding degrees of freedom. Dense optimizer: Portable PyTorch implementation of adam-atan2 0.0.3 upstream reference formula; fused CUDA build unavailable on Windows. Native sparse optimizer: CastedSparseEmbeddingSignSGD_Distributed, world_size=1.

Saved validation pool: 2,048 puzzles; saved test pool: 1,024. Configured endpoint size: 1,024; evaluation batch size: 128. Every arm, mode, and weight version uses the same saved examples. A panel equal to a pool's size evaluates that full saved pool; a smaller panel uses its prefix. The saved pools themselves are bounded sequential subsets of the official dataset, not the complete published benchmark.

## Saved evaluations

Unknown-cell accuracy excludes given clues. Exact accuracy requires the whole saved target solution. Given agreement measures copied clues; valid-grid fraction requires a completed legal Sudoku that also preserves all givens. All-cell accuracy includes copied clues and can look strong when unknown cells remain wrong. The clue-copy baseline exposes that inflation. A uniform guess over nine digits gives an 11.11% unknown-cell reference, rather than a trained solver.

| Arm | Weights | Mode | Split | Puzzles | Unknown-cell acc | Exact puzzle acc | Given agreement | Valid grids | All-cell acc | Clue-copy baseline | Stablemax per cell |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Native A | raw | off | val | 1,024 | 39.76% | 0.00% | 100.00% | 0.00% | 58.40% | 30.95% | 0.9801 |
| Native A | raw | off | test | 1,024 | 40.65% | 0.00% | 100.00% | 0.00% | 59.24% | 31.33% | 0.9710 |
| Native A | ema | off | val | 1,024 | 35.21% | 0.00% | 100.00% | 0.00% | 55.27% | 30.95% | 1.0101 |
| Native A | ema | off | test | 1,024 | 35.77% | 0.00% | 100.00% | 0.00% | 55.89% | 31.33% | 1.0061 |
| Static joint | raw | static | val | 1,024 | 39.75% | 0.00% | 100.00% | 0.00% | 58.40% | 30.95% | 1.0534 |
| Static joint | raw | static | test | 1,024 | 40.13% | 0.00% | 100.00% | 0.00% | 58.89% | 31.33% | 1.0484 |
| Static joint | ema | static | val | 1,024 | 32.48% | 0.00% | 100.00% | 0.00% | 53.38% | 30.95% | 1.0609 |
| Static joint | ema | static | test | 1,024 | 33.17% | 0.00% | 100.00% | 0.00% | 54.11% | 31.33% | 1.0572 |
| Dynamic joint | raw | dynamic | val | 1,024 | 38.73% | 0.00% | 100.00% | 0.00% | 57.70% | 30.95% | 1.0390 |
| Dynamic joint | raw | dynamic | test | 1,024 | 39.19% | 0.00% | 100.00% | 0.00% | 58.24% | 31.33% | 1.0296 |
| Dynamic joint | ema | dynamic | val | 1,024 | 31.95% | 0.00% | 100.00% | 0.00% | 53.01% | 30.95% | 1.0608 |
| Dynamic joint | ema | dynamic | test | 1,024 | 32.76% | 0.00% | 100.00% | 0.00% | 53.82% | 31.33% | 1.0535 |
| Dynamic joint | raw | off | val | 1,024 | 38.82% | 0.00% | 100.00% | 0.00% | 57.75% | 30.95% | 1.0329 |
| Dynamic joint | raw | off | test | 1,024 | 39.26% | 0.00% | 100.00% | 0.00% | 58.29% | 31.33% | 1.0247 |
| Dynamic joint | raw | fixed | val | 1,024 | 38.80% | 0.00% | 100.00% | 0.00% | 57.74% | 30.95% | 1.0349 |
| Dynamic joint | raw | fixed | test | 1,024 | 39.17% | 0.00% | 100.00% | 0.00% | 58.23% | 31.33% | 1.0256 |

Evaluation checkpoint policy: raw fixed final; EMA also evaluated separately with mu=0.999.

Raw final weights are primary. EMA is secondary and is kept separate in every table; its configured rate is 0.9990. EMA can lag during learning, so raw and EMA results are never averaged. Endpoint loss is normalized per-cell Stablemax loss. Training curves log per-puzzle normalized Stablemax task loss plus 0.5 × halting BCE on the sampled training batch. The plotted history contains sampled training measurements; it does not invent held-out validation curves or a validation-best checkpoint selection.

Raw dynamic-checkpoint interventions compare identical A/B weights and test examples. Positive gains favor evolving dynamic gates over the indicated inference mode:

| Panel | Comparator mode | Dynamic unknown-cell gain (pp) | Dynamic exact gain (pp) | Dynamic Stablemax decrease |
| --- | --- | --- | --- | --- |
| val | off | -0.082 | +0.000 | -0.00609 |
| val | fixed | -0.061 | +0.000 | -0.00408 |
| test | off | -0.072 | +0.000 | -0.00493 |
| test | fixed | +0.011 | +0.000 | -0.00401 |

## Exposure and local resources

Loader advancement, ACT batch-slot presentations, and newly admitted source puzzles are different quantities. An unhalted carry can continue an earlier puzzle. Training augmentations create new symmetry views of existing puzzles, not independent new puzzles. Use the admission and unique-source counters below to assess actual data coverage.

| Arm | Executed/requested steps | New puzzle admissions | Unique source puzzles | Train-pool coverage | Distinct admitted augmentation views | ACT-slot presentations | Trainable parameters | Seconds | Complete |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Native A | 2000/2000 | 16,000 | 14,574 | 22.24% | 16,000 | 256,000 | 6,828,034 | 611.9 | True |
| Static joint | 2000/2000 | 16,000 | 14,574 | 22.24% | 16,000 | 256,000 | 6,844,426 | 654.9 | True |
| Dynamic joint | 2000/2000 | 16,000 | 14,574 | 22.24% | 16,000 | 256,000 | 7,654,154 | 810.1 | True |

Peak CUDA tensor allocation: 5.17 GiB. GPU: `NVIDIA GeForce RTX 5070 Ti`. Tensor allocation excludes driver and other-process memory. Timing includes whatever the runner records as stage work and is not a dedicated hardware benchmark.

| Saved matching / inference check | Passed |
| --- | --- |
| all configured arms recorded | True |
| same initial full state | True |
| all stages completed requested steps | True |
| equal completed step budgets | True |
| same proposed candidate stream | True |
| all evaluations use 16 ACT steps | True |
| evaluation parameter states unchanged | True |
| evaluation runtime gates cleared | True |
| logged training reasoning calls 21 | True |
| logged training gradient enabled calls 7 | True |
| logged training controller calls match mode | True |

All recorded matching/inference checks passed: **True**. Actual admitted puzzle streams equal: **True**. ACT admissions can differ as models halt at different times. An admission difference is a consequence of native ACT and must be considered when attributing gains; equal candidate batches alone do not establish equal realized training data.

## Data and integrity

The saved data evidence records pinned source identity, stream selection, split sizes, hashes, and integrity checks. Data labels are training/evaluation targets; B receives no solutions at inference.

```json
{
  "repository": "sapientinc/sudoku-extreme",
  "revision": "58942f96baeb572ca3127e2a9e9c70f330783d6b",
  "selection": "Official test sequential prefix first; official train sequential prefix supplies train then validation; reject invalid rows and any already-selected puzzle OR solution hash",
  "requested_sizes": {
    "train": 65536,
    "val": 2048,
    "test": 1024
  },
  "splits": {
    "train": {
      "n": 65536,
      "shape": [
        65536,
        81
      ],
      "official_source_split": "train",
      "original_index_min": 0,
      "original_index_max": 65535,
      "source_counts": {
        "puzzles4_forum_hardest_1905": 29701,
        "puzzles1_unbiased": 14182,
        "puzzles0_kaggle": 1372,
        "puzzles5_forum_hardest_1905_11+": 673,
        "01_file1": 18836,
        "puzzles2_17_clue": 740,
        "puzzles6_forum_hardest_1106": 5,
        "puzzles3_magictour_top1465": 27
      },
      "rating_min": 0,
      "rating_max": 286,
      "valid_solution_count": 65536,
      "clue_consistent_count": 65536,
      "unique_puzzle_sha256": 65536,
      "unique_solution_sha256": 65536
    },
    "val": {
      "n": 2048,
      "shape": [
        2048,
        81
      ],
      "official_source_split": "train",
      "original_index_min": 65536,
      "original_index_max": 67583,
      "source_counts": {
        "puzzles1_unbiased": 543,
        "puzzles4_forum_hardest_1905": 1131,
        "puzzles2_17_clue": 28,
        "puzzles0_kaggle": 48,
        "01_file1": 267,
        "puzzles5_forum_hardest_1905_11+": 30,
        "puzzles3_magictour_top1465": 1
      },
      "rating_min": 0,
      "rating_max": 230,
      "valid_solution_count": 2048,
      "clue_consistent_count": 2048,
      "unique_puzzle_sha256": 2048,
      "unique_solution_sha256": 2048
    },
    "test": {
      "n": 1024,
      "shape": [
        1024,
        81
      ],
      "official_source_split": "test",
      "original_index_min": 0,
      "original_index_max": 1023,
      "source_counts": {
        "puzzles1_unbiased": 218,
        "puzzles4_forum_hardest_1905": 439,
        "puzzles0_kaggle": 21,
        "puzzles2_17_clue": 6,
        "01_file1": 333,
        "puzzles5_forum_hardest_1905_11+": 7
      },
      "rating_min": 0,
      "rating_max": 112,
      "valid_solution_count": 1024,
      "clue_consistent_count": 1024,
      "unique_puzzle_sha256": 1024,
      "unique_solution_sha256": 1024
    }
  },
  "encoding": {
    "pad": 0,
    "blank": 1,
    "digits_1_to_9": [
      2,
      10
    ],
    "vocab_size": 11,
    "puzzle_identifier": 0,
    "seq_len": 81
  },
  "limitations": [
    "Sequential prefixes can be source- and difficulty-biased; these are not random samples of the full official dataset.",
    "Validation is an adjacent held-out portion of the official training prefix, not a separately published benchmark split.",
    "Exact puzzle and solution hash separation is enforced; mathematical equivalence of local train/validation puzzles is not independently solved.",
    "Official card claims unique solutions and mathematically inequivalent train/test sets; local checks validate provided solutions and clues, not independent uniqueness proofs.",
    "Pool size is not training exposure; the runner must report actual base puzzles seen and repeated augmented presentations."
  ]
}
```

Saved independent verification passed: **True**. Recorded check count: 339. Scope: Independent source/data/hash/protocol/accounting and saved metric identity checks. Upstream source comparison permits Windows CRLF normalization. No Torch import; no checkpoint tensor reconstruction or independent per-step ACT admission ledger.

Separate saved-tensor verification passed: **True**, with 1,071/1,071 checks. Scope: Independent saved-tensor hashes, finite values and expected trained/frozen components; CPU Torch only. Does not replay training or independently recompute predictions.

| Arm | Changed A tensors | Changed B tensors | Changed adapter tensors | Changed static gates |
| --- | --- | --- | --- | --- |
| Native A | 13 | 0 | 0 | 0 |
| Static joint | 13 | 0 | 4 | 1 |
| Dynamic joint | 13 | 43 | 4 | 0 |

Shared final local validation records **56 tests passed** in 6.13 seconds using `python -m pytest -q`. Discovery: not recorded. This reporter does not rerun those tests.

The [isolated A-weight probe: longer_probe_seed29](../longer_probe_seed29/probe_report.md) is reported separately. It freezes donor A and adapts common fresh B plus U/V; source-learning scores in this report are not substituted for those adaptation results.

The [isolated A-weight probe: longer_probe_seed43](../longer_probe_seed43/probe_report.md) is reported separately. It freezes donor A and adapts common fresh B plus U/V; source-learning scores in this report are not substituted for those adaptation results.

The [isolated A-weight probe: longer_probe_comparison](../longer_probe_comparison/aggregate_report.md) is reported separately. It freezes donor A and adapts common fresh B plus U/V; source-learning scores in this report are not substituted for those adaptation results.

## Figures

![Local TRM evidence](plots/sudoku_endpoints.png)

![Local TRM evidence](plots/sudoku_training.png)

## Interpretation and limits

A begins from scratch. A dynamic/static/native difference can reflect ordinary task learning, controller dependence, initialization, optimization noise, or unequal effective work if any matching check fails. It is not an isolated test of whether A learned to become easier to adapt. This single-seed diagnostic shares its puzzle collection across arms; examples and augmented views must not be treated as independent model-training replicates.

The native README's smallest full Sudoku experiment takes about 18 hours on a 48 GB L40S. This bounded local run uses its own recorded data and optimization protocol. It does not reproduce the complete published benchmark. Source-task accuracy alone establishes neither broad retention nor H3; the separate probe must compare donor initial states, learning changes, static-joint controls, and independent A-training seeds.

Upstream references: [official code and native training protocol](https://github.com/SamsungSAILMontreal/TinyRecursiveModels), [primary paper](https://arxiv.org/html/2510.04871v1). These links describe the upstream research; the local saved evidence establishes this run's behavior.

The reporter imports no Torch, loads no model weights, performs no training, and publishes nothing. `report_summary.json` preserves normalized saved metric rows; `report_manifest.json` hashes evidence and the reporter. Original configuration, logs, data audit, checkpoints, and source snapshots remain the owning evidence.
