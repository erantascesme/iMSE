# Reproduction checks

Run these from the repository root after installing the package.

**Detector implementation**

```bash
python checks/check_imse.py
```

The standalone iMSE implementation agrees with the direct Gaussian log-likelihood ratio at `k=16` and `k=all`. With the original displacement and neighborhood source folders supplied to `--history` and `--neighborhood`, the same check compares against the archived BOSA/LMSE generator and iMSE extension. The local comparison matched finite-k shifts, OAS covariance, and all-neighbor or capped-neighborhood scatter exactly on the tested inputs. The check also verifies that the README contains the same implementation as the Python module.

**1. Cohorts, aggregation and table audit**

```bash
python checks/reproduce_tables.py
```

This verifies the archived data manifests and checkpoint locks, complete extraction layouts, validation/test separation, all 18 checkpoint identities, and the 15 paper detector rows. It recomputes the OOD dataset macros and checkpoint/backbone aggregation. The local check matched all 2,310 checkpoint result rows and 2,970 audited table cells within floating-point roundoff.

**2. Original score replay**

```bash
python checks/replay_original_scores.py --imagenet /path/to/imagenet_scores.zip --cifar10 /path/to/cifar10_published_scores.zip
```

Supply the original verification archives `OpenOOD_verification_scores_20260924T143947098769Z.zip` and `CIFAR10_Published_Scores_Verification.zip`. This verifies the original score file hashes and recomputes AUROC/FPR95, including Near/Far/All, from the per-image scores. The local check reproduced 820 metric rows for the ten original baselines on all eight ImageNet backbones and the two published CIFAR-10 backbones. Maximum absolute differences were (2.85\times 10^{-14}) percentage points for AUROC and (1.43\times 10^{-14}) for FPR95. These are external archives and are not packaged here.

The CPU smoke check exercised the feature cache, validation-based selection, all 15 primary detector rows, sensitivity and ablation score generation, and completion replay on a small fixture. The paired bootstrap ROC implementation was also checked against explicitly resampled ROC calculations. Seven available original CIFAR checkpoints loaded strictly on CPU and produced finite logits that exactly matched their saved linear heads on an input probe.

**Fresh run verification**

```bash
python -m imse all --dataset CIFAR10 --require-paper-match
```

The run saves `reference_comparison.csv` with unrounded metric differences and a two-decimal match flag for each checkpoint and OOD scope. The strict option exits with an error if a row differs at this precision. Roundoff differences within (10^{-10}) percentage points count as matches. The numerical comparison uses the original completed tables as its reference.

The full pixel-to-table run across all 18 checkpoints has not been executed in this workspace. The score replay covers the score arrays present in the supplied archives. The iMSE, ViM, ReAct and SCALE checkpoint results in `reference/` come from the reviewed experiment exports. Cross-device kernel and reduction differences can affect tied or nearly tied scores, particularly after fresh feature extraction. Use the original software record and authenticated feature caches for the closest numerical replay.

**Paper analyses**

Add `--sensitivity` or `--ablation` to the evaluation command to save the sensitivity grid or covariance controls in each checkpoint's `supplementary_metrics.csv`. Use a separate `--output` directory if changing these switches after a completed evaluation. The CIFAR-100 sweep treats 512 and all as the same effective neighborhood in the archived summary.

The paper's paired bootstrap uses 2,000 resamples with seed 42, reuses the original cohort seeds, and compares iMSE at (k=16) to its matched ID-only control and MDS++. Run it after a complete evaluation of the chosen dataset.

```bash
python -m imse.analysis.bootstrap --dataset CIFAR10 --scores work/results
```

The archived appendix results are provided in `reference/sensitivity_summary.csv`, `reference/ablation_comparisons.csv`, `reference/paired_intervals.csv`, and `reference/nnguide_controls.csv`.
