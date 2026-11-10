# Experiment sources

The package was assembled from the completed experiment notebooks and reviewed exports. The original checkpoint hashes and ordered manifests are in `src/imse/assets`. The table audit is tied to the 9 November iMSE table build used by the final manuscript.

| Component | Original experiment |
|---|---|
| ImageNet-1K data and features | Canonical download and extraction notebooks, followed by the eight-backbone OpenOOD-aligned evaluation with the strict Swin fix |
| CIFAR-10 official ResNet18 | OpenOOD download and feature notebooks, followed by the baseline evaluation with the final validation fix |
| CIFAR-10 DenseNet and ViT | Published-checkpoint experiment and its corrected reference evaluation |
| CIFAR-100 | OpenOOD data and feature experiment and the reviewed detector evaluation, restricted to the three paper backbones |
| ViM, ReAct and SCALE | The reviewed frozen-checkpoint extension with the original validation grids |
| iMSE k=16 | The validated displacement score with a shared ID class mean, full parent population and no class shift |
| iMSE all | The later leave-one-out full-class-mean experiment, with the same ID geometry |
| Sensitivity, covariance controls, paired uncertainty | The reviewed neighborhood sweep, all-backbone ablation and seed-42 paired bootstrap |

The historical finite-neighbor arithmetic is retained for the original iMSE results. With every parent included, the old neighborhood ranking affects only the order of FP64 scatter reductions. The all-neighbor branch computes leave-one-out means by class sums. Published settings and controls are kept in separate result files.

The implementation retains the OpenOOD loader convention in the main NNGuide row and provides the pre-existing CIFAR-10 ResNet18 random-bank control separately. The targeted diagnostic validated the transformed activations and classifier scoring for ReAct on ImageNet ResNet50 and RegNetY-16GF and for SCALE on EfficientNetV2-S. It also identified the single-class bank in the original CIFAR-10 ResNet18 NNGuide run. Those effects belong to the reported protocol and checkpoint settings.

Baseline references: [OpenOOD](https://github.com/Jingkang50/OpenOOD/tree/3c35632ee91b54b09d1f085d04f94744cece7d0b), [NNGuide](https://github.com/roomo7time/nnguide), [ViM](https://github.com/haoqiwang/vim), [ReAct](https://github.com/deeplearning-wisc/react), and [SCALE](https://github.com/kai422/SCALE). ImageNet and CIFAR implementations are kept separate where the completed runs used different numerical kernels or bank selection.
