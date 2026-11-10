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

The historical finite-neighbor arithmetic is retained for the original iMSE results. With every parent included, the old neighborhood ranking affects only the order of FP64 scatter reductions. The all-neighbor branch computes leave-one-out means by class sums.

The targeted diagnostic validated the transformed activations and classifier scoring for ReAct on ImageNet ResNet50 and RegNetY-16GF and for SCALE on EfficientNetV2-S. It also identified the single-class bank in the original CIFAR-10 ResNet18 NNGuide run. The benchmark uses the original NNGuide bank policy from the paper's results tables.

Baseline references include [OpenOOD](https://github.com/Jingkang50/OpenOOD/tree/3c35632ee91b54b09d1f085d04f94744cece7d0b), [NNGuide](https://github.com/roomo7time/nnguide), [ViM](https://github.com/haoqiwang/vim), [ReAct](https://github.com/deeplearning-wisc/react), and [SCALE](https://github.com/kai422/SCALE). Each detector module preserves the numerical kernels and bank selection used in the completed runs.
