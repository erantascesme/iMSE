# Benchmark protocol

| ID dataset | Frozen checkpoints | Near OOD | Far OOD |
|---|---|---|---|
| CIFAR-10 | OpenOOD ResNet18 s0–s2, ODIN DenseNet, ViT-B16 | CIFAR-100, Tiny ImageNet | MNIST, SVHN, Textures, Places365 |
| CIFAR-100 | OpenOOD ResNet18 s0–s2, ResNet34, ViT-B16 | CIFAR-10, Tiny ImageNet | MNIST, SVHN, Textures, Places365 |
| ImageNet-1K | SwinV2-B, DeiT3-B16, ConvNeXtV2-B, ViT-B16, ResNet50, EfficientNetV2-S, Vim-Base, RegNetY-16GF | NINCO, SSB-Hard | iNaturalist, Textures, OpenImage-O |

The shipped gzip manifests carry the original ordered file lists and labels. The CIFAR archives and every checkpoint are verified against observed SHA-256 values from the completed experiments. For ImageNet, the official image sources are resolved against the pinned manifests. The package stores the released checkpoints' revisions, heads and transform records under `src/imse/assets`. Inference uses raw FP32 inputs to the trained head and its original logits. Vim-Base uses the complete EMA state. SwinV2 regenerates only verified matching, nonpersistent attention buffers.

Fit on the full ID training sets. Validation contains 5,000 ID and 1,763 OOD images on ImageNet-1K, 1,000 ID and 1,000 OOD images on CIFAR-10, and 1,000 ID and 972 OOD images on CIFAR-100. The final ID populations contain 45,000 or 9,000 images respectively. KNN, ViM, ReAct and SCALE select by validation AUROC, with the first candidate winning a tie. ID validation activations determine ReAct thresholds. Neither iMSE setting fits on OOD or test data.

| Detector | Configuration |
|---|---|
| MSP, Energy | Temperature 1 |
| LogitGap | Top logit minus mean of ranks 2–200 for ImageNet, or all competing logits for CIFAR |
| KNN | Full ID training bank, exact search on normalized features, K in 50/100/200/500/1000 |
| NNGuide | K=100, nominal 1% full-batch bank. ImageNet uses shuffled official manifest order with seed 0; CIFAR uses the original ordered first 600 rows |
| fDBD | Classifier boundary distance divided by distance to the training feature mean |
| MDS and RMDS | Single-layer empirical Mahalanobis and relative scores. CIFAR raw scores retain the pinned OpenOOD FP32 reduction order. ImageNet uses the original FP64 pseudoinverse |
| MDS++ and RMDS++ | Corresponding Gaussian estimators with per-sample L2 normalization |
| ViM | Principal dimension 256 or 1000 when smaller than feature dimension; full ID training fit |
| ReAct | 85/90/95/99 percentile; Energy on upper-clipped activations |
| SCALE | 65/70/75/80/85/90/95 percentile; Energy on scaled original activations |
| iMSE | Cosine, k=16 or all other same-class samples, tau=1, one synthetic sample per parent, OAS ID covariance and pooled shift covariance |

FPR95 is ID rejection at 95% OOD recall. Compute AUROC and FPR95 separately for each OOD dataset, average the datasets equally within Near/Far/All, then average checkpoints within a backbone before averaging backbones. All averages the individual OOD datasets. The original CIFAR-10 ResNet18 NNGuide bank covers one class. A random 600-row control at the same K is exported separately and does not replace the paper table.

## Fresh data and existing caches

The download stage uses the original OpenOOD CIFAR archive IDs. If an archive is already available, place it in `work/data/cifar/<asset>.zip`. Put a manually obtained matching OpenOOD checkpoint ZIP or pinned model file in `work/weights/<dataset>/<backbone>/` using the filename in the checkpoint lock.

For ImageNet-1K, accept the access conditions on [ILSVRC/imagenet-1k](https://huggingface.co/datasets/ILSVRC/imagenet-1k) and authenticate with Hugging Face. The downloader uses its original `script` archives and the benchmark OOD image sources. A local archive or unpacked image tree can instead be supplied at `work/data/imagenet/{train,val,NINCO,SSB_Hard,iNaturalist,Textures,OpenImage_O}`. Use the five canonical train tar files with images directly inside the tar, or a pre-extracted official train tree. Choose either the archive or the unpacked tree for an asset. The released ImageNet manifests define identities, but the experiment export does not provide full original image-byte locks for these sources. For the closest numeric reproduction, import the original feature banks.

`python -m imse import-cache --dataset CIFAR10 --legacy-root "/path/to/Colab Notebooks"` reads the original archive-authenticated feature banks and preserves their original row order. The destination must be outside the original experiment root. Use fast local storage for feature caches. Store the repo and the resulting tables wherever convenient.

For Vim-Base use Linux with a compatible CUDA toolkit and compiler. Install `ninja`, `packaging`, `einops`, and `transformers` before extracting that backbone. The loader builds the pinned Vim extensions when absent and checks the reviewed source patches and classifier head. The original ImageNet software record is in `reference/environment_imagenet.json`.

The source of the CIFAR ResNet implementation is [pinned OpenOOD](https://github.com/Jingkang50/OpenOOD/tree/3c35632ee91b54b09d1f085d04f94744cece7d0b). The restricted historical DenseNet loader uses the [ODIN release](https://github.com/facebookresearch/odin/tree/64e97962ccaed1fe979f43a089c0feb4d8b002fd). Their notices are under `src/imse/assets/upstream`.
