"""Settings for the paper's core benchmark."""

OOD_RECALL = 0.95
IMSE_SETTINGS = {"iMSE16": (16, 1.0), "iMSEall": ("all", 1.0)}
LOGIT_GAP_TOP = {"CIFAR10": None, "CIFAR100": None, "ImageNet1K": 200}
MAHALANOBIS_VARIANTS = (("", False), ("++", True))

VALIDATION_GRIDS = {
    "KNN": (50, 100, 200, 500, 1000),
    "ViM": (256, 1000),
    "ReAct": (85, 90, 95, 99),
    "SCALE": (65, 70, 75, 80, 85, 90, 95),
}
KNN_TEST_COMPANION = {"CIFAR10": 50, "CIFAR100": 200, "ImageNet1K": 1000}

NNGUIDE_K = 100
NNGUIDE_BANK_FRACTION = 0.01
NNGUIDE_BANK_BATCH = 200
NNGUIDE_IMAGENET_SEED = 0
NNGUIDE_CIFAR_COMPANION_K = 10
NNGUIDE_BANK_POLICY = {
    "CIFAR10": "ordered prefix",
    "CIFAR100": "ordered prefix",
    "ImageNet1K": "shuffled official manifest",
}

DETECTORS = (
    "MSP", "Energy", "LogitGap", "KNN", "NNGuide", "fDBD", "Mahalanobis",
    "Mahalanobis++", "Relative-Mahalanobis", "Relative-Mahalanobis++",
    "ViM", "ReAct", "SCALE", *IMSE_SETTINGS,
)
