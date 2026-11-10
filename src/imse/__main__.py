import argparse
from pathlib import Path

import pandas as pd

from .backbones import runs
from .common import require
from .datasets import DATASETS, download
from .benchmark.report import compare, display_table, write_reports


def main():
    parser = argparse.ArgumentParser(description="Reproduce the iMSE benchmark")
    parser.add_argument("stage", choices=("replay", "download", "extract", "import-cache", "evaluate", "all"))
    parser.add_argument("--dataset", choices=DATASETS, nargs="+", default=list(DATASETS))
    parser.add_argument("--backbone", help="Optional paper backbone identifier")
    parser.add_argument("--data", type=Path, default=Path("work/data"))
    parser.add_argument("--weights", type=Path, default=Path("work/weights"))
    parser.add_argument("--features", type=Path, default=Path("work/features"))
    parser.add_argument("--output", type=Path, default=Path("work/results"))
    parser.add_argument("--reference", type=Path, default=Path("reference/checkpoints.csv"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sensitivity", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--require-paper-match", action="store_true",
                        help="Fail when any result differs at the paper's two-decimal precision")
    parser.add_argument("--legacy-root", type=Path, help="Root of the existing experiment feature caches")
    args = parser.parse_args()
    if args.stage in ("download", "all"):
        for ds in args.dataset:
            download(ds, args.data)
        if args.stage == "download":
            return
    tasks = [(ds, model, run) for ds in args.dataset for model, run in runs(ds, args.backbone)]
    require(tasks, "No paper dataset/backbone matches the selection")
    if args.stage == "import-cache":
        from .backbones.cache import import_cache

        require(args.legacy_root, "--legacy-root is required for import-cache")
        for ds, model, run in tasks:
            import_cache(ds, model, run, args.legacy_root, args.features / ds / model / run)
        return
    if args.stage in ("extract", "all"):
        from .backbones.features import extract

        for ds, model, run in tasks:
            extract(ds, model, run, args.data, args.weights, args.features / ds / model / run, args.device)
        if args.stage == "extract":
            return
    reference = pd.read_csv(args.reference)
    task_keys = set(tasks)
    ref = reference.loc[[k in task_keys for k in zip(reference.Dataset, reference.Backbone, reference.Run)]]
    if args.stage == "replay":
        rows = ref
    else:
        from .benchmark.evaluate import evaluate

        rows = pd.concat([evaluate(ds, model, run, args.features / ds / model / run,
                                   args.output / ds / model / run, args.device, args.sensitivity,
                                   args.ablation)
                          for ds, model, run in tasks], ignore_index=True)
    runs_table, pairs, avg = write_reports(rows, args.output)
    delta = compare(runs_table, ref)
    delta.to_csv(args.output / "reference_comparison.csv", index=False)
    for ds in dict.fromkeys(row[0] for row in tasks):
        print("\n", ds, "\n", display_table(pairs, avg if args.backbone is None else None, ds).to_string())
    for metric in ("AUROC", "FPR95"):
        print(f"{metric}: {delta[metric + '_same_2dp'].sum()}/{len(delta)} rows at paper precision, "
              f"max absolute delta {delta[metric + '_delta_pp'].abs().max():.9g} pp")
    if args.require_paper_match:
        require(delta[["AUROC_same_2dp", "FPR95_same_2dp"]].to_numpy().all(),
                "Result drift at paper precision. See reference_comparison.csv.")


if __name__ == "__main__":
    main()
