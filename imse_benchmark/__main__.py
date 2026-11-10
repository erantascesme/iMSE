import argparse
from pathlib import Path

import pandas as pd

from .backbones.checkpoints import runs
from .common import require
from .datasets import DATASETS, download
from .benchmark.report import display_table, write_reports


def main():
    parser = argparse.ArgumentParser(description="Reproduce the iMSE benchmark")
    parser.add_argument("stage", choices=("download", "extract", "evaluate", "all"))
    parser.add_argument("--dataset", choices=DATASETS, nargs="+", default=list(DATASETS))
    parser.add_argument("--backbone", help="Optional paper backbone identifier")
    parser.add_argument("--work-dir", type=Path, default=Path("work"))
    parser.add_argument("--data", type=Path, help="Override the dataset directory")
    parser.add_argument("--weights", type=Path, help="Override the checkpoint directory")
    parser.add_argument("--features", type=Path, help="Override the feature cache directory")
    parser.add_argument("--output", type=Path, help="Override the evaluation output directory")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    data = args.data or args.work_dir / "data"
    weights = args.weights or args.work_dir / "weights"
    features = args.features or args.work_dir / "features"
    output = args.output or args.work_dir / "results"
    tasks = [(ds, model, run) for ds in dict.fromkeys(args.dataset)
             for model, run in runs(ds, args.backbone)]
    require(tasks, "No paper dataset/backbone matches the selection")
    datasets = list(dict.fromkeys(ds for ds, _, _ in tasks))
    if args.stage in ("download", "all"):
        for ds in datasets:
            download(ds, data)
        if args.stage == "download":
            return
    if args.stage in ("extract", "all"):
        from .backbones.features import extract

        for ds, model, run in tasks:
            extract(ds, model, run, data, weights, features / ds / model / run, args.device)
        if args.stage == "extract":
            return
    from .benchmark.evaluate import evaluate

    rows = pd.concat([evaluate(ds, model, run, features / ds / model / run,
                               output / ds / model / run, args.device)
                      for ds, model, run in tasks], ignore_index=True)
    _, pairs, avg = write_reports(rows, output)
    for ds in datasets:
        for scope in ("All", "Near", "Far"):
            table = display_table(pairs, avg if args.backbone is None else None, ds, scope)
            print(f"\n{ds} | {scope} | AUROC / FPR95 (%)\n{table.to_string()}")
    print(f"\nSaved metrics to {output.resolve()}")


if __name__ == "__main__":
    main()
