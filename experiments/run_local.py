"""
run_local.py — Run TaskAug experiments locally (CPU or GPU).

Usage
-----
    # Full run (all methods, all tasks, all N, 15 splits)
    python experiments/run_local.py --ptb_root ~/data/ptb-xl-1.0.3

    # Quick smoke-test (1 split, 2 epochs, small N)
    python experiments/run_local.py --ptb_root ~/data/ptb-xl-1.0.3 \\
        --methods no_aug taskaug --tasks MI --n_sizes 200 \\
        --n_splits 1 --epochs 2

    # Ablations only
    python experiments/run_local.py --ptb_root ~/data/ptb-xl-1.0.3 \\
        --methods frozen_policy global_mag --n_sizes 1000
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── Locate repo root and exec the shared setup ──────────────────────────────
REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

exec(open(f"{REPO_ROOT}/experiments/colab_setup.py").read())
# After exec: DEVICE, TASKS, N_SIZES, N_SPLITS, EPOCHS, BATCH,
#             build_loaders, run_split, binary_metrics_fn, etc.


# ── CLI ──────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="TaskAug local experiment runner")
    p.add_argument(
        "--ptb_root",
        default=os.environ.get("PTB_XL_ROOT", str(Path.home() / "data" / "ptb-xl-1.0.3")),
        help="Path to PTB-XL root (must contain ptbxl_database.csv + records500/)",
    )
    p.add_argument(
        "--results_dir",
        default=str(Path(__file__).parent / "results"),
        help="Directory for JSON result files",
    )
    p.add_argument(
        "--methods",
        nargs="+",
        default=["no_aug", "frozen_policy", "global_mag", "taskaug"],
        choices=["no_aug", "frozen_policy", "global_mag", "taskaug"],
        help="Methods to run",
    )
    p.add_argument(
        "--tasks",
        nargs="+",
        default=TASKS,
        choices=TASKS,
        help="ECG classification tasks",
    )
    p.add_argument(
        "--n_sizes",
        nargs="+",
        type=int,
        default=N_SIZES,
        help="Training set sizes",
    )
    p.add_argument(
        "--n_splits",
        type=int,
        default=N_SPLITS,
        help="Number of random splits (default: 15)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Training epochs per split (default: 50)",
    )
    p.add_argument(
        "--batch",
        type=int,
        default=BATCH,
        help="Batch size (default: 64)",
    )
    return p.parse_args()


# ── Runner ───────────────────────────────────────────────────────────────────
def run_experiment(method, tasks, n_sizes, n_splits, epochs, ptb_root, results_dir):
    # Override globals so run_split picks up CLI values
    global EPOCHS, BATCH
    EPOCHS = epochs

    os.makedirs(results_dir, exist_ok=True)
    all_results = {}

    for task in tasks:
        for n in n_sizes:
            # Ablations only run at N=1000
            if method in ("frozen_policy", "global_mag") and n != 1000:
                continue

            key = f"{method}__{task}__n{n}"
            result_file = os.path.join(results_dir, f"{key}.json")

            if os.path.isfile(result_file):
                print(f"[SKIP] {key} — already done")
                with open(result_file) as f:
                    all_results[key] = json.load(f)
                continue

            print(f"\n>>> {key}  (device={DEVICE})")
            split_scores = []
            for seed in range(n_splits):
                t0 = time.time()
                scores = run_split(
                    task, n, seed, method, ptb_root,
                    os.path.join(results_dir, "ckpts"),
                )
                elapsed = time.time() - t0
                split_scores.append(scores)
                print(
                    f"  split {seed:02d} | "
                    f"AUROC={scores['roc_auc']:.4f}  "
                    f"AUPRC={scores['pr_auc']:.4f} | "
                    f"{elapsed:.0f}s"
                )

            agg = {
                "roc_auc_mean": float(np.mean([s["roc_auc"] for s in split_scores])),
                "roc_auc_std":  float(np.std( [s["roc_auc"] for s in split_scores])),
                "pr_auc_mean":  float(np.mean([s["pr_auc"]  for s in split_scores])),
                "pr_auc_std":   float(np.std( [s["pr_auc"]  for s in split_scores])),
                "splits": split_scores,
            }
            all_results[key] = agg
            with open(result_file, "w") as f:
                json.dump(agg, f, indent=2)
            print(
                f"  → AUROC {agg['roc_auc_mean']:.4f} ± {agg['roc_auc_std']:.4f}"
                f" | AUPRC {agg['pr_auc_mean']:.4f} ± {agg['pr_auc_std']:.4f}"
            )

    return all_results


def print_table(results_dir, methods, tasks, n_sizes):
    try:
        import pandas as pd
    except ImportError:
        print("(install pandas to see formatted table)")
        return

    for n in n_sizes:
        print(f"\n{'='*60}")
        print(f"AUROC  (N={n})")
        print(f"{'='*60}")
        rows = []
        for m in methods:
            if m in ("frozen_policy", "global_mag") and n != 1000:
                continue
            row = {"Method": m}
            for t in tasks:
                p = os.path.join(results_dir, f"{m}__{t}__n{n}.json")
                if os.path.isfile(p):
                    with open(p) as f:
                        r = json.load(f)
                    row[t] = f"{r['roc_auc_mean']:.3f}±{r['roc_auc_std']:.3f}"
                else:
                    row[t] = "TBD"
            rows.append(row)
        if rows:
            print(pd.DataFrame(rows).to_string(index=False))


# ── Main ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    args = parse_args()

    print(f"\nPTB-XL root : {args.ptb_root}")
    print(f"Results dir : {args.results_dir}")
    print(f"Methods     : {args.methods}")
    print(f"Tasks       : {args.tasks}")
    print(f"N sizes     : {args.n_sizes}")
    print(f"Splits      : {args.n_splits}")
    print(f"Epochs      : {args.epochs}")
    print(f"Device      : {DEVICE}\n")

    # Ensure ptbxl-pyhealth.csv exists
    meta = os.path.join(args.ptb_root, "ptbxl-pyhealth.csv")
    if not os.path.isfile(meta):
        print("Building ptbxl-pyhealth.csv ...")
        PTBXLDataset.prepare_metadata(args.ptb_root)

    for method in args.methods:
        run_experiment(
            method=method,
            tasks=args.tasks,
            n_sizes=args.n_sizes,
            n_splits=args.n_splits,
            epochs=args.epochs,
            ptb_root=args.ptb_root,
            results_dir=args.results_dir,
        )

    print("\n" + "="*60)
    print("All experiments complete.")
    print_table(args.results_dir, args.methods, args.tasks, args.n_sizes)
