"""TaskAug on PTB-XL — Complete End-to-End Example.

Reproduces the bi-level augmentation policy training from:
    Raghu et al. (2022). Data Augmentation for Electrocardiograms.
    CHIL 2022, PMLR 174. https://proceedings.mlr.press/v174/raghu22a.html

Tested on Google Colab Pro (T4 GPU, 16 GB VRAM).
Set ``PTB_XL_ROOT`` to your local PTB-XL 1.0.3 directory.
Download from: https://physionet.org/content/ptb-xl/1.0.3/

Usage (local)::

    python examples/taskaug_ptbxl.py --task MI --n 1000 --epochs 50

Usage (Colab)::

    !python examples/taskaug_ptbxl.py --task MI --n 1000 --epochs 50
"""

import argparse
import os

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(description="TaskAug on PTB-XL")
parser.add_argument(
    "--root",
    default=os.environ.get("PTB_XL_ROOT", "/path/to/ptb-xl/1.0.3"),
    help="Path to the PTB-XL 1.0.3 root directory (contains ptbxl_database.csv).",
)
parser.add_argument(
    "--task",
    default="MI",
    choices=["MI", "HYP", "STTC", "CD"],
    help="Diagnostic superclass binary task to train on. Default: MI.",
)
parser.add_argument(
    "--n",
    type=int,
    default=1000,
    help="Number of patients to use (paper uses 1000 or 5000). Default: 1000.",
)
parser.add_argument(
    "--epochs",
    type=int,
    default=50,
    help="Number of training epochs. Default: 50.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=64,
    help="Batch size. Default: 64.",
)
parser.add_argument(
    "--output",
    default="./output/taskaug",
    help="Directory for checkpoints and logs. Default: ./output/taskaug.",
)
args = parser.parse_args()

# ---------------------------------------------------------------------------
# Stage 1 — Dataset
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print(f"  TaskAug | Task: {args.task} | N≈{args.n} patients")
print(f"{'='*60}\n")

from pyhealth.datasets import PTBXLDataset  # noqa: E402

print("[Stage 1] Loading PTBXLDataset …")
dataset = PTBXLDataset(
    root=args.root,
    dev=False,  # set True for a quick smoke-test with 1 000 patients
)

# ---------------------------------------------------------------------------
# Stage 2 — Task
# ---------------------------------------------------------------------------
print(f"\n[Stage 2] Setting task: {args.task} binary classification …")

from pyhealth.tasks.ecg_classification_ptbxl import ECGBinaryClassificationPTBXL  # noqa: E402

task_fn = ECGBinaryClassificationPTBXL(task=args.task)
sample_ds = dataset.set_task(task_fn)

print(f"  Samples: {len(sample_ds)}")

# ---------------------------------------------------------------------------
# Stage 3 — DataLoader
# ---------------------------------------------------------------------------
print("\n[Stage 3] Splitting and building DataLoaders …")

from pyhealth.datasets import get_dataloader, split_by_patient  # noqa: E402

# Paper protocol: 80/10/10 patient-level split
train_ds, val_ds, test_ds = split_by_patient(sample_ds, [0.8, 0.1, 0.1])

# Sub-sample to ~N patients for ablation experiments
# (the paper uses N=1000 and N=5000 from the training fold)
if args.n < len(train_ds):
    from pyhealth.datasets import sample_balanced  # noqa: E402
    train_ds = sample_balanced(train_ds, args.n)
    print(f"  Subsampled train → {len(train_ds)} samples")

train_loader = get_dataloader(train_ds, batch_size=args.batch_size, shuffle=True)
val_loader   = get_dataloader(val_ds,   batch_size=args.batch_size, shuffle=False)
test_loader  = get_dataloader(test_ds,  batch_size=args.batch_size, shuffle=False)

print(f"  Train batches: {len(train_loader)}")
print(f"  Val   batches: {len(val_loader)}")
print(f"  Test  batches: {len(test_loader)}")

# ---------------------------------------------------------------------------
# Stage 4 — Model + Policy + BiLevelTrainer
# ---------------------------------------------------------------------------
print("\n[Stage 4] Building model, policy, and bi-level trainer …")

from pyhealth.models.taskaug_resnet import (  # noqa: E402
    BiLevelTrainer,
    ResNet1D,
    TaskAugPolicy,
)

model  = ResNet1D(
    dataset=sample_ds,
    feature_keys=["signal"],
    label_key="label",
    mode="binary",
)
policy = TaskAugPolicy(n_stages=2, n_ops=7)

print(f"  ResNet1D params: {sum(p.numel() for p in model.parameters()):,}")
print(f"  Policy params  : {sum(p.numel() for p in policy.parameters()):,}")

trainer = BiLevelTrainer(
    model=model,
    policy=policy,
    metrics=["roc_auc", "pr_auc"],
    output_path=args.output,
    exp_name=f"{args.task}_n{args.n}",
)

print(f"\n[Training] Bi-level optimisation for {args.epochs} epochs …")
trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=args.epochs,
    inner_lr=1e-3,
    outer_lr=1e-2,
    neumann_order=3,
    monitor="roc_auc",
)

# ---------------------------------------------------------------------------
# Stage 5 — Evaluation
# ---------------------------------------------------------------------------
print("\n[Stage 5] Evaluating on test set …")

from pyhealth.metrics import binary_metrics_fn  # noqa: E402

y_true, y_prob, mean_loss = trainer.inference(test_loader)
scores = binary_metrics_fn(y_true, y_prob, metrics=["roc_auc", "pr_auc"])

print(f"\n{'='*60}")
print(f"  Task: {args.task} | N≈{args.n}")
print(f"  AUROC : {scores['roc_auc']:.4f}")
print(f"  AUPRC : {scores['pr_auc']:.4f}")
print(f"{'='*60}\n")
