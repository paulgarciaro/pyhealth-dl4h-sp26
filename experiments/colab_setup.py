"""
colab_setup.py — Import firewall + experiment utilities for TaskAug on PTB-XL.

Usage
-----
Colab notebook:
    exec(open(f'{REPO_ROOT}/experiments/colab_setup.py').read())

Local Python:
    from experiments.colab_setup import *   # after setting REPO_ROOT in env
    # or just run experiments/run_local.py

After exec/import the following are available:
    DEVICE, TASKS, N_SIZES, N_SPLITS, EPOCHS, BATCH
    get_dataloader, split_by_patient, PTBXLDataset
    ECGBinaryClassificationPTBXL, binary_metrics_fn
    ResNet1D, TaskAugPolicy, BiLevelTrainer
    build_loaders(task, n, seed, ptb_root)
    run_split(task, n, seed, method, ptb_root, output_base)
"""

from pathlib import Path as _Path
import importlib.util as _ilu

# ── Auto-detect REPO_ROOT if not already defined ────────────────────────────
# When exec()'d from Colab the caller sets REPO_ROOT before calling us.
# When run directly or imported, we infer it from this file's location.
if "REPO_ROOT" not in dir():
    REPO_ROOT = str(_Path(__file__).resolve().parent.parent)
import sys
import types
import os
import json
import time
import warnings

import numpy as np
import torch
import torch.nn as _nn

warnings.filterwarnings("ignore")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[colab_setup] device = {DEVICE}")

# ============================================================================
# 1. Dependency firewall
# ============================================================================
# pyhealth's __init__ files eagerly import every model/task/metric, dragging
# in optional heavy packages (rdkit, transformers, torch_geometric …) that are
# not available in stock Colab.  We block those chains by:
#   a) stubbing the problematic packages in sys.modules before anything runs
#   b) loading only the specific .py files we need via importlib

# -- a) Stub optional heavy packages ----------------------------------------
_STUBS = [
    "rdkit", "rdkit.Chem", "rdkit.Chem.rdMolDescriptors",
    "torch_geometric", "torch_geometric.nn", "torch_geometric.data",
    "torch_geometric.utils",
    "ogb", "ogb.graphproppred",
    "linear_attention_transformer",
    "transformers",
    "dgl",
]
for _s in _STUBS:
    sys.modules.setdefault(_s, types.ModuleType(_s))


# -- b) Minimal BaseModel stand-in (only used at import resolution time) -----
class _BaseModelStub(_nn.Module):
    def __init__(self, dataset=None):
        super().__init__()
        self.dataset = dataset
        self.feature_keys = (
            list(dataset.input_schema.keys())
            if dataset and hasattr(dataset, "input_schema")
            else []
        )
        self.label_keys = (
            list(dataset.output_schema.keys())
            if dataset and hasattr(dataset, "output_schema")
            else []
        )
        self.mode = "binary"


# -- c) Helper: load a .py file without touching any __init__.py -------------
def _load_file(mod_name, filepath):
    spec = _ilu.spec_from_file_location(mod_name, filepath)
    mod = _ilu.module_from_spec(spec)
    sys.modules[mod_name] = mod        # register BEFORE exec (prevents re-entry)
    spec.loader.exec_module(mod)
    return mod


# -- d) Stub pyhealth.models -------------------------------------------------
_bm_stub = types.ModuleType("pyhealth.models.base_model")
_bm_stub.BaseModel = _BaseModelStub
sys.modules.setdefault("pyhealth.models.base_model", _bm_stub)

_models_stub = types.ModuleType("pyhealth.models")
_models_stub.BaseModel = _BaseModelStub
sys.modules.setdefault("pyhealth.models", _models_stub)

# -- e) Stub pyhealth.tasks and load only ecg_classification_ptbxl -----------
_bt = _load_file(
    "pyhealth.tasks.base_task",
    f"{REPO_ROOT}/pyhealth/tasks/base_task.py",
)
_tasks_stub = types.ModuleType("pyhealth.tasks")
_tasks_stub.BaseTask = _bt.BaseTask
sys.modules.setdefault("pyhealth.tasks", _tasks_stub)

_et = _load_file(
    "pyhealth.tasks.ecg_classification_ptbxl",
    f"{REPO_ROOT}/pyhealth/tasks/ecg_classification_ptbxl.py",
)
ECGBinaryClassificationPTBXL = _et.ECGBinaryClassificationPTBXL

# -- f) Load taskaug_resnet (bypasses models/__init__.py) --------------------
_tr = _load_file(
    "taskaug_resnet",
    f"{REPO_ROOT}/pyhealth/models/taskaug_resnet.py",
)
ResNet1D = _tr.ResNet1D
TaskAugPolicy = _tr.TaskAugPolicy
BiLevelTrainer = _tr.BiLevelTrainer

assert issubclass(ResNet1D, _nn.Module), "ResNet1D failed to load"
print("[colab_setup] ResNet1D / TaskAugPolicy / BiLevelTrainer ✓")

# ============================================================================
# 2. Safe leaf imports
# ============================================================================
from pyhealth.datasets.utils import get_dataloader          # noqa: E402
from pyhealth.datasets.splitter import split_by_patient     # noqa: E402
from pyhealth.datasets.ptbxl import PTBXLDataset            # noqa: E402
from pyhealth.metrics.binary import binary_metrics_fn       # noqa: E402

print("[colab_setup] dataset / metrics utilities ✓")

# ============================================================================
# 3. Experiment constants
# ============================================================================
TASKS    = ["MI", "HYP", "STTC", "CD"]
N_SIZES  = [1000, 5000]
N_SPLITS = 15   # 15 random 80/10/10 patient-level splits (paper protocol)
EPOCHS   = 50
BATCH    = 64

# ============================================================================
# 4. Experiment utilities
# ============================================================================

def build_loaders(task: str, n: int, seed: int, ptb_root: str):
    """Load PTB-XL, apply task, split by patient, sub-sample training fold.

    Returns
    -------
    train_loader, val_loader, test_loader, sample_ds
    """
    dataset   = PTBXLDataset(root=ptb_root, dev=False)
    task_fn   = ECGBinaryClassificationPTBXL(task=task)
    sample_ds = dataset.set_task(task_fn)

    train_ds, val_ds, test_ds = split_by_patient(
        sample_ds, [0.8, 0.1, 0.1], seed=seed
    )

    # Sub-sample training fold to exactly n patients
    rng    = np.random.default_rng(seed)
    pids   = list({s["patient_id"] for s in train_ds})
    chosen = set(rng.choice(pids, size=min(n, len(pids)), replace=False))
    train_ds = [s for s in train_ds if s["patient_id"] in chosen]

    train_loader = get_dataloader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader   = get_dataloader(val_ds,   batch_size=BATCH, shuffle=False)
    test_loader  = get_dataloader(test_ds,  batch_size=BATCH, shuffle=False)
    return train_loader, val_loader, test_loader, sample_ds


def run_split(task, n, seed, method, ptb_root, output_base):
    """Train one method on one random split; return metric dict."""
    train_loader, val_loader, test_loader, sample_ds = build_loaders(
        task, n, seed, ptb_root
    )

    model  = ResNet1D(
        dataset=sample_ds, feature_keys=["signal"],
        label_key="label", mode="binary",
    ).to(DEVICE)
    policy = TaskAugPolicy(n_stages=2, n_ops=7).to(DEVICE)

    if method == "no_aug":
        # Plain supervised training — no augmentation policy
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        model.train()
        for _ in range(EPOCHS):
            for batch in train_loader:
                batch = {
                    k: v.to(DEVICE) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }
                loss = model(**batch)["loss"]
                opt.zero_grad()
                loss.backward()
                opt.step()
        trainer = BiLevelTrainer(
            model=model, policy=policy, device=DEVICE,
            output_path=output_base,
            exp_name=f"{method}_{task}_n{n}_s{seed}",
        )

    elif method == "frozen_policy":
        for p in policy.parameters():
            p.requires_grad_(False)
        trainer = BiLevelTrainer(
            model=model, policy=policy, device=DEVICE,
            output_path=output_base,
            exp_name=f"{method}_{task}_n{n}_s{seed}",
        )
        trainer.train(train_loader, val_loader, epochs=EPOCHS, monitor="roc_auc")

    elif method == "global_mag":
        # Disable class-specific magnitudes by tying neg == pos at init
        with torch.no_grad():
            policy.magnitudes_pos.copy_(policy.magnitudes_neg)
        trainer = BiLevelTrainer(
            model=model, policy=policy, device=DEVICE,
            output_path=output_base,
            exp_name=f"{method}_{task}_n{n}_s{seed}",
        )
        trainer.train(train_loader, val_loader, epochs=EPOCHS, monitor="roc_auc")

    else:  # full TaskAug
        trainer = BiLevelTrainer(
            model=model, policy=policy, device=DEVICE,
            output_path=output_base,
            exp_name=f"{method}_{task}_n{n}_s{seed}",
        )
        trainer.train(
            train_loader, val_loader,
            epochs=EPOCHS, inner_lr=1e-3, outer_lr=1e-2,
            neumann_order=3, monitor="roc_auc",
        )

    y_true, y_prob, _ = trainer.inference(test_loader)
    return binary_metrics_fn(y_true, y_prob, metrics=["roc_auc", "pr_auc"])


print("[colab_setup] build_loaders / run_split ✓")
print("[colab_setup] Ready — DEVICE:", DEVICE)
