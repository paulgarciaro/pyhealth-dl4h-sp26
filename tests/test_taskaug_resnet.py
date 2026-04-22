"""Unit tests for TaskAug model components.

Tests cover:
  * All seven augmentation transforms
  * TaskAugPolicy (shape preservation, gradient flow)
  * ResNet1D forward-pass output contract
  * BiLevelTrainer.neumann_hypergrad

All tests are CPU-only, use tiny tensors for speed, and require no real
PTB-XL data. Tests complete in < 30 s.

Imports are done directly from pyhealth.models.taskaug_resnet to avoid
pulling in all of pyhealth.models (which requires 'transformers', 'rdkit', etc.).

Run with:
    python -m pytest tests/test_taskaug_resnet.py -v
"""

import importlib.util
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Stub out optional heavy dependencies that block import in CI / lightweight envs
# ---------------------------------------------------------------------------
for _stub in ("transformers", "rdkit", "rdkit.Chem", "rdkit.Chem.rdMolDescriptors",
              "torch_geometric", "torch_geometric.nn", "torch_geometric.data",
              "ogb", "ogb.graphproppred", "linear_attention_transformer"):
    if _stub not in sys.modules:
        sys.modules[_stub] = types.ModuleType(_stub)

# ---------------------------------------------------------------------------
# Load taskaug_resnet directly from its source file (bypasses __init__.py)
# ---------------------------------------------------------------------------
_repo_root = Path(__file__).parent.parent
_module_path = _repo_root / "pyhealth" / "models" / "taskaug_resnet.py"

_spec = importlib.util.spec_from_file_location("taskaug_resnet", str(_module_path))
_taskaug = importlib.util.module_from_spec(_spec)
# Pre-register a stub for pyhealth.models.base_model to avoid chain imports
_base_model_stub = types.ModuleType("pyhealth.models.base_model")

import torch.nn as _nn  # noqa: E402

class _BaseModelStub(_nn.Module):
    def __init__(self, dataset=None):
        super().__init__()
        self.dataset = dataset
        self.feature_keys = []
        self.label_keys = []
        if dataset and hasattr(dataset, "input_schema"):
            self.feature_keys = list(dataset.input_schema.keys())
        if dataset and hasattr(dataset, "output_schema"):
            self.label_keys = list(dataset.output_schema.keys())
        self.mode = "binary"

_base_model_stub.BaseModel = _BaseModelStub
sys.modules.setdefault("pyhealth.models.base_model", _base_model_stub)

_models_pkg_stub = types.ModuleType("pyhealth.models")
_models_pkg_stub.BaseModel = _BaseModelStub
sys.modules.setdefault("pyhealth.models", _models_pkg_stub)

_spec.loader.exec_module(_taskaug)

TemporalWarp = _taskaug.TemporalWarp
BaselineWander = _taskaug.BaselineWander
GaussianNoise = _taskaug.GaussianNoise
MagnitudeScaling = _taskaug.MagnitudeScaling
TimeMasking = _taskaug.TimeMasking
TemporalDisplacement = _taskaug.TemporalDisplacement
NoOp = _taskaug.NoOp
TaskAugPolicy = _taskaug.TaskAugPolicy
ResNet1D = _taskaug.ResNet1D
BiLevelTrainer = _taskaug.BiLevelTrainer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_BATCH = 2
_LEADS = 12
_T = 256  # short time dimension for fast CPU tests


def _random_batch(B: int = _BATCH, C: int = _LEADS, T: int = _T) -> torch.Tensor:
    return torch.randn(B, C, T)


def _random_labels(B: int = _BATCH) -> torch.Tensor:
    return torch.randint(0, 2, (B,))


def _make_mock_sample_dataset():
    """Return a minimal mock SampleDataset accepted by BaseModel.__init__."""
    ds = MagicMock()
    ds.input_schema = {"signal": "tensor"}
    ds.output_schema = {"label": "binary"}
    return ds


# ---------------------------------------------------------------------------
# 1. Augmentation transforms
# ---------------------------------------------------------------------------

class TestAugmentationTransforms(unittest.TestCase):
    """Each transform must return a tensor of the same shape as its input."""

    def _assert_shape_preserved(self, transform: nn.Module, mag: float = 0.3) -> None:
        x = _random_batch()
        out = transform(x, magnitude=mag)
        self.assertEqual(out.shape, x.shape, f"{type(transform).__name__} changed shape")

    def test_temporal_warp(self) -> None:
        self._assert_shape_preserved(TemporalWarp())

    def test_baseline_wander(self) -> None:
        self._assert_shape_preserved(BaselineWander())

    def test_gaussian_noise(self) -> None:
        self._assert_shape_preserved(GaussianNoise())

    def test_magnitude_scaling(self) -> None:
        self._assert_shape_preserved(MagnitudeScaling())

    def test_time_masking(self) -> None:
        self._assert_shape_preserved(TimeMasking())

    def test_temporal_displacement(self) -> None:
        self._assert_shape_preserved(TemporalDisplacement())

    def test_noop(self) -> None:
        x = _random_batch()
        out = NoOp()(x, magnitude=0.5)
        self.assertTrue(torch.equal(out, x), "NoOp changed the signal")

    def test_zero_magnitude_is_benign(self) -> None:
        """All transforms with magnitude=0 should return output with correct shape."""
        for cls in [BaselineWander, GaussianNoise, MagnitudeScaling,
                    TemporalDisplacement, TimeMasking]:
            x = _random_batch()
            out = cls()(x, magnitude=0.0)
            self.assertEqual(out.shape, x.shape)


# ---------------------------------------------------------------------------
# 2. TaskAugPolicy
# ---------------------------------------------------------------------------

class TestTaskAugPolicy(unittest.TestCase):

    def test_output_shape_preserved(self) -> None:
        """policy(x, y) returns same shape as x."""
        policy = TaskAugPolicy(n_stages=2, n_ops=7)
        x = _random_batch()
        y = _random_labels()
        out = policy(x, y)
        self.assertEqual(out.shape, x.shape)

    def test_gradient_flows_through_policy(self) -> None:
        """Output of policy has a grad_fn (differentiable w.r.t. policy params)."""
        policy = TaskAugPolicy(n_stages=2, n_ops=7)
        x = _random_batch().requires_grad_(False)
        y = _random_labels()
        out = policy(x, y, temperature=1.0)
        loss = out.mean()
        self.assertIsNotNone(loss.grad_fn, "Policy output has no grad_fn")

    def test_policy_params_receive_grads(self) -> None:
        """After loss.backward(), policy parameters accumulate gradients."""
        policy = TaskAugPolicy(n_stages=2, n_ops=7)
        x = _random_batch()
        y = _random_labels()
        out = policy(x, y)
        out.mean().backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in policy.parameters()
        )
        self.assertTrue(has_grad, "No policy parameter accumulated a gradient")

    def test_temperature_annealing(self) -> None:
        """Low temperature (τ→0) still returns a tensor of correct shape."""
        policy = TaskAugPolicy()
        out = policy(_random_batch(), _random_labels(), temperature=0.1)
        self.assertEqual(out.shape, (_BATCH, _LEADS, _T))

    def test_learnable_parameters(self) -> None:
        """Policy exposes logits, magnitudes_neg, magnitudes_pos parameters."""
        policy = TaskAugPolicy(n_stages=2, n_ops=7)
        param_names = [n for n, _ in policy.named_parameters()]
        self.assertIn("logits", param_names)
        self.assertIn("magnitudes_neg", param_names)
        self.assertIn("magnitudes_pos", param_names)


# ---------------------------------------------------------------------------
# 3. ResNet1D
# ---------------------------------------------------------------------------

class TestResNet1D(unittest.TestCase):

    def _make_model(self):
        ds = _make_mock_sample_dataset()
        return ResNet1D(dataset=ds, feature_keys=["signal"], label_key="label", mode="binary")

    def test_forward_output_contract(self) -> None:
        """forward() returns dict with loss, y_prob, y_true, logit."""
        model = self._make_model()
        batch = {
            "signal": _random_batch(),
            "label": _random_labels(),
        }
        out = model(**batch)
        for key in ("loss", "y_prob", "y_true", "logit"):
            self.assertIn(key, out, f"Missing key '{key}' in model output")

    def test_loss_is_scalar(self) -> None:
        model = self._make_model()
        out = model(signal=_random_batch(), label=_random_labels())
        self.assertEqual(out["loss"].shape, torch.Size([]))

    def test_y_prob_shape(self) -> None:
        """y_prob has shape (B, 1)."""
        model = self._make_model()
        B = _BATCH
        out = model(signal=_random_batch(B=B), label=_random_labels(B=B))
        self.assertEqual(out["y_prob"].shape, (B, 1))

    def test_y_prob_in_unit_interval(self) -> None:
        """All y_prob values lie in [0, 1]."""
        model = self._make_model()
        out = model(signal=_random_batch(), label=_random_labels())
        probs = out["y_prob"].detach()
        self.assertTrue((probs >= 0).all() and (probs <= 1).all())

    def test_gradients_exist_after_backward(self) -> None:
        """loss.backward() populates gradients on model parameters."""
        model = self._make_model()
        out = model(signal=_random_batch(), label=_random_labels())
        out["loss"].backward()
        has_grad = any(
            p.grad is not None for p in model.parameters() if p.requires_grad
        )
        self.assertTrue(has_grad, "No parameter accumulated a gradient after backward()")

    def test_no_label_no_loss(self) -> None:
        """When label is not in batch, output has no 'loss' or 'y_true' key."""
        model = self._make_model()
        out = model(signal=_random_batch())
        self.assertNotIn("loss", out)
        self.assertNotIn("y_true", out)

    def test_forward_from_embedding(self) -> None:
        """forward_from_embedding() gives identical shape output to forward()."""
        model = self._make_model()
        signal = _random_batch()
        label = _random_labels()
        out1 = model(signal=signal, label=label)
        out2 = model.forward_from_embedding(signal=signal, label=label)
        self.assertEqual(out1["y_prob"].shape, out2["y_prob"].shape)


# ---------------------------------------------------------------------------
# 4. BiLevelTrainer
# ---------------------------------------------------------------------------

class TestBiLevelTrainer(unittest.TestCase):

    def _make_components(self):
        ds = _make_mock_sample_dataset()
        model = ResNet1D(dataset=ds, feature_keys=["signal"], label_key="label")
        policy = TaskAugPolicy(n_stages=2, n_ops=7)
        trainer = BiLevelTrainer(
            model=model,
            policy=policy,
            device="cpu",
            output_path=tempfile.mkdtemp(),
        )
        return trainer

    def _make_dataloader(self, n_batches: int = 2):
        """Create an in-memory DataLoader with synthetic ECG batches."""
        from torch.utils.data import DataLoader, TensorDataset

        signals = torch.randn(n_batches * _BATCH, _LEADS, _T)
        labels = torch.randint(0, 2, (n_batches * _BATCH,))
        ds = TensorDataset(signals, labels)

        def collate(batch):
            sigs = torch.stack([b[0] for b in batch])
            lbls = torch.stack([b[1] for b in batch])
            return {"signal": sigs, "label": lbls}

        return DataLoader(ds, batch_size=_BATCH, collate_fn=collate)

    def test_neumann_hypergrad_runs(self) -> None:
        """neumann_hypergrad() completes without error on tiny tensors."""
        p1 = nn.Parameter(torch.randn(3))
        p2 = nn.Parameter(torch.randn(3))
        phi = nn.Parameter(torch.randn(2))

        loss_train = (p1 ** 2 + p2 ** 2).sum() + phi.sum() * 0
        loss_val = (p1 ** 2).sum()

        grads = BiLevelTrainer.neumann_hypergrad(
            loss_val=loss_val,
            loss_train=loss_train,
            model_params=[p1, p2],
            policy_params=[phi],
            lr=1e-3,
            order=2,
        )
        self.assertEqual(len(grads), 1)
        self.assertIsNotNone(grads[0])

    def test_evaluate_returns_metrics(self) -> None:
        """evaluate() returns a dict containing roc_auc."""
        trainer = self._make_components()
        loader = self._make_dataloader()
        scores = trainer.evaluate(loader)
        self.assertIn("roc_auc", scores)

    def test_inference_shapes(self) -> None:
        """inference() returns y_true and y_prob arrays of matching length."""
        trainer = self._make_components()
        loader = self._make_dataloader(n_batches=2)
        y_true, y_prob, loss = trainer.inference(loader)
        self.assertEqual(y_true.shape, y_prob.shape)
        self.assertGreater(len(y_true), 0)

    def test_one_epoch_completes(self) -> None:
        """train() for 1 epoch with 2 train batches and val finishes without error."""
        trainer = self._make_components()
        train_loader = self._make_dataloader(n_batches=2)
        val_loader = self._make_dataloader(n_batches=2)

        trainer.train(
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            epochs=1,
            neumann_order=1,
        )

    def test_checkpoint_roundtrip(self) -> None:
        """save_checkpoint / load_checkpoint preserves model weights."""
        trainer = self._make_components()
        original_param = next(trainer.model.parameters()).data.clone()

        trainer.save_checkpoint("test.ckpt")
        with torch.no_grad():
            next(trainer.model.parameters()).fill_(999.0)

        trainer.load_checkpoint(os.path.join(trainer.exp_path, "test.ckpt"))
        restored_param = next(trainer.model.parameters()).data
        self.assertTrue(
            torch.allclose(original_param, restored_param),
            "Model weights not restored after load_checkpoint",
        )


if __name__ == "__main__":
    unittest.main()
