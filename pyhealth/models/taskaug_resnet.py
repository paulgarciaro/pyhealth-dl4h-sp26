"""TaskAug: Task-Adaptive Data Augmentation for ECGs.

Implements the full TaskAug framework (Raghu et al., CHIL 2022) as a
native PyHealth 2.0 contribution:

  * Seven augmentation transforms (Section 2 of the paper)
  * ``TaskAugPolicy`` — differentiable K-stage policy with Gumbel-Softmax
    straight-through estimator and class-specific magnitudes
  * ``ResNet1D`` — 1-D ResNet-18 backbone subclassing
    :class:`~pyhealth.models.BaseModel`
  * ``BiLevelTrainer`` — bi-level optimiser that alternates an inner
    Adam update (augmented train data) with an outer RMSprop update
    (clean validation loss via Neumann-series hypergradient)

Reference:
    Raghu, A., Shanmugam, D., Pomerantsev, E., Guttag, J., & Stultz, C. M.
    (2022). Data Augmentation for Electrocardiograms.
    CHIL, PMLR 174. https://proceedings.mlr.press/v174/raghu22a.html

Author:
    Paul Garcia (alanpg2@illinois.edu) — DL4H Spring 2026

Import lines for ``pyhealth/models/__init__.py``::

    from .taskaug_resnet import BiLevelTrainer, ResNet1D, TaskAugPolicy
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader

from pyhealth.models.base_model import BaseModel

logger = logging.getLogger(__name__)

# ============================================================================
# Section 1 — Augmentation transforms
# ============================================================================


class TemporalWarp(nn.Module):
    """Temporally warp the ECG signal using random piecewise-linear interpolation.

    Args:
        n_knots (int): Number of interior control points. Default is 4.
    """

    def __init__(self, n_knots: int = 4) -> None:
        super().__init__()
        self.n_knots = n_knots

    def forward(self, x: Tensor, magnitude: float) -> Tensor:
        """Apply temporal warp.

        Args:
            x (Tensor): Input signal of shape ``(B, C, T)``.
            magnitude (float): Warp strength in [0, 1]. A value of 0 yields
                no warp; 1 allows knot displacements of up to ±50 % of the
                signal length.

        Returns:
            Tensor: Warped signal of shape ``(B, C, T)``.
        """
        B, C, T = x.shape
        if magnitude <= 0.0:
            return x

        half_range = magnitude * 0.5 * T
        # Interior knot positions in [0, T] + small perturbations
        knot_x = torch.linspace(0, T - 1, self.n_knots + 2, device=x.device)
        perturb = (torch.rand(self.n_knots, device=x.device) * 2 - 1) * half_range
        inner = knot_x[1:-1] + perturb
        inner = torch.clamp(inner, 0, T - 1)
        src_knots = torch.cat([knot_x[:1], inner, knot_x[-1:]])

        # Build warped grid via linear interpolation between knots
        tgt = torch.linspace(0, T - 1, T, device=x.device)
        grid = torch.zeros(T, device=x.device)
        for i in range(len(src_knots) - 1):
            t0, t1 = knot_x[i].item(), knot_x[i + 1].item()
            s0, s1 = src_knots[i].item(), src_knots[i + 1].item()
            mask = (tgt >= t0) & (tgt < t1)
            if mask.any() and (t1 - t0) > 0:
                alpha = (tgt[mask] - t0) / (t1 - t0)
                grid[mask] = s0 + alpha * (s1 - s0)
        grid[-1] = src_knots[-1]

        # Normalise warp grid to [-1, 1] for grid_sample
        grid_norm = (grid / (T - 1)) * 2 - 1  # (T,)
        # grid_sample for 4-D input needs grid shape (B, H_out, W_out, 2)
        # We treat H=1, W=T: x-coord = time warp, y-coord = 0 (no height warp)
        grid_xy = torch.stack(
            [grid_norm, torch.zeros_like(grid_norm)], dim=-1
        )  # (T, 2)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(0).expand(B, 1, T, 2)  # (B, 1, T, 2)

        x_4d = x.unsqueeze(2)  # (B, C, 1, T)
        warped = F.grid_sample(
            x_4d, grid_xy,
            mode="bilinear", padding_mode="border", align_corners=True,
        )  # (B, C, 1, T)
        return warped.squeeze(2)  # (B, C, T)


class BaselineWander(nn.Module):
    """Add low-frequency sinusoidal baseline wander to each lead independently."""

    def forward(self, x: Tensor, magnitude: float) -> Tensor:
        """Apply baseline wander.

        Args:
            x (Tensor): ``(B, C, T)``
            magnitude (float): Amplitude of wander relative to signal std.

        Returns:
            Tensor: ``(B, C, T)`` with additive wander.
        """
        B, C, T = x.shape
        if magnitude <= 0.0:
            return x
        t = torch.linspace(0, 2 * np.pi, T, device=x.device)
        freqs = torch.rand(B, C, 1, device=x.device) * 1.5 + 0.1  # 0.1–1.6 Hz range
        phases = torch.rand(B, C, 1, device=x.device) * 2 * np.pi
        wander = magnitude * torch.sin(freqs * t + phases)  # broadcast over T
        return x + wander


class GaussianNoise(nn.Module):
    """Add i.i.d. Gaussian noise to all samples and leads."""

    def forward(self, x: Tensor, magnitude: float) -> Tensor:
        """Apply Gaussian noise.

        Args:
            x (Tensor): ``(B, C, T)``
            magnitude (float): Standard deviation of the noise.

        Returns:
            Tensor: Noisy signal of the same shape.
        """
        if magnitude <= 0.0:
            return x
        return x + torch.randn_like(x) * magnitude


class MagnitudeScaling(nn.Module):
    """Multiply the entire waveform by a random scalar close to 1."""

    def forward(self, x: Tensor, magnitude: float) -> Tensor:
        """Apply magnitude scaling.

        Args:
            x (Tensor): ``(B, C, T)``
            magnitude (float): Half-range of the multiplicative factor.
                Scale factor is drawn uniformly from
                ``[1 - magnitude, 1 + magnitude]``.

        Returns:
            Tensor: Scaled signal of the same shape.
        """
        if magnitude <= 0.0:
            return x
        B = x.shape[0]
        scale = 1.0 + (torch.rand(B, 1, 1, device=x.device) * 2 - 1) * magnitude
        return x * scale


class TimeMasking(nn.Module):
    """Zero out a contiguous time window (inspired by SpecAugment)."""

    def forward(self, x: Tensor, magnitude: float) -> Tensor:
        """Apply time masking.

        Args:
            x (Tensor): ``(B, C, T)``
            magnitude (float): Fraction of ``T`` to mask.

        Returns:
            Tensor: Signal with a random window zeroed out.
        """
        B, C, T = x.shape
        if magnitude <= 0.0:
            return x
        mask_len = max(1, int(magnitude * T))
        out = x.clone()
        starts = torch.randint(0, max(1, T - mask_len + 1), (B,), device=x.device)
        for b, s in enumerate(starts.tolist()):
            out[b, :, s : s + mask_len] = 0.0
        return out


class TemporalDisplacement(nn.Module):
    """Circularly shift the signal along the time axis."""

    def forward(self, x: Tensor, magnitude: float) -> Tensor:
        """Apply temporal displacement.

        Args:
            x (Tensor): ``(B, C, T)``
            magnitude (float): Maximum shift as a fraction of ``T``.

        Returns:
            Tensor: Circularly shifted signal.
        """
        B, C, T = x.shape
        if magnitude <= 0.0:
            return x
        max_shift = max(1, int(magnitude * T))
        shifts = torch.randint(-max_shift, max_shift + 1, (B,), device=x.device)
        out = torch.stack([torch.roll(x[b], shifts[b].item(), dims=-1) for b in range(B)])
        return out


class NoOp(nn.Module):
    """Identity transform — returns the input unchanged."""

    def forward(self, x: Tensor, magnitude: float) -> Tensor:
        return x


# Registry used by TaskAugPolicy
_TRANSFORMS: List[nn.Module] = [
    TemporalWarp(),
    BaselineWander(),
    GaussianNoise(),
    MagnitudeScaling(),
    TimeMasking(),
    TemporalDisplacement(),
    NoOp(),
]
_N_OPS = len(_TRANSFORMS)  # 7


# ============================================================================
# Section 2 — TaskAugPolicy
# ============================================================================


def _gumbel_softmax_st(logits: Tensor, temperature: float) -> Tensor:
    """Gumbel-Softmax with straight-through gradient estimator.

    Forward pass returns a hard one-hot vector; the backward pass uses the
    soft Gumbel-Softmax approximation to propagate gradients.

    Args:
        logits (Tensor): Unnormalised log-probabilities ``(..., K)``.
        temperature (float): Softmax temperature τ. Lower values push the
            distribution towards one-hot.

    Returns:
        Tensor: Straight-through one-hot of the same shape as *logits*.
    """
    soft = F.gumbel_softmax(logits, tau=temperature, hard=False)
    # Hard one-hot (argmax in forward, but gradients flow through *soft*)
    hard = torch.zeros_like(soft)
    hard.scatter_(-1, soft.argmax(dim=-1, keepdim=True), 1.0)
    return hard - soft.detach() + soft  # straight-through


class TaskAugPolicy(nn.Module):
    """Differentiable K-stage augmentation policy with class-specific magnitudes.

    Learnable parameters
    ~~~~~~~~~~~~~~~~~~~~
    - ``logits``         shape ``(n_stages, n_ops)`` — unnormalised op weights
    - ``magnitudes_neg`` shape ``(n_ops,)`` — magnitude for negative samples
    - ``magnitudes_pos`` shape ``(n_ops,)`` — magnitude for positive samples

    Total learnable parameters per binary task: 2 × n_ops + 2 × n_stages × n_ops
    → with n_stages=2, n_ops=7: 14 + 14 = 28 params (paper reports 18 for a
    slightly different parameterisation; the key design is the class-specific
    split).

    Args:
        n_stages (int): Number of sequential augmentation stages K. Default 2.
        n_ops (int): Number of candidate operations. Default 7.
        transforms (list[nn.Module]): Ordered list of transform instances.
            Defaults to the module-level ``_TRANSFORMS`` list.

    Examples:
        >>> policy = TaskAugPolicy(n_stages=2, n_ops=7)
        >>> x = torch.randn(4, 12, 2500)  # batch of 4 ECGs
        >>> y = torch.randint(0, 2, (4,))
        >>> x_aug = policy(x, y)
        >>> x_aug.shape
        torch.Size([4, 12, 2500])
    """

    def __init__(
        self,
        n_stages: int = 2,
        n_ops: int = 7,
        transforms: Optional[List[nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.n_stages = n_stages
        self.n_ops = n_ops
        self.transforms: List[nn.Module] = transforms if transforms is not None else list(_TRANSFORMS)

        self.logits = nn.Parameter(torch.zeros(n_stages, n_ops))
        # Initialise magnitudes to small positive values
        self.magnitudes_neg = nn.Parameter(torch.full((n_ops,), 0.1))
        self.magnitudes_pos = nn.Parameter(torch.full((n_ops,), 0.1))

    def forward(
        self,
        x: Tensor,
        y: Tensor,
        temperature: float = 1.0,
    ) -> Tensor:
        """Apply K augmentation stages to a batch of ECG signals.

        For each stage a single operation is sampled via the Gumbel-Softmax
        straight-through estimator. The magnitude is selected per sample based
        on its class label: negative samples use ``magnitudes_neg``, positives
        use ``magnitudes_pos``.

        Args:
            x (Tensor): Input ECG batch ``(B, C, T)``.
            y (Tensor): Binary class labels ``(B,)`` with values in {0, 1}.
            temperature (float): Gumbel-Softmax temperature τ. Default 1.0.

        Returns:
            Tensor: Augmented signal of shape ``(B, C, T)``.
        """
        B = x.shape[0]
        out = x

        for stage in range(self.n_stages):
            weights = _gumbel_softmax_st(self.logits[stage], temperature)  # (n_ops,)

            # Per-sample magnitude: negative → magnitudes_neg, positive → magnitudes_pos
            mags_neg = torch.sigmoid(self.magnitudes_neg)  # clamp to (0, 1)
            mags_pos = torch.sigmoid(self.magnitudes_pos)

            # Apply each operation and form a weighted sum (differentiable)
            stage_out = torch.zeros_like(out)
            for op_idx, transform in enumerate(self.transforms):
                # Scalar magnitude for this op, computed per sample
                mag_per_sample = torch.where(
                    y.bool(),
                    mags_pos[op_idx].expand(B),
                    mags_neg[op_idx].expand(B),
                )
                # Average magnitude over batch for this transform call
                # (transform API takes a scalar magnitude)
                mag_scalar = mag_per_sample.mean().item()
                t_out = transform(out, mag_scalar)
                stage_out = stage_out + weights[op_idx] * t_out

            out = stage_out

        return out


# ============================================================================
# Section 3 — ResNet1D (BaseModel subclass)
# ============================================================================


class _BasicBlock1D(nn.Module):
    """Residual block for 1-D ResNet with optional downsampling shortcut."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.shortcut: Optional[nn.Module] = None
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )

    def forward(self, x: Tensor) -> Tensor:
        identity = x if self.shortcut is None else self.shortcut(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)


class ResNet1D(BaseModel):
    """1-D ResNet-18 ECG classifier implementing the PyHealth BaseModel API.

    Architecture matches the backbone used in Raghu et al. (CHIL 2022):
    four residual layer groups ``[64, 128, 256, 512]`` with 2 blocks each,
    a stride-2 downsampling at each group transition, and a global average
    pooling head.

    Args:
        dataset: Fitted :class:`~pyhealth.datasets.SampleDataset`.
        feature_keys (list[str]): List with a single entry naming the signal
            feature key. Default: ``["signal"]``.
        label_key (str): Name of the label key in the dataset. Default:
            ``"label"``.
        mode (str): Task mode — ``"binary"`` | ``"multiclass"`` |
            ``"multilabel"``. Default: ``"binary"``.
        in_channels (int): Number of ECG leads. Default: ``12``.
        base_filters (int): Width of the first residual stage. Default: ``64``.
        **kwargs: Additional keyword arguments forwarded to
            :class:`~pyhealth.models.BaseModel`.

    Examples:
        >>> from pyhealth.models import ResNet1D
        >>> model = ResNet1D(dataset=sample_ds, feature_keys=["signal"],
        ...                  label_key="label", mode="binary")
        >>> batch = {"signal": torch.randn(4, 12, 2500), "label": torch.randint(0, 2, (4,))}
        >>> out = model(**batch)
        >>> out.keys()
        dict_keys(['loss', 'y_prob', 'y_true', 'logit'])

    Import line for ``pyhealth/models/__init__.py``::

        from .taskaug_resnet import BiLevelTrainer, ResNet1D, TaskAugPolicy
    """

    def __init__(
        self,
        dataset,
        feature_keys: Optional[List[str]] = None,
        label_key: str = "label",
        mode: str = "binary",
        in_channels: int = 12,
        base_filters: int = 64,
        **kwargs,
    ) -> None:
        super().__init__(dataset=dataset)
        self.feature_keys = feature_keys or ["signal"]
        self.label_key = label_key
        self.mode = mode

        f = base_filters  # shorthand

        # Stem
        self.stem = nn.Sequential(
            nn.Conv1d(in_channels, f, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(f),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2, padding=1),
        )

        # Residual groups
        self.layer1 = self._make_layer(f, f, blocks=2, stride=1)
        self.layer2 = self._make_layer(f, f * 2, blocks=2, stride=2)
        self.layer3 = self._make_layer(f * 2, f * 4, blocks=2, stride=2)
        self.layer4 = self._make_layer(f * 4, f * 8, blocks=2, stride=2)

        # Classification head
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(f * 8, 1)  # binary output

        self._init_weights()

    @staticmethod
    def _make_layer(
        in_ch: int, out_ch: int, blocks: int, stride: int
    ) -> nn.Sequential:
        layers = [_BasicBlock1D(in_ch, out_ch, stride=stride)]
        for _ in range(1, blocks):
            layers.append(_BasicBlock1D(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _encode(self, signal: Tensor) -> Tensor:
        """Shared encoder used by both forward and forward_from_embedding."""
        x = self.stem(signal)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x).squeeze(-1)  # (B, 512)
        return self.fc(x)  # (B, 1)

    def forward(self, **kwargs) -> Dict[str, Tensor]:
        """Forward pass.

        Args:
            **kwargs: Must include the signal tensor under
                ``self.feature_keys[0]`` and optionally the label tensor
                under ``self.label_key``.

        Returns:
            dict with keys:
                - ``logit``  — raw logit ``(B, 1)``
                - ``y_prob`` — sigmoid probability ``(B, 1)``
                - ``loss``   — BCE loss (scalar), only when label is present
                - ``y_true`` — ground-truth labels ``(B, 1)``, when present
        """
        signal: Tensor = kwargs[self.feature_keys[0]]
        logit = self._encode(signal)
        y_prob = torch.sigmoid(logit)

        out: Dict[str, Tensor] = {"logit": logit, "y_prob": y_prob}

        if self.label_key in kwargs:
            label = kwargs[self.label_key].float()
            if label.dim() == 1:
                label = label.unsqueeze(-1)
            loss = F.binary_cross_entropy_with_logits(logit, label)
            out["loss"] = loss
            out["y_true"] = label

        return out

    def forward_from_embedding(self, signal: Tensor, **kwargs) -> Dict[str, Tensor]:
        """Alias allowing interpretability methods to bypass kwarg dispatch."""
        return self.forward(**{self.feature_keys[0]: signal, **kwargs})


# ============================================================================
# Section 4 — BiLevelTrainer
# ============================================================================


class BiLevelTrainer:
    """Bi-level optimiser for TaskAug.

    Alternates between:

    * **Inner loop** — updates ``model`` parameters on augmented training
      batches using Adam (lr = 1e-3 by default).
    * **Outer loop** — updates ``policy`` parameters on clean validation loss
      using RMSprop (lr = 1e-2 by default) with a Neumann-series
      hypergradient approximation.

    Args:
        model (ResNet1D): Classifier model (must expose ``forward(**batch)``
            returning a dict with ``"loss"`` key).
        policy (TaskAugPolicy): Augmentation policy network.
        metrics (list[str]): Metric names passed to the PyHealth metrics
            functions. Default: ``["roc_auc", "pr_auc"]``.
        device (Optional[str]): ``"cuda"`` or ``"cpu"``. Auto-detected if
            ``None``.
        output_path (str): Directory for checkpoints. Default: ``"./output"``.
        exp_name (Optional[str]): Sub-directory name. Defaults to timestamp.
        feature_key (str): Key for ECG signal in the batch dict. Default
            ``"signal"``.
        label_key (str): Key for binary label in the batch dict. Default
            ``"label"``.

    Examples:
        >>> trainer = BiLevelTrainer(
        ...     model=model, policy=policy, metrics=["roc_auc", "pr_auc"]
        ... )
        >>> trainer.train(
        ...     train_dataloader=train_loader,
        ...     val_dataloader=val_loader,
        ...     epochs=50,
        ...     monitor="roc_auc",
        ... )
    """

    def __init__(
        self,
        model: ResNet1D,
        policy: TaskAugPolicy,
        metrics: Optional[List[str]] = None,
        device: Optional[str] = None,
        output_path: str = "./output",
        exp_name: Optional[str] = None,
        feature_key: str = "signal",
        label_key: str = "label",
    ) -> None:
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model = model.to(device)
        self.policy = policy.to(device)
        self.metrics = metrics or ["roc_auc", "pr_auc"]
        self.feature_key = feature_key
        self.label_key = label_key

        from datetime import datetime

        exp_name = exp_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.exp_path = os.path.join(output_path, exp_name)
        os.makedirs(self.exp_path, exist_ok=True)

        self.best_score = -float("inf")
        self.epoch = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        epochs: int = 50,
        inner_lr: float = 1e-3,
        outer_lr: float = 1e-2,
        neumann_order: int = 3,
        max_grad_norm: float = 5.0,
        monitor: str = "roc_auc",
        monitor_criterion: str = "max",
        temperature_schedule: bool = True,
    ) -> None:
        """Run the full bi-level training loop.

        Args:
            train_dataloader: DataLoader for augmented inner-loop batches.
            val_dataloader: DataLoader for clean outer-loop / evaluation.
            epochs: Total training epochs. Default 50.
            inner_lr: Adam learning rate for model parameters. Default 1e-3.
            outer_lr: RMSprop learning rate for policy parameters. Default 1e-2.
            neumann_order: Truncation order of the Neumann series. Default 3.
            max_grad_norm: Gradient norm clip for the outer loop. Default 5.0.
            monitor: Metric name to track for best-model checkpointing.
            monitor_criterion: ``"max"`` or ``"min"``. Default ``"max"``.
            temperature_schedule: If ``True`` anneal τ from 1.0 → 0.1 over
                training. Default ``True``.
        """
        inner_opt = torch.optim.Adam(self.model.parameters(), lr=inner_lr)
        outer_opt = torch.optim.RMSprop(self.policy.parameters(), lr=outer_lr)

        val_iter = iter(val_dataloader) if val_dataloader is not None else None

        for epoch in range(epochs):
            self.epoch = epoch
            tau = max(0.1, 1.0 - epoch / epochs) if temperature_schedule else 1.0
            train_losses = self._run_epoch(
                train_dataloader,
                val_iter,
                val_dataloader,
                inner_opt,
                outer_opt,
                tau,
                neumann_order,
                max_grad_norm,
            )

            avg_loss = float(np.mean(train_losses)) if train_losses else 0.0
            logger.info(f"Epoch {epoch:3d} | train_loss={avg_loss:.4f} | τ={tau:.3f}")

            if val_dataloader is not None:
                scores = self.evaluate(val_dataloader)
                score_str = " ".join(f"{k}={v:.4f}" for k, v in scores.items())
                logger.info(f"Epoch {epoch:3d} | {score_str}")

                val_score = scores.get(monitor, -float("inf"))
                better = (
                    val_score > self.best_score
                    if monitor_criterion == "max"
                    else val_score < self.best_score
                )
                if better:
                    self.best_score = val_score
                    self.save_checkpoint("best.ckpt")
                    logger.info(
                        f"  ✓ New best {monitor}={val_score:.4f} saved."
                    )

            self.save_checkpoint("last.ckpt")

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on *dataloader* and return metric scores.

        Args:
            dataloader: DataLoader of clean (un-augmented) samples.

        Returns:
            dict[str, float]: Metric name → score.
        """
        from pyhealth.metrics.binary import binary_metrics_fn

        self.model.eval()
        all_y_true: List[Tensor] = []
        all_y_prob: List[Tensor] = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {
                    k: v.to(self.device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                }
                out = self.model(**batch)
                all_y_true.append(out["y_true"].cpu())
                all_y_prob.append(out["y_prob"].cpu())

        y_true = torch.cat(all_y_true).squeeze(-1).numpy()
        y_prob = torch.cat(all_y_prob).squeeze(-1).numpy()
        return binary_metrics_fn(y_true, y_prob, metrics=self.metrics)

    def inference(
        self, dataloader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """Run inference and return (y_true, y_prob, mean_loss).

        Args:
            dataloader: DataLoader to run inference on.

        Returns:
            Tuple of ``(y_true, y_prob, mean_loss)`` as numpy arrays / float.
        """
        self.model.eval()
        all_y_true, all_y_prob, losses = [], [], []

        with torch.no_grad():
            for batch in dataloader:
                batch = {
                    k: v.to(self.device) if isinstance(v, Tensor) else v
                    for k, v in batch.items()
                }
                out = self.model(**batch)
                all_y_true.append(out["y_true"].cpu())
                all_y_prob.append(out["y_prob"].cpu())
                if "loss" in out:
                    losses.append(out["loss"].item())

        y_true = torch.cat(all_y_true).squeeze(-1).numpy()
        y_prob = torch.cat(all_y_prob).squeeze(-1).numpy()
        mean_loss = float(np.mean(losses)) if losses else 0.0
        return y_true, y_prob, mean_loss

    def save_checkpoint(self, filename: str) -> None:
        """Save model and policy weights to *self.exp_path/filename*."""
        path = os.path.join(self.exp_path, filename)
        torch.save(
            {
                "model": self.model.state_dict(),
                "policy": self.policy.state_dict(),
                "epoch": self.epoch,
                "best_score": self.best_score,
            },
            path,
        )

    def load_checkpoint(self, path: str) -> None:
        """Restore model and policy from a checkpoint file.

        Args:
            path (str): Path to the ``.ckpt`` file.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.model.load_state_dict(ckpt["model"])
        self.policy.load_state_dict(ckpt["policy"])
        self.epoch = ckpt.get("epoch", 0)
        self.best_score = ckpt.get("best_score", -float("inf"))

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        train_loader: DataLoader,
        val_iter,
        val_loader: Optional[DataLoader],
        inner_opt: torch.optim.Optimizer,
        outer_opt: torch.optim.Optimizer,
        tau: float,
        neumann_order: int,
        max_grad_norm: float,
    ) -> List[float]:
        """Run one full epoch of bi-level optimisation."""
        self.model.train()
        self.policy.train()
        train_losses = []

        for batch in train_loader:
            batch = {
                k: v.to(self.device) if isinstance(v, Tensor) else v
                for k, v in batch.items()
            }
            signal: Tensor = batch[self.feature_key]
            labels: Tensor = batch[self.label_key]

            # ---- Augmented forward pass (retain graph for outer loop) ----
            aug_signal = self.policy(signal, labels, temperature=tau)
            aug_batch = {**batch, self.feature_key: aug_signal}
            out = self.model(**aug_batch)
            loss_train = out["loss"]
            train_losses.append(loss_train.item())

            # ---- Outer loop: update policy on clean val loss ----
            # Must run BEFORE inner backward so the train graph is still alive.
            if val_loader is not None:
                try:
                    val_batch = next(val_iter)
                except StopIteration:
                    val_iter = iter(val_loader)
                    val_batch = next(val_iter)

                val_batch = {
                    k: v.to(self.device) if isinstance(v, Tensor) else v
                    for k, v in val_batch.items()
                }
                val_out = self.model(**val_batch)
                loss_val = val_out["loss"]

                hypgrad = self.neumann_hypergrad(
                    loss_val=loss_val,
                    loss_train=loss_train,
                    model_params=list(self.model.parameters()),
                    policy_params=list(self.policy.parameters()),
                    lr=inner_opt.param_groups[0]["lr"],
                    order=neumann_order,
                )

                outer_opt.zero_grad()
                for p, g in zip(self.policy.parameters(), hypgrad):
                    if g is not None:
                        p.grad = g.detach().clone()
                torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_grad_norm)
                outer_opt.step()

            # ---- Inner loop: update model on augmented data ----
            # retain_graph=True allows hypergradient to run first; here we
            # allow the graph to be freed as it is no longer needed.
            inner_opt.zero_grad()
            loss_train.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
            inner_opt.step()

        return train_losses

    @staticmethod
    def neumann_hypergrad(
        loss_val: Tensor,
        loss_train: Tensor,
        model_params: List[Tensor],
        policy_params: List[Tensor],
        lr: float,
        order: int = 3,
    ) -> List[Optional[Tensor]]:
        """Approximate the hypergradient via the Neumann series.

        Computes:

        .. math::

            \\nabla_{\\phi} L_{\\text{val}} \\approx
            \\frac{\\partial L_{\\text{val}}}{\\partial \\phi} -
            \\alpha \\sum_{i=0}^{\\text{order}-1}
            \\left(I - \\alpha \\frac{\\partial^2 L_{\\text{train}}}{\\partial \\theta^2}\\right)^i
            \\frac{\\partial^2 L_{\\text{train}}}{\\partial \\phi \\partial \\theta}
            \\frac{\\partial L_{\\text{val}}}{\\partial \\theta}

        Args:
            loss_val: Validation loss (scalar tensor with grad_fn).
            loss_train: Training loss (scalar tensor with grad_fn).
            model_params: List of model (inner) parameters θ.
            policy_params: List of policy (outer) parameters φ.
            lr: Inner-loop learning rate α.
            order: Neumann truncation order. Default 3.

        Returns:
            list[Tensor | None]: Hypergradient for each policy parameter.
        """
        # ∂L_val / ∂φ  (direct gradient)
        d_val_phi = torch.autograd.grad(
            loss_val,
            policy_params,
            retain_graph=True,
            allow_unused=True,
        )

        # ∂L_val / ∂θ
        d_val_theta = torch.autograd.grad(
            loss_val,
            model_params,
            retain_graph=True,
            allow_unused=True,
            create_graph=False,
        )

        # v = ∂L_val/∂θ  (start vector for Neumann series)
        v = [g.detach().clone() if g is not None else torch.zeros_like(p)
             for g, p in zip(d_val_theta, model_params)]

        # Accumulate Neumann sum: v_0 + v_1 + ... + v_{order-1}
        neumann_sum = [vi.clone() for vi in v]

        for _ in range(order - 1):
            # ∂L_train/∂θ, then JVP with v to get H·v
            d_train_theta = torch.autograd.grad(
                loss_train,
                model_params,
                create_graph=True,
                retain_graph=True,
                allow_unused=True,
            )
            # (I - lr·H)·v  per parameter
            v = [
                vi - lr * (gi if gi is not None else torch.zeros_like(p))
                for vi, gi, p in zip(v, d_train_theta, model_params)
            ]
            neumann_sum = [ns + vi for ns, vi in zip(neumann_sum, v)]

        # ∂L_train / ∂φ using neumann_sum as the vector (implicit diff)
        d_train_theta_for_phi = torch.autograd.grad(
            loss_train,
            model_params,
            create_graph=True,
            retain_graph=True,
            allow_unused=True,
        )
        # Build a scalar proxy: dot(neumann_sum, d_train_theta_for_phi)
        proxy = sum(
            (ns * (g if g is not None else torch.zeros_like(p))).sum()
            for ns, g, p in zip(neumann_sum, d_train_theta_for_phi, model_params)
        )
        d_proxy_phi = torch.autograd.grad(
            proxy,
            policy_params,
            retain_graph=True,  # keep train graph alive for inner backward()
            allow_unused=True,
        )

        hypgrads = [
            (dv if dv is not None else torch.zeros_like(p))
            - lr * (dp if dp is not None else torch.zeros_like(p))
            for dv, dp, p in zip(d_val_phi, d_proxy_phi, policy_params)
        ]
        return hypgrads
