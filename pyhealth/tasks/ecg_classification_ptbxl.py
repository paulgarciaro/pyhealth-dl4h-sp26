"""PyHealth task for binary ECG classification on PTB-XL.

Exposes four independent binary classification tasks that mirror the
diagnostic superclasses used by Raghu et al. (CHIL 2022):

  * MI   — Myocardial Infarction
  * HYP  — Hypertrophy
  * STTC — ST/T-Change
  * CD   — Conduction Disturbance

Each task produces one sample per ECG recording. The 12-lead waveform is
loaded with ``wfdb``, downsampled from 500 Hz to 250 Hz, per-lead z-score
normalised, and returned as a ``torch.FloatTensor`` of shape ``(12, 2500)``.

Reference:
    Raghu, A., et al. (2022). Data Augmentation for Electrocardiograms.
    CHIL, PMLR 174. https://proceedings.mlr.press/v174/raghu22a.html

Author:
    Paul Garcia (alanpg2@illinois.edu) — DL4H Spring 2026
"""

import logging
import os
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from pyhealth.tasks import BaseTask

logger = logging.getLogger(__name__)

_VALID_TASKS = ("MI", "HYP", "STTC", "CD")
_LABEL_COLS = {
    "MI": "superclass_mi",
    "HYP": "superclass_hyp",
    "STTC": "superclass_sttc",
    "CD": "superclass_cd",
}


class ECGBinaryClassificationPTBXL(BaseTask):
    """Binary ECG classification task for PTB-XL.

    For each ECG recording the waveform is loaded, preprocessed to
    ``(12, 2500)`` and labelled with a binary flag for the requested
    diagnostic superclass.

    Attributes:
        task_name (str): ``"ECGBinaryClassificationPTBXL"``.
        input_schema (dict[str, str]): ``{"signal": "tensor"}``.
        output_schema (dict[str, str]): ``{"label": "binary"}``.

    Args:
        task (str): One of ``"MI"``, ``"HYP"``, ``"STTC"``, ``"CD"``.
        signal_length (int): Expected signal length after resampling. Default
            is ``2500`` (10 s × 250 Hz).
        fs_original (int): Original sampling frequency of PTB-XL 500 Hz
            recordings. Default is ``500``.
        fs_target (int): Target sampling frequency after downsampling. Default
            is ``250``.

    Raises:
        ValueError: If *task* is not one of the four valid superclass names.

    Examples:
        >>> from pyhealth.datasets import PTBXLDataset
        >>> from pyhealth.tasks import ECGBinaryClassificationPTBXL
        >>> dataset = PTBXLDataset(root="/path/to/ptb-xl/1.0.3")
        >>> mi_task = ECGBinaryClassificationPTBXL(task="MI")
        >>> sample_ds = dataset.set_task(mi_task)
        >>> sample_ds.samples[0]
        {'patient_id': '...', 'ecg_id': '...', 'signal': tensor(...), 'label': 0}

    Import line for ``pyhealth/tasks/__init__.py``::

        from .ecg_classification_ptbxl import (
            ECGBinaryClassificationPTBXL,
            MIClassificationPTBXL,
            HYPClassificationPTBXL,
            STTCClassificationPTBXL,
            CDClassificationPTBXL,
        )
    """

    task_name: str = "ECGBinaryClassificationPTBXL"
    input_schema: Dict[str, str] = {"signal": "tensor"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(
        self,
        task: str,
        signal_length: int = 2500,
        fs_original: int = 500,
        fs_target: int = 250,
    ) -> None:
        task = task.upper()
        if task not in _VALID_TASKS:
            raise ValueError(
                f"Invalid task '{task}'. Must be one of {_VALID_TASKS}."
            )
        self.task = task
        self.signal_length = signal_length
        self.fs_original = fs_original
        self.fs_target = fs_target
        self._label_col = _LABEL_COLS[task]
        super().__init__()

    def __call__(self, patient: Any) -> List[Dict]:
        """Process a single patient and return a list of ECG samples.

        Args:
            patient: PyHealth ``Patient`` object with events of type
                ``"records500"``.

        Returns:
            List[Dict]: Each dict contains:
                - ``patient_id`` (str)
                - ``ecg_id`` (str)
                - ``signal`` (torch.FloatTensor, shape ``(12, 2500)``)
                - ``label`` (int, 0 or 1)
        """
        import wfdb
        from scipy.signal import resample

        events = patient.get_events(event_type="records500")
        root = self._get_root(patient)
        samples: List[Dict] = []

        for event in events:
            try:
                filename_hr: str = event["filename_hr"]
                label: int = int(event[self._label_col])
            except (KeyError, TypeError):
                logger.warning(
                    f"Skipping event for patient {patient.patient_id}: "
                    "missing filename_hr or label attribute."
                )
                continue

            record_path = os.path.join(root, filename_hr) if root else filename_hr

            try:
                record = wfdb.rdrecord(record_path)
            except Exception as exc:
                logger.warning(
                    f"Could not load waveform '{record_path}': {exc}. Skipping."
                )
                continue

            # record.p_signal: (n_samples, n_leads) — e.g. (5000, 12)
            signal_np: np.ndarray = record.p_signal.T  # → (12, n_samples)

            # Replace NaN/Inf from bad leads with zeros
            if not np.isfinite(signal_np).all():
                signal_np = np.nan_to_num(signal_np, nan=0.0, posinf=0.0, neginf=0.0)

            # Downsample 500 → 250 Hz
            n_target = int(signal_np.shape[1] * self.fs_target / self.fs_original)
            signal_np = resample(signal_np, n_target, axis=1)

            # Crop / pad to exactly signal_length samples
            if signal_np.shape[1] > self.signal_length:
                signal_np = signal_np[:, : self.signal_length]
            elif signal_np.shape[1] < self.signal_length:
                pad = self.signal_length - signal_np.shape[1]
                signal_np = np.pad(signal_np, ((0, 0), (0, pad)))

            # Per-lead z-score normalisation
            mean = signal_np.mean(axis=1, keepdims=True)
            std = signal_np.std(axis=1, keepdims=True) + 1e-8
            signal_np = (signal_np - mean) / std

            signal_tensor = torch.from_numpy(signal_np.astype(np.float32))  # (12, 2500)

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "ecg_id": str(event["ecg_id"]),
                    "signal": signal_tensor,
                    "label": label,
                }
            )

        return samples

    @staticmethod
    def _get_root(patient: Any) -> Optional[str]:
        """Extract the dataset root directory from the patient's data source.

        Attempts to read ``patient.data_source`` (a Polars DataFrame) for the
        ``root`` column injected by :class:`~pyhealth.datasets.BaseDataset`, or
        falls back to ``None`` so callers can still form absolute paths.
        """
        try:
            return patient.data_source.select("root").row(0)[0]
        except Exception:
            return None


# ---------------------------------------------------------------------------
# Factory helpers — one per superclass
# ---------------------------------------------------------------------------

def MIClassificationPTBXL(**kwargs) -> ECGBinaryClassificationPTBXL:
    """Return a pre-configured MI binary classification task.

    Examples:
        >>> from pyhealth.tasks import MIClassificationPTBXL
        >>> task = MIClassificationPTBXL()
    """
    return ECGBinaryClassificationPTBXL(task="MI", **kwargs)


def HYPClassificationPTBXL(**kwargs) -> ECGBinaryClassificationPTBXL:
    """Return a pre-configured HYP binary classification task.

    Examples:
        >>> from pyhealth.tasks import HYPClassificationPTBXL
        >>> task = HYPClassificationPTBXL()
    """
    return ECGBinaryClassificationPTBXL(task="HYP", **kwargs)


def STTCClassificationPTBXL(**kwargs) -> ECGBinaryClassificationPTBXL:
    """Return a pre-configured STTC binary classification task.

    Examples:
        >>> from pyhealth.tasks import STTCClassificationPTBXL
        >>> task = STTCClassificationPTBXL()
    """
    return ECGBinaryClassificationPTBXL(task="STTC", **kwargs)


def CDClassificationPTBXL(**kwargs) -> ECGBinaryClassificationPTBXL:
    """Return a pre-configured CD binary classification task.

    Examples:
        >>> from pyhealth.tasks import CDClassificationPTBXL
        >>> task = CDClassificationPTBXL()
    """
    return ECGBinaryClassificationPTBXL(task="CD", **kwargs)
