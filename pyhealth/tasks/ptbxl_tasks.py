from __future__ import annotations

import ast
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .base_task import BaseTask


class _PTBXLBinaryTask(BaseTask):
    """Base class for PTB-XL binary ECG classification tasks.

    Subclasses set ``_target_superclass`` to select which SCP
    diagnostic superclass is treated as the positive label.

    Args:
        scp_statements: Mapping from SCP code to its
            ``diagnostic_superclass`` string (e.g.
            ``{"MI": "MI", "LVH": "HYP", ...}``).  Mutually
            exclusive with *root*.
        root: Path to the PTB-XL root directory.  When provided
            (and *scp_statements* is ``None``), the mapping is
            loaded from ``scp_statements.csv``.
        target_rate: Target sampling rate in Hz after resampling.
            Defaults to ``250``.
    """

    _target_superclass: str
    input_schema: Dict[str, str] = {"waveform": "signal"}
    output_schema: Dict[str, str] = {"label": "binary"}

    def __init__(
        self,
        scp_statements: Optional[Dict[str, str]] = None,
        root: Optional[str] = None,
        target_rate: int = 250,
    ) -> None:
        self.target_rate = target_rate
        if scp_statements is not None:
            self._scp_statements = scp_statements
        elif root is not None:
            self._scp_statements = self._load_scp_statements(root)
        else:
            raise ValueError(
                "Provide either scp_statements dict or root path"
            )
        super().__init__()

    @staticmethod
    def _load_scp_statements(root: str) -> Dict[str, str]:
        """Load SCP-code-to-superclass mapping from CSV."""
        df = pd.read_csv(
            f"{root}/scp_statements.csv", index_col=0
        )
        return df["diagnostic_superclass"].dropna().to_dict()

    def __call__(
        self, patient: Any
    ) -> List[Dict[str, Any]]:
        try:
            import wfdb
        except ImportError:
            raise ImportError(
                "wfdb is required for PTB-XL tasks. "
                "Install it with: pip install wfdb"
            )
        try:
            from scipy.signal import resample
        except ImportError:
            raise ImportError(
                "scipy is required for PTB-XL tasks. "
                "Install it with: pip install scipy"
            )

        rec_events = patient.get_events(event_type="recordings")
        scp_events = patient.get_events(event_type="scp_codes")

        ecg_scp_map: Dict[str, str] = {}
        for ev in scp_events:
            ecg_scp_map[ev["ecg_id"]] = ev["scp_codes"]

        samples: List[Dict[str, Any]] = []
        for rec in rec_events:
            ecg_id = rec["ecg_id"]
            filename_hr = rec["filename_hr"]
            scp_str = ecg_scp_map.get(ecg_id, "{}")
            codes = ast.literal_eval(scp_str)

            label = int(
                any(
                    self._scp_statements.get(c)
                    == self._target_superclass
                    for c in codes
                )
            )

            signal, fields = wfdb.rdsamp(filename_hr)
            source_rate = fields.get("fs", 500)
            target_len = int(
                signal.shape[0] * self.target_rate / source_rate
            )
            if signal.shape[0] != target_len:
                signal = resample(signal, target_len, axis=0)

            waveform = signal.T.astype(np.float32)

            mean = waveform.mean(axis=1, keepdims=True)
            std = waveform.std(axis=1, keepdims=True)
            std[std == 0] = 1.0
            waveform = (waveform - mean) / std

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": str(ecg_id),
                    "waveform": waveform,
                    "label": label,
                }
            )

        return samples


class PTBXLMITask(_PTBXLBinaryTask):
    """Myocardial Infarction detection on PTB-XL.

    Binary classification: positive when any SCP code maps to the
    ``"MI"`` diagnostic superclass.

    Attributes:
        task_name: ``"ptbxl_mi"``
        input_schema: ``{"waveform": "signal"}``
        output_schema: ``{"label": "binary"}``
    """

    task_name: str = "ptbxl_mi"
    _target_superclass: str = "MI"


class PTBXLHYPTask(_PTBXLBinaryTask):
    """Hypertrophy detection on PTB-XL.

    Binary classification: positive when any SCP code maps to the
    ``"HYP"`` diagnostic superclass.

    Attributes:
        task_name: ``"ptbxl_hyp"``
        input_schema: ``{"waveform": "signal"}``
        output_schema: ``{"label": "binary"}``
    """

    task_name: str = "ptbxl_hyp"
    _target_superclass: str = "HYP"


class PTBXLSTTCTask(_PTBXLBinaryTask):
    """ST/T Change detection on PTB-XL.

    Binary classification: positive when any SCP code maps to the
    ``"STTC"`` diagnostic superclass.

    Attributes:
        task_name: ``"ptbxl_sttc"``
        input_schema: ``{"waveform": "signal"}``
        output_schema: ``{"label": "binary"}``
    """

    task_name: str = "ptbxl_sttc"
    _target_superclass: str = "STTC"


class PTBXLCDTask(_PTBXLBinaryTask):
    """Conduction Disturbance detection on PTB-XL.

    Binary classification: positive when any SCP code maps to the
    ``"CD"`` diagnostic superclass.

    Attributes:
        task_name: ``"ptbxl_cd"``
        input_schema: ``{"waveform": "signal"}``
        output_schema: ``{"label": "binary"}``
    """

    task_name: str = "ptbxl_cd"
    _target_superclass: str = "CD"
