from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import narwhals as pl

from pyhealth.tasks.ptbxl_tasks import PTBXLMITask
from .base_dataset import BaseDataset

logger = logging.getLogger(__name__)


class PTBXLDataset(BaseDataset):
    """PTB-XL ECG Dataset.

    Dataset URL: https://physionet.org/content/ptb-xl/1.0.3/

    PTB-XL is a large publicly available electrocardiography dataset
    containing 21,837 clinical 12-lead ECG records from 18,885
    patients of 10-second length.  The raw waveforms are stored as
    WFDB ``.hea`` / ``.dat`` pairs at 100 Hz and 500 Hz.

    Args:
        root: Root directory containing the PTB-XL dataset files
            (must include ``ptbxl_database.csv`` and the
            ``records500/`` folder).
        dataset_name: Optional dataset name, defaults to
            ``"ptbxl"``.
        config_path: Optional path to a YAML config file.  When
            ``None``, the bundled ``ptbxl.yaml`` is used.
        sampling_rate: Target sampling rate in Hz after
            resampling.  The TaskAug paper uses 250 Hz (down from
            500 Hz).  Defaults to ``250``.

    Attributes:
        task: The current task set on this dataset.
        samples: The processed sample dataset after calling
            :meth:`set_task`.
        patient_to_index: Mapping from patient ID to sample
            indices.
        visit_to_index: Mapping from visit (ECG) ID to sample
            indices.

    Example:
        >>> from pyhealth.datasets import PTBXLDataset
        >>> dataset = PTBXLDataset(
        ...     root="/path/to/ptb-xl/1.0.3"
        ... )
        >>> dataset.stats()
    """

    def __init__(
        self,
        root: str,
        dataset_name: str | None = None,
        config_path: str | None = None,
        sampling_rate: int = 250,
        **kwargs,
    ) -> None:
        if config_path is None:
            logger.info(
                "No config path provided, using default config"
            )
            config_path = (
                Path(__file__).parent / "configs" / "ptbxl.yaml"
            )

        self.sampling_rate = sampling_rate
        default_tables = ["recordings", "scp_codes"]

        super().__init__(
            root=root,
            tables=default_tables,
            dataset_name=dataset_name or "ptbxl",
            config_path=config_path,
            **kwargs,
        )

    def preprocess_recordings(
        self, df: pl.LazyFrame
    ) -> pl.LazyFrame:
        """Prepend the dataset root to waveform file paths."""
        root = str(Path(self.root).resolve())
        for col in ("filename_lr", "filename_hr"):
            if col in df.collect_schema().names():
                df = df.with_columns(
                    pl.concat_str(
                        [
                            pl.lit(root),
                            pl.lit(os.sep),
                            pl.col(col),
                        ]
                    ).alias(col)
                )
        return df

    @staticmethod
    def normalize_waveform(waveform):
        """Apply per-lead z-score normalization.

        Args:
            waveform: Array of shape ``(leads, samples)``.

        Returns:
            Normalized array with per-lead mean 0 and std 1.
        """
        import numpy as np

        waveform = waveform.astype(np.float32)
        mean = waveform.mean(axis=1, keepdims=True)
        std = waveform.std(axis=1, keepdims=True)
        std[std == 0] = 1.0
        return (waveform - mean) / std

    @property
    def default_task(self) -> PTBXLMITask:
        """Returns the default MI classification task.

        Returns:
            PTBXLMITask: Task instance configured with the
                ``scp_statements.csv`` from the dataset root.
        """
        return PTBXLMITask(root=self.root)
