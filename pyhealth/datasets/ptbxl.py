"""PyHealth dataset for the PTB-XL electrocardiography dataset.

Dataset:
    PTB-XL, a large publicly available electrocardiography dataset.
    21,837 12-lead ECGs at 500 Hz (10 s), 18,869 unique patients.
    Available at: https://physionet.org/content/ptb-xl/1.0.3/
    No registration required.

Reference:
    Wagner, P., Strodthoff, N., Bousseljot, R.-D., et al. (2020).
    PTB-XL, a large publicly available electrocardiography dataset.
    Scientific Data, 7(1), 154. https://doi.org/10.1038/s41597-020-0495-6

Author:
    Paul Garcia (alanpg2@illinois.edu) — DL4H Spring 2026
"""

import ast
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd

from pyhealth.datasets import BaseDataset

logger = logging.getLogger(__name__)

_SUPERCLASSES = ("MI", "HYP", "STTC", "CD")


class PTBXLDataset(BaseDataset):
    """PyHealth 2.0 dataset class for PTB-XL ECG data.

    PTB-XL contains 21,837 12-lead, 10-second clinical ECG recordings from
    18,869 patients collected at the Physikalisch-Technische Bundesanstalt (PTB)
    between 1989 and 1996. Each recording is annotated with SCP-ECG statements
    which are mapped to five diagnostic superclasses (NORM, MI, HYP, STTC, CD).
    This dataset exposes the four pathological superclasses as binary labels,
    enabling four independent binary classification tasks.

    Dataset URL: https://physionet.org/content/ptb-xl/1.0.3/

    Args:
        root (str): Root directory containing the raw PTB-XL files, i.e. the
            directory that holds ``ptbxl_database.csv`` and ``scp_statements.csv``
            as well as the ``records500/`` waveform folder.
        dataset_name (Optional[str]): Name for the dataset instance. Defaults
            to ``"ptbxl"``.
        config_path (Optional[str]): Path to the YAML config file. If ``None``
            the bundled ``configs/ptbxl.yaml`` is used.
        dev (bool): When ``True`` only the first 1 000 patients are loaded
            (useful for rapid iteration). Defaults to ``False``.
        **kwargs: Additional keyword arguments forwarded to
            :class:`~pyhealth.datasets.BaseDataset`.

    Attributes:
        task (Optional[str]): Name of the active task. ``None`` before
            :meth:`set_task` is called.
        samples (Optional[list[dict]]): Task-specific sample list populated
            by :meth:`set_task`.
        patient_to_index (Optional[dict[str, list[int]]]): Maps ``patient_id``
            to sample indices populated by :meth:`set_task`.
        visit_to_index (Optional[dict[str, list[int]]]): Maps ``ecg_id`` to
            sample indices populated by :meth:`set_task`.

    Examples:
        >>> from pyhealth.datasets import PTBXLDataset
        >>> dataset = PTBXLDataset(
        ...     root="/path/to/ptb-xl/1.0.3",
        ... )
        >>> dataset.stats()

    Import line for ``pyhealth/datasets/__init__.py``::

        from .ptbxl import PTBXLDataset
    """

    def __init__(
        self,
        root: str,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        dev: bool = False,
        **kwargs,
    ) -> None:
        if config_path is None:
            config_path = str(Path(__file__).parent / "configs" / "ptbxl.yaml")

        metadata_path = os.path.join(root, "ptbxl-pyhealth.csv")
        if not os.path.exists(metadata_path):
            logger.info(
                "ptbxl-pyhealth.csv not found — running prepare_metadata()..."
            )
            self.prepare_metadata(root)

        super().__init__(
            root=root,
            tables=["records500"],
            dataset_name=dataset_name or "ptbxl",
            config_path=config_path,
            dev=dev,
            **kwargs,
        )

    # ------------------------------------------------------------------
    # Metadata preparation
    # ------------------------------------------------------------------

    @classmethod
    def prepare_metadata(cls, root: str) -> None:
        """Build ``ptbxl-pyhealth.csv`` from the raw PTB-XL metadata files.

        Reads ``ptbxl_database.csv`` and ``scp_statements.csv`` from *root*,
        resolves SCP diagnostic codes to the four binary superclass labels
        (MI, HYP, STTC, CD), and writes the result to
        ``<root>/ptbxl-pyhealth.csv``.

        Args:
            root (str): Root directory containing ``ptbxl_database.csv`` and
                ``scp_statements.csv``.

        Raises:
            FileNotFoundError: If either source CSV is missing from *root*.
        """
        db_path = os.path.join(root, "ptbxl_database.csv")
        scp_path = os.path.join(root, "scp_statements.csv")

        for p in (db_path, scp_path):
            if not os.path.isfile(p):
                raise FileNotFoundError(
                    f"Required file not found: {p}\n"
                    "Download PTB-XL from https://physionet.org/content/ptb-xl/1.0.3/"
                )

        db = pd.read_csv(db_path, index_col="ecg_id")
        scp = pd.read_csv(scp_path, index_col=0)

        # Map each SCP code that is diagnostic to its superclass
        diagnostic_mask = scp["diagnostic"] == 1.0
        sc_map: dict[str, str] = (
            scp.loc[diagnostic_mask, "diagnostic_class"].dropna().to_dict()
        )

        # Parse scp_codes column (stored as a stringified dict)
        db["_scp_dict"] = db["scp_codes"].apply(ast.literal_eval)

        for sc in _SUPERCLASSES:
            codes = {k for k, v in sc_map.items() if v == sc}
            db[f"superclass_{sc.lower()}"] = db["_scp_dict"].apply(
                lambda d, c=codes: int(any(k in d for k in c))
            )

        out = db.reset_index()[
            [
                "patient_id",
                "ecg_id",
                "filename_hr",
                "superclass_mi",
                "superclass_hyp",
                "superclass_sttc",
                "superclass_cd",
            ]
        ].copy()

        out["patient_id"] = out["patient_id"].astype(str)
        out["ecg_id"] = out["ecg_id"].astype(str)

        out_path = os.path.join(root, "ptbxl-pyhealth.csv")
        out.to_csv(out_path, index=False)
        logger.info(
            f"Wrote {len(out):,} records to {out_path} "
            f"(MI={out.superclass_mi.sum()}, HYP={out.superclass_hyp.sum()}, "
            f"STTC={out.superclass_sttc.sum()}, CD={out.superclass_cd.sum()})"
        )

    # ------------------------------------------------------------------
    # Default task
    # ------------------------------------------------------------------

    @property
    def default_task(self):
        """Returns the default task (MI binary classification).

        Returns:
            ECGBinaryClassificationPTBXL: MI task instance.
        """
        from pyhealth.tasks.ecg_classification_ptbxl import (
            ECGBinaryClassificationPTBXL,
        )

        return ECGBinaryClassificationPTBXL(task="MI")
