"""Unit tests for PTBXLDataset.

Uses synthetic data only — no real PTB-XL files required.
All tests are CPU-only and complete in < 5 s.

Run with:
    python -m pytest tests/test_ptbxl_dataset.py -v
"""

import os
import tempfile
import unittest

import pandas as pd


def _write_synthetic_csvs(root: str) -> None:
    """Write minimal synthetic ptbxl_database.csv and scp_statements.csv."""
    # --- scp_statements.csv ---
    scp = pd.DataFrame(
        {
            "description": ["Myocardial infarction", "Hypertrophy", "ST change", "Conduction"],
            "diagnostic": [1.0, 1.0, 1.0, 1.0],
            "form": [None, None, None, None],
            "rhythm": [None, None, None, None],
            "diagnostic_class": ["MI", "HYP", "STTC", "CD"],
            "diagnostic_subclass": [None, None, None, None],
        },
        index=pd.Index(["IMI", "LVH", "NDT", "CRBBB"], name=None),
    )
    scp.index.name = None  # match real file — unnamed first column
    scp.to_csv(os.path.join(root, "scp_statements.csv"))

    # --- ptbxl_database.csv ---
    records = []
    for ecg_id in range(1, 6):  # 5 synthetic patients
        records.append(
            {
                "ecg_id": ecg_id,
                "patient_id": ecg_id,
                "age": 50 + ecg_id,
                "sex": ecg_id % 2,
                "filename_lr": f"records100/00000/0000{ecg_id}_lr",
                "filename_hr": f"records500/00000/0000{ecg_id}_hr",
                "scp_codes": str({"IMI": 100.0} if ecg_id % 2 == 0 else {"LVH": 80.0}),
                "strat_fold": ecg_id,
            }
        )
    db = pd.DataFrame(records)
    db.to_csv(os.path.join(root, "ptbxl_database.csv"), index=False)


class TestPTBXLDatasetPrepareMetadata(unittest.TestCase):
    """Tests for PTBXLDataset.prepare_metadata()."""

    def setUp(self) -> None:
        self.tmp = tempfile.mkdtemp()
        _write_synthetic_csvs(self.tmp)

    def test_prepare_metadata_creates_csv(self) -> None:
        """prepare_metadata() writes ptbxl-pyhealth.csv to root."""
        from pyhealth.datasets.ptbxl import PTBXLDataset

        PTBXLDataset.prepare_metadata(self.tmp)
        out_path = os.path.join(self.tmp, "ptbxl-pyhealth.csv")
        self.assertTrue(os.path.isfile(out_path), "ptbxl-pyhealth.csv not found")

    def test_prepare_metadata_columns(self) -> None:
        """Output CSV contains all required columns with correct dtypes."""
        from pyhealth.datasets.ptbxl import PTBXLDataset

        PTBXLDataset.prepare_metadata(self.tmp)
        df = pd.read_csv(os.path.join(self.tmp, "ptbxl-pyhealth.csv"))

        expected_cols = {
            "patient_id",
            "ecg_id",
            "filename_hr",
            "superclass_mi",
            "superclass_hyp",
            "superclass_sttc",
            "superclass_cd",
        }
        self.assertEqual(expected_cols, set(df.columns))

    def test_prepare_metadata_row_count(self) -> None:
        """Output CSV has the same number of rows as ptbxl_database.csv."""
        from pyhealth.datasets.ptbxl import PTBXLDataset

        PTBXLDataset.prepare_metadata(self.tmp)
        df_out = pd.read_csv(os.path.join(self.tmp, "ptbxl-pyhealth.csv"))
        df_in = pd.read_csv(os.path.join(self.tmp, "ptbxl_database.csv"))
        self.assertEqual(len(df_out), len(df_in))

    def test_prepare_metadata_binary_labels(self) -> None:
        """Binary superclass columns contain only {0, 1}."""
        from pyhealth.datasets.ptbxl import PTBXLDataset

        PTBXLDataset.prepare_metadata(self.tmp)
        df = pd.read_csv(os.path.join(self.tmp, "ptbxl-pyhealth.csv"))
        for col in ["superclass_mi", "superclass_hyp", "superclass_sttc", "superclass_cd"]:
            unique_vals = set(df[col].unique())
            self.assertTrue(unique_vals.issubset({0, 1}), f"{col} contains non-binary values: {unique_vals}")

    def test_missing_source_file_raises(self) -> None:
        """FileNotFoundError raised when ptbxl_database.csv is absent."""
        from pyhealth.datasets.ptbxl import PTBXLDataset

        bad_root = tempfile.mkdtemp()  # empty directory
        with self.assertRaises(FileNotFoundError):
            PTBXLDataset.prepare_metadata(bad_root)

    def test_idempotent(self) -> None:
        """Running prepare_metadata() twice does not raise an error."""
        from pyhealth.datasets.ptbxl import PTBXLDataset

        PTBXLDataset.prepare_metadata(self.tmp)
        PTBXLDataset.prepare_metadata(self.tmp)  # second call — overwrites existing file
        self.assertTrue(os.path.isfile(os.path.join(self.tmp, "ptbxl-pyhealth.csv")))


if __name__ == "__main__":
    unittest.main()
