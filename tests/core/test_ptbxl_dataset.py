from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pyhealth.datasets import PTBXLDataset


class TestPTBXLDataset(unittest.TestCase):
    """Test PTB-XL dataset with synthetic test data."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.root = Path(self.temp_dir)
        self.num_patients = 3

        ptbxl_data = {
            "ecg_id": [1, 2, 3, 4, 5],
            "patient_id": [
                "P101", "P101", "P102", "P103", "P103",
            ],
            "filename_lr": [
                f"records100/00000/0000{i}_lr"
                for i in range(1, 6)
            ],
            "filename_hr": [
                f"records500/00000/0000{i}_hr"
                for i in range(1, 6)
            ],
            "scp_codes": [
                "{'NORM':100.0}",
                "{'MI':80.0}",
                "{'LVH':50.0}",
                "{'ISC_':100.0}",
                "{'CLBBB':100.0}",
            ],
            "age": [56, 56, 45, 60, 60],
            "sex": [0, 0, 1, 0, 0],
        }
        pd.DataFrame(ptbxl_data).to_csv(
            self.root / "ptbxl_database.csv", index=False
        )

        scp_df = pd.DataFrame(
            {
                "description": [
                    "normal ECG",
                    "myocardial infarction",
                    "left ventricular hypertrophy",
                    "ischemic ST-T changes",
                    "complete left bundle branch block",
                ],
                "diagnostic_superclass": [
                    "NORM",
                    "MI",
                    "HYP",
                    "STTC",
                    "CD",
                ],
            },
            index=pd.Index(
                ["NORM", "MI", "LVH", "ISC_", "CLBBB"],
                name="",
            ),
        )
        scp_df.to_csv(self.root / "scp_statements.csv")

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_dataset_initialization(self):
        dataset = PTBXLDataset(root=str(self.root))
        self.assertIsNotNone(dataset)
        self.assertEqual(dataset.dataset_name, "ptbxl")
        self.assertEqual(dataset.root, str(self.root))

    def test_patient_count(self):
        dataset = PTBXLDataset(root=str(self.root))
        self.assertEqual(
            len(dataset.unique_patient_ids), self.num_patients
        )

    def test_stats_method(self):
        dataset = PTBXLDataset(root=str(self.root))
        dataset.stats()

    def test_get_patient(self):
        dataset = PTBXLDataset(root=str(self.root))
        patient = dataset.get_patient("P101")
        self.assertIsNotNone(patient)
        self.assertEqual(patient.patient_id, "P101")

    def test_get_patient_not_found(self):
        dataset = PTBXLDataset(root=str(self.root))
        with self.assertRaises(AssertionError):
            dataset.get_patient("P999")

    def test_available_tables(self):
        dataset = PTBXLDataset(root=str(self.root))
        expected = ["recordings", "scp_codes"]
        self.assertEqual(
            sorted(dataset.tables), sorted(expected)
        )


if __name__ == "__main__":
    unittest.main()
