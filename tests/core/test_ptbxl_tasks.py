from __future__ import annotations

import sys
import unittest
from datetime import datetime
from unittest import mock

import numpy as np
import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks.ptbxl_tasks import (
    PTBXLCDTask,
    PTBXLHYPTask,
    PTBXLMITask,
    PTBXLSTTCTask,
)


class TestPTBXLTasks(unittest.TestCase):
    """Test PTB-XL binary ECG task classes."""

    def setUp(self):
        self.scp_statements = {
            "NORM": "NORM",
            "MI": "MI",
            "LVH": "HYP",
            "ISC_": "STTC",
            "CLBBB": "CD",
        }

        rng = np.random.RandomState(42)
        self.mock_signal = rng.randn(5000, 12).astype(
            np.float32
        )
        self.mock_fields = {"fs": 500}

        self.mock_wfdb = mock.MagicMock()
        self.mock_wfdb.rdsamp.return_value = (
            self.mock_signal,
            self.mock_fields,
        )

        ts = pl.Series(
            [datetime(2020, 1, 1), datetime(2020, 1, 2)],
            dtype=pl.Datetime("ms"),
        )
        rec_df = pl.DataFrame(
            {
                "patient_id": ["101", "101"],
                "event_type": ["recordings", "recordings"],
                "timestamp": ts,
                "recordings/ecg_id": ["1", "2"],
                "recordings/filename_hr": [
                    "/tmp/r1",
                    "/tmp/r2",
                ],
                "recordings/filename_lr": [
                    "/tmp/l1",
                    "/tmp/l2",
                ],
                "recordings/age": ["56", "56"],
                "recordings/sex": ["0", "0"],
            }
        )
        scp_df = pl.DataFrame(
            {
                "patient_id": ["101", "101"],
                "event_type": ["scp_codes", "scp_codes"],
                "timestamp": ts,
                "scp_codes/ecg_id": ["1", "2"],
                "scp_codes/scp_codes": [
                    "{'MI':80.0}",
                    "{'NORM':100.0}",
                ],
            }
        )
        combined = pl.concat(
            [rec_df, scp_df], how="diagonal"
        )
        self.patient = Patient(
            patient_id="101", data_source=combined
        )

    def _run(self, task):
        with mock.patch.dict(
            sys.modules, {"wfdb": self.mock_wfdb}
        ):
            return task(self.patient)

    def _assert_samples(self, samples):
        self.assertGreater(len(samples), 0)
        for s in samples:
            self.assertIn("patient_id", s)
            self.assertIn("visit_id", s)
            self.assertIn("waveform", s)
            self.assertIn("label", s)
            self.assertEqual(s["waveform"].shape, (12, 2500))
            self.assertIn(s["label"], (0, 1))
            wf = s["waveform"]
            np.testing.assert_allclose(
                wf.mean(axis=1), 0, atol=0.1
            )
            np.testing.assert_allclose(
                wf.std(axis=1), 1, atol=0.1
            )

    def test_mi_task(self):
        task = PTBXLMITask(
            scp_statements=self.scp_statements
        )
        samples = self._run(task)
        self._assert_samples(samples)
        labels = {
            s["visit_id"]: s["label"] for s in samples
        }
        self.assertEqual(labels["1"], 1)
        self.assertEqual(labels["2"], 0)

    def test_hyp_task(self):
        task = PTBXLHYPTask(
            scp_statements=self.scp_statements
        )
        samples = self._run(task)
        self._assert_samples(samples)

    def test_sttc_task(self):
        task = PTBXLSTTCTask(
            scp_statements=self.scp_statements
        )
        samples = self._run(task)
        self._assert_samples(samples)

    def test_cd_task(self):
        task = PTBXLCDTask(
            scp_statements=self.scp_statements
        )
        samples = self._run(task)
        self._assert_samples(samples)


if __name__ == "__main__":
    unittest.main()
