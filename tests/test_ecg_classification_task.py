"""Unit tests for ECGBinaryClassificationPTBXL task.

All tests use mocked wfdb and synthetic Patient objects — no real PTB-XL
waveforms are loaded. Tests are CPU-only and complete in < 5 s.

Run with:
    python -m pytest tests/test_ecg_classification_task.py -v
"""

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch


def _make_mock_patient(task: str = "MI") -> MagicMock:
    """Build a minimal mock Patient with one synthetic records500 event."""
    patient = MagicMock()
    patient.patient_id = "42"

    label_col = {
        "MI": "superclass_mi",
        "HYP": "superclass_hyp",
        "STTC": "superclass_sttc",
        "CD": "superclass_cd",
    }[task]

    event = MagicMock()
    event.__getitem__ = lambda self, key: {
        "filename_hr": "records500/00000/00001_hr",
        "ecg_id": "1",
        label_col: 1,
        "superclass_mi": 1,
        "superclass_hyp": 0,
        "superclass_sttc": 0,
        "superclass_cd": 0,
    }.get(key, 0)

    patient.get_events.return_value = [event]

    # data_source.select("root").row(0)[0] → dummy root
    ds = MagicMock()
    ds.select.return_value.row.return_value = ("/tmp/ptbxl",)
    patient.data_source = ds

    return patient


def _make_mock_record(n_samples: int = 5000, n_leads: int = 12) -> MagicMock:
    """Return a fake wfdb Record with synthetic p_signal."""
    record = MagicMock()
    record.p_signal = np.random.randn(n_samples, n_leads).astype(np.float32)
    return record


class TestECGBinaryClassificationPTBXL(unittest.TestCase):
    """Tests for the ECGBinaryClassificationPTBXL callable class."""

    @patch("wfdb.rdrecord")
    def test_mi_task_returns_samples(self, mock_rdrecord: MagicMock) -> None:
        """__call__ returns a non-empty list of dicts for a valid patient."""
        from pyhealth.tasks.ecg_classification_ptbxl import ECGBinaryClassificationPTBXL

        mock_rdrecord.return_value = _make_mock_record()
        task = ECGBinaryClassificationPTBXL(task="MI")
        patient = _make_mock_patient("MI")
        samples = task(patient)

        self.assertIsInstance(samples, list)
        self.assertGreater(len(samples), 0)

    @patch("wfdb.rdrecord")
    def test_sample_keys(self, mock_rdrecord: MagicMock) -> None:
        """Each returned sample contains the required keys."""
        from pyhealth.tasks.ecg_classification_ptbxl import ECGBinaryClassificationPTBXL

        mock_rdrecord.return_value = _make_mock_record()
        task = ECGBinaryClassificationPTBXL(task="MI")
        patient = _make_mock_patient("MI")
        samples = task(patient)

        required_keys = {"patient_id", "ecg_id", "signal", "label"}
        for sample in samples:
            self.assertEqual(required_keys, set(sample.keys()))

    @patch("wfdb.rdrecord")
    def test_signal_shape(self, mock_rdrecord: MagicMock) -> None:
        """Signal tensor has shape (12, 2500) after downsampling."""
        from pyhealth.tasks.ecg_classification_ptbxl import ECGBinaryClassificationPTBXL

        mock_rdrecord.return_value = _make_mock_record(n_samples=5000, n_leads=12)
        task = ECGBinaryClassificationPTBXL(task="MI")
        patient = _make_mock_patient("MI")
        samples = task(patient)

        signal = samples[0]["signal"]
        self.assertIsInstance(signal, torch.Tensor)
        self.assertEqual(signal.shape, (12, 2500))

    @patch("wfdb.rdrecord")
    def test_signal_dtype_float32(self, mock_rdrecord: MagicMock) -> None:
        """Signal tensor uses float32 dtype."""
        from pyhealth.tasks.ecg_classification_ptbxl import ECGBinaryClassificationPTBXL

        mock_rdrecord.return_value = _make_mock_record()
        task = ECGBinaryClassificationPTBXL(task="HYP")
        patient = _make_mock_patient("HYP")
        samples = task(patient)
        self.assertEqual(samples[0]["signal"].dtype, torch.float32)

    @patch("wfdb.rdrecord")
    def test_label_is_binary_int(self, mock_rdrecord: MagicMock) -> None:
        """Label value is 0 or 1."""
        from pyhealth.tasks.ecg_classification_ptbxl import ECGBinaryClassificationPTBXL

        mock_rdrecord.return_value = _make_mock_record()
        task = ECGBinaryClassificationPTBXL(task="MI")
        patient = _make_mock_patient("MI")
        samples = task(patient)
        for s in samples:
            self.assertIn(s["label"], (0, 1))

    def test_invalid_task_raises_value_error(self) -> None:
        """ECGBinaryClassificationPTBXL raises ValueError for unknown tasks."""
        from pyhealth.tasks.ecg_classification_ptbxl import ECGBinaryClassificationPTBXL

        with self.assertRaises(ValueError):
            ECGBinaryClassificationPTBXL(task="INVALID")

    def test_case_insensitive_task_name(self) -> None:
        """Task name is normalised to upper-case; lowercase input is accepted."""
        from pyhealth.tasks.ecg_classification_ptbxl import ECGBinaryClassificationPTBXL

        task = ECGBinaryClassificationPTBXL(task="mi")
        self.assertEqual(task.task, "MI")

    @patch("wfdb.rdrecord", side_effect=Exception("file not found"))
    def test_missing_waveform_skipped(self, _mock: MagicMock) -> None:
        """Records whose waveforms cannot be loaded are silently skipped."""
        from pyhealth.tasks.ecg_classification_ptbxl import ECGBinaryClassificationPTBXL

        task = ECGBinaryClassificationPTBXL(task="MI")
        patient = _make_mock_patient("MI")
        samples = task(patient)
        self.assertEqual(samples, [])

    def test_factory_functions_instantiate(self) -> None:
        """All four factory helper functions return a valid task object."""
        from pyhealth.tasks.ecg_classification_ptbxl import (
            CDClassificationPTBXL,
            ECGBinaryClassificationPTBXL,
            HYPClassificationPTBXL,
            MIClassificationPTBXL,
            STTCClassificationPTBXL,
        )

        for fn, expected_task in [
            (MIClassificationPTBXL, "MI"),
            (HYPClassificationPTBXL, "HYP"),
            (STTCClassificationPTBXL, "STTC"),
            (CDClassificationPTBXL, "CD"),
        ]:
            task = fn()
            self.assertIsInstance(task, ECGBinaryClassificationPTBXL)
            self.assertEqual(task.task, expected_task)

    def test_input_output_schema(self) -> None:
        """Task declares the correct input and output schemas."""
        from pyhealth.tasks.ecg_classification_ptbxl import ECGBinaryClassificationPTBXL

        task = ECGBinaryClassificationPTBXL(task="CD")
        self.assertEqual(task.input_schema, {"signal": "tensor"})
        self.assertEqual(task.output_schema, {"label": "binary"})


if __name__ == "__main__":
    unittest.main()
