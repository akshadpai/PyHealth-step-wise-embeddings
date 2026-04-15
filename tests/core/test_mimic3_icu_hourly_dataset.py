"""Lightweight tests for MIMIC3ICUHourlyDataset config wiring."""

import tempfile
import unittest
from pathlib import Path

from pyhealth.datasets import MIMIC3ICUHourlyDataset


class TestMIMIC3ICUHourlyDataset(unittest.TestCase):
    def test_default_config_exists(self) -> None:
        cfg = (
            Path(__file__).resolve().parents[2]
            / "pyhealth"
            / "datasets"
            / "configs"
            / "mimic3_icu_hourly.yaml"
        )
        self.assertTrue(cfg.is_file())

    def test_init_does_not_touch_parquet(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            d = MIMIC3ICUHourlyDataset(root=tmp, tables=["labevents"])
            self.assertEqual(d.dataset_name, "mimic3_icu_hourly")
            self.assertIn("labevents", d.tables)


if __name__ == "__main__":
    unittest.main()
