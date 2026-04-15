"""Fast tests for FTTransformer on tiny synthetic SampleDataset."""

import unittest
from datetime import datetime, timedelta

import numpy as np
import torch

from pyhealth.datasets import InMemorySampleDataset
from pyhealth.models.ft_transformer import FTTransformer
from pyhealth.tasks.base_task import BaseTask


class _TinyTSTask(BaseTask):
    task_name = "tiny_ts"
    input_schema = {"x": "timeseries"}
    output_schema = {"y": "binary"}

    def __call__(self, patient):
        return []


class TestFTTransformer(unittest.TestCase):
    def setUp(self) -> None:
        t0 = datetime(2200, 1, 1, 0, 0, 0)
        times = [t0 + timedelta(hours=i) for i in range(4)]
        arr = np.random.randn(4, 3).astype(np.float32)
        samples = [
            {"patient_id": "a", "visit_id": "0", "x": (times, arr), "y": 1},
            {"patient_id": "b", "visit_id": "1", "x": (times, arr * 0.1), "y": 0},
        ]
        self.ds = InMemorySampleDataset(
            samples=samples,
            input_schema=_TinyTSTask.input_schema,
            output_schema=_TinyTSTask.output_schema,
        )

    def test_forward_backward_grouped(self) -> None:
        m = FTTransformer(self.ds, hidden_dim=8, n_heads=2, depth=1, dropout=0.0)
        batch = {
            "x": torch.stack([self.ds[0]["x"], self.ds[1]["x"]]),
            "y": torch.tensor([[1.0], [0.0]]),
        }
        out = m(**batch)
        self.assertIn("loss", out)
        out["loss"].backward()

    def test_direct_mode(self) -> None:
        m = FTTransformer(
            self.ds,
            hidden_dim=8,
            n_heads=1,
            depth=1,
            dropout=0.0,
            feature_grouping=False,
        )
        batch = {
            "x": torch.stack([self.ds[0]["x"], self.ds[1]["x"]]),
            "y": torch.tensor([[1.0], [0.0]]),
        }
        out = m(**batch)
        self.assertEqual(out["logit"].shape, (2, 1))


if __name__ == "__main__":
    unittest.main()
