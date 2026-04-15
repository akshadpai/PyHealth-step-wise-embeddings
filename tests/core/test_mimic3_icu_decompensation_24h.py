"""Tests for MIMIC3ICUDecompensation24h (synthetic Patient only)."""

from datetime import datetime, timedelta
import unittest

import polars as pl

from pyhealth.data import Patient
from pyhealth.tasks.mimic3_icu_decompensation_24h import MIMIC3ICUDecompensation24h


def _make_patient_positive_lactate() -> Patient:
    """ICU stay 72h; lactate spike in prediction window -> label 1."""
    base = datetime(2199, 1, 1, 12, 0, 0)
    intime = base
    outtime = base + timedelta(hours=72)
    rows = [
        {
            "patient_id": "p1",
            "event_type": "icustays",
            "timestamp": intime,
            "icustays/icustay_id": "100",
            "icustays/hadm_id": "500",
            "icustays/outtime": outtime.isoformat(),
        },
    ]
    # observation window labs (low lactate)
    for h in [1, 5, 10]:
        rows.append(
            {
                "patient_id": "p1",
                "event_type": "labevents",
                "timestamp": intime + timedelta(hours=h),
                "labevents/itemid": "50813",
                "labevents/valuenum": "1.5",
                "labevents/hadm_id": "500",
            }
        )
    # spike after 24h
    rows.append(
        {
            "patient_id": "p1",
            "event_type": "labevents",
            "timestamp": intime + timedelta(hours=30),
            "labevents/itemid": "50813",
            "labevents/valuenum": "5.0",
            "labevents/hadm_id": "500",
        }
    )
    df = pl.DataFrame(rows)
    return Patient("p1", df)


class TestMIMIC3ICUDecompensation24h(unittest.TestCase):
    def test_positive_lactate(self) -> None:
        task = MIMIC3ICUDecompensation24h()
        patient = _make_patient_positive_lactate()
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["decompensation_24h"], 1)
        ts, arr = samples[0]["icu_hourly_labs"]
        self.assertEqual(arr.shape[1], len(task.itemids))
        self.assertGreater(len(ts), 5)

    def test_negative(self) -> None:
        base = datetime(2199, 2, 1, 8, 0, 0)
        outtime = base + timedelta(hours=72)
        rows = [
            {
                "patient_id": "p2",
                "event_type": "icustays",
                "timestamp": base,
                "icustays/icustay_id": "101",
                "icustays/hadm_id": "501",
                "icustays/outtime": outtime.isoformat(),
            },
            {
                "patient_id": "p2",
                "event_type": "labevents",
                "timestamp": base + timedelta(hours=2),
                "labevents/itemid": "50813",
                "labevents/valuenum": "1.1",
                "labevents/hadm_id": "501",
            },
            {
                "patient_id": "p2",
                "event_type": "labevents",
                "timestamp": base + timedelta(hours=30),
                "labevents/itemid": "50813",
                "labevents/valuenum": "1.2",
                "labevents/hadm_id": "501",
            },
        ]
        patient = Patient("p2", pl.DataFrame(rows))
        task = MIMIC3ICUDecompensation24h()
        samples = task(patient)
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["decompensation_24h"], 0)


if __name__ == "__main__":
    unittest.main()
