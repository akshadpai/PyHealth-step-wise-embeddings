# Contributors: Akshad Pai, Matthew Ruth (NetIDs on PR)
# Paper: On the Importance of Step-wise Embeddings for Heterogeneous Clinical
# Time-Series (Kuznetsova et al., ML4H 2023)
# Paper URL: https://proceedings.mlr.press/v225/kuznetsova23a.html
# Task: 24h ICU lab window -> binary label for acute worsening in next 24h
# (simplified proxy inspired by decompensation / TLS-style MIMIC benchmarks).

"""ICU hourly lab series with a simplified 48h decompensation-style label."""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import polars as pl

from .base_task import BaseTask

# MIMIC-III itemids (labs) used as a compact heterogeneous panel.
DEFAULT_ITEMIDS: Tuple[int, ...] = (
    50813,  # lactate
    50912,  # creatinine
    51301,  # WBC
    51221,  # hemoglobin
)


def _parse_dt(val: Any) -> Optional[datetime]:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return None
    if isinstance(val, datetime):
        return val
    ts = pd.to_datetime(val, errors="coerce")
    if pd.isna(ts):
        return None
    return ts.to_pydatetime()


class MIMIC3ICUDecompensation24h(BaseTask):
    """Binary task on ICU stays using a 24h observation lab grid.

    For each ICU stay, take labs from ``intime`` to ``intime + 24h`` and build
    an hourly multivariate vector (forward-filled within the hour grid). The
    label is **1** if, between ``intime + 24h`` and ``intime + 48h`` (clipped to
    ``outtime``), lactate reaches **≥ 4.0** OR creatinine rises by **≥ 0.3**
    above the observation-window median. This is a **course-scale proxy** for
    acute deterioration; it is not a drop-in replacement for Harutyunyan labels.

    ``input_schema`` uses ``timeseries`` so :class:`TimeseriesProcessor` can
    resample to a uniform hourly grid for models such as :class:`FTTransformer`.

    Attributes:
        task_name: Logical task name.
        input_schema: Maps ``icu_hourly_labs`` to ``timeseries``.
        output_schema: Binary ``decompensation_24h`` label.
    """

    task_name: str = "MIMIC3ICUDecompensation24h"
    input_schema: Dict[str, str] = {"icu_hourly_labs": "timeseries"}
    output_schema: Dict[str, str] = {"decompensation_24h": "binary"}

    def __init__(
        self,
        itemids: Optional[Sequence[int]] = None,
        lactate_itemid: int = 50813,
        creatinine_itemid: int = 50912,
        lactate_threshold: float = 4.0,
        creatinine_delta: float = 0.3,
    ) -> None:
        """Args tune the proxy label; defaults match MIMIC-III itemids above."""
        super().__init__()
        self.itemids: Tuple[int, ...] = tuple(itemids) if itemids else DEFAULT_ITEMIDS
        self.lactate_itemid = lactate_itemid
        self.creatinine_itemid = creatinine_itemid
        self.lactate_threshold = lactate_threshold
        self.creatinine_delta = creatinine_delta
        self._itemid_to_idx = {int(i): k for k, i in enumerate(self.itemids)}

    def __call__(self, patient: Any) -> List[Dict[str, Any]]:
        samples: List[Dict[str, Any]] = []
        stays = patient.get_events(event_type="icustays")
        if not stays:
            return samples

        for stay in stays:
            intime = stay.timestamp
            outtime = _parse_dt(stay.attr_dict.get("outtime"))
            if outtime is None or intime is None or outtime <= intime:
                continue
            obs_end = intime + timedelta(hours=24)
            pred_end = min(intime + timedelta(hours=48), outtime)
            if pred_end <= obs_end:
                continue

            icustay_id = stay.attr_dict.get("icustay_id")
            hadm_id = stay.attr_dict.get("hadm_id")

            lab_df = patient.get_events(
                event_type="labevents",
                start=intime,
                end=pred_end,
                return_df=True,
            )
            if isinstance(lab_df, list) or len(lab_df) == 0:
                continue

            timestamps, values = self._build_hourly_series(
                lab_df, intime, obs_end, self.itemids, hadm_id
            )
            if timestamps is None or values is None:
                continue
            if values.shape[0] < 2:
                continue

            label = self._compute_label(
                lab_df,
                intime,
                obs_end,
                pred_end,
                values,
                hadm_id,
            )
            if label is None:
                continue

            samples.append(
                {
                    "patient_id": patient.patient_id,
                    "visit_id": str(icustay_id),
                    "hadm_id": str(hadm_id) if hadm_id is not None else "",
                    "icustay_id": str(icustay_id) if icustay_id is not None else "",
                    "icu_hourly_labs": (timestamps, values),
                    "decompensation_24h": int(label),
                }
            )
        return samples

    def _build_hourly_series(
        self,
        lab_df: pl.DataFrame,
        start: datetime,
        end: datetime,
        itemids: Sequence[int],
        hadm_id: Any,
    ) -> Tuple[Optional[List[datetime]], Optional[np.ndarray]]:
        """Aggregate labs into one value per hour per itemid (mean)."""
        n_hours = int((end - start) / timedelta(hours=1))
        if n_hours <= 0:
            return None, None
        f = len(itemids)
        grid = np.full((n_hours, f), np.nan, dtype=np.float64)
        times: List[datetime] = [start + timedelta(hours=h) for h in range(n_hours)]

        id_col = "labevents/itemid"
        val_col = "labevents/valuenum"
        hadm_col = "labevents/hadm_id"
        if id_col not in lab_df.columns or val_col not in lab_df.columns:
            return None, None

        for row in lab_df.iter_rows(named=True):
            if hadm_col in row and hadm_id is not None:
                row_h = row.get(hadm_col)
                if row_h is not None and str(row_h) != str(hadm_id):
                    continue
            ts = row.get("timestamp")
            if ts is None:
                continue
            if isinstance(ts, str):
                ts = _parse_dt(ts)
            if ts is None or ts < start or ts >= end:
                continue
            try:
                itemid = int(float(row[id_col]))
            except (TypeError, ValueError):
                continue
            if itemid not in self._itemid_to_idx:
                continue
            j = self._itemid_to_idx[itemid]
            raw = row.get(val_col)
            if raw is None or raw == "":
                continue
            try:
                v = float(raw)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v):
                continue
            hi = int((ts - start) / timedelta(hours=1))
            if hi < 0 or hi >= n_hours:
                continue
            prev = grid[hi, j]
            grid[hi, j] = v if np.isnan(prev) else 0.5 * (prev + v)

        for j in range(f):
            last = np.nan
            for i in range(n_hours):
                if np.isnan(grid[i, j]):
                    grid[i, j] = last
                else:
                    last = grid[i, j]
        if np.all(np.isnan(grid)):
            return None, None
        grid = np.nan_to_num(grid, nan=0.0)
        return times, grid

    def _compute_label(
        self,
        lab_df: pl.DataFrame,
        _intime: datetime,
        obs_end: datetime,
        pred_end: datetime,
        obs_values: np.ndarray,
        hadm_id: Any,
    ) -> Optional[int]:
        """Positive if lactate spike or creatinine jump in (obs_end, pred_end]."""
        id_col = "labevents/itemid"
        val_col = "labevents/valuenum"
        hadm_col = "labevents/hadm_id"
        if id_col not in lab_df.columns:
            return None

        cr_idx = self._itemid_to_idx.get(self.creatinine_itemid)
        lac_idx = self._itemid_to_idx.get(self.lactate_itemid)
        base_cr = None
        if cr_idx is not None:
            col = obs_values[:, cr_idx]
            finite = col[np.isfinite(col) & (col > 0)]
            if len(finite):
                base_cr = float(np.median(finite))

        max_lac = None
        max_cr = None
        for row in lab_df.iter_rows(named=True):
            if hadm_col in row and hadm_id is not None:
                row_h = row.get(hadm_col)
                if row_h is not None and str(row_h) != str(hadm_id):
                    continue
            ts = row.get("timestamp")
            if isinstance(ts, str):
                ts = _parse_dt(ts)
            if ts is None or ts <= obs_end or ts > pred_end:
                continue
            try:
                itemid = int(float(row[id_col]))
            except (TypeError, ValueError):
                continue
            raw = row.get(val_col)
            if raw is None or raw == "":
                continue
            try:
                v = float(raw)
            except (TypeError, ValueError):
                continue
            if not np.isfinite(v):
                continue
            if itemid == self.lactate_itemid:
                max_lac = v if max_lac is None else max(max_lac, v)
            if itemid == self.creatinine_itemid:
                max_cr = v if max_cr is None else max(max_cr, v)

        if max_lac is not None and max_lac >= self.lactate_threshold:
            return 1
        if (
            base_cr is not None
            and max_cr is not None
            and (max_cr - base_cr) >= self.creatinine_delta
        ):
            return 1
        return 0
