# Contributors: Akshad Pai, Matthew Ruth (NetIDs on PR)
# Paper: On the Importance of Step-wise Embeddings for Heterogeneous Clinical
# Time-Series (Kuznetsova et al., ML4H 2023)
# Paper URL: https://proceedings.mlr.press/v225/kuznetsova23a.html
# Dataset: MIMIC-III tables needed for ICU hourly lab series and decomp tasks.

"""MIMIC-III ICU + labevents slice for hourly multivariate series workflows."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

from .mimic3 import MIMIC3Dataset

if TYPE_CHECKING:
    from ..tasks import BaseTask

logger = logging.getLogger(__name__)


class MIMIC3ICUHourlyDataset(MIMIC3Dataset):
    """MIMIC-III loader preset for ICU stays with joined lab events.

    This subclasses :class:`MIMIC3Dataset` but pins ``config_path`` to
    ``mimic3_icu_hourly.yaml`` (patients, admissions, icustays, labevents) so
    contributors can build reproducibility pipelines around hourly lab grids
    without pulling unrelated tables by default.

    Args:
        root: Root directory of unpacked MIMIC-III CSV files (e.g. mimic-iii/1.4).
        tables: Extra tables beyond the ICU-hourly defaults; defaults to
            ``["labevents"]`` which together with core tables matches the yaml.
        dataset_name: Optional logical name (default ``mimic3_icu_hourly``).
        config_path: Optional override; normally the bundled ICU-hourly yaml.
    """

    def __init__(
        self,
        root: str,
        tables: Optional[List[str]] = None,
        dataset_name: Optional[str] = None,
        config_path: Optional[str] = None,
        **kwargs,
    ) -> None:
        if config_path is None:
            config_path = str(
                Path(__file__).resolve().parent / "configs" / "mimic3_icu_hourly.yaml"
            )
        if tables is None:
            tables = ["labevents"]
        super().__init__(
            root=root,
            tables=tables,
            dataset_name=dataset_name or "mimic3_icu_hourly",
            config_path=config_path,
            **kwargs,
        )

    @property
    def default_task(self) -> Optional["BaseTask"]:
        """Suggested pairing: 24h ICU lab grid with a decomp-style proxy label."""

        from pyhealth.tasks.mimic3_icu_decompensation_24h import (
            MIMIC3ICUDecompensation24h,
        )

        return MIMIC3ICUDecompensation24h()
