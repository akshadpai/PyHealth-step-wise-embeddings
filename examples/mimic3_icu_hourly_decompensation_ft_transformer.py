# Contributors: Akshad Pai, Matthew Ruth (NetIDs on PR)
# Paper: Kuznetsova et al., ML4H 2023 — step-wise embeddings for clinical series.
# Ablation: compare feature_grouping True vs False; compare n_heads in {1, 2}.

"""Minimal end-to-end demo on synthetic hourly lab tensors (no real MIMIC I/O).

For full MIMIC-III runs, point ``MIMIC3ICUHourlyDataset(root=...)`` at a
PhysioNet download and keep ``dev=True`` while iterating.
"""

from __future__ import annotations

import os
import tempfile
from datetime import datetime, timedelta

import numpy as np
import torch
from torch.utils.data import DataLoader

from pyhealth.datasets import InMemorySampleDataset
from pyhealth.models.ft_transformer import FTTransformer
from pyhealth.tasks.mimic3_icu_decompensation_24h import MIMIC3ICUDecompensation24h


def _synth_samples(n: int = 8, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    t0 = datetime(2200, 1, 1, 0, 0, 0)
    out = []
    for i in range(n):
        times = [t0 + timedelta(hours=h) for h in range(8)]
        x = rng.normal(size=(8, 4)).astype(np.float32)
        y = int(rng.random() > 0.5)
        out.append(
            {
                "patient_id": f"s{i}",
                "visit_id": str(i),
                "icu_hourly_labs": (times, x),
                "decompensation_24h": y,
            }
        )
    return out


def _run_epoch(model: FTTransformer, loader: DataLoader, device: torch.device) -> float:
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for batch in loader:
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        opt.zero_grad()
        o = model(**batch)
        o["loss"].backward()
        opt.step()
        losses.append(float(o["loss"].detach().cpu()))
    return sum(losses) / max(len(losses), 1)


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    task = MIMIC3ICUDecompensation24h()
    samples = _synth_samples(16, seed=1)
    ds = InMemorySampleDataset(
        samples=samples,
        input_schema=task.input_schema,
        output_schema=task.output_schema,
        dataset_name="synth_mimic3_icu",
        task_name=task.task_name,
    )
    loader = DataLoader(ds, batch_size=4, shuffle=True)

    results = {}
    for grouping in (True, False):
        for heads in (1, 2):
            m = FTTransformer(
                ds,
                hidden_dim=16,
                n_heads=heads,
                depth=1,
                dropout=0.0,
                feature_grouping=grouping,
            ).to(device)
            loss = _run_epoch(m, loader, device)
            key = f"group={grouping}_heads={heads}"
            results[key] = loss
            print(f"{key}: mean_loss={loss:.4f}")

    tmp = tempfile.mkdtemp(prefix="pyhealth_ftt_demo_")
    path = os.path.join(tmp, "results.txt")
    with open(path, "w", encoding="utf-8") as f:
        for k, v in results.items():
            f.write(f"{k}\t{v:.6f}\n")
    print(f"Wrote {path}")


if __name__ == "__main__":
    main()
