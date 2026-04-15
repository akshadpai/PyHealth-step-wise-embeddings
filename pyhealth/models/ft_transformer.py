# Contributors: Akshad Pai, Matthew Ruth (NetIDs on PR)
# Paper: On the Importance of Step-wise Embeddings for Heterogeneous Clinical
# Time-Series (Kuznetsova et al., ML4H 2023)
# Paper URL: https://proceedings.mlr.press/v225/kuznetsova23a.html
# Model: Feature-wise linear tokenizer + temporal Transformer (FT-style).

"""Feature tokenizer + Transformer encoder for uniform multivariate time grids."""

from __future__ import annotations

from typing import Dict, cast

import torch
import torch.nn as nn

from pyhealth.datasets import SampleDataset

from .base_model import BaseModel
from .transformer import TransformerBlock


class FTTransformer(BaseModel):
    """Feature-tokenizer Transformer for continuous ICU hourly grids.

    Expects a single ``timeseries`` input (post-:class:`TimeseriesProcessor`) of
    shape ``[batch, time, n_features]``. Each scalar channel is lifted with a
    **per-feature** linear (Gorishniy-style) when ``feature_grouping`` is True,
    or a **shared** scalar linear when False ("direct" pooling). Embeddings are
    summed across features to form a step-wise representation, then a stack of
    :class:`TransformerBlock` modules models temporal interactions.

    Args:
        dataset: Fitted :class:`SampleDataset` with one ``timeseries`` feature
            and a single binary label.
        hidden_dim: Token dimension ``d``.
        n_heads: Attention heads on the temporal Transformer.
        depth: Number of temporal Transformer blocks.
        dropout: Dropout inside attention and feed-forward blocks.
        feature_grouping: If True, each feature uses its own ``Linear(1, d)``.
            If False, one shared ``Linear(1, d)`` maps every channel (direct).
    """

    def __init__(
        self,
        dataset: SampleDataset,
        hidden_dim: int = 32,
        n_heads: int = 2,
        depth: int = 1,
        dropout: float = 0.1,
        feature_grouping: bool = True,
    ) -> None:
        super().__init__(dataset=dataset)
        if len(self.feature_keys) != 1:
            raise ValueError("FTTransformer expects exactly one input feature key.")
        if len(self.label_keys) != 1:
            raise ValueError("FTTransformer expects exactly one label key.")
        self.feature_key = self.feature_keys[0]
        self.label_key = self.label_keys[0]
        self.mode = self.dataset.output_schema[self.label_key]

        proc = self.dataset.input_processors[self.feature_key]
        n_features = proc.size()
        if n_features is None or n_features < 1:
            raise ValueError("TimeseriesProcessor must be fitted (n_features).")

        self.hidden_dim = hidden_dim
        self.n_features = int(n_features)
        self.feature_grouping = feature_grouping
        if depth < 1:
            raise ValueError("depth must be >= 1.")

        if feature_grouping:
            self.feature_linears = nn.ModuleList(
                [nn.Linear(1, hidden_dim) for _ in range(self.n_features)]
            )
        else:
            self.direct_linear = nn.Linear(1, hidden_dim)

        self.temporal_blocks = nn.ModuleList(
            [
                TransformerBlock(hidden_dim, n_heads, dropout)
                for _ in range(depth)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        out_dim = self.get_output_size()
        self.head = nn.Linear(hidden_dim, out_dim)

    def _tokenize(self, x: torch.Tensor) -> torch.Tensor:
        """Map ``[B, T, F]`` -> ``[B, T, hidden]``."""
        b, t, f = x.shape
        if f != self.n_features:
            raise ValueError(f"Expected {self.n_features} features, got {f}.")
        if self.feature_grouping:
            pieces = [
                self.feature_linears[j](x[..., j : j + 1]) for j in range(f)
            ]
        else:
            pieces = [self.direct_linear(x[..., j : j + 1]) for j in range(f)]
        stacked = torch.stack(pieces, dim=2)
        return stacked.sum(dim=2)

    def _time_mask(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, T]`` float mask, 1 for valid steps."""
        m = x.abs().sum(dim=-1) > 1e-6
        if not m.any(dim=1).all():
            m[:, 0] = True
        return m.float()

    def forward(
        self,
        **kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        x = kwargs[self.feature_key]
        if isinstance(x, tuple):
            x = x[0]
        x = cast(torch.Tensor, x).to(self.device).float()

        h = self._tokenize(x)
        h = self.dropout(h)
        mask = self._time_mask(x)
        attn_mask = torch.einsum("bt,bu->btu", mask, mask)

        for blk in self.temporal_blocks:
            h = blk(h, attn_mask, register_hook=False)

        denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        pooled = (h * mask.unsqueeze(-1)).sum(dim=1) / denom
        logits = self.head(pooled)
        y_prob = self.prepare_y_prob(logits)
        out: Dict[str, torch.Tensor] = {"logit": logits, "y_prob": y_prob}
        if self.label_key in kwargs:
            y_true = kwargs[self.label_key].to(self.device).float()
            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(-1)
            loss = self.get_loss_function()(logits, y_true)
            out["loss"] = loss
            out["y_true"] = y_true
        return out
