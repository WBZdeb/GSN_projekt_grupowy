from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

CellType = Literal["rnn", "gru", "lstm"]
Pooling = Literal["last", "attn"]


@dataclass
class ModelSpec:
    """Specyfikacja modelu sekwencyjnego."""
    cell_type: CellType = "gru"
    pooling: Pooling = "last"  # 'last' albo 'attn'
    input_size: int = 4
    hidden_size: int = 32
    num_layers: int = 1
    num_classes: int = 2
    dropout: float = 0.0


class SeqClassifier(nn.Module):
    """RNN/GRU/LSTM + pooling + linear head.

    pooling='last':
      klasyczne many‑to‑one: bierzesz tylko stan z ostatniego kroku h_T.

    pooling='attn':
      attention pooling: uczysz się wag alfa_t i bierzesz sumę Σ alfa_t h_t.
      To skraca drogę sygnału uczenia do wczesnych kroków (często pomaga w zadaniach pamięci).
    """

    def __init__(self, spec: ModelSpec):
        super().__init__()
        self.spec = spec

        if spec.cell_type == "rnn":
            self.rnn = nn.RNN(
                input_size=spec.input_size,
                hidden_size=spec.hidden_size,
                num_layers=spec.num_layers,
                batch_first=True,
                nonlinearity="tanh",
                dropout=spec.dropout if spec.num_layers > 1 else 0.0,
            )
        elif spec.cell_type == "gru":
            self.rnn = nn.GRU(
                input_size=spec.input_size,
                hidden_size=spec.hidden_size,
                num_layers=spec.num_layers,
                batch_first=True,
                dropout=spec.dropout if spec.num_layers > 1 else 0.0,
            )
        elif spec.cell_type == "lstm":
            self.rnn = nn.LSTM(
                input_size=spec.input_size,
                hidden_size=spec.hidden_size,
                num_layers=spec.num_layers,
                batch_first=True,
                dropout=spec.dropout if spec.num_layers > 1 else 0.0,
            )
        else:
            raise ValueError(f"Unknown cell_type: {spec.cell_type}")

        # Attention pooling: score_t = v^T tanh(W h_t)
        if spec.pooling == "attn":
            self.attn_W = nn.Linear(spec.hidden_size, spec.hidden_size)
            self.attn_v = nn.Linear(spec.hidden_size, 1, bias=False)

        self.head = nn.Linear(spec.hidden_size, spec.num_classes)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: (B, T, V)
        out, _ = self.rnn(x)  # out: (B, T, H)

        if self.spec.pooling == "last":
            h = out[:, -1, :]
            logits = self.head(h)
            return logits, None

        # Attention pooling
        scores = self.attn_v(torch.tanh(self.attn_W(out))).squeeze(-1)  # (B, T)
        alpha = F.softmax(scores, dim=1)  # (B, T)
        ctx = torch.sum(out * alpha.unsqueeze(-1), dim=1)  # (B, H)
        logits = self.head(ctx)
        return logits, alpha
