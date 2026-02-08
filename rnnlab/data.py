from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import torch


Variant = Literal["distractors", "all-binary"]


@dataclass(frozen=True)
class TaskSpec:
    """Opis zadania.

    variant:
      - 'distractors': pierwszy token jest sygnałem 0/1, reszta to dystraktory 2/3
      - 'all-binary': cała sekwencja jest 0/1 (trudniej)
    """
    variant: Variant = "distractors"
    vocab_size: int = 4
    num_classes: int = 2


def _sample_tokens(rng: np.random.RandomState, *, seq_len: int, variant: Variant) -> np.ndarray:
    if variant == "distractors":
        first = rng.randint(0, 2, size=(1,))
        rest = rng.randint(2, 4, size=(seq_len - 1,))
        return np.concatenate([first, rest], axis=0)
    if variant == "all-binary":
        return rng.randint(0, 2, size=(seq_len,))
    raise ValueError(f"Unknown variant: {variant}")


def sample_batch(
    *,
    batch_size: int,
    seq_len: int,
    task: TaskSpec,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Zwraca (x, y).

    x: one-hot (B, T, V)
    y: etykieta to pierwszy symbol (B,)
    """
    rng = np.random.RandomState(seed)
    tokens = np.stack(
        [_sample_tokens(rng, seq_len=seq_len, variant=task.variant) for _ in range(batch_size)],
        axis=0,
    )
    y = tokens[:, 0].astype(np.int64)

    x = np.zeros((batch_size, seq_len, task.vocab_size), dtype=np.float32)
    x[np.arange(batch_size)[:, None], np.arange(seq_len)[None, :], tokens] = 1.0

    return torch.from_numpy(x).to(device), torch.from_numpy(y).to(device)


def make_fixed_testset(
    *,
    test_size: int,
    seq_len: int,
    task: TaskSpec,
    seed: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Stały testset = porównywalność seedów i optimizerów."""
    return sample_batch(batch_size=test_size, seq_len=seq_len, task=task, seed=seed, device=device)
