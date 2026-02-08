import os
import random
from typing import Any, Dict

import numpy as np
import torch


def set_seed(seed: int) -> None:
    """Ustaw wszystkie ziarna losowości (reproducibility)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(prefer: str = "auto") -> torch.device:
    """Wybierz device.

    - prefer='auto'  -> cuda jeśli jest, inaczej cpu
    - prefer='cpu'   -> zawsze cpu
    - prefer='cuda'  -> cuda jeśli jest, inaczej cpu (bez crasha)
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def cpu_friendly(max_threads: int = 4) -> None:
    """Zmniejsz overhead na laptopach (czasem PyTorch odpala za dużo wątków)."""
    try:
        torch.set_num_threads(min(max_threads, os.cpu_count() or 1))
    except Exception:
        pass


def count_runs(*, seq_lens, hidden_sizes, models, seeds, protocols=1, optimizers=None, lrs=1) -> int:
    """Policz ile runów uruchomisz (żeby nie kliknąć 'Run all' na 300 runów).

    To jest najprostsza kontrola wykonalności na CPU.
    """
    if optimizers is None:
        optimizers = [None]
    if not isinstance(optimizers, (list, tuple)):
        optimizers = [optimizers]
    return (
        len(seq_lens)
        * len(hidden_sizes)
        * len(models)
        * len(seeds)
        * int(protocols)
        * len(optimizers)
        * int(lrs)
    )


def now_tag() -> str:
    import datetime
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
