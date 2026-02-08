from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TrainConfig:
    """Konfiguracja treningu.

    Wersja CPU-friendly:
    - mały testset
    - rzadsza ewaluacja
    - early_stop (żeby nie trenować, gdy już jest sukces)
    """
    steps: int = 600
    batch_size: int = 64
    lr: float = 3e-3
    optimizer: str = "adam"  # 'adam' | 'rmsprop' | 'sgd_momentum'
    momentum: float = 0.9
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    eval_every: int = 100
    test_size: int = 512
    fixed_test_seed: int = 123

    early_stop: bool = True
    early_stop_threshold: float = 0.9
    early_stop_patience: int = 2
    min_steps: int = 200


def make_optimizer(model: nn.Module, cfg: TrainConfig) -> torch.optim.Optimizer:
    if cfg.optimizer == "adam":
        return torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "rmsprop":
        return torch.optim.RMSprop(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    if cfg.optimizer == "sgd_momentum":
        return torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
    raise ValueError(f"Unknown optimizer: {cfg.optimizer}")


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    logits, _ = model(x)
    pred = logits.argmax(dim=-1)
    return (pred == y).float().mean().item()


def train_run(
    *,
    model: nn.Module,
    sample_batch_fn,
    make_testset_fn,
    cfg: TrainConfig,
    device: torch.device,
    seed: int,
) -> Dict:
    """Jeden run: trenuje model i zwraca historię + podsumowanie.

    Zwraca:
      - history: lista wpisów co eval_every kroków
      - best_test_acc: najlepszy test_acc w trakcie treningu
      - steps_ran: ile kroków faktycznie wykonano (early_stop skraca)
    """
    model.to(device)
    opt = make_optimizer(model, cfg)

    # Stały testset => wyniki porównywalne (seed nie zmienia testu)
    test_x, test_y = make_testset_fn(test_size=cfg.test_size, seed=cfg.fixed_test_seed)

    best_test_acc = 0.0
    hist = []
    good_hits = 0
    t0 = time.time()

    for step in range(1, cfg.steps + 1):
        model.train()
        x, y = sample_batch_fn(batch_size=cfg.batch_size, seed=seed * 1_000_000 + step)

        logits, _ = model(x)
        loss = F.cross_entropy(logits, y)

        opt.zero_grad(set_to_none=True)
        loss.backward()

        # Stabilizacja: clipping pomaga, szczególnie dla długich sekwencji
        if cfg.grad_clip and cfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)

        opt.step()

        # szybkie train_acc na batchu (diagnostyczne)
        with torch.no_grad():
            train_acc = (logits.argmax(dim=-1) == y).float().mean().item()

        if step % cfg.eval_every == 0 or step == 1:
            test_acc = evaluate(model, test_x, test_y)
            best_test_acc = max(best_test_acc, test_acc)

            hist.append({
                "step": step,
                "loss": float(loss.item()),
                "train_acc": float(train_acc),
                "test_acc": float(test_acc),
            })

            # Early stop: jeśli przez 2 ewaluacje z rzędu jesteś powyżej progu, kończ
            if cfg.early_stop and step >= cfg.min_steps:
                if test_acc >= cfg.early_stop_threshold:
                    good_hits += 1
                else:
                    good_hits = 0
                if good_hits >= cfg.early_stop_patience:
                    break

    t1 = time.time()
    return {
        "history": hist,
        "best_test_acc": float(best_test_acc),
        "steps_ran": int(step),
        "train_time_sec": float(t1 - t0),
        "cfg": cfg.__dict__.copy(),
    }
