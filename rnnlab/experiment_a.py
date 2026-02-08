from __future__ import annotations

from typing import Sequence

import pandas as pd
import torch

from .data import TaskSpec, make_fixed_testset, sample_batch
from .models import ModelSpec, SeqClassifier
from .train import TrainConfig, train_run
from .utils import set_seed


def sweep_frontier(
    *,
    task_variant: str,
    models: Sequence[str],
    seq_lens: Sequence[int],
    hidden_sizes: Sequence[int],
    seeds: Sequence[int],
    pooling: str,
    device: torch.device,
    train_cfg: TrainConfig,
) -> pd.DataFrame:
    """Projekt A: uruchom siatkę (model × seq_len × hidden × seed).

    Zwraca tabelę runów, którą łatwo scalać między studentami:
    - każdy student może zrobić tylko fragment siatki,
    - prowadzący scala CSV (concat) i liczy podsumowania.
    """
    runs = []
    task = TaskSpec(variant=task_variant)

    for cell_type in models:
        for hidden in hidden_sizes:
            for seq_len in seq_lens:
                for seed in seeds:
                    set_seed(seed)
                    spec = ModelSpec(
                        cell_type=cell_type,
                        pooling=pooling,
                        input_size=task.vocab_size,
                        hidden_size=hidden,
                        num_classes=task.num_classes,
                    )
                    model = SeqClassifier(spec)

                    def _sample(batch_size: int, seed: int):
                        return sample_batch(
                            batch_size=batch_size,
                            seq_len=seq_len,
                            task=task,
                            seed=seed,
                            device=device,
                        )

                    def _testset(test_size: int, seed: int):
                        return make_fixed_testset(
                            test_size=test_size,
                            seq_len=seq_len,
                            task=task,
                            seed=seed,
                            device=device,
                        )

                    out = train_run(
                        model=model,
                        sample_batch_fn=_sample,
                        make_testset_fn=_testset,
                        cfg=train_cfg,
                        device=device,
                        seed=seed,
                    )

                    best = out["best_test_acc"]
                    success = 1 if best >= train_cfg.early_stop_threshold else 0

                    runs.append({
                        "model": cell_type,
                        "pooling": pooling,
                        "task_variant": task_variant,
                        "seq_len": int(seq_len),
                        "hidden": int(hidden),
                        "seed": int(seed),
                        "best_test_acc": float(best),
                        "success": int(success),
                        "steps_ran": int(out["steps_ran"]),
                        "train_time_sec": float(out["train_time_sec"]),
                    })

    return pd.DataFrame(runs)


def summarize_frontier(runs_df: pd.DataFrame) -> pd.DataFrame:
    """Agregacja runów do mapy fazowej."""
    g = runs_df.groupby(["model", "pooling", "seq_len", "hidden"], as_index=False)
    out = g.agg(
        P_success=("success", "mean"),
        mean_best_acc=("best_test_acc", "mean"),
        std_best_acc=("best_test_acc", "std"),
        mean_steps=("steps_ran", "mean"),
    )
    out["P_success"] = out["P_success"].fillna(0.0)
    out["std_best_acc"] = out["std_best_acc"].fillna(0.0)
    return out
