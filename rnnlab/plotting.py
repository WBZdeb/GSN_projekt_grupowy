from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_heatmap(summary_df: pd.DataFrame, *, value_col: str, title: str, out_path: Optional[str] = None):
    """Heatmap dla jednego modelu: wiersze=HIDDEN, kolumny=SEQ_LEN."""
    pivot = summary_df.pivot(index="hidden", columns="seq_len", values=value_col)
    fig, ax = plt.subplots(figsize=(6, 4))
    im = ax.imshow(pivot.values, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(pivot.shape[1]))
    ax.set_xticklabels(list(pivot.columns))
    ax.set_yticks(np.arange(pivot.shape[0]))
    ax.set_yticklabels(list(pivot.index))
    ax.set_xlabel("SEQ_LEN")
    ax.set_ylabel("HIDDEN")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    if out_path:
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
    return fig, ax


def plot_lines_acc(summary_df: pd.DataFrame, *, title: str, out_path: Optional[str] = None):
    """Linie mean(best_acc) vs SEQ_LEN; osobna linia per model."""
    fig, ax = plt.subplots(figsize=(6, 4))
    for model in sorted(summary_df["model"].unique()):
        dfm = summary_df[summary_df["model"] == model].sort_values("seq_len")
        ax.errorbar(dfm["seq_len"], dfm["mean_best_acc"], yerr=dfm["std_best_acc"], marker="o", label=model)
    ax.set_xlabel("SEQ_LEN")
    ax.set_ylabel("mean(best_test_acc)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if out_path:
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
    return fig, ax


def plot_acc_over_steps(history_df: pd.DataFrame, *, title: str, out_path: Optional[str] = None):
    """test_acc vs step (mean over seeds) dla kilku krzywych."""
    fig, ax = plt.subplots(figsize=(7, 4))
    for key in sorted(history_df["curve_id"].unique()):
        d = history_df[history_df["curve_id"] == key].sort_values("step")
        ax.plot(d["step"], d["test_acc_mean"], marker="o", label=key)
    ax.set_xlabel("step")
    ax.set_ylabel("test_acc (mean over seeds)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    if out_path:
        fig.tight_layout()
        fig.savefig(out_path, dpi=160)
    return fig, ax
