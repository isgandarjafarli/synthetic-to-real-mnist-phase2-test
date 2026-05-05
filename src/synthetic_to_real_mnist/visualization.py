"""Plotting helpers for Phase 2 analysis."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def plot_sample_grid(data: np.ndarray, labels: np.ndarray, title: str, save_path: str | Path | None = None):
    fig, axes = plt.subplots(2, 10, figsize=(14, 3.2))
    fig.suptitle(title, fontsize=14, fontweight="bold")
    for digit in range(10):
        idxs = np.where(labels == digit)[0]
        for row in range(2):
            ax = axes[row, digit]
            if len(idxs) > row:
                ax.imshow(data[idxs[row]], cmap="gray", vmin=0, vmax=1)
            ax.set_title(str(digit) if row == 0 else "", fontsize=10)
            ax.axis("off")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_history(history: Dict[str, List[float]], title: str, save_path: str | Path | None = None):
    epochs = np.arange(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(epochs, history["train_loss"], marker="o", label="train")
    axes[0].plot(epochs, history["val_loss"], marker="o", label="validation")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, history["train_accuracy"], marker="o", label="train")
    axes[1].plot(epochs, history["val_accuracy"], marker="o", label="validation")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy (%)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_confusion_matrix(cm: np.ndarray, title: str, save_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, interpolation="nearest")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    max_val = cm.max() if cm.size else 0
    for i in range(10):
        for j in range(10):
            ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", fontsize=8,
                    color="white" if cm[i, j] > max_val * 0.55 else "black")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_per_class_accuracy(per_class: np.ndarray, title: str, save_path: str | Path | None = None):
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(np.arange(10), per_class)
    ax.set_ylim(0, 100)
    ax.set_xticks(np.arange(10))
    ax.set_xlabel("Digit")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, value in enumerate(per_class):
        ax.text(i, min(100, value + 2), f"{value:.1f}", ha="center", fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_experiment_summary(df: pd.DataFrame, save_path: str | Path | None = None):
    order = df.sort_values("real_test_accuracy", ascending=True)
    fig, ax = plt.subplots(figsize=(11, max(4, 0.42 * len(order))))
    ax.barh(order["experiment"], order["real_test_accuracy"])
    ax.set_xlabel("Real MNIST test accuracy (%)")
    ax.set_title("Phase 2 experiment comparison", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    for idx, (_, row) in enumerate(order.iterrows()):
        ax.text(row["real_test_accuracy"] + 0.4, idx, f"{row['real_test_accuracy']:.2f}%", va="center", fontsize=8)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def plot_conv1_filters(model, title: str, save_path: str | Path | None = None, max_filters: int = 32):
    weights = model.conv1.weight.detach().cpu().numpy()[:max_filters, 0]
    cols = 8
    rows = int(np.ceil(len(weights) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.2, rows * 1.2))
    axes = np.asarray(axes).reshape(-1)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < len(weights):
            ax.imshow(weights[i], cmap="gray")
            ax.set_title(str(i), fontsize=7)
    fig.suptitle(title, fontsize=13, fontweight="bold")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig


def show_prediction_examples(dataset, y_true: np.ndarray, y_pred: np.ndarray, title: str, save_path: str | Path | None = None):
    correct = np.where(y_true == y_pred)[0]
    incorrect = np.where(y_true != y_pred)[0]
    fig, axes = plt.subplots(2, 10, figsize=(14, 3.2))
    fig.suptitle(title, fontsize=13, fontweight="bold")

    for row, indices, row_name in [(0, correct, "correct"), (1, incorrect, "wrong")]:
        for col in range(10):
            ax = axes[row, col]
            ax.axis("off")
            if len(indices) == 0 or col >= len(indices):
                continue
            idx = int(indices[col])
            image, label = dataset[idx]
            if torch.is_tensor(image):
                image = image.squeeze().detach().cpu().numpy()
            ax.imshow(image, cmap="gray", vmin=0, vmax=1)
            ax.set_title(f"T:{int(y_true[idx])} P:{int(y_pred[idx])}", fontsize=8)
        axes[row, 0].set_ylabel(row_name, fontsize=10)
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=200, bbox_inches="tight")
    return fig
