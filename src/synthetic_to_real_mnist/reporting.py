"""Reporting helpers that convert experiment outputs into markdown summaries."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    display_cols = [
        "experiment",
        "train_source",
        "synthetic_samples",
        "real_fraction",
        "epochs",
        "val_accuracy",
        "real_test_accuracy",
        "generalization_gap",
    ]
    cols = [c for c in display_cols if c in df.columns]
    out = df[cols].copy()
    for col in ["val_accuracy", "real_test_accuracy", "generalization_gap"]:
        if col in out.columns:
            out[col] = out[col].map(lambda x: "" if pd.isna(x) else f"{float(x):.2f}")
    return out.to_markdown(index=False)


def build_interpretation(df: pd.DataFrame) -> str:
    if df.empty or "real_test_accuracy" not in df.columns:
        return "No completed experiments were found. Run the Phase 2 notebook or `python run_phase2.py --mode full`."

    best = df.sort_values("real_test_accuracy", ascending=False).iloc[0]
    lines: List[str] = []
    lines.append(f"The best real-MNIST test accuracy came from **{best['experiment']}** with **{best['real_test_accuracy']:.2f}%**.")

    def get_acc(name: str):
        row = df[df["experiment"] == name]
        return None if row.empty else float(row.iloc[0]["real_test_accuracy"])

    basic10 = get_acc("synthetic_basic_10k")
    basic50 = get_acc("synthetic_basic_50k")
    improved50 = get_acc("synthetic_improved_50k")
    real = get_acc("real_mnist_baseline")

    if basic10 is not None and basic50 is not None:
        delta = basic50 - basic10
        direction = "improved" if delta >= 0 else "decreased"
        lines.append(f"Scaling the basic synthetic dataset from 10k to 50k {direction} transfer by **{delta:+.2f} percentage points**.")
    if basic50 is not None and improved50 is not None:
        delta = improved50 - basic50
        direction = "helped" if delta >= 0 else "hurt"
        lines.append(f"Adding Phase 2 realism operations - blur/noise/stroke/intensity/shift/shear variation - {direction} by **{delta:+.2f} percentage points** relative to 50k basic synthetic data.")
    if real is not None and improved50 is not None:
        gap = real - improved50
        lines.append(f"The remaining gap to the real-data upper bound is **{gap:.2f} percentage points**, which estimates the unresolved synthetic-to-real domain gap.")

    hybrid_rows = df[df["train_source"] == "hybrid"].sort_values("real_fraction")
    if len(hybrid_rows) > 0:
        best_hybrid = hybrid_rows.sort_values("real_test_accuracy", ascending=False).iloc[0]
        lines.append(f"Among hybrid runs, **{best_hybrid['experiment']}** was strongest, suggesting how much real data was needed to anchor the synthetic variation.")

    ablations = df[df["experiment_group"].astype(str).str.contains("ablation", na=False)] if "experiment_group" in df.columns else pd.DataFrame()
    if not ablations.empty:
        strongest_ablation = ablations.sort_values("real_test_accuracy", ascending=False).iloc[0]
        weakest_ablation = ablations.sort_values("real_test_accuracy", ascending=True).iloc[0]
        lines.append(f"The ablation study ranged from **{weakest_ablation['experiment']}** ({weakest_ablation['real_test_accuracy']:.2f}%) to **{strongest_ablation['experiment']}** ({strongest_ablation['real_test_accuracy']:.2f}%), showing that generation parameters affect transfer rather than merely synthetic validation accuracy.")

    return "\n\n".join(lines)


def build_limitations(df: pd.DataFrame) -> str:
    return (
        "1. The synthetic generator is still hand-coded, so it cannot fully reproduce natural motor patterns, digit-writing conventions, or pen-pressure effects.\n"
        "2. MNIST is grayscale, centered, and low-resolution; conclusions should not be generalized to harder vision domains without new experiments.\n"
        "3. If synthetic validation accuracy is high but real test accuracy is lower, the model has learned generator-specific shortcuts. This is a central failure mode, not just noise.\n"
        "4. Full Phase 2 results depend on random seeds, number of epochs, and available hardware. The notebook fixes seeds, but small differences can still occur on GPU.\n"
        "5. Hybrid experiments use small labeled real subsets from MNIST, so they no longer test purely synthetic learning; they answer a different but useful question about how much real data is needed to reduce the gap."
    )


def write_phase2_report(df: pd.DataFrame, output_path: str | Path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = "# Phase 2 Results Report\n\n"
    text += "## Experiment Table\n\n"
    text += dataframe_to_markdown(df) + "\n\n"
    text += "## Interpretation\n\n" + build_interpretation(df) + "\n\n"
    text += "## Limitations and Failures\n\n" + build_limitations(df) + "\n"
    output_path.write_text(text, encoding="utf-8")
    return text


def update_readme_results(readme_path: str | Path, df: pd.DataFrame):
    readme_path = Path(readme_path)
    original = readme_path.read_text(encoding="utf-8")
    start = "<!-- RESULTS_START -->"
    end = "<!-- RESULTS_END -->"
    if start not in original or end not in original:
        raise ValueError("README markers not found")
    replacement = start + "\n" + dataframe_to_markdown(df) + "\n\n" + build_interpretation(df) + "\n" + end
    before = original.split(start)[0]
    after = original.split(end)[1]
    readme_path.write_text(before + replacement + after, encoding="utf-8")
