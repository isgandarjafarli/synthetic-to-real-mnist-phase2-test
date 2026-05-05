"""Phase 2 experiment orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .config import SyntheticConfig, TrainConfig
from .datasets import hybrid_train_val_test_loaders, real_train_val_test_loaders, synthetic_train_val_loaders
from .model import get_model
from .train import evaluate, fit_model, get_device, save_history, set_seed
from .visualization import (
    ensure_dir,
    plot_confusion_matrix,
    plot_conv1_filters,
    plot_experiment_summary,
    plot_history,
    plot_per_class_accuracy,
)


@dataclass
class ExperimentSpec:
    experiment: str
    train_source: str  # "synthetic", "real", or "hybrid"
    experiment_group: str
    synthetic_samples: int = 0
    synth_config: SyntheticConfig | None = None
    real_fraction: float | None = None
    real_subset_count: int | None = None
    epochs: int = 4
    seed: int = 42


def build_phase2_specs(mode: str = "full") -> List[ExperimentSpec]:
    """Return the Phase 2 experiment plan.

    mode="full" matches the project plan.  mode="smoke" runs tiny versions for
    debugging imports and notebook execution before committing to the full run.
    """

    if mode not in {"full", "smoke"}:
        raise ValueError("mode must be 'full' or 'smoke'")

    if mode == "smoke":
        syn10, syn50, syn_abl = 1000, 2000, 1000
        core_epochs, ablation_epochs = 1, 1
        real_count = 1500
    else:
        syn10, syn50, syn_abl = 10000, 50000, 20000
        core_epochs, ablation_epochs = 4, 2
        real_count = None

    basic = SyntheticConfig.basic(rotation_range=15, thickness_range=(2, 4))
    improved = SyntheticConfig.improved(rotation_range=20, thickness_range=(1, 5))

    specs: List[ExperimentSpec] = [
        ExperimentSpec(
            experiment="real_mnist_baseline",
            train_source="real",
            experiment_group="baseline",
            epochs=core_epochs,
            real_subset_count=real_count,
            seed=42,
        ),
        ExperimentSpec(
            experiment="synthetic_basic_10k",
            train_source="synthetic",
            experiment_group="scale",
            synthetic_samples=syn10,
            synth_config=basic,
            epochs=core_epochs,
            seed=43,
        ),
        ExperimentSpec(
            experiment="synthetic_basic_50k",
            train_source="synthetic",
            experiment_group="scale",
            synthetic_samples=syn50,
            synth_config=basic,
            epochs=core_epochs,
            seed=44,
        ),
        ExperimentSpec(
            experiment="synthetic_improved_50k",
            train_source="synthetic",
            experiment_group="improved_generation",
            synthetic_samples=syn50,
            synth_config=improved,
            epochs=core_epochs,
            seed=45,
        ),
        ExperimentSpec(
            experiment="hybrid_improved_real_5pct",
            train_source="hybrid",
            experiment_group="hybrid",
            synthetic_samples=syn50,
            synth_config=improved,
            real_fraction=0.05,
            epochs=core_epochs,
            seed=46,
        ),
        ExperimentSpec(
            experiment="hybrid_improved_real_10pct",
            train_source="hybrid",
            experiment_group="hybrid",
            synthetic_samples=syn50,
            synth_config=improved,
            real_fraction=0.10,
            epochs=core_epochs,
            seed=47,
        ),
        ExperimentSpec(
            experiment="hybrid_improved_real_20pct",
            train_source="hybrid",
            experiment_group="hybrid",
            synthetic_samples=syn50,
            synth_config=improved,
            real_fraction=0.20,
            epochs=core_epochs,
            seed=48,
        ),
    ]

    # Ablation 1: rotation range, with other improved parameters fixed.
    for rot in [0, 10, 20, 35]:
        specs.append(
            ExperimentSpec(
                experiment=f"ablation_rotation_{rot}",
                train_source="synthetic",
                experiment_group="ablation_rotation",
                synthetic_samples=syn_abl,
                synth_config=SyntheticConfig.improved(rotation_range=rot, thickness_range=(1, 5)),
                epochs=ablation_epochs,
                seed=100 + rot,
            )
        )

    # Ablation 2: stroke thickness, with rotation fixed.
    for name, thickness in [("thin", (1, 2)), ("medium", (2, 4)), ("wide", (4, 6))]:
        specs.append(
            ExperimentSpec(
                experiment=f"ablation_thickness_{name}",
                train_source="synthetic",
                experiment_group="ablation_thickness",
                synthetic_samples=syn_abl,
                synth_config=SyntheticConfig.improved(rotation_range=20, thickness_range=thickness),
                epochs=ablation_epochs,
                seed=200 + thickness[0],
            )
        )

    return specs


def _loaders_for_spec(spec: ExperimentSpec, train_cfg: TrainConfig, data_root: str):
    cfg = TrainConfig(
        batch_size=train_cfg.batch_size,
        epochs=spec.epochs,
        learning_rate=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
        val_fraction=train_cfg.val_fraction,
        num_workers=train_cfg.num_workers,
        seed=spec.seed,
        pin_memory=train_cfg.pin_memory,
    )
    if spec.train_source == "real":
        return (*real_train_val_test_loaders(cfg, root=data_root, subset_count=spec.real_subset_count, seed=spec.seed), cfg)
    if spec.train_source == "synthetic":
        train_loader, val_loader, _syn_dataset = synthetic_train_val_loaders(spec.synthetic_samples, spec.synth_config, cfg, seed=spec.seed)
        # Real MNIST test set is still the target domain.
        _, _, real_test_loader, _, _, real_test_dataset = real_train_val_test_loaders(cfg, root=data_root, subset_count=1, seed=spec.seed)
        return train_loader, val_loader, real_test_loader, None, None, real_test_dataset, cfg
    if spec.train_source == "hybrid":
        return (*hybrid_train_val_test_loaders(spec.synthetic_samples, spec.synth_config, spec.real_fraction, cfg, root=data_root, seed=spec.seed), cfg)
    raise ValueError(f"Unknown train_source: {spec.train_source}")


def run_experiment(
    spec: ExperimentSpec,
    train_cfg: TrainConfig,
    output_dir: str | Path = "outputs",
    data_root: str = "./data",
    device=None,
    save_model: bool = True,
):
    output_dir = ensure_dir(output_dir)
    history_dir = ensure_dir(output_dir / "histories")
    plot_dir = ensure_dir(output_dir / "plots")
    model_dir = ensure_dir(output_dir / "models")

    set_seed(spec.seed)
    device = device or get_device()
    print("\n" + "=" * 80)
    print(f"Running: {spec.experiment} on {device}")
    print("=" * 80)

    train_loader, val_loader, real_test_loader, train_ds, val_ds, real_test_dataset, cfg = _loaders_for_spec(spec, train_cfg, data_root)
    model = get_model(device=device)
    model, history = fit_model(model, train_loader, val_loader, cfg, device=device)
    save_history(history_dir / f"{spec.experiment}.json", history)
    plot_history(history, spec.experiment, plot_dir / f"{spec.experiment}_history.png")

    criterion = nn.CrossEntropyLoss()
    val_metrics = evaluate(model, val_loader, criterion, device, collect_predictions=False)
    test_metrics = evaluate(model, real_test_loader, criterion, device, collect_predictions=True)

    cm = test_metrics["confusion_matrix"]
    per_class = test_metrics["per_class_accuracy"]
    plot_confusion_matrix(cm, f"{spec.experiment}: real MNIST confusion matrix", plot_dir / f"{spec.experiment}_confusion_matrix.png")
    plot_per_class_accuracy(per_class, f"{spec.experiment}: per-class real MNIST accuracy", plot_dir / f"{spec.experiment}_per_class.png")

    if save_model:
        torch.save(
            {
                "experiment": spec.experiment,
                "model_state_dict": model.state_dict(),
                "history": history,
                "spec": _spec_to_dict(spec),
                "real_test_accuracy": float(test_metrics["accuracy"]),
            },
            model_dir / f"{spec.experiment}.pt",
        )

    row = {
        "experiment": spec.experiment,
        "train_source": spec.train_source,
        "experiment_group": spec.experiment_group,
        "synthetic_samples": spec.synthetic_samples,
        "real_fraction": spec.real_fraction if spec.real_fraction is not None else "",
        "epochs": spec.epochs,
        "seed": spec.seed,
        "val_accuracy": float(val_metrics["accuracy"]),
        "real_test_accuracy": float(test_metrics["accuracy"]),
        "generalization_gap": float(val_metrics["accuracy"]) - float(test_metrics["accuracy"]),
        "synth_config_name": spec.synth_config.name if spec.synth_config else "",
    }
    return row, model, test_metrics, history


def _spec_to_dict(spec: ExperimentSpec) -> dict:
    d = asdict(spec)
    if spec.synth_config is not None:
        d["synth_config"] = asdict(spec.synth_config)
    return d


def run_phase2_suite(
    mode: str = "full",
    output_dir: str | Path = "outputs",
    data_root: str = "./data",
    base_train_cfg: TrainConfig | None = None,
    save_model: bool = True,
):
    output_dir = ensure_dir(output_dir)
    base_train_cfg = base_train_cfg or TrainConfig()
    device = get_device()
    specs = build_phase2_specs(mode)
    rows: List[dict] = []
    trained_models: Dict[str, object] = {}
    detailed: Dict[str, dict] = {}

    for spec in specs:
        row, model, test_metrics, history = run_experiment(
            spec,
            base_train_cfg,
            output_dir=output_dir,
            data_root=data_root,
            device=device,
            save_model=save_model,
        )
        rows.append(row)
        # Keep only two models in memory for feature visualization.
        if spec.experiment in {"real_mnist_baseline", "synthetic_improved_50k"}:
            trained_models[spec.experiment] = model
        detailed[spec.experiment] = {"test_metrics": test_metrics, "history": history}
        pd.DataFrame(rows).to_csv(output_dir / "phase2_results_partial.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(output_dir / "phase2_results.csv", index=False)
    plot_experiment_summary(df, output_dir / "plots" / "phase2_experiment_summary.png")

    if "real_mnist_baseline" in trained_models:
        plot_conv1_filters(trained_models["real_mnist_baseline"], "Conv1 filters: real MNIST baseline", output_dir / "plots" / "conv1_filters_real_baseline.png")
    if "synthetic_improved_50k" in trained_models:
        plot_conv1_filters(trained_models["synthetic_improved_50k"], "Conv1 filters: improved synthetic model", output_dir / "plots" / "conv1_filters_synthetic_improved.png")

    return df, trained_models, detailed
