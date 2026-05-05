"""Dataset and DataLoader helpers for synthetic, real, and hybrid experiments."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, random_split

from .config import SyntheticConfig, TrainConfig
from .synthetic_data import SyntheticMNISTDataset, generate_synthetic_dataset


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)


def make_loader(dataset: Dataset, cfg: TrainConfig, shuffle: bool) -> DataLoader:
    generator = torch.Generator().manual_seed(cfg.seed)

    def safe_collate(batch):
        images, labels = zip(*batch)
        images = torch.stack([
            img if torch.is_tensor(img) else torch.tensor(img)
            for img in images
        ])
        labels = torch.tensor([int(label) for label in labels], dtype=torch.long)
        return images, labels

    return DataLoader(
        dataset,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available() and cfg.pin_memory,
        worker_init_fn=None,
        generator=generator,
        collate_fn=safe_collate,
    )


def split_train_val(dataset: Dataset, val_fraction: float, seed: int) -> Tuple[Dataset, Dataset]:
    val_size = int(round(len(dataset) * val_fraction))
    train_size = len(dataset) - val_size
    return random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(seed))


def synthetic_train_val_loaders(
    num_samples: int,
    synth_cfg: SyntheticConfig,
    train_cfg: TrainConfig,
    seed: int,
) -> Tuple[DataLoader, DataLoader, Dataset]:
    data, labels = generate_synthetic_dataset(num_samples, synth_cfg, seed=seed, balanced=True)
    dataset = SyntheticMNISTDataset(data, labels)
    train_ds, val_ds = split_train_val(dataset, train_cfg.val_fraction, seed)
    return make_loader(train_ds, train_cfg, shuffle=True), make_loader(val_ds, train_cfg, shuffle=False), dataset


def load_mnist_datasets(root: str = "./data"):
    """Load torchvision MNIST datasets.

    Importing torchvision is delayed so synthetic-only smoke tests still work in
    environments where torchvision is unavailable or mismatched.
    """

    try:
        from torchvision import datasets, transforms
    except Exception as exc:  # pragma: no cover - environment-specific
        raise RuntimeError(
            "Could not import torchvision. In Google Colab, run `pip install torch torchvision`. "
            f"Original error: {exc}"
        ) from exc

    transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.MNIST(root=root, train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root=root, train=False, download=True, transform=transform)
    return train_dataset, test_dataset


def class_balanced_subset(dataset: Dataset, fraction: float | None = None, count: int | None = None, seed: int = 42) -> Subset:
    """Return a class-balanced subset of a labeled MNIST-like dataset.

    Works with torchvision MNIST because it exposes `targets`.
    """

    if fraction is None and count is None:
        raise ValueError("Either fraction or count must be provided")
    if fraction is not None and not (0 < fraction <= 1):
        raise ValueError("fraction must be in (0, 1]")

    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = [int(dataset[i][1]) for i in range(len(dataset))]
    targets = np.asarray(targets, dtype=np.int64)

    rng = np.random.default_rng(seed)
    by_class = defaultdict(list)
    for idx, label in enumerate(targets):
        by_class[int(label)].append(idx)

    selected = []
    if count is not None:
        per_class = max(1, count // 10)
        remainder = max(0, count - per_class * 10)
        extra_classes = set(rng.choice(np.arange(10), size=remainder, replace=False)) if remainder else set()
        for label in range(10):
            k = per_class + (1 if label in extra_classes else 0)
            indices = np.array(by_class[label])
            selected.extend(rng.choice(indices, size=min(k, len(indices)), replace=False).tolist())
    else:
        for label in range(10):
            indices = np.array(by_class[label])
            k = max(1, int(round(len(indices) * float(fraction))))
            selected.extend(rng.choice(indices, size=k, replace=False).tolist())

    rng.shuffle(selected)
    return Subset(dataset, selected)


def real_train_val_test_loaders(
    train_cfg: TrainConfig,
    root: str = "./data",
    subset_fraction: float | None = None,
    subset_count: int | None = None,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dataset, Dataset, Dataset]:
    real_train, real_test = load_mnist_datasets(root=root)
    if subset_fraction is not None or subset_count is not None:
        real_train = class_balanced_subset(real_train, fraction=subset_fraction, count=subset_count, seed=seed)
    train_ds, val_ds = split_train_val(real_train, train_cfg.val_fraction, seed)
    return (
        make_loader(train_ds, train_cfg, shuffle=True),
        make_loader(val_ds, train_cfg, shuffle=False),
        make_loader(real_test, train_cfg, shuffle=False),
        train_ds,
        val_ds,
        real_test,
    )


def hybrid_train_val_test_loaders(
    num_synthetic: int,
    synth_cfg: SyntheticConfig,
    real_fraction: float,
    train_cfg: TrainConfig,
    root: str = "./data",
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, Dataset, Dataset, Dataset]:
    syn_data, syn_labels = generate_synthetic_dataset(num_synthetic, synth_cfg, seed=seed, balanced=True)
    syn_dataset = SyntheticMNISTDataset(syn_data, syn_labels)
    real_train, real_test = load_mnist_datasets(root=root)
    real_subset = class_balanced_subset(real_train, fraction=real_fraction, seed=seed)
    combined = ConcatDataset([syn_dataset, real_subset])
    train_ds, val_ds = split_train_val(combined, train_cfg.val_fraction, seed)
    return (
        make_loader(train_ds, train_cfg, shuffle=True),
        make_loader(val_ds, train_cfg, shuffle=False),
        make_loader(real_test, train_cfg, shuffle=False),
        train_ds,
        val_ds,
        real_test,
    )
