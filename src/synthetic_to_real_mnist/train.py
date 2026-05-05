"""Training and evaluation utilities."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from tqdm.auto import tqdm

from .config import TrainConfig


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer, device, epoch: int | None = None) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    desc = "train" if epoch is None else f"train epoch {epoch}"
    for images, labels in tqdm(loader, desc=desc, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        pred = logits.argmax(dim=1)
        total += labels.size(0)
        correct += int((pred == labels).sum().item())
    return {"loss": total_loss / max(total, 1), "accuracy": 100.0 * correct / max(total, 1)}


@torch.no_grad()
def evaluate(model, loader, criterion, device, collect_predictions: bool = False) -> Dict[str, object]:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    y_true: List[int] = []
    y_pred: List[int] = []

    for images, labels in tqdm(loader, desc="evaluate", leave=False):
        images = images.to(device)
        labels = labels.to(device)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss.item()) * labels.size(0)
        pred = logits.argmax(dim=1)
        total += labels.size(0)
        correct += int((pred == labels).sum().item())
        if collect_predictions:
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(pred.cpu().numpy().tolist())

    result: Dict[str, object] = {
        "loss": total_loss / max(total, 1),
        "accuracy": 100.0 * correct / max(total, 1),
    }
    if collect_predictions:
        result["y_true"] = np.asarray(y_true, dtype=np.int64)
        result["y_pred"] = np.asarray(y_pred, dtype=np.int64)
        result["confusion_matrix"] = confusion_matrix(y_true, y_pred, labels=list(range(10)))
        cm = result["confusion_matrix"].astype(float)
        per_class = np.divide(np.diag(cm), cm.sum(axis=1), out=np.zeros(10), where=cm.sum(axis=1) != 0) * 100.0
        result["per_class_accuracy"] = per_class
    return result


def fit_model(model, train_loader, val_loader, train_cfg: TrainConfig, device=None) -> Tuple[object, Dict[str, List[float]]]:
    device = device or get_device()
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=train_cfg.learning_rate, weight_decay=train_cfg.weight_decay)

    history: Dict[str, List[float]] = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    for epoch in range(1, train_cfg.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch=epoch)
        val_metrics = evaluate(model, val_loader, criterion, device, collect_predictions=False)
        history["train_loss"].append(float(train_metrics["loss"]))
        history["train_accuracy"].append(float(train_metrics["accuracy"]))
        history["val_loss"].append(float(val_metrics["loss"]))
        history["val_accuracy"].append(float(val_metrics["accuracy"]))
        print(
            f"epoch {epoch:02d}/{train_cfg.epochs} | "
            f"train acc {train_metrics['accuracy']:.2f}% | "
            f"val acc {val_metrics['accuracy']:.2f}%"
        )
    return model, history


def save_history(path: str | Path, history: Dict[str, List[float]]):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(history, indent=2))
