"""Backward-compatible wrapper for Phase 1 imports."""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_to_real_mnist.train import evaluate, fit_model, get_device, save_history, set_seed, train_one_epoch  # noqa: F401
from synthetic_to_real_mnist.visualization import plot_history as plot_training_history  # noqa: F401
