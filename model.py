"""Backward-compatible wrapper for Phase 1 imports."""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_to_real_mnist.model import SimpleCNN, count_parameters, get_model  # noqa: F401
