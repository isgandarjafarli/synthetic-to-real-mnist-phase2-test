"""Backward-compatible wrapper for Phase 1 imports."""
from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from synthetic_to_real_mnist.config import SyntheticConfig  # noqa: F401
from synthetic_to_real_mnist.synthetic_data import (  # noqa: F401
    SyntheticDigitGenerator,
    SyntheticMNISTDataset,
    generate_synthetic_dataset,
    load_npz,
    save_npz,
)


class SyntheticMNIST:
    """Compatibility class matching the Phase 1 API."""

    def __init__(self, num_samples=50000, img_size=28):
        self.num_samples = num_samples
        self.img_size = img_size
        self.data = None
        self.labels = None

    def generate_dataset(self):
        cfg = SyntheticConfig.basic()
        data, labels = generate_synthetic_dataset(self.num_samples, cfg, seed=42, balanced=True)
        self.data = data
        self.labels = labels
        return data, labels

    def save_dataset(self, filepath="synthetic_mnist.npz"):
        save_npz(filepath, self.data, self.labels, SyntheticConfig.basic())

    @staticmethod
    def load_dataset(filepath="synthetic_mnist.npz"):
        return load_npz(filepath)
