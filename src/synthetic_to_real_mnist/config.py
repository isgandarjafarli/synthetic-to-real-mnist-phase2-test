"""Configuration dataclasses used across experiments."""

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SyntheticConfig:
    """Controls how synthetic digit images are generated.

    The default values are intentionally close to Phase 1.  The "improved"
    classmethod adds the extra variation used in Phase 2.
    """

    img_size: int = 28
    rotation_range: float = 15.0
    thickness_range: Tuple[int, int] = (2, 4)
    shift_range: int = 2
    shear_range: float = 0.0
    blur_prob: float = 0.0
    blur_radius_range: Tuple[float, float] = (0.1, 0.8)
    gaussian_noise_std: float = 0.0
    salt_pepper_prob: float = 0.0
    intensity_range: Tuple[int, int] = (235, 255)
    random_scale_range: Tuple[float, float] = (0.95, 1.05)
    elastic_like_prob: float = 0.0
    name: str = "basic"

    @classmethod
    def basic(cls, *, rotation_range: float = 15.0, thickness_range: Tuple[int, int] = (2, 4)):
        return cls(
            rotation_range=rotation_range,
            thickness_range=thickness_range,
            shift_range=2,
            shear_range=0.0,
            blur_prob=0.0,
            gaussian_noise_std=0.0,
            salt_pepper_prob=0.0,
            intensity_range=(245, 255),
            random_scale_range=(0.95, 1.05),
            name=f"basic_rot{rotation_range}_thick{thickness_range[0]}-{thickness_range[1]}",
        )

    @classmethod
    def improved(cls, *, rotation_range: float = 20.0, thickness_range: Tuple[int, int] = (1, 5)):
        return cls(
            rotation_range=rotation_range,
            thickness_range=thickness_range,
            shift_range=3,
            shear_range=0.18,
            blur_prob=0.45,
            blur_radius_range=(0.1, 0.7),
            gaussian_noise_std=0.055,
            salt_pepper_prob=0.004,
            intensity_range=(180, 255),
            random_scale_range=(0.88, 1.12),
            elastic_like_prob=0.35,
            name=f"improved_rot{rotation_range}_thick{thickness_range[0]}-{thickness_range[1]}",
        )


@dataclass(frozen=True)
class TrainConfig:
    """Training hyperparameters shared by all experiments."""

    batch_size: int = 128
    epochs: int = 4
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    val_fraction: float = 0.10
    num_workers: int = 2
    seed: int = 42
    pin_memory: bool = True
