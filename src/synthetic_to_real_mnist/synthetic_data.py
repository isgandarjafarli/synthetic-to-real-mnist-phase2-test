"""Synthetic MNIST-like digit generation.

Phase 1 used simple geometric primitives.  Phase 2 keeps that idea but adds
controlled parameters so we can study scale, rotation, thickness, blur, noise,
shift, shear, and stroke variation in a reproducible way.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Callable, Dict, Iterable, List, Tuple

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFilter
from torch.utils.data import Dataset

from .config import SyntheticConfig


class SyntheticDigitGenerator:
    """Generate individual 28x28 grayscale digit images using drawing primitives."""

    def __init__(self, config: SyntheticConfig | None = None, seed: int | None = None):
        self.config = config or SyntheticConfig.basic()
        self.rng = np.random.default_rng(seed)
        self.generators: Dict[int, Callable[[], Image.Image]] = {
            0: self._digit_0,
            1: self._digit_1,
            2: self._digit_2,
            3: self._digit_3,
            4: self._digit_4,
            5: self._digit_5,
            6: self._digit_6,
            7: self._digit_7,
            8: self._digit_8,
            9: self._digit_9,
        }

    @property
    def img_size(self) -> int:
        return self.config.img_size

    def _new_canvas(self) -> Tuple[Image.Image, ImageDraw.ImageDraw]:
        img = Image.new("L", (self.img_size, self.img_size), 0)
        return img, ImageDraw.Draw(img)

    def _intensity(self) -> int:
        lo, hi = self.config.intensity_range
        return int(self.rng.integers(lo, hi + 1))

    def _thickness(self) -> int:
        lo, hi = self.config.thickness_range
        return int(self.rng.integers(lo, hi + 1))

    def _jitter(self, value: int, amount: int = 2) -> int:
        return int(value + self.rng.integers(-amount, amount + 1))

    def _line(self, draw: ImageDraw.ImageDraw, xy: Iterable[int], width: int | None = None):
        draw.line(list(map(int, xy)), fill=self._intensity(), width=width or self._thickness(), joint="curve")

    def _arc(self, draw: ImageDraw.ImageDraw, box: Iterable[int], start: int, end: int, width: int | None = None):
        draw.arc(list(map(int, box)), start=start, end=end, fill=self._intensity(), width=width or self._thickness())

    def _ellipse(self, draw: ImageDraw.ImageDraw, box: Iterable[int], width: int | None = None):
        draw.ellipse(list(map(int, box)), outline=self._intensity(), width=width or self._thickness())

    def _digit_0(self) -> Image.Image:
        img, draw = self._new_canvas()
        w = self._thickness()
        cx = self._jitter(14, 2)
        cy = self._jitter(14, 2)
        rx = int(self.rng.integers(7, 11))
        ry = int(self.rng.integers(8, 12))
        self._ellipse(draw, [cx - rx, cy - ry, cx + rx, cy + ry], w)
        return img

    def _digit_1(self) -> Image.Image:
        img, draw = self._new_canvas()
        w = self._thickness()
        x = self._jitter(14, 3)
        y0 = int(self.rng.integers(4, 8))
        y1 = int(self.rng.integers(21, 25))
        self._line(draw, [x, y0, x, y1], w)
        if self.rng.random() < 0.65:
            self._line(draw, [x - int(self.rng.integers(2, 6)), y0 + int(self.rng.integers(1, 5)), x, y0], max(1, w - 1))
        if self.rng.random() < 0.40:
            self._line(draw, [x - 4, y1, x + 4, y1], max(1, w - 1))
        return img

    def _digit_2(self) -> Image.Image:
        img, draw = self._new_canvas()
        w = self._thickness()
        x0, y0, x1, y1 = 7, 4, 21, 15
        self._arc(draw, [self._jitter(x0), self._jitter(y0), self._jitter(x1), self._jitter(y1)], 190, 25, w)
        self._line(draw, [self._jitter(20), self._jitter(10), self._jitter(8), self._jitter(22)], w)
        self._line(draw, [self._jitter(8), self._jitter(22), self._jitter(21), self._jitter(22)], w)
        return img

    def _digit_3(self) -> Image.Image:
        img, draw = self._new_canvas()
        w = self._thickness()
        self._arc(draw, [self._jitter(8), self._jitter(4), self._jitter(22), self._jitter(14)], 205, 30, w)
        self._arc(draw, [self._jitter(8), self._jitter(13), self._jitter(22), self._jitter(24)], 330, 160, w)
        if self.rng.random() < 0.35:
            self._line(draw, [self._jitter(15), self._jitter(14), self._jitter(20), self._jitter(14)], max(1, w - 1))
        return img

    def _digit_4(self) -> Image.Image:
        img, draw = self._new_canvas()
        w = self._thickness()
        self._line(draw, [self._jitter(18), self._jitter(4), self._jitter(18), self._jitter(24)], w)
        self._line(draw, [self._jitter(9), self._jitter(5), self._jitter(9), self._jitter(16)], w)
        self._line(draw, [self._jitter(9), self._jitter(16), self._jitter(22), self._jitter(16)], w)
        if self.rng.random() < 0.45:
            self._line(draw, [self._jitter(10), self._jitter(5), self._jitter(18), self._jitter(16)], max(1, w - 1))
        return img

    def _digit_5(self) -> Image.Image:
        img, draw = self._new_canvas()
        w = self._thickness()
        self._line(draw, [self._jitter(9), self._jitter(5), self._jitter(21), self._jitter(5)], w)
        self._line(draw, [self._jitter(9), self._jitter(5), self._jitter(9), self._jitter(14)], w)
        self._line(draw, [self._jitter(9), self._jitter(14), self._jitter(17), self._jitter(14)], w)
        self._arc(draw, [self._jitter(8), self._jitter(12), self._jitter(23), self._jitter(25)], 285, 150, w)
        return img

    def _digit_6(self) -> Image.Image:
        img, draw = self._new_canvas()
        w = self._thickness()
        self._arc(draw, [self._jitter(8), self._jitter(4), self._jitter(20), self._jitter(22)], 95, 270, w)
        self._ellipse(draw, [self._jitter(9), self._jitter(13), self._jitter(22), self._jitter(24)], w)
        if self.rng.random() < 0.55:
            self._line(draw, [self._jitter(13), self._jitter(13), self._jitter(18), self._jitter(13)], max(1, w - 1))
        return img

    def _digit_7(self) -> Image.Image:
        img, draw = self._new_canvas()
        w = self._thickness()
        self._line(draw, [self._jitter(7), self._jitter(5), self._jitter(22), self._jitter(5)], w)
        self._line(draw, [self._jitter(22), self._jitter(5), self._jitter(10), self._jitter(24)], w)
        if self.rng.random() < 0.35:
            self._line(draw, [self._jitter(13), self._jitter(15), self._jitter(18), self._jitter(15)], max(1, w - 1))
        return img

    def _digit_8(self) -> Image.Image:
        img, draw = self._new_canvas()
        w = self._thickness()
        self._ellipse(draw, [self._jitter(9), self._jitter(4), self._jitter(20), self._jitter(14)], w)
        self._ellipse(draw, [self._jitter(8), self._jitter(13), self._jitter(22), self._jitter(25)], w)
        if self.rng.random() < 0.35:
            self._line(draw, [self._jitter(11), self._jitter(13), self._jitter(19), self._jitter(14)], max(1, w - 1))
        return img

    def _digit_9(self) -> Image.Image:
        img, draw = self._new_canvas()
        w = self._thickness()
        self._ellipse(draw, [self._jitter(8), self._jitter(4), self._jitter(22), self._jitter(16)], w)
        self._arc(draw, [self._jitter(12), self._jitter(7), self._jitter(23), self._jitter(25)], 270, 92, w)
        if self.rng.random() < 0.50:
            self._line(draw, [self._jitter(17), self._jitter(15), self._jitter(12), self._jitter(15)], max(1, w - 1))
        return img

    def _apply_global_transforms(self, img: Image.Image) -> Image.Image:
        cfg = self.config
        # Scale around center by resizing then pasting back to 28x28.
        scale = float(self.rng.uniform(*cfg.random_scale_range))
        if abs(scale - 1.0) > 1e-3:
            new_size = max(14, min(34, int(round(self.img_size * scale))))
            resized = img.resize((new_size, new_size), Image.Resampling.BILINEAR)
            canvas = Image.new("L", (self.img_size, self.img_size), 0)
            left = (self.img_size - new_size) // 2
            top = (self.img_size - new_size) // 2
            canvas.paste(resized, (left, top))
            img = canvas.crop((0, 0, self.img_size, self.img_size))

        # Shear creates slanted handwriting-like variation.
        if cfg.shear_range > 0:
            shear = float(self.rng.uniform(-cfg.shear_range, cfg.shear_range))
            try:
                transform_mode = Image.Transform.AFFINE
            except AttributeError:  # older Pillow
                transform_mode = Image.AFFINE
            img = img.transform(
                (self.img_size, self.img_size),
                transform_mode,
                (1, shear, -shear * self.img_size / 2, 0, 1, 0),
                resample=Image.Resampling.BILINEAR,
                fillcolor=0,
            )

        # Rotation and translation were already used in Phase 1; now parameterized.
        angle = float(self.rng.uniform(-cfg.rotation_range, cfg.rotation_range))
        img = img.rotate(angle, resample=Image.Resampling.BILINEAR, fillcolor=0)

        if cfg.shift_range > 0:
            dx = int(self.rng.integers(-cfg.shift_range, cfg.shift_range + 1))
            dy = int(self.rng.integers(-cfg.shift_range, cfg.shift_range + 1))
            shifted = Image.new("L", (self.img_size, self.img_size), 0)
            shifted.paste(img, (dx, dy))
            img = shifted

        if cfg.blur_prob > 0 and self.rng.random() < cfg.blur_prob:
            radius = float(self.rng.uniform(*cfg.blur_radius_range))
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))

        return img

    def _postprocess_array(self, img: Image.Image) -> np.ndarray:
        arr = np.asarray(img, dtype=np.float32) / 255.0
        cfg = self.config

        if cfg.gaussian_noise_std > 0:
            noise = self.rng.normal(0.0, cfg.gaussian_noise_std, size=arr.shape).astype(np.float32)
            arr = arr + noise

        if cfg.salt_pepper_prob > 0:
            mask = self.rng.random(arr.shape)
            arr = np.where(mask < cfg.salt_pepper_prob / 2, 0.0, arr)
            arr = np.where(mask > 1 - cfg.salt_pepper_prob / 2, 1.0, arr)

        if cfg.elastic_like_prob > 0 and self.rng.random() < cfg.elastic_like_prob:
            # Cheap local distortion: roll a few rows/columns by one pixel.
            for _ in range(int(self.rng.integers(1, 4))):
                row = int(self.rng.integers(5, self.img_size - 5))
                arr[row : row + 2] = np.roll(arr[row : row + 2], int(self.rng.choice([-1, 1])), axis=1)
            for _ in range(int(self.rng.integers(1, 3))):
                col = int(self.rng.integers(5, self.img_size - 5))
                arr[:, col : col + 1] = np.roll(arr[:, col : col + 1], int(self.rng.choice([-1, 1])), axis=0)

        return np.clip(arr, 0.0, 1.0).astype(np.float32)

    def generate_one(self, digit: int) -> np.ndarray:
        if digit not in self.generators:
            raise ValueError(f"digit must be in 0..9, got {digit}")
        img = self.generators[int(digit)]()
        img = self._apply_global_transforms(img)
        return self._postprocess_array(img)

    def config_dict(self) -> dict:
        return asdict(self.config)


class SyntheticMNISTDataset(Dataset):
    """PyTorch Dataset wrapper around generated numpy arrays."""

    def __init__(self, data: np.ndarray, labels: np.ndarray):
        if data.ndim != 3:
            raise ValueError(f"Expected data shape [N,H,W], got {data.shape}")
        self.data = torch.from_numpy(data.astype(np.float32)).unsqueeze(1)
        self.labels = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return int(self.labels.shape[0])

    def __getitem__(self, idx: int):
        return self.data[idx], self.labels[idx]


def _balanced_labels(num_samples: int, rng: np.random.Generator) -> np.ndarray:
    repeats = num_samples // 10
    remainder = num_samples % 10
    labels = np.repeat(np.arange(10), repeats)
    if remainder:
        labels = np.concatenate([labels, rng.choice(np.arange(10), size=remainder, replace=False)])
    rng.shuffle(labels)
    return labels.astype(np.int64)


def generate_synthetic_dataset(
    num_samples: int,
    config: SyntheticConfig | None = None,
    seed: int = 42,
    balanced: bool = True,
    progress: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a full synthetic dataset as numpy arrays.

    Returns:
        data: float32 array of shape [N, 28, 28] in [0, 1]
        labels: int64 array of shape [N]
    """

    cfg = config or SyntheticConfig.basic()
    rng = np.random.default_rng(seed)
    generator = SyntheticDigitGenerator(cfg, seed=seed)
    labels = _balanced_labels(num_samples, rng) if balanced else rng.integers(0, 10, size=num_samples, dtype=np.int64)
    data = np.empty((num_samples, cfg.img_size, cfg.img_size), dtype=np.float32)

    for i, digit in enumerate(labels):
        data[i] = generator.generate_one(int(digit))
        if progress and num_samples >= 10000 and (i + 1) % 10000 == 0:
            print(f"generated {i + 1:,}/{num_samples:,} synthetic images")

    return data, labels


def save_npz(path: str, data: np.ndarray, labels: np.ndarray, config: SyntheticConfig | None = None):
    meta = asdict(config) if config is not None else {}
    np.savez_compressed(path, data=data, labels=labels, meta=str(meta))


def load_npz(path: str) -> Tuple[np.ndarray, np.ndarray]:
    loaded = np.load(path)
    return loaded["data"], loaded["labels"]
