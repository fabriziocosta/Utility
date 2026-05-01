from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class SpiralDataset:
    x_train: FloatArray
    y_train: IntArray
    x_test: FloatArray
    y_test: IntArray
    x_oracle: FloatArray
    y_oracle: IntArray


def _spiral_points(
    n_per_class: int,
    *,
    turns: float,
    noise: float,
    rng: np.random.Generator,
    t_min: float,
    t_max: float,
) -> tuple[FloatArray, IntArray]:
    t = np.linspace(t_min, t_max, n_per_class)
    radius = t / t_max

    x0 = np.column_stack([radius * np.cos(turns * t), radius * np.sin(turns * t)])
    x1 = -x0
    x = np.vstack([x0, x1])
    if noise > 0:
        x = x + rng.normal(scale=noise, size=x.shape)
    y = np.concatenate(
        [np.zeros(n_per_class, dtype=int), np.ones(n_per_class, dtype=int)]
    )
    return x.astype(float), y


def _sparse_indices(
    n_per_class: int,
    train_per_class: int,
    *,
    rng: np.random.Generator,
) -> IntArray:
    if train_per_class > n_per_class:
        raise ValueError("train_per_class cannot exceed n_per_class")

    # Stratified sparse sampling with mild jitter. This leaves visible gaps along
    # the spiral, which is where transfer-difference samples can help.
    anchors = np.linspace(0, n_per_class - 1, train_per_class, dtype=int)
    step = max(1, n_per_class // train_per_class)
    jitter = rng.integers(-step // 3, step // 3 + 1, size=train_per_class)
    idx = np.clip(anchors + jitter, 0, n_per_class - 1)
    return np.unique(idx)


def make_two_spirals(
    *,
    train_per_class: int = 18,
    test_per_class: int = 2_000,
    oracle_per_class: int = 5_000,
    turns: float = 3.6,
    noise: float = 0.025,
    seed: int = 0,
) -> SpiralDataset:
    """Build sparse-train and dense-test two-spiral data.

    The oracle set is dense and generated from the same process. Metrics use it
    only as a high-resolution stand-in for the data manifold, not as training
    data for downstream augmentation.
    """

    rng = np.random.default_rng(seed)
    x_oracle, y_oracle = _spiral_points(
        oracle_per_class,
        turns=turns,
        noise=noise,
        rng=rng,
        t_min=0.45,
        t_max=4.2,
    )
    x_test, y_test = _spiral_points(
        test_per_class,
        turns=turns,
        noise=noise,
        rng=rng,
        t_min=0.45,
        t_max=4.2,
    )

    class0 = _sparse_indices(oracle_per_class, train_per_class, rng=rng)
    class1 = class0 + oracle_per_class
    train_idx = np.concatenate([class0, class1])

    return SpiralDataset(
        x_train=x_oracle[train_idx].copy(),
        y_train=y_oracle[train_idx].copy(),
        x_test=x_test,
        y_test=y_test,
        x_oracle=x_oracle,
        y_oracle=y_oracle,
    )
