from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray
from sklearn.neighbors import NearestNeighbors


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


class Generator(Protocol):
    name: str

    def sample(
        self,
        x: FloatArray,
        y: IntArray,
        n_per_class: int,
        *,
        rng: np.random.Generator,
    ) -> tuple[FloatArray, IntArray]:
        ...


def _class_neighbors(x_class: FloatArray, k: int) -> tuple[NearestNeighbors, int]:
    if len(x_class) < 2:
        raise ValueError("at least two samples per class are required")
    k_eff = min(k + 1, len(x_class))
    nn = NearestNeighbors(n_neighbors=k_eff).fit(x_class)
    return nn, k_eff


def _choose_neighbor(
    nn: NearestNeighbors,
    x_class: FloatArray,
    row: int,
    *,
    rng: np.random.Generator,
    exclude: set[int] | None = None,
) -> int:
    neighbors = nn.kneighbors(x_class[row : row + 1], return_distance=False)[0]
    neighbors = neighbors[neighbors != row]
    if exclude:
        neighbors = np.array([idx for idx in neighbors if int(idx) not in exclude])
    if len(neighbors) == 0:
        raise ValueError("no eligible neighbor found")
    return int(rng.choice(neighbors))


@dataclass(frozen=True)
class SmoteGenerator:
    """Interpolates between a point and one of its same-class neighbors."""

    k: int = 4
    alpha_low: float = 0.05
    alpha_high: float = 0.95
    name: str = "SMOTE interpolation"

    def sample(
        self,
        x: FloatArray,
        y: IntArray,
        n_per_class: int,
        *,
        rng: np.random.Generator,
    ) -> tuple[FloatArray, IntArray]:
        xs: list[FloatArray] = []
        ys: list[IntArray] = []

        for label in np.unique(y):
            x_class = x[y == label]
            nn, _ = _class_neighbors(x_class, self.k)
            rows = rng.integers(0, len(x_class), size=n_per_class)
            out = np.empty((n_per_class, x.shape[1]), dtype=float)
            for i, row in enumerate(rows):
                neighbor = _choose_neighbor(nn, x_class, int(row), rng=rng)
                alpha = rng.uniform(self.alpha_low, self.alpha_high)
                out[i] = x_class[row] + alpha * (x_class[neighbor] - x_class[row])
            xs.append(out)
            ys.append(np.full(n_per_class, label, dtype=int))

        return np.vstack(xs), np.concatenate(ys)


@dataclass(frozen=True)
class TransferDifferenceGenerator:
    """Transfers a local same-class difference vector to another point.

    For A, B, C in the same class with B near A and C near B, the sample is
    A + alpha * (C - B). Unlike interpolation, this can leave the convex hull of
    the local pair while still using a tangent-like local displacement.
    """

    k_ab: int = 2
    k_bc: int = 4
    alpha_low: float = 0.05
    alpha_high: float = 0.70
    name: str = "Transferred local differences"

    def sample(
        self,
        x: FloatArray,
        y: IntArray,
        n_per_class: int,
        *,
        rng: np.random.Generator,
    ) -> tuple[FloatArray, IntArray]:
        xs: list[FloatArray] = []
        ys: list[IntArray] = []

        for label in np.unique(y):
            x_class = x[y == label]
            nn_ab, _ = _class_neighbors(x_class, self.k_ab)
            nn_bc, _ = _class_neighbors(x_class, self.k_bc)
            rows = rng.integers(0, len(x_class), size=n_per_class)
            out = np.empty((n_per_class, x.shape[1]), dtype=float)
            for i, row in enumerate(rows):
                b = _choose_neighbor(nn_ab, x_class, int(row), rng=rng)
                c = _choose_neighbor(nn_bc, x_class, b, rng=rng, exclude={int(row)})
                alpha = rng.uniform(self.alpha_low, self.alpha_high)
                out[i] = x_class[row] + alpha * (x_class[c] - x_class[b])
            xs.append(out)
            ys.append(np.full(n_per_class, label, dtype=int))

        return np.vstack(xs), np.concatenate(ys)


@dataclass(frozen=True)
class NoiseGenerator:
    """Negative control: samples from a broad bounding box and assigns labels."""

    scale: float = 1.15
    name: str = "Random noise"

    def sample(
        self,
        x: FloatArray,
        y: IntArray,
        n_per_class: int,
        *,
        rng: np.random.Generator,
    ) -> tuple[FloatArray, IntArray]:
        mins = x.min(axis=0) * self.scale
        maxs = x.max(axis=0) * self.scale
        labels = np.unique(y)
        xs = []
        ys = []
        for label in labels:
            xs.append(rng.uniform(mins, maxs, size=(n_per_class, x.shape[1])))
            ys.append(np.full(n_per_class, label, dtype=int))
        return np.vstack(xs), np.concatenate(ys)


def default_generators() -> list[Generator]:
    return [SmoteGenerator(), TransferDifferenceGenerator(), NoiseGenerator()]
