from __future__ import annotations

from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from .data import SpiralDataset
from .experiment import generator_label


FloatArray = NDArray[np.float64]


@dataclass(frozen=True)
class MnistConfig:
    train_per_class: int = 100
    test_per_class: int = 300
    oracle_per_class: int = 1_000
    classes: tuple[int, ...] = tuple(range(10))
    seed: int = 7
    normalize: bool = True


def _take_per_class(
    x: FloatArray,
    y: NDArray[np.int_],
    n_per_class: int,
    *,
    classes: tuple[int, ...],
    rng: np.random.Generator,
) -> tuple[FloatArray, NDArray[np.int_]]:
    xs = []
    ys = []
    for label in classes:
        idx = np.flatnonzero(y == label)
        if len(idx) < n_per_class:
            raise ValueError(
                f"class {label} has {len(idx)} samples, need {n_per_class}"
            )
        chosen = rng.choice(idx, size=n_per_class, replace=False)
        xs.append(x[chosen])
        ys.append(y[chosen])
    return np.vstack(xs), np.concatenate(ys)


def _take_indices_per_class(
    y: NDArray[np.int_],
    n_per_class: int,
    *,
    classes: tuple[int, ...],
    rng: np.random.Generator,
) -> NDArray[np.int_]:
    chosen = []
    for label in classes:
        idx = np.flatnonzero(y == label)
        if len(idx) < n_per_class:
            raise ValueError(
                f"class {label} has {len(idx)} samples, need {n_per_class}"
            )
        chosen.append(rng.choice(idx, size=n_per_class, replace=False))
    return np.concatenate(chosen)


def make_mnist_dataset(config: MnistConfig) -> SpiralDataset:
    """Load MNIST and return the same dataset container used by the experiment.

    `x_train` is intentionally sparse and is the only real data used by the
    generators. `x_test` measures downstream augmentation gain. `x_oracle` is a
    larger held-out real sample used for quality and distributional diagnostics.
    """

    mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
    x = mnist.data.astype(float)
    if config.normalize:
        x = x / 255.0
    y = mnist.target.astype(int)

    keep = np.isin(y, np.asarray(config.classes))
    x = x[keep]
    y = y[keep]

    x_pool, x_test_source, y_pool, y_test_source = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=config.seed,
    )

    rng = np.random.default_rng(config.seed)
    train_idx = _take_indices_per_class(
        y_pool,
        config.train_per_class,
        classes=config.classes,
        rng=rng,
    )
    x_train = x_pool[train_idx]
    y_train = y_pool[train_idx]
    x_test, y_test = _take_per_class(
        x_test_source,
        y_test_source,
        config.test_per_class,
        classes=config.classes,
        rng=rng,
    )

    remaining_mask = np.ones(len(x_pool), dtype=bool)
    remaining_mask[train_idx] = False
    x_oracle_source = x_pool[remaining_mask]
    y_oracle_source = y_pool[remaining_mask]

    x_oracle, y_oracle = _take_per_class(
        x_oracle_source,
        y_oracle_source,
        config.oracle_per_class,
        classes=config.classes,
        rng=rng,
    )

    return SpiralDataset(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        x_oracle=x_oracle,
        y_oracle=y_oracle,
    )


def plot_mnist_samples(
    generated: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    n_per_generator: int = 12,
    image_shape: tuple[int, int] = (28, 28),
    seed: int | None = 0,
) -> plt.Figure:
    rng = np.random.default_rng(seed)
    nrows = len(generated)
    fig, axes = plt.subplots(
        nrows,
        n_per_generator,
        figsize=(1.15 * n_per_generator, 1.35 * nrows),
        squeeze=False,
    )

    for row, (name, (x_gen, y_gen)) in enumerate(generated.items()):
        labels = np.unique(y_gen)
        if n_per_generator % len(labels) != 0:
            raise ValueError(
                "n_per_generator must be divisible by the number of classes"
            )
        n_per_class = n_per_generator // len(labels)

        selected = []
        for label in labels:
            label_idx = np.flatnonzero(y_gen == label)
            if len(label_idx) < n_per_class:
                raise ValueError(
                    f"class {label} has {len(label_idx)} samples, need {n_per_class}"
                )
            selected.append(rng.choice(label_idx, size=n_per_class, replace=False))
        selected_idx = rng.permutation(np.concatenate(selected))

        for col, idx in enumerate(selected_idx):
            ax = axes[row, col]
            ax.imshow(x_gen[idx].reshape(image_shape), cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(
                    generator_label(name),
                    rotation=0,
                    ha="right",
                    va="center",
                )
            ax.set_title(str(int(y_gen[idx])), fontsize=8)

    fig.tight_layout()
    return fig
