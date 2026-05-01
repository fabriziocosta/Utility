from __future__ import annotations

from dataclasses import asdict, dataclass

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import SpiralDataset, make_two_spirals
from .generators import (
    Generator,
    NoiseGenerator,
    SmoteGenerator,
    TransferDifferenceGenerator,
)
from .metrics import MetricResult, evaluate_generator


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 7
    train_per_class: int = 100
    test_per_class: int = 2_000
    oracle_per_class: int = 3_000
    generated_per_class: int = 500
    smote_neighbors: int = 4
    transfer_ab_neighbors: int = 2
    transfer_bc_neighbors: int = 4
    rf_n_estimators: int = 100
    rf_max_depth: int | None = None
    rf_min_samples_leaf: int = 4
    rf_max_features: str | float | int | None = "sqrt"
    rf_max_real_samples: int = 2_000
    rf_n_jobs: int | None = -1
    turns: float = 2.5
    noise: float = 0.08


def default_generators(config: ExperimentConfig) -> list[Generator]:
    return [
        SmoteGenerator(k=config.smote_neighbors),
        TransferDifferenceGenerator(
            k_ab=config.transfer_ab_neighbors,
            k_bc=config.transfer_bc_neighbors,
        ),
        NoiseGenerator(),
    ]


def run_experiment(
    config: ExperimentConfig,
    generators: list[Generator] | None = None,
) -> tuple[SpiralDataset, dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    data = make_two_spirals(
        train_per_class=config.train_per_class,
        test_per_class=config.test_per_class,
        oracle_per_class=config.oracle_per_class,
        turns=config.turns,
        noise=config.noise,
        seed=config.seed,
    )
    return run_on_dataset(config, data, generators=generators)


def run_on_dataset(
    config: ExperimentConfig,
    data: SpiralDataset,
    generators: list[Generator] | None = None,
) -> tuple[SpiralDataset, dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    generated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[MetricResult] = []

    if generators is None:
        generators = default_generators(config)

    for offset, generator in enumerate(generators):
        rng = np.random.default_rng(config.seed + 10_000 + offset)
        x_gen, y_gen = generator.sample(
            data.x_train,
            data.y_train,
            config.generated_per_class,
            rng=rng,
        )
        generated[generator.name] = (x_gen, y_gen)
        rows.append(
            evaluate_generator(
                generator.name,
                data.x_train,
                data.y_train,
                x_gen,
                y_gen,
                data.x_test,
                data.y_test,
                data.x_oracle,
                data.y_oracle,
                seed=config.seed + offset,
                rf_n_estimators=config.rf_n_estimators,
                rf_max_depth=config.rf_max_depth,
                rf_min_samples_leaf=config.rf_min_samples_leaf,
                rf_max_features=config.rf_max_features,
                rf_max_real_samples=config.rf_max_real_samples,
                rf_n_jobs=config.rf_n_jobs,
            )
        )

    result = pd.DataFrame([asdict(row) for row in rows])
    result.insert(0, "seed", config.seed)
    return data, generated, result


def run_many(
    seeds: list[int],
    base_config: ExperimentConfig,
    generators: list[Generator] | None = None,
) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        config = ExperimentConfig(**{**asdict(base_config), "seed": seed})
        _, _, result = run_experiment(config, generators=generators)
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "quality",
        "utility_gain",
        "utility_augmented_accuracy",
        "baseline_accuracy",
        "similarity_to_train",
        "fid_to_oracle",
        "precision",
        "recall",
        "distinguishability_auc",
    ]
    summary = results.groupby("generator")[metrics].agg(["mean", "std"])
    return summary.sort_values(("utility_gain", "mean"), ascending=False)


def plot_spirals(
    data: SpiralDataset,
    generated: dict[str, tuple[np.ndarray, np.ndarray]],
    *,
    max_oracle: int = 2_000,
) -> plt.Figure:
    ncols = 1 + len(generated)
    fig, axes = plt.subplots(1, ncols, figsize=(4.2 * ncols, 4.0), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)

    oracle_idx = np.linspace(0, len(data.x_oracle) - 1, min(max_oracle, len(data.x_oracle)), dtype=int)
    axes[0].scatter(
        data.x_oracle[oracle_idx, 0],
        data.x_oracle[oracle_idx, 1],
        c=data.y_oracle[oracle_idx],
        s=2,
        cmap="coolwarm",
        alpha=0.18,
        linewidths=0,
    )
    axes[0].scatter(
        data.x_train[:, 0],
        data.x_train[:, 1],
        c=data.y_train,
        s=38,
        cmap="coolwarm",
        edgecolor="black",
        linewidth=0.5,
    )
    axes[0].set_title("Sparse train over dense manifold")

    for ax, (name, (x_gen, y_gen)) in zip(axes[1:], generated.items()):
        ax.scatter(
            data.x_oracle[oracle_idx, 0],
            data.x_oracle[oracle_idx, 1],
            c=data.y_oracle[oracle_idx],
            s=2,
            cmap="coolwarm",
            alpha=0.10,
            linewidths=0,
        )
        ax.scatter(
            x_gen[:, 0],
            x_gen[:, 1],
            c=y_gen,
            s=12,
            cmap="coolwarm",
            alpha=0.75,
            linewidths=0,
        )
        ax.scatter(
            data.x_train[:, 0],
            data.x_train[:, 1],
            c=data.y_train,
            s=24,
            cmap="coolwarm",
            edgecolor="black",
            linewidth=0.4,
        )
        ax.set_title(name)

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig


def plot_metric_bars(results: pd.DataFrame) -> plt.Figure:
    plot_df = results.copy()
    metrics = [
        ("utility_gain", "Utility gain"),
        ("quality", "Quality"),
        ("similarity_to_train", "Similarity to train"),
        ("fid_to_oracle", "FID-like distance"),
        ("precision", "Precision"),
        ("recall", "Recall"),
        ("distinguishability_auc", "Distinguishability AUC"),
    ]

    fig, axes = plt.subplots(2, 4, figsize=(15, 7))
    axes = axes.ravel()
    order = (
        plot_df.groupby("generator")["utility_gain"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    for ax, (metric, title) in zip(axes, metrics):
        grouped = (
            plot_df.groupby("generator")[metric]
            .agg(["mean", "std"])
            .fillna(0.0)
            .loc[order]
        )
        ax.bar(grouped.index, grouped["mean"], yerr=grouped["std"], capsize=3, color="#4C78A8")
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=35)
        ax.axhline(0, color="black", linewidth=0.8)

    axes[-1].axis("off")
    fig.tight_layout()
    return fig
