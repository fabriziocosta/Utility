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

METRIC_DIRECTIONS = {
    "quality": "up",
    "utility": "up",
    "indistinguishability": "up",
    "similarity": "up",
    "baseline_accuracy": "up",
    "generated_only_accuracy": "up",
    "real_augmented_accuracy": "up",
    "generated_augmented_accuracy": "up",
    "real_augmentation_gain": "up",
    "generated_augmentation_gain": "up",
    "fid_to_oracle": "down",
    "precision": "up",
    "recall": "up",
    "distinguishability_accuracy": "down",
}

METRIC_LABELS = {
    "quality": "Quality",
    "utility": "Utility",
    "indistinguishability": "Indistinguishability",
    "similarity": "Similarity",
    "baseline_accuracy": "Baseline accuracy",
    "generated_only_accuracy": "Generated-only accuracy",
    "real_augmented_accuracy": "Real-augmented accuracy",
    "generated_augmented_accuracy": "Generated-augmented accuracy",
    "real_augmentation_gain": "Real augmentation gain",
    "generated_augmentation_gain": "Generated augmentation gain",
    "fid_to_oracle": "FID-like distance",
    "precision": "Precision",
    "recall": "Recall",
    "distinguishability_accuracy": "Distinguishability accuracy",
}

METRIC_ARROWS = {
    "up": "↑",
    "down": "↓",
}

GENERATOR_LABELS = {
    "SMOTE interpolation": "SMOTE",
    "Transferred local differences": "TLD",
    "Random noise": "Noise",
}


@dataclass(frozen=True)
class ExperimentConfig:
    seed: int = 7
    train_per_class: int = 100
    test_per_class: int = 2_000
    oracle_per_class: int = 3_000
    generated_per_class: int = 500
    generator_latent_components: int | None = 4
    smote_neighbors: int = 4
    smote_lambda: float = 0.5
    transfer_ab_neighbors: int = 2
    transfer_bc_neighbors: int = 4
    transfer_lambda: float = 1.0
    rf_n_estimators: int = 100
    rf_max_depth: int | None = None
    rf_min_samples_leaf: int = 4
    rf_max_features: str | float | int | None = "sqrt"
    rf_max_real_samples: int = 2_000
    rf_n_jobs: int | None = -1
    turns: float = 2.5
    noise: float = 0.08


def default_generators(
    config: ExperimentConfig,
    *,
    include_noise_model: bool = False,
) -> list[Generator]:
    generators: list[Generator] = [
        SmoteGenerator(
            k=config.smote_neighbors,
            latent_components=config.generator_latent_components,
            lambda_=config.smote_lambda,
        ),
        TransferDifferenceGenerator(
            k_ab=config.transfer_ab_neighbors,
            k_bc=config.transfer_bc_neighbors,
            latent_components=config.generator_latent_components,
            lambda_=config.transfer_lambda,
        ),
    ]
    if include_noise_model:
        generators.append(NoiseGenerator())
    return generators


def run_experiment(
    config: ExperimentConfig,
    generators: list[Generator] | None = None,
    *,
    include_noise_model: bool = False,
) -> tuple[SpiralDataset, dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    data = make_two_spirals(
        train_per_class=config.train_per_class,
        test_per_class=config.test_per_class,
        oracle_per_class=config.oracle_per_class,
        turns=config.turns,
        noise=config.noise,
        seed=config.seed,
    )
    return run_on_dataset(
        config,
        data,
        generators=generators,
        include_noise_model=include_noise_model,
    )


def run_on_dataset(
    config: ExperimentConfig,
    data: SpiralDataset,
    generators: list[Generator] | None = None,
    *,
    include_noise_model: bool = False,
) -> tuple[SpiralDataset, dict[str, tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    generated: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    rows: list[MetricResult] = []

    if generators is None:
        generators = default_generators(
            config,
            include_noise_model=include_noise_model,
        )

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
    *,
    include_noise_model: bool = False,
) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        config = ExperimentConfig(**{**asdict(base_config), "seed": seed})
        _, _, result = run_experiment(
            config,
            generators=generators,
            include_noise_model=include_noise_model,
        )
        frames.append(result)
    return pd.concat(frames, ignore_index=True)


DEFAULT_SUMMARY_METRICS = [
    "quality",
    "utility",
    "indistinguishability",
    "similarity",
]


def summarize(
    results: pd.DataFrame,
    metrics: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    if metrics is None:
        metrics = DEFAULT_SUMMARY_METRICS
    summary = results.groupby("generator")[metrics].agg(["mean", "std"])
    return summary.sort_values(("utility", "mean"), ascending=False)


def metric_label(metric: str, *, human_readable: bool = True) -> str:
    label = METRIC_LABELS.get(metric, metric) if human_readable else metric
    direction = METRIC_DIRECTIONS.get(metric)
    if direction is None:
        return label
    return f"{label} {METRIC_ARROWS[direction]}"


def label_metric_columns(
    df: pd.DataFrame,
    *,
    human_readable: bool = False,
) -> pd.DataFrame:
    labeled = df.copy()
    if isinstance(labeled.columns, pd.MultiIndex):
        labeled.columns = pd.MultiIndex.from_tuples(
            (
                metric_label(column[0], human_readable=human_readable),
                *column[1:],
            )
            if column and column[0] in METRIC_DIRECTIONS
            else column
            for column in labeled.columns
        )
    else:
        labeled = labeled.rename(
            columns={
                column: metric_label(column, human_readable=human_readable)
                for column in labeled.columns
                if column in METRIC_DIRECTIONS
            }
        )
    return labeled


def generator_label(generator: str) -> str:
    return GENERATOR_LABELS.get(generator, generator)


def _autoscale_y_range(ax: plt.Axes, values: pd.Series, errors: pd.Series) -> None:
    lower = float((values - errors).min())
    upper = float((values + errors).max())
    if lower == upper:
        margin = max(abs(lower) * 0.05, 0.01)
    else:
        margin = (upper - lower) * 0.08
    ax.set_ylim(lower - margin, upper + margin)


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
        ax.set_title(generator_label(name))

    for ax in axes:
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])

    fig.tight_layout()
    return fig


def plot_metric_bars(results: pd.DataFrame) -> plt.Figure:
    plot_df = results.copy()
    metrics = [
        "quality",
        "utility",
        "indistinguishability",
        "similarity",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(13, 3.5))
    axes = axes.ravel()
    order = (
        plot_df.groupby("generator")["utility"]
        .mean()
        .sort_values(ascending=False)
        .index
        .tolist()
    )

    for ax, metric in zip(axes, metrics):
        grouped = (
            plot_df.groupby("generator")[metric]
            .agg(["mean", "std"])
            .fillna(0.0)
            .loc[order]
        )
        ax.bar(
            [generator_label(generator) for generator in grouped.index],
            grouped["mean"],
            yerr=grouped["std"],
            capsize=3,
            color="#4C78A8",
        )
        ax.set_title(metric_label(metric))
        ax.tick_params(axis="x", rotation=35)
        ax.axhline(0, color="black", linewidth=0.8)
        _autoscale_y_range(ax, grouped["mean"], grouped["std"])

    fig.tight_layout()
    return fig
