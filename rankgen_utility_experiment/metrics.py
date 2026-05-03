from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import sqrtm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.svm import SVC


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class MetricResult:
    generator: str
    n_generated: int
    quality: float
    utility: float
    indistinguishability: float
    similarity: float
    baseline_accuracy: float
    generated_only_accuracy: float
    real_augmented_accuracy: float
    generated_augmented_accuracy: float
    real_augmentation_gain: float
    generated_augmentation_gain: float
    fid_to_oracle: float
    precision: float
    recall: float
    distinguishability_accuracy: float


def make_task_classifier(seed: int = 0) -> SVC:
    return SVC(C=12.0, gamma="scale", kernel="rbf", random_state=seed)


def task_accuracy(
    x_train: FloatArray,
    y_train: IntArray,
    x_test: FloatArray,
    y_test: IntArray,
    *,
    seed: int,
) -> float:
    clf = make_task_classifier(seed)
    clf.fit(x_train, y_train)
    return float(accuracy_score(y_test, clf.predict(x_test)))


def downstream_utility(
    x_train: FloatArray,
    y_train: IntArray,
    x_generated: FloatArray,
    y_generated: IntArray,
    x_real_aug: FloatArray,
    y_real_aug: IntArray,
    x_test: FloatArray,
    y_test: IntArray,
    *,
    seed: int,
) -> tuple[float, float, float, float, float, float]:
    baseline = task_accuracy(x_train, y_train, x_test, y_test, seed=seed)
    x_real = np.vstack([x_train, x_real_aug])
    y_real = np.concatenate([y_train, y_real_aug])
    real_augmented = task_accuracy(x_real, y_real, x_test, y_test, seed=seed)
    x_gen = np.vstack([x_train, x_generated])
    y_gen = np.concatenate([y_train, y_generated])
    generated_augmented = task_accuracy(x_gen, y_gen, x_test, y_test, seed=seed)
    real_gain = real_augmented - baseline
    generated_gain = generated_augmented - baseline
    if real_gain <= 1.0e-12:
        utility = 0.0
    else:
        utility = np.clip(generated_gain / real_gain, 0.0, 1.0)
    return (
        float(utility),
        float(baseline),
        float(real_augmented),
        float(generated_augmented),
        float(real_gain),
        float(generated_gain),
    )


def generated_quality(
    x_train: FloatArray,
    y_train: IntArray,
    x_generated: FloatArray,
    y_generated: IntArray,
    x_test: FloatArray,
    y_test: IntArray,
    *,
    seed: int,
) -> tuple[float, float, float]:
    baseline = task_accuracy(x_train, y_train, x_test, y_test, seed=seed)
    generated_only = task_accuracy(
        x_generated,
        y_generated,
        x_test,
        y_test,
        seed=seed,
    )
    if baseline <= 1.0e-12:
        quality = 0.0
    else:
        quality = np.clip(generated_only / baseline, 0.0, 1.0)
    return float(quality), float(generated_only), float(baseline)


def mean_nearest_train_distance(x_generated: FloatArray, x_train: FloatArray) -> float:
    nn = NearestNeighbors(n_neighbors=1).fit(x_train)
    dist, _ = nn.kneighbors(x_generated)
    return float(dist.mean())


def frechet_distance(x_real: FloatArray, x_generated: FloatArray) -> float:
    mu_real = x_real.mean(axis=0)
    mu_gen = x_generated.mean(axis=0)
    cov_real = np.cov(x_real, rowvar=False)
    cov_gen = np.cov(x_generated, rowvar=False)
    cov_prod_sqrt = sqrtm(cov_real @ cov_gen)
    if np.iscomplexobj(cov_prod_sqrt):
        cov_prod_sqrt = cov_prod_sqrt.real
    return float(
        np.sum((mu_real - mu_gen) ** 2)
        + np.trace(cov_real + cov_gen - 2.0 * cov_prod_sqrt)
    )


def manifold_precision_recall(
    x_real: FloatArray,
    x_generated: FloatArray,
    *,
    k: int = 5,
) -> tuple[float, float]:
    real_nn = NearestNeighbors(n_neighbors=min(k + 1, len(x_real))).fit(x_real)
    real_dist, _ = real_nn.kneighbors(x_real)
    real_radius = real_dist[:, -1]

    gen_nn = NearestNeighbors(n_neighbors=min(k + 1, len(x_generated))).fit(
        x_generated
    )
    gen_dist, _ = gen_nn.kneighbors(x_generated)
    gen_radius = gen_dist[:, -1]

    real_to_gen = NearestNeighbors(n_neighbors=1).fit(x_real).kneighbors(
        x_generated, return_distance=True
    )[0][:, 0]
    gen_to_real = NearestNeighbors(n_neighbors=1).fit(x_generated).kneighbors(
        x_real, return_distance=True
    )[0][:, 0]

    nearest_real = NearestNeighbors(n_neighbors=1).fit(x_real).kneighbors(
        x_generated, return_distance=False
    )[:, 0]
    nearest_gen = NearestNeighbors(n_neighbors=1).fit(x_generated).kneighbors(
        x_real, return_distance=False
    )[:, 0]

    precision = np.mean(real_to_gen <= real_radius[nearest_real])
    recall = np.mean(gen_to_real <= gen_radius[nearest_gen])
    return float(precision), float(recall)


def random_forest_distinguishability_accuracy(
    x_real: FloatArray,
    x_generated: FloatArray,
    *,
    seed: int,
    max_real: int = 2_000,
    n_estimators: int = 100,
    max_depth: int | None = None,
    min_samples_leaf: int = 4,
    max_features: str | float | int | None = "sqrt",
    n_jobs: int | None = -1,
) -> float:
    if len(x_real) > max_real:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(x_real), size=max_real, replace=False)
        x_real = x_real[idx]

    x = np.vstack([x_real, x_generated])
    y = np.concatenate(
        [np.zeros(len(x_real), dtype=int), np.ones(len(x_generated), dtype=int)]
    )
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        max_features=max_features,
        n_jobs=n_jobs,
        random_state=seed,
        class_weight="balanced",
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    pred = cross_val_predict(clf, x, y, cv=cv, method="predict")
    accuracy = accuracy_score(y, pred)
    return float(max(accuracy, 1.0 - accuracy))


def indistinguishability_score(domain_accuracy: float) -> float:
    return float(np.clip(1.0 - 2.0 * abs(domain_accuracy - 0.5), 0.0, 1.0))


def same_domain_neighbor_entropy(
    x_real: FloatArray,
    x_generated: FloatArray,
    *,
    k: int = 10,
) -> float:
    x = np.vstack([x_real, x_generated])
    domain = np.concatenate(
        [np.zeros(len(x_real), dtype=int), np.ones(len(x_generated), dtype=int)]
    )
    if len(x) <= 1:
        return 0.0
    n_neighbors = min(k + 1, len(x))
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine").fit(x)
    neighbors = nn.kneighbors(x, return_distance=False)
    entropies = []
    for row, neighbor_idx in enumerate(neighbors):
        neighbor_idx = neighbor_idx[neighbor_idx != row][:k]
        if len(neighbor_idx) == 0:
            entropies.append(0.0)
            continue
        p = float(np.mean(domain[neighbor_idx] == domain[row]))
        if p <= 0.0 or p >= 1.0:
            entropies.append(0.0)
        else:
            entropies.append(-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) / np.log(2.0))
    return float(np.mean(entropies))


def sample_real_augmentation_like(
    x_real: FloatArray,
    y_real: IntArray,
    y_reference: IntArray,
    *,
    rng: np.random.Generator,
) -> tuple[FloatArray, IntArray]:
    xs: list[FloatArray] = []
    ys: list[IntArray] = []
    for label in np.unique(y_reference):
        n_label = int(np.sum(y_reference == label))
        candidates = np.flatnonzero(y_real == label)
        if len(candidates) == 0:
            raise ValueError(f"no held-out real samples found for class {label}")
        chosen = rng.choice(candidates, size=n_label, replace=len(candidates) < n_label)
        xs.append(x_real[chosen])
        ys.append(np.full(n_label, label, dtype=int))
    return np.vstack(xs), np.concatenate(ys)


def evaluate_generator(
    generator: str,
    x_train: FloatArray,
    y_train: IntArray,
    x_generated: FloatArray,
    y_generated: IntArray,
    x_test: FloatArray,
    y_test: IntArray,
    x_oracle: FloatArray,
    y_oracle: IntArray,
    *,
    seed: int,
    rf_n_estimators: int = 100,
    rf_max_depth: int | None = None,
    rf_min_samples_leaf: int = 4,
    rf_max_features: str | float | int | None = "sqrt",
    rf_max_real_samples: int = 2_000,
    rf_n_jobs: int | None = -1,
) -> MetricResult:
    rng = np.random.default_rng(seed + 50_000)
    x_real_aug, y_real_aug = sample_real_augmentation_like(
        x_oracle,
        y_oracle,
        y_generated,
        rng=rng,
    )
    quality, generated_only_acc, quality_baseline_acc = generated_quality(
        x_train,
        y_train,
        x_generated,
        y_generated,
        x_test,
        y_test,
        seed=seed,
    )
    (
        utility,
        baseline_acc,
        real_augmented_acc,
        generated_augmented_acc,
        real_gain,
        generated_gain,
    ) = downstream_utility(
        x_train,
        y_train,
        x_generated,
        y_generated,
        x_real_aug,
        y_real_aug,
        x_test,
        y_test,
        seed=seed,
    )
    if abs(quality_baseline_acc - baseline_acc) > 1.0e-12:
        raise RuntimeError("quality and utility baselines disagree")
    precision, recall = manifold_precision_recall(x_oracle, x_generated)
    domain_accuracy = random_forest_distinguishability_accuracy(
        x_oracle,
        x_generated,
        seed=seed,
        max_real=rf_max_real_samples,
        n_estimators=rf_n_estimators,
        max_depth=rf_max_depth,
        min_samples_leaf=rf_min_samples_leaf,
        max_features=rf_max_features,
        n_jobs=rf_n_jobs,
    )
    return MetricResult(
        generator=generator,
        n_generated=len(x_generated),
        quality=quality,
        utility=utility,
        indistinguishability=indistinguishability_score(domain_accuracy),
        similarity=same_domain_neighbor_entropy(x_train, x_generated),
        baseline_accuracy=baseline_acc,
        generated_only_accuracy=generated_only_acc,
        real_augmented_accuracy=real_augmented_acc,
        generated_augmented_accuracy=generated_augmented_acc,
        real_augmentation_gain=real_gain,
        generated_augmentation_gain=generated_gain,
        fid_to_oracle=frechet_distance(x_oracle, x_generated),
        precision=precision,
        recall=recall,
        distinguishability_accuracy=domain_accuracy,
    )
