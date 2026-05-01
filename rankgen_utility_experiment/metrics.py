from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.linalg import sqrtm
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int_]


@dataclass(frozen=True)
class MetricResult:
    generator: str
    n_generated: int
    quality: float
    utility_gain: float
    utility_augmented_accuracy: float
    baseline_accuracy: float
    similarity_to_train: float
    fid_to_oracle: float
    precision: float
    recall: float
    distinguishability_auc: float


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


def downstream_utility_gain(
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
    x_aug = np.vstack([x_train, x_generated])
    y_aug = np.concatenate([y_train, y_generated])
    augmented = task_accuracy(x_aug, y_aug, x_test, y_test, seed=seed)
    return augmented - baseline, augmented, baseline


def oracle_quality(
    x_generated: FloatArray,
    y_generated: IntArray,
    x_oracle: FloatArray,
    y_oracle: IntArray,
    *,
    k: int = 9,
) -> float:
    oracle = KNeighborsClassifier(n_neighbors=k, weights="distance")
    oracle.fit(x_oracle, y_oracle)
    return float(accuracy_score(y_generated, oracle.predict(x_generated)))


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


def distinguishability_auc(
    x_real: FloatArray,
    x_generated: FloatArray,
    *,
    seed: int,
) -> float:
    x = np.vstack([x_real, x_generated])
    y = np.concatenate(
        [np.zeros(len(x_real), dtype=int), np.ones(len(x_generated), dtype=int)]
    )
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1_000, class_weight="balanced", random_state=seed),
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    scores = cross_val_predict(clf, x, y, cv=cv, method="decision_function")
    auc = roc_auc_score(y, scores)
    return float(max(auc, 1.0 - auc))


def random_forest_distinguishability_auc(
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
    scores = cross_val_predict(clf, x, y, cv=cv, method="predict_proba")[:, 1]
    auc = roc_auc_score(y, scores)
    return float(max(auc, 1.0 - auc))


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
    utility_gain, augmented_acc, baseline_acc = downstream_utility_gain(
        x_train,
        y_train,
        x_generated,
        y_generated,
        x_test,
        y_test,
        seed=seed,
    )
    precision, recall = manifold_precision_recall(x_oracle, x_generated)
    return MetricResult(
        generator=generator,
        n_generated=len(x_generated),
        quality=oracle_quality(x_generated, y_generated, x_oracle, y_oracle),
        utility_gain=utility_gain,
        utility_augmented_accuracy=augmented_acc,
        baseline_accuracy=baseline_acc,
        similarity_to_train=mean_nearest_train_distance(x_generated, x_train),
        fid_to_oracle=frechet_distance(x_oracle, x_generated),
        precision=precision,
        recall=recall,
        distinguishability_auc=random_forest_distinguishability_auc(
            x_oracle,
            x_generated,
            seed=seed,
            max_real=rf_max_real_samples,
            n_estimators=rf_n_estimators,
            max_depth=rf_max_depth,
            min_samples_leaf=rf_min_samples_leaf,
            max_features=rf_max_features,
            n_jobs=rf_n_jobs,
        ),
    )
