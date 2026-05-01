# RankGen Utility Two-Spirals Experiment

This workspace contains a compact synthetic experiment for testing whether a
Utility-style probe separates informative augmentation from near-copying when
quality and simple distributional fit look similar.

The thin notebooks are `notebooks/two_spirals_utility.ipynb` and
`notebooks/mnist_utility.ipynb`. Reusable code lives in
`rankgen_utility_experiment/`.

Run the notebook, or execute the core experiment directly:

```bash
python -m pip install numpy scipy scikit-learn pandas matplotlib nbformat
python scripts/run_two_spirals.py
```

The default configuration uses 100 sparse training points per class against a
dense 3,000-point-per-class oracle/test manifold. Generator neighbor counts are
controlled by `ExperimentConfig.smote_neighbors`,
`ExperimentConfig.transfer_ab_neighbors`, and
`ExperimentConfig.transfer_bc_neighbors`. The random-forest distinguishability
probe is controlled by `ExperimentConfig.rf_n_estimators`,
`ExperimentConfig.rf_max_depth`, `ExperimentConfig.rf_min_samples_leaf`,
`ExperimentConfig.rf_max_features`, and
`ExperimentConfig.rf_max_real_samples`, and `ExperimentConfig.rf_n_jobs`.
The intended comparison is:

- `SMOTE interpolation`: same-class convex interpolation between sparse train
  points.
- `Transferred local differences`: applies a same-class local difference vector
  from one neighborhood to another point, allowing samples outside a local
  convex segment while staying tangent-like.
- `Random noise`: negative control.

The main outcome is `utility_gain`, defined as dense-test accuracy after
augmentation minus dense-test accuracy from the sparse train set alone.

`test_per_class` controls the held-out set used to measure downstream
augmentation gain. `oracle_per_class` controls a separate dense real sample used
only for diagnostics such as quality, FID-like distance, precision/recall, and
real-vs-generated distinguishability.
