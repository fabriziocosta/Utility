# RankGen Utility Experiments

This workspace contains compact experiments for testing whether a
Utility-style probe separates informative augmentation from near-copying when
quality and simple distributional fit look similar. The code compares synthetic
generators that can look similarly plausible, but differ in whether their
samples improve a downstream classifier when used as augmentation.

The thin notebooks are `notebooks/two_spirals_utility.ipynb` and
`notebooks/mnist_utility.ipynb`. Reusable code lives in
`rankgen_utility_experiment/`.

Run the notebook, or execute the core experiment directly:

```bash
python -m pip install numpy scipy scikit-learn pandas matplotlib nbformat
python scripts/run_two_spirals.py
```

## Experiments

### Two spirals

`notebooks/two_spirals_utility.ipynb` runs a controlled two-class spiral
experiment. The sparse training set contains only a small number of labelled
points per class, while a dense oracle/test manifold provides the real-data
reference. This makes it easy to inspect whether a generator only interpolates
near known samples or produces samples that help recover the underlying spiral
structure.

The notebook creates:

- a single-run visual comparison of the sparse training data, oracle manifold,
  and generated samples;
- a single-run CSV of all metrics for the configured seed;
- a multi-seed metric summary for `quality`, `utility`,
  `indistinguishability`, and `similarity`;
- a bar plot of the multi-seed metric means and standard deviations.

### MNIST

`notebooks/mnist_utility.ipynb` runs the same generator and metric pipeline on a
selected subset of MNIST digit classes. The sparse real training set is sampled
per class, with separate held-out test and oracle samples. This experiment is a
higher-dimensional sanity check: generated images are inspected directly, and a
UMAP projection compares generated samples with held-out real digits.

The notebook creates:

- a grid of generated digit samples;
- a UMAP comparison of real oracle samples and generated samples;
- a single-run CSV of all metrics for the configured MNIST seed;
- a multi-seed CSV of all repeated MNIST runs;
- a multi-seed metric summary CSV with means and standard deviations;
- a bar plot of the reported metrics with standard-deviation error bars.

## Output files

Notebook outputs are saved relative to the directory where the notebook is run.
If you run the notebooks from the `notebooks/` directory, retrieve the latest
paper artifacts from:

```text
notebooks/figures/
notebooks/results/
```

The two-spirals notebook writes:

```text
notebooks/figures/two_spirals.pdf
notebooks/figures/two_spirals_metrics.pdf
notebooks/results/two_spirals_one_run.csv
notebooks/results/two_spirals_all_seeds.csv
notebooks/results/two_spirals_summary.csv
```

The MNIST notebook writes:

```text
notebooks/figures/mnist_samples.pdf
notebooks/figures/mnist_umap.pdf
notebooks/figures/mnist_metrics.pdf
notebooks/results/mnist_result.csv
notebooks/results/mnist_all_seeds.csv
notebooks/results/mnist_summary.csv
```

If you run `python scripts/run_two_spirals.py` from the repository root, the
same two-spirals artifacts are written to root-level `figures/` and `results/`
directories instead.

Figures are saved as PDFs for direct inclusion in a conference paper. Results
are saved as CSV files so that parameter sweeps can be rerun in the notebooks
and the corresponding tables can be retrieved without copying from notebook
outputs.

## Configuration

Generator neighbor counts are controlled by `ExperimentConfig.smote_neighbors`,
`ExperimentConfig.transfer_ab_neighbors`, and
`ExperimentConfig.transfer_bc_neighbors`. Generator interpolation scales are
controlled by `ExperimentConfig.smote_lambda` and
`ExperimentConfig.transfer_lambda`.

For the two-spirals experiment, `ExperimentConfig.train_per_class`,
`ExperimentConfig.test_per_class`, and `ExperimentConfig.oracle_per_class`
control the sparse training set, held-out test set, and dense real reference
manifold. `ExperimentConfig.turns` and `ExperimentConfig.noise` control the
spiral geometry.

For MNIST, `MnistConfig.classes`, `MnistConfig.train_per_class`,
`MnistConfig.test_per_class`, and `MnistConfig.oracle_per_class` control the
selected digits and per-class real samples. The downstream metric settings are
still controlled by `ExperimentConfig`.

The random-forest distinguishability probe is controlled by
`ExperimentConfig.rf_n_estimators`, `ExperimentConfig.rf_max_depth`,
`ExperimentConfig.rf_min_samples_leaf`, `ExperimentConfig.rf_max_features`,
`ExperimentConfig.rf_max_real_samples`, and `ExperimentConfig.rf_n_jobs`.

## Generators and metrics

The intended comparison is:

- `SMOTE interpolation`: same-class convex interpolation between sparse train
  points.
- `Transferred local differences`: applies a same-class local difference vector
  from one neighborhood to another point, allowing samples outside a local
  convex segment while staying tangent-like.
- `Random noise`: negative control.

The main reported profile contains the four RankGen base metrics from the paper:
`quality`, `utility`, `indistinguishability`, and `similarity`. Quality is the
generated-only task score relative to the real-only baseline, and Utility is the
generated augmentation gain relative to an equally sized held-out real
augmentation gain.

`test_per_class` controls the held-out set used to measure downstream
task scores. `oracle_per_class` controls a separate dense real sample used for
the real augmentation reference and secondary diagnostics such as FID-like
distance, precision/recall, and real-vs-generated distinguishability.
