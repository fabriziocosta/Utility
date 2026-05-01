from __future__ import annotations

from rankgen_utility_experiment.experiment import (
    ExperimentConfig,
    run_many,
    summarize,
)


def main() -> None:
    config = ExperimentConfig()
    results = run_many(list(range(7, 12)), config)
    summary = summarize(results)
    columns = [
        ("utility_gain", "mean"),
        ("utility_gain", "std"),
        ("quality", "mean"),
        ("similarity_to_train", "mean"),
        ("fid_to_oracle", "mean"),
        ("precision", "mean"),
        ("recall", "mean"),
    ]
    print(summary[columns].round(4).to_string())


if __name__ == "__main__":
    main()
