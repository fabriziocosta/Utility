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
        ("quality", "mean"),
        ("quality", "std"),
        ("utility", "mean"),
        ("utility", "std"),
        ("indistinguishability", "mean"),
        ("indistinguishability", "std"),
        ("similarity", "mean"),
        ("similarity", "std"),
    ]
    print(summary[columns].round(4).to_string())


if __name__ == "__main__":
    main()
