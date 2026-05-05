from __future__ import annotations

from rankgen_utility_experiment.experiment import (
    ExperimentConfig,
    plot_metric_bars,
    plot_spirals,
    run_experiment,
    run_many,
    summarize,
)
from rankgen_utility_experiment.figures import save_paper_pdf, save_results_csv


def main() -> None:
    config = ExperimentConfig()
    data, generated, one_run = run_experiment(config)
    one_run_path = save_results_csv(one_run, "two_spirals_one_run", index=False)
    spiral_path = save_paper_pdf(plot_spirals(data, generated), "two_spirals")

    results = run_many(list(range(7, 12)), config)
    results_path = save_results_csv(results, "two_spirals_all_seeds", index=False)
    metrics_path = save_paper_pdf(plot_metric_bars(results), "two_spirals_metrics")
    summary = summarize(results)
    summary_path = save_results_csv(summary, "two_spirals_summary")
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
    print(f"\nSaved paper figures:\n- {spiral_path}\n- {metrics_path}")
    print(
        "\nSaved CSV results:"
        f"\n- {one_run_path}"
        f"\n- {results_path}"
        f"\n- {summary_path}"
    )


if __name__ == "__main__":
    main()
