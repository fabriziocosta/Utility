from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PAPER_FIGURE_DIR = Path("figures")
PAPER_RESULTS_DIR = Path("results")


def save_paper_pdf(
    fig: plt.Figure,
    filename: str | Path,
    *,
    output_dir: str | Path = PAPER_FIGURE_DIR,
) -> Path:
    """Save a Matplotlib figure as a conference-paper-ready PDF."""

    path = Path(filename)
    if path.suffix and path.suffix.lower() != ".pdf":
        path = path.with_suffix(".pdf")
    elif not path.suffix:
        path = path.with_suffix(".pdf")

    if not path.is_absolute():
        path = Path(output_dir) / path

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        path,
        format="pdf",
        bbox_inches="tight",
        pad_inches=0.02,
        dpi=300,
        facecolor="white",
        metadata={"Creator": "rankgen-utility-experiment"},
    )
    return path


def save_results_csv(
    df: pd.DataFrame,
    filename: str | Path,
    *,
    output_dir: str | Path = PAPER_RESULTS_DIR,
    index: bool = True,
) -> Path:
    """Save an experiment results DataFrame as a CSV file."""

    path = Path(filename)
    if path.suffix and path.suffix.lower() != ".csv":
        path = path.with_suffix(".csv")
    elif not path.suffix:
        path = path.with_suffix(".csv")

    if not path.is_absolute():
        path = Path(output_dir) / path

    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=index)
    return path
