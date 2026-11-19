"""F1: the main model's TPR gap against the share of women in each occupation."""

import json
from pathlib import Path

import numpy as np

from hiring_bias_xai import plotting
from hiring_bias_xai.config import RESULTS_DIR

HEIGHT_INCHES = 3.2


def _read(results_dir: Path, name: str):
    return json.loads((results_dir / name).read_text())


def build(results_dir: Path):
    per_occupation = _read(results_dir, "metrics_deberta.json")["test"]["per_occupation"]
    shares = _read(results_dir, "gap_statistics.json")["main_model"]["occupations"]
    names = list(per_occupation)
    x = np.array([shares[name]["share_women"] for name in names])
    gap = np.array([per_occupation[name]["tpr_gap"]["value"] for name in names])
    low = np.array([per_occupation[name]["tpr_gap"]["ci_low"] for name in names])
    high = np.array([per_occupation[name]["tpr_gap"]["ci_high"] for name in names])

    fig, ax = plotting.new_figure(HEIGHT_INCHES)
    ax.errorbar(x, gap, yerr=[gap - low, high - gap], fmt="none", ecolor=plotting.NEUTRAL, lw=0.6)
    ax.scatter(x, gap, s=12, color=plotting.NEUTRAL, zorder=3)
    ax.set_xlabel("Share of women in the training subsample")
    ax.set_ylabel("TPR gap (women minus men)")
    return fig


def main(results_dir: Path = RESULTS_DIR):
    fig = build(results_dir)
    plotting.save(fig, "f1_gap_vs_share", results_dir / "figures")
    return fig


if __name__ == "__main__":
    main()
