#!/usr/bin/env python3
"""Create paper-ready figures for the CMT successor-lambda ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _save_pair(fig, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")


def _series(
    rows: list[dict[str, Any]],
    *,
    lambda_value: float,
    discriminator: str,
    value: str,
) -> list[dict[str, Any]]:
    return sorted(
        (
            row
            for row in rows
            if float(row["lambda"]) == float(lambda_value)
            and row.get(discriminator) == value
        ),
        key=lambda row: int(row.get("step", 0)),
    )


def _plot_final(summary, output_dir, plt, np) -> None:
    benchmarks = summary["benchmarks"]
    lambdas = summary["expected_lambdas"]
    rows = summary["final_eval_summary"]
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), squeeze=False)
    for axis, benchmark in zip(axes.flat, benchmarks):
        points = sorted(
            (row for row in rows if row["benchmark"] == benchmark),
            key=lambda row: float(row["lambda"]),
        )
        if points:
            x = np.asarray([row["lambda"] for row in points], dtype=float)
            y = np.asarray([row["mean"] for row in points], dtype=float)
            error = np.asarray([row["std"] for row in points], dtype=float)
            axis.errorbar(x, y, yerr=error, marker="o", capsize=3, linewidth=1.8)
            axis.set_xticks(lambdas)
        else:
            axis.text(0.5, 0.5, "No evaluation data", ha="center", va="center")
        axis.set_title(benchmark)
        axis.set_xlabel(r"$\lambda$")
        axis.set_ylabel(summary["metric"])
        axis.grid(alpha=0.25)
    fig.suptitle(r"Final benchmark performance vs. successor weight $\lambda$")
    fig.tight_layout()
    _save_pair(fig, output_dir, "lambda_final_benchmarks")
    plt.close(fig)


def _plot_eval_curves(summary, output_dir, plt, np, colors) -> None:
    benchmarks = summary["benchmarks"]
    rows = summary["eval_curve_summary"]
    fig, axes = plt.subplots(2, 3, figsize=(12.4, 7.0), squeeze=False)
    for axis, benchmark in zip(axes.flat, benchmarks):
        for index, lambda_value in enumerate(summary["expected_lambdas"]):
            curve = _series(
                rows,
                lambda_value=lambda_value,
                discriminator="benchmark",
                value=benchmark,
            )
            if not curve:
                continue
            step = np.asarray([row["step"] for row in curve], dtype=float)
            mean = np.asarray([row["mean"] for row in curve], dtype=float)
            std = np.asarray([row["std"] for row in curve], dtype=float)
            color = colors[index % len(colors)]
            axis.plot(step, mean, label=fr"$\lambda={lambda_value:g}$", color=color)
            if np.any(std > 0):
                axis.fill_between(step, mean - std, mean + std, color=color, alpha=0.16)
        axis.set_title(benchmark)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel(summary["metric"])
        axis.grid(alpha=0.25)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Evaluation curves (mean ± std across seeds)", y=1.02)
    fig.tight_layout()
    _save_pair(fig, output_dir, "lambda_eval_curves")
    plt.close(fig)


def _plot_diagnostic_grid(
    summary,
    output_dir,
    plt,
    np,
    colors,
    *,
    diagnostics: tuple[tuple[str, str], ...],
    stem: str,
    title: str,
    shape: tuple[int, int],
) -> None:
    rows = summary["training_summary"]
    fig, axes = plt.subplots(*shape, figsize=(12.2, 6.8), squeeze=False)
    for axis, (diagnostic, label) in zip(axes.flat, diagnostics):
        any_curve = False
        for index, lambda_value in enumerate(summary["expected_lambdas"]):
            curve = _series(
                rows,
                lambda_value=lambda_value,
                discriminator="diagnostic",
                value=diagnostic,
            )
            if not curve:
                continue
            any_curve = True
            step = np.asarray([row["step"] for row in curve], dtype=float)
            mean = np.asarray([row["mean"] for row in curve], dtype=float)
            std = np.asarray([row["std"] for row in curve], dtype=float)
            color = colors[index % len(colors)]
            axis.plot(step, mean, label=fr"$\lambda={lambda_value:g}$", color=color)
            if np.any(std > 0):
                axis.fill_between(step, mean - std, mean + std, color=color, alpha=0.16)
        if not any_curve:
            axis.text(0.5, 0.5, "Not logged", ha="center", va="center")
        axis.set_title(label)
        axis.set_xlabel("Optimizer step")
        axis.grid(alpha=0.25)
    for axis in axes.flat[len(diagnostics) :]:
        axis.set_visible(False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle(title, y=1.02)
    fig.tight_layout()
    _save_pair(fig, output_dir, stem)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = tuple(plt.get_cmap("tab10").colors)
    _plot_final(summary, args.output_dir, plt, np)
    _plot_eval_curves(summary, args.output_dir, plt, np, colors)
    _plot_diagnostic_grid(
        summary,
        args.output_dir,
        plt,
        np,
        colors,
        diagnostics=(
            ("allocation_kl_final", "Final allocation KL"),
            ("normalized_ess", "Normalized ESS"),
            ("weight_mean", "Mean final weight"),
            ("weight_std", "Std. final weight"),
            ("weight_final_max", "Maximum final weight"),
            ("fraction_at_weight_max", "Fraction at upper bound"),
        ),
        stem="lambda_allocation_diagnostics",
        title="Allocation behavior across successor weights",
        shape=(2, 3),
    )
    _plot_diagnostic_grid(
        summary,
        args.output_dir,
        plt,
        np,
        colors,
        diagnostics=(
            ("lambda_d_over_kappa_q95", r"q95($|\lambda D_t|$) / $\kappa$"),
            ("lambda_d_over_kappa_q99", r"q99($|\lambda D_t|$) / $\kappa$"),
            ("saturation_fraction_gt_1", r"Fraction $|\lambda D_t|\geq\kappa$"),
            ("saturation_fraction_gt_2", r"Fraction $|\lambda D_t|\geq2\kappa$"),
        ),
        stem="lambda_sequential_saturation",
        title=r"Sequential magnitude and tanh$_{q99}$ saturation",
        shape=(2, 2),
    )
    print(f"Wrote PNG and PDF figures to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
