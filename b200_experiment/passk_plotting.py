from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


BENCHMARKS = ("AIME24", "AIME25", "AMC23")


def _slug(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("._-")
    return cleaned.lower() or "unnamed"


def load_passk_rows(summary_paths: Iterable[str | Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw_path in summary_paths:
        path = Path(raw_path).expanduser().resolve()
        payload = json.loads(path.read_text(encoding="utf-8"))
        source_rows = payload.get("results") if isinstance(payload, dict) else payload
        if not isinstance(source_rows, list):
            raise ValueError(f"Pass@K summary has no results list: {path}")
        for row in source_rows:
            missing = {
                "model_group",
                "method",
                "checkpoint",
                "benchmark",
                "k",
                "pass_at_k",
            } - set(row)
            if missing:
                raise ValueError(f"Malformed Pass@K row in {path}: missing {missing}")
            normalized = dict(row)
            normalized["k"] = int(normalized["k"])
            normalized["pass_at_k"] = float(normalized["pass_at_k"])
            normalized.setdefault("label", normalized["method"])
            normalized["_source"] = str(path)
            rows.append(normalized)
    if not rows:
        raise ValueError("No Pass@K result rows were found")
    return rows


def _unique_output_stem(output_dir: Path, requested: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    candidate = output_dir / requested
    if (
        not candidate.with_suffix(".png").exists()
        and not candidate.with_suffix(".pdf").exists()
    ):
        return candidate
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    candidate = output_dir / f"{requested}_{timestamp}"
    suffix = 1
    while (
        candidate.with_suffix(".png").exists() or candidate.with_suffix(".pdf").exists()
    ):
        candidate = output_dir / f"{requested}_{timestamp}_{suffix}"
        suffix += 1
    return candidate


def plot_passk_results(
    rows: Iterable[dict[str, Any]],
    output_dir: str | Path,
    *,
    tag: str,
) -> dict[str, dict[str, str]]:
    # Aggregation remains importable on compute nodes where Matplotlib is not
    # installed; only the plotting call itself needs the optional dependency.
    try:
        import matplotlib.pyplot as plt
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "Pass@K plotting requires matplotlib (pip install matplotlib)"
        ) from error

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["model_group"])].append(dict(row))
    if not grouped:
        raise ValueError("No Pass@K rows were provided for plotting")

    output_dir = Path(output_dir).expanduser().resolve()
    outputs: dict[str, dict[str, str]] = {}
    for model_group, model_rows in sorted(grouped.items()):
        figure, axes = plt.subplots(1, 3, figsize=(15.3, 4.5), constrained_layout=True)
        curve_keys = sorted(
            {
                (
                    str(row.get("label") or row["method"]),
                    str(row["method"]),
                    str(row["checkpoint"]),
                )
                for row in model_rows
            }
        )
        label_counts: dict[str, int] = defaultdict(int)
        for label, _, _ in curve_keys:
            label_counts[label] += 1

        for axis, benchmark in zip(axes, BENCHMARKS):
            benchmark_rows = [
                row for row in model_rows if row["benchmark"] == benchmark
            ]
            for label, method, checkpoint in curve_keys:
                curve = sorted(
                    (
                        row
                        for row in benchmark_rows
                        if str(row.get("label") or row["method"]) == label
                        and str(row["method"]) == method
                        and str(row["checkpoint"]) == checkpoint
                    ),
                    key=lambda row: int(row["k"]),
                )
                if not curve:
                    continue
                legend_label = label
                if label_counts[label] > 1:
                    legend_label = f"{label} ({Path(checkpoint).name})"
                axis.plot(
                    [int(row["k"]) for row in curve],
                    [100.0 * float(row["pass_at_k"]) for row in curve],
                    marker="o",
                    markersize=5.5,
                    linewidth=2.0,
                    label=legend_label,
                )
            axis.set_title(benchmark, fontsize=12, fontweight="semibold")
            axis.set_xlabel("K")
            axis.set_ylabel("Pass@K (%)" if benchmark == BENCHMARKS[0] else "")
            axis.grid(True, linestyle="--", alpha=0.32)
            axis.set_axisbelow(True)
            if benchmark_rows:
                axis.set_xticks(sorted({int(row["k"]) for row in benchmark_rows}))
            else:
                axis.text(
                    0.5,
                    0.5,
                    "No results",
                    transform=axis.transAxes,
                    ha="center",
                    va="center",
                    color="0.45",
                )
        handles, labels = axes[0].get_legend_handles_labels()
        if not handles:
            for axis in axes[1:]:
                handles, labels = axis.get_legend_handles_labels()
                if handles:
                    break
        if handles:
            figure.legend(
                handles,
                labels,
                loc="outside upper center",
                ncol=max(1, min(4, len(labels))),
                frameon=False,
            )
        figure.suptitle(f"Pass@K — {model_group}", fontsize=14, fontweight="semibold")
        stem = _unique_output_stem(
            output_dir, f"passk_{_slug(model_group)}_{_slug(tag)}"
        )
        png_path = stem.with_suffix(".png")
        pdf_path = stem.with_suffix(".pdf")
        figure.savefig(png_path, dpi=300, bbox_inches="tight")
        figure.savefig(pdf_path, bbox_inches="tight")
        plt.close(figure)
        outputs[model_group] = {
            "png": str(png_path),
            "pdf": str(pdf_path),
        }
    return outputs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot cached Pass@K summaries without model inference"
    )
    parser.add_argument("--summary", action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--tag", required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = load_passk_rows(args.summary)
    outputs = plot_passk_results(rows, args.output_dir, tag=args.tag)
    for model_group, paths in outputs.items():
        print(f"{model_group}: {paths['png']}")
        print(f"{model_group}: {paths['pdf']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
