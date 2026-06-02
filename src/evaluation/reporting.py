from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def save_metric_table(results: dict[str, Any], path: str | Path) -> None:
    """Flatten nested experiment metrics into a publication-friendly CSV."""
    rows: list[dict[str, Any]] = []
    for experiment, models in results.items():
        for model, metrics in models.items():
            rows.append(_metric_row(experiment, model, "all", metrics))
            for family, family_metrics in metrics.get("per_family", {}).items():
                rows.append(_metric_row(experiment, model, family, family_metrics))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)


def _metric_row(experiment: str, model: str, attack_family: str, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "experiment": experiment,
        "model": model,
        "attack_family": attack_family,
        **{key: value for key, value in metrics.items() if key not in {"per_family", "confusion_matrix"}},
        "confusion_matrix": str(metrics.get("confusion_matrix", "")),
    }


def save_baseline_summary(results: dict[str, Any], path: str | Path) -> None:
    """Write a compact Markdown table for Phase 4 result review."""
    lines = [
        "# Phase 4 Baseline Summary",
        "",
        "Thresholds are calibrated from benign fit scores only. Higher anomaly scores indicate more suspicious requests.",
        "",
        "| Experiment | Model | AUROC | AUPRC | FPR @ 95% TPR | Precision | Recall | F1 | Fit rows | Fit seconds | P50 ms | P95 ms | Req/s |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for experiment, models in results.items():
        for model, metrics in models.items():
            latency = metrics["latency"]
            lines.append(
                "| {experiment} | {model} | {auroc:.4f} | {auprc:.4f} | {fpr:.4f} | {precision:.4f} | {recall:.4f} | "
                "{f1:.4f} | {fit_rows} | {fit_seconds:.2f} | {p50:.3f} | {p95:.3f} | {rps:.2f} |".format(
                    experiment=experiment,
                    model=model,
                    auroc=metrics["auroc"],
                    auprc=metrics["auprc"],
                    fpr=metrics["fpr_at_95_tpr"],
                    precision=metrics["precision"],
                    recall=metrics["recall"],
                    f1=metrics["f1"],
                    fit_rows=metrics["fit_sample_count"],
                    fit_seconds=metrics["fit_seconds"],
                    p50=latency["p50_ms"],
                    p95=latency["p95_ms"],
                    rps=latency["requests_per_second"],
                )
            )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
