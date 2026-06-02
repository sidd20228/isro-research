from pathlib import Path

from src.evaluation.reporting import save_baseline_summary


def test_save_baseline_summary_writes_readable_comparison_table(tmp_path: Path) -> None:
    output = tmp_path / "summary.md"
    save_baseline_summary(
        {
            "experiment_c": {
                "tfidf": {
                    "auroc": 0.75,
                    "auprc": 0.5,
                    "fpr_at_95_tpr": 0.25,
                    "precision": 0.6,
                    "recall": 0.7,
                    "f1": 0.64,
                    "fit_sample_count": 100,
                    "fit_seconds": 1.5,
                    "latency": {"p50_ms": 2.0, "p95_ms": 3.0, "requests_per_second": 400.0},
                }
            }
        },
        output,
    )
    content = output.read_text(encoding="utf-8")
    assert "# Phase 4 Baseline Summary" in content
    assert "| experiment_c | tfidf | 0.7500 | 0.5000 |" in content
