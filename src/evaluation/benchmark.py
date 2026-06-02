from __future__ import annotations

import logging
import statistics
import time
import tracemalloc
from dataclasses import asdict, dataclass
from typing import Callable

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatencyReport:
    """CPU inference latency summary."""

    p50_ms: float
    p95_ms: float
    p99_ms: float
    requests_per_second: float
    peak_memory_mb: float
    sample_count: int


def benchmark_inference(infer: Callable[[list[str]], object], samples: list[str], warmup: int = 10) -> dict[str, float | int]:
    """Measure per-request CPU inference latency and Python allocation peak."""
    for sample in samples[:warmup]:
        infer([sample])
    tracemalloc.start()
    durations: list[float] = []
    for sample in samples:
        start = time.perf_counter()
        infer([sample])
        durations.append((time.perf_counter() - start) * 1000)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    ordered = sorted(durations)
    report = LatencyReport(
        p50_ms=statistics.median(ordered),
        p95_ms=_percentile(ordered, 0.95),
        p99_ms=_percentile(ordered, 0.99),
        requests_per_second=1000 / statistics.mean(ordered),
        peak_memory_mb=peak / 1024 / 1024,
        sample_count=len(samples),
    )
    LOGGER.info("Benchmarked %d requests: %.2f req/s", len(samples), report.requests_per_second)
    return asdict(report)


def _percentile(ordered: list[float], quantile: float) -> float:
    if not ordered:
        raise ValueError("At least one timing sample is required")
    return ordered[min(round((len(ordered) - 1) * quantile), len(ordered) - 1)]
