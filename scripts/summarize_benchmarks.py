#!/usr/bin/env python3
"""Aggregate real llama.cpp benchmark JSON files into a comparison report."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    loaded = [json.loads(path.read_text(encoding="utf-8")) for path in args.reports]
    summary = {
        "report_type": "llama.cpp quantization comparison",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_reports": [str(path) for path in args.reports],
        "runs": [_run_row(report) for report in loaded],
        "comparison": _comparison(loaded),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _run_row(report: dict[str, Any]) -> dict[str, Any]:
    model = report.get("model") or {}
    latency = report.get("latency_ms") or {}
    ttft = report.get("ttft_ms") or {}
    cpu_memory = report.get("server_memory_mib") or {}
    gpu_memory = report.get("gpu_memory_mib") or {}
    return {
        "label": report.get("label"),
        "quantization": model.get("quantization"),
        "model_file_size_mib": model.get("file_size_mib"),
        "requests": report.get("requests"),
        "concurrency": report.get("concurrency"),
        "streaming": report.get("streaming"),
        "success": report.get("success"),
        "failure": report.get("failure"),
        "completion_tokens": report.get("completion_tokens"),
        "throughput_tokens_per_second": report.get("throughput_tokens_per_second"),
        "latency_p50_ms": latency.get("median"),
        "latency_p95_ms": latency.get("p95"),
        "ttft_p50_ms": ttft.get("median"),
        "ttft_p95_ms": ttft.get("p95"),
        "structured_json_pass_rate": report.get("structured_json_pass_rate"),
        "cpu_rss_peak_mib": cpu_memory.get("peak_mib"),
        "gpu_vram_peak_mib": gpu_memory.get("peak_mib"),
        "environment": report.get("environment"),
        "runtime": report.get("runtime"),
    }


def _comparison(reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [_run_row(report) for report in reports]
    q4_rows = [row for row in rows if row.get("quantization") == "Q4"]
    q8_rows = [row for row in rows if row.get("quantization") == "Q8"]
    if not q4_rows or not q8_rows:
        return {"status": "pending", "reason": "Need both Q4 and Q8 reports."}
    pairs = []
    for q4 in q4_rows:
        q8 = next(
            (
                row
                for row in q8_rows
                if row.get("concurrency") == q4.get("concurrency")
                and row.get("streaming") == q4.get("streaming")
            ),
            None,
        )
        if q8 is None:
            continue
        pairs.append(
            {
                "concurrency": q4.get("concurrency"),
                "q4_label": q4.get("label"),
                "q8_label": q8.get("label"),
                "q8_minus_q4_latency_p95_ms": _difference(
                    q8.get("latency_p95_ms"), q4.get("latency_p95_ms")
                ),
                "q8_minus_q4_cpu_rss_peak_mib": _difference(
                    q8.get("cpu_rss_peak_mib"), q4.get("cpu_rss_peak_mib")
                ),
                "q8_minus_q4_gpu_vram_peak_mib": _difference(
                    q8.get("gpu_vram_peak_mib"), q4.get("gpu_vram_peak_mib")
                ),
                "q8_to_q4_throughput_ratio": _ratio(
                    q8.get("throughput_tokens_per_second"),
                    q4.get("throughput_tokens_per_second"),
                ),
            }
        )
    return {
        "status": "complete" if pairs else "pending",
        "pairs": pairs,
    }


def _difference(left: Any, right: Any) -> float | None:
    if not isinstance(left, (int, float)) or not isinstance(right, (int, float)):
        return None
    return round(float(left) - float(right), 2)


def _ratio(numerator: Any, denominator: Any) -> float | None:
    if not isinstance(numerator, (int, float)) or not isinstance(denominator, (int, float)):
        return None
    if denominator == 0:
        return None
    return round(float(numerator) / float(denominator), 4)


if __name__ == "__main__":
    raise SystemExit(main())
