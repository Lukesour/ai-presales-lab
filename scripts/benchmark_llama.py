#!/usr/bin/env python3
"""Benchmark a running llama.cpp OpenAI-compatible server."""

from __future__ import annotations

import argparse
import json
import math
import statistics
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ai_presales_lab.llama_client import LlamaClient, LlamaClientError

ROOT = Path(__file__).resolve().parents[1]


PROMPTS = [
    "请用 JSON 输出：客户要求内网部署、数据不能出域、峰值 5 并发。请列出需要澄清的三个问题。",
    "请用 JSON 比较云端 API 与本地量化模型在数据安全、延迟、成本和运维方面的取舍。",
    "请用 JSON 输出一个企业知识库 PoC 的四步实施计划，并标记不能直接承诺的指标。",
    "请用 JSON 判断：只有一台 Apple Silicon 工作站是否可以直接承诺 100 并发生产服务？说明理由。",
    "请用 JSON 输出模型量化评测报告应包含的字段。不要编造任何实际数值。",
]


def one_request(client: LlamaClient, prompt: str, stream: bool) -> dict[str, object]:
    started = time.perf_counter()
    try:
        messages = [
            {"role": "system", "content": "你是严谨的 AI 基础设施售前顾问，只输出有效 JSON。"},
            {"role": "user", "content": prompt},
        ]
        result = (
            client.chat_stream(messages, max_tokens=256)
            if stream
            else client.chat(messages, max_tokens=256)
        )
        elapsed = (time.perf_counter() - started) * 1000
        tokens = result.completion_tokens
        valid_json = _is_json(result.text)
        return {
            "ok": True,
            "latency_ms": round(elapsed, 2),
            "ttft_ms": result.time_to_first_token_ms,
            "completion_tokens": tokens,
            "tokens_per_second": round(tokens / (elapsed / 1000), 2)
            if tokens and elapsed
            else None,
            "text_chars": len(result.text),
            "valid_json": valid_json,
        }
    except LlamaClientError as exc:
        return {
            "ok": False,
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
            "error": str(exc),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--requests", type=int, default=5)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--no-stream", action="store_true")
    parser.add_argument("--label", default="unlabeled")
    parser.add_argument(
        "--server-pid", type=int, default=None, help="PID of llama-server for RSS sampling"
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.requests < 1 or args.concurrency < 1:
        parser.error("--requests and --concurrency must be positive")
    client = LlamaClient(base_url=args.base_url)
    jobs = [PROMPTS[index % len(PROMPTS)] for index in range(args.requests)]
    started = time.perf_counter()
    results: list[dict[str, object]] = []
    monitor = RssMonitor(args.server_pid) if args.server_pid else None
    if monitor:
        monitor.start()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one_request, client, prompt, not args.no_stream) for prompt in jobs]
        for future in as_completed(futures):
            results.append(future.result())
    if monitor:
        monitor.stop()
    wall_ms = round((time.perf_counter() - started) * 1000, 2)
    successful = [result for result in results if result.get("ok")]
    latencies = [float(result["latency_ms"]) for result in successful]
    ttfts = [float(result["ttft_ms"]) for result in successful if result.get("ttft_ms") is not None]
    valid_json_count = sum(bool(result.get("valid_json")) for result in successful)
    report = {
        "label": args.label,
        "base_url": client.base_url,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "streaming": not args.no_stream,
        "wall_time_ms": wall_ms,
        "success": len(successful),
        "failure": len(results) - len(successful),
        "structured_json_pass": valid_json_count,
        "structured_json_pass_rate": round(valid_json_count / len(successful), 4)
        if successful
        else 0.0,
        "latency_ms": _stats(latencies),
        "ttft_ms": _stats(ttfts),
        "server_memory_mib": monitor.stats() if monitor else None,
        "results": results,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if len(successful) == args.requests else 1


def _stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "mean": None, "median": None, "p95": None, "min": None, "max": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * 0.95) - 1))
    return {
        "count": len(values),
        "mean": round(statistics.mean(values), 2),
        "median": round(statistics.median(values), 2),
        "p95": round(ordered[p95_index], 2),
        "min": round(min(values), 2),
        "max": round(max(values), 2),
    }


def _is_json(text: str) -> bool:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.removeprefix("```").removeprefix("json").removesuffix("```").strip()
    try:
        json.loads(candidate)
    except json.JSONDecodeError:
        return False
    return True


class RssMonitor:
    """Sample the llama-server resident set size without third-party packages."""

    def __init__(self, pid: int, interval_s: float = 0.05):
        self.pid = pid
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[float] = []

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name="rss-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)

    def stats(self) -> dict[str, float | None]:
        if not self._samples:
            return {"samples": 0, "before_mib": None, "peak_mib": None, "after_mib": None}
        return {
            "samples": len(self._samples),
            "before_mib": round(self._samples[0], 2),
            "peak_mib": round(max(self._samples), 2),
            "after_mib": round(self._samples[-1], 2),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            rss_mib = _rss_mib(self.pid)
            if rss_mib is not None:
                self._samples.append(rss_mib)
            self._stop.wait(self.interval_s)


def _rss_mib(pid: int) -> float | None:
    try:
        output = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True).strip()
        rss_kib = int(output)
    except (OSError, ValueError, subprocess.CalledProcessError):
        return None
    return rss_kib / 1024


if __name__ == "__main__":
    raise SystemExit(main())
