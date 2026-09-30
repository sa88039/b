from __future__ import annotations

import json
import math
import os
import random
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments.json"
RESULTS = ROOT / "results"
HISTORY = RESULTS / "history"


@dataclass
class Metrics:
    mean: float
    stdev: float
    minimum: float
    maximum: float


def draw_uniform_without_replacement(rng: random.Random, population: int, sample_size: int) -> list[int]:
    return rng.sample(range(1, population + 1), sample_size)


def overlap(a: list[int], b: list[int]) -> int:
    return len(set(a).intersection(b))


def summarize(values: list[float]) -> Metrics:
    if not values:
        return Metrics(0.0, 0.0, 0.0, 0.0)
    return Metrics(
        mean=statistics.fmean(values),
        stdev=statistics.pstdev(values) if len(values) > 1 else 0.0,
        minimum=min(values),
        maximum=max(values),
    )


def run_experiment(exp: dict[str, Any]) -> dict[str, Any]:
    exp_id = str(exp["id"])
    engine = str(exp["engine"])
    population = int(exp["population"])
    sample_size = int(exp["sample_size"])
    trials = int(exp.get("trials", 1000))
    seed = int(exp.get("seed", 1))

    if not (1 <= sample_size <= population):
        raise ValueError("sample_size must be between 1 and population")
    if trials < 2:
        raise ValueError("trials must be >= 2")

    rng = random.Random(seed)
    draws: list[list[int]] = []

    for _ in range(trials):
        if engine == "uniform_without_replacement":
            current = draw_uniform_without_replacement(rng, population, sample_size)
        elif engine == "continuous_state_with_skip":
            max_skip = int(exp.get("max_skip", 63))
            for _ in range(rng.randrange(max_skip + 1)):
                rng.random()
            current = draw_uniform_without_replacement(rng, population, sample_size)
        else:
            raise ValueError(f"unknown engine: {engine}")
        draws.append(current)

    overlaps = [overlap(draws[i - 1], draws[i]) for i in range(1, len(draws))]
    sums = [sum(d) for d in draws]
    overlap_metrics = summarize([float(x) for x in overlaps])
    sum_metrics = summarize([float(x) for x in sums])

    expected_overlap = (sample_size * sample_size) / population

    return {
        "id": exp_id,
        "status": "ok",
        "engine": engine,
        "trials": trials,
        "metrics": {
            "adjacent_overlap_mean": overlap_metrics.mean,
            "adjacent_overlap_sd": overlap_metrics.stdev,
            "expected_overlap": expected_overlap,
            "overlap_delta": overlap_metrics.mean - expected_overlap,
            "draw_sum_mean": sum_metrics.mean,
            "draw_sum_sd": sum_metrics.stdev
        }
    }


def main() -> int:
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    HISTORY.mkdir(parents=True, exist_ok=True)

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    experiments = config.get("experiments", [])

    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for exp in experiments:
        exp_id = str(exp.get("id", "UNKNOWN"))
        try:
            results.append(run_experiment(exp))
        except Exception as exc:
            failure = {
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "experiment": exp_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "retry_count": 0
            }
            failures.append(failure)
            results.append({"id": exp_id, "status": "failed"})

    payload = {
        "schema_version": 1,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 3),
        "experiment_count": len(experiments),
        "success_count": sum(1 for r in results if r.get("status") == "ok"),
        "failure_count": len(failures),
        "results": results
    }

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    (RESULTS / "latest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (HISTORY / f"{stamp}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (RESULTS / "failure_report.json").write_text(json.dumps({
        "generated_utc": payload["generated_utc"],
        "failures": failures
    }, indent=2), encoding="utf-8")

    print(json.dumps({
        "experiment_count": payload["experiment_count"],
        "success_count": payload["success_count"],
        "failure_count": payload["failure_count"],
        "runtime_seconds": payload["runtime_seconds"]
    }))

    # Per design, individual experiment failures do not fail the whole batch.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
