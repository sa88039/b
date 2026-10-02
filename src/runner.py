from __future__ import annotations

import hashlib
import json
import math
import os
import random
import statistics
import time
from pathlib import Path
from typing import Any

from composite_lab import simulate_pipeline

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments.json"
RESULTS = ROOT / "results"
HISTORY = RESULTS / "history"
STATE = RESULTS / "runner_state.json"


def work_hash() -> str:
    h = hashlib.sha256()
    h.update(CONFIG.read_bytes())
    h.update(Path(__file__).read_bytes())
    h.update((ROOT / "src" / "composite_lab.py").read_bytes())
    return h.hexdigest()


def should_skip_scheduled_run(current_hash: str) -> bool:
    if os.environ.get("GITHUB_EVENT_NAME") != "schedule":
        return False
    if not STATE.exists():
        return False
    try:
        state = json.loads(STATE.read_text(encoding="utf-8"))
    except Exception:
        return False
    return state.get("last_successful_work_hash") == current_hash


def hypergeom_pmf(population: int, success: int, draws: int) -> tuple[list[int], list[float]]:
    lo = max(0, draws - (population - success))
    hi = min(draws, success)
    xs = list(range(lo, hi + 1))
    den = math.comb(population, draws)
    ws = [
        math.comb(success, x) * math.comb(population - success, draws - x) / den
        for x in xs
    ]
    return xs, ws


def simulate_holdout_means(rng: random.Random, n: int, reps: int, population: int, sample_size: int) -> list[float]:
    xs, ws = hypergeom_pmf(population, sample_size, sample_size)
    means: list[float] = []
    for _ in range(reps):
        values = rng.choices(xs, weights=ws, k=n)
        means.append(statistics.fmean(values))
    return means


def quantile(values: list[float], q: float) -> float:
    s = sorted(values)
    idx = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[idx]


def run_experiment(exp: dict[str, Any]) -> dict[str, Any]:
    exp_id = str(exp["id"])
    engine = str(exp["engine"])
    seed = int(exp.get("seed", 1))
    rng = random.Random(seed)

    population = int(exp.get("population", 80))
    sample_size = int(exp.get("sample_size", 20))
    expected = sample_size * sample_size / population

    if engine == "null_holdout_distribution":
        n = int(exp.get("n", 203))
        reps = int(exp.get("reps", 20000))
        means = simulate_holdout_means(rng, n, reps, population, sample_size)
        return {
            "id": exp_id,
            "status": "ok",
            "engine": engine,
            "metrics": {
                "n": n,
                "reps": reps,
                "expected": expected,
                "mean": statistics.fmean(means),
                "p90": quantile(means, 0.90),
                "p95": quantile(means, 0.95),
                "p975": quantile(means, 0.975),
                "p99": quantile(means, 0.99),
            },
        }

    if engine == "two_holdout_consistency":
        n = int(exp.get("n", 203))
        reps = int(exp.get("reps", 15000))
        thresholds = [float(x) for x in exp.get("thresholds", [5.05, 5.10, 5.15, 5.20])]
        a = simulate_holdout_means(rng, n, reps, population, sample_size)
        b = simulate_holdout_means(rng, n, reps, population, sample_size)
        both = {}
        for th in thresholds:
            both[str(th)] = sum(x >= th and y >= th for x, y in zip(a, b)) / reps
        return {
            "id": exp_id,
            "status": "ok",
            "engine": engine,
            "metrics": {"n": n, "reps": reps, "both_holdouts_ge": both},
        }

    if engine == "screening_false_positive":
        n = int(exp.get("n", 203))
        reps = int(exp.get("reps", 3000))
        candidates = int(exp.get("candidates", 50))
        means = simulate_holdout_means(rng, n, reps * candidates, population, sample_size)
        maxima = [
            max(means[i * candidates:(i + 1) * candidates])
            for i in range(reps)
        ]
        return {
            "id": exp_id,
            "status": "ok",
            "engine": engine,
            "metrics": {
                "n": n,
                "reps": reps,
                "candidates": candidates,
                "max_mean_avg": statistics.fmean(maxima),
                "max_p95": quantile(maxima, 0.95),
                "max_p99": quantile(maxima, 0.99),
            },
        }

    if engine == "composite_pipeline":
        trials = int(exp.get("trials", 50000))
        spec = dict(exp.get("spec", {}))
        metrics = simulate_pipeline(spec, trials, seed)
        return {
            "id": exp_id,
            "status": "ok",
            "engine": engine,
            "metrics": metrics,
        }

    if engine == "sequence_fingerprint":
        trials = int(exp.get("trials", 100000))
        max_skip = int(exp.get("max_skip", 0))
        prev: set[int] | None = None
        overlaps: list[int] = []
        sums: list[int] = []
        for _ in range(trials):
            for _ in range(rng.randrange(max_skip + 1) if max_skip > 0 else 0):
                rng.random()
            cur = rng.sample(range(1, population + 1), sample_size)
            sums.append(sum(cur))
            cs = set(cur)
            if prev is not None:
                overlaps.append(len(prev & cs))
            prev = cs
        return {
            "id": exp_id,
            "status": "ok",
            "engine": engine,
            "metrics": {
                "trials": trials,
                "max_skip": max_skip,
                "overlap_mean": statistics.fmean(overlaps),
                "overlap_sd": statistics.pstdev(overlaps),
                "sum_mean": statistics.fmean(sums),
                "sum_sd": statistics.pstdev(sums),
                "expected_overlap": expected,
            },
        }

    raise ValueError(f"unknown engine: {engine}")


def main() -> int:
    started = time.time()
    RESULTS.mkdir(parents=True, exist_ok=True)
    HISTORY.mkdir(parents=True, exist_ok=True)

    current_hash = work_hash()
    if should_skip_scheduled_run(current_hash):
        print(json.dumps({
            "status": "no_new_work",
            "reason": "same successful research batch already completed",
            "work_hash": current_hash[:12],
        }))
        return 0

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    experiments = config.get("experiments", [])
    results: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for exp in experiments:
        exp_id = str(exp.get("id", "UNKNOWN"))
        try:
            results.append(run_experiment(exp))
        except Exception as exc:
            failures.append({
                "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "experiment": exp_id,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "retry_count": 0,
            })
            results.append({"id": exp_id, "status": "failed"})

    payload = {
        "schema_version": 2,
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "runtime_seconds": round(time.time() - started, 3),
        "experiment_count": len(experiments),
        "success_count": sum(r.get("status") == "ok" for r in results),
        "failure_count": len(failures),
        "results": results,
    }

    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    body = json.dumps(payload, indent=2)
    (RESULTS / "latest.json").write_text(body, encoding="utf-8")
    (HISTORY / f"{stamp}.json").write_text(body, encoding="utf-8")
    (RESULTS / "failure_report.json").write_text(
        json.dumps({"generated_utc": payload["generated_utc"], "failures": failures}, indent=2),
        encoding="utf-8",
    )
    if not failures:
        STATE.write_text(
            json.dumps({
                "last_successful_work_hash": current_hash,
                "completed_utc": payload["generated_utc"],
                "experiment_count": payload["experiment_count"],
            }, indent=2),
            encoding="utf-8",
        )
    print(json.dumps({
        "experiment_count": payload["experiment_count"],
        "success_count": payload["success_count"],
        "failure_count": payload["failure_count"],
        "runtime_seconds": payload["runtime_seconds"],
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
