from __future__ import annotations

import json
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "experiments.json"
LATEST = ROOT / "results" / "latest.json"
MAX_GENERIC_BATCH = 28

BATCH_RE = re.compile(r"EXP_C([0-9]+)")


def current_batch(config: dict) -> int:
    batches = []
    for exp in config.get("experiments", []):
        m = BATCH_RE.fullmatch(str(exp.get("id", "")))
        if m:
            batches.append(int(m.group(1)) // 100)
    return max(batches, default=0)


def exp(exp_id: str, seed: int, spec: dict, trials: int = 18000) -> dict:
    return {
        "id": exp_id,
        "engine": "composite_pipeline",
        "trials": trials,
        "seed": seed,
        "spec": spec,
    }


def build_batch(batch: int) -> list[dict]:
    out: list[dict] = []
    base = batch * 1000

    if batch == 10:
        # Replication: same leading architecture across independent seeds,
        # plus nearby controls. This measures whether fingerprints are stable.
        specs = [
            {"rng_a": "family_11", "mapper": "fisher_yates", "reseed_every": 203, "max_skip": 15},
            {"rng_a": "family_11", "mapper": "fisher_yates", "reseed_every": 203, "max_skip": 7},
            {"rng_a": "family_11", "mapper": "fisher_yates", "reseed_every": 203, "max_skip": 31},
            {"rng_a": "family_06", "rng_b": "family_11", "mapper": "swap_remove", "reseed_every": 203, "max_skip": 15},
        ]
        for i in range(20):
            out.append(exp(f"EXP_C10{i+1:02d}", base + i + 1, specs[i % len(specs)]))
        return out

    if batch == 11:
        # Mapping robustness around the strongest families.
        mappers = ["fisher_yates", "swap_remove", "rejection"]
        families = ["family_11", "family_06", "family_07", "family_04"]
        for i in range(20):
            spec = {
                "rng_a": families[i % len(families)],
                "mapper": mappers[i % len(mappers)],
                "reseed_every": [101, 203, 406, 609][(i // 3) % 4],
                "max_skip": [0, 3, 7, 15, 31][i % 5],
            }
            out.append(exp(f"EXP_C11{i+1:02d}", base + i + 1, spec))
        return out

    if batch == 12:
        # State/reseed/skip sensitivity.
        for i in range(20):
            spec = {
                "rng_a": "family_11",
                "mapper": "fisher_yates",
                "reseed_every": [0, 101, 203, 406, 609][i % 5],
                "max_skip": [0, 1, 3, 7, 15, 31, 63, 127][i % 8],
                "background": [0, 1, 3, 5][(i // 4) % 4],
            }
            out.append(exp(f"EXP_C12{i+1:02d}", base + i + 1, spec))
        return out

    if batch == 13:
        # Mixed-family interaction stress test.
        pairs = [
            ("family_06", "family_11"),
            ("family_04", "family_07"),
            ("family_09", "family_11"),
            ("family_01", "family_11"),
            ("family_08", "family_10"),
        ]
        mappers = ["swap_remove", "fisher_yates", "rejection"]
        for i in range(20):
            a, b = pairs[i % len(pairs)]
            spec = {
                "rng_a": a,
                "rng_b": b,
                "mapper": mappers[i % 3],
                "reseed_every": [0, 203, 406][(i // 3) % 3],
                "max_skip": [0, 7, 15, 31][i % 4],
            }
            out.append(exp(f"EXP_C13{i+1:02d}", base + i + 1, spec))
        return out

    # Continuing generic robustness search. Each later batch changes seed,
    # family pairing, mapping, reseed cadence, skip and background deterministically.
    families = [f"family_{i:02d}" for i in range(1, 12)]
    mappers = ["swap_remove", "fisher_yates", "rejection"]
    reseeds = [0, 101, 203, 406, 609]
    skips = [0, 1, 3, 7, 15, 31, 63, 127]
    for i in range(20):
        a = families[(batch + i * 3) % len(families)]
        b = families[(batch * 2 + i * 5) % len(families)]
        spec = {
            "rng_a": a,
            "mapper": mappers[(batch + i) % 3],
            "reseed_every": reseeds[(batch + i) % len(reseeds)],
            "max_skip": skips[(batch * 3 + i) % len(skips)],
            "background": [0, 1, 3, 5][(batch + i) % 4],
        }
        if a != b and i % 2:
            spec["rng_b"] = b
        out.append(exp(f"EXP_C{batch}{i+1:02d}", base + i + 1, spec))
    return out


def main() -> int:
    # The public worker only advances generic simulation batches.
    # It never consumes private target data or produces wagering advice.
    if os.environ.get("GITHUB_EVENT_NAME") not in {"schedule", "workflow_dispatch", "push"}:
        return 0
    if not CONFIG.exists() or not LATEST.exists():
        return 0

    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    latest = json.loads(LATEST.read_text(encoding="utf-8"))
    if int(latest.get("failure_count", 1)) != 0:
        print("autopilot: latest batch has failures; not advancing")
        return 0
    if int(latest.get("success_count", 0)) != int(latest.get("experiment_count", -1)):
        print("autopilot: latest batch incomplete; not advancing")
        return 0

    current = current_batch(config)
    if current < 9:
        print(f"autopilot: batch C{current} is below autonomous handoff point")
        return 0

    if current >= MAX_GENERIC_BATCH:
        print(
            f"autopilot: reached generic batch cap C{MAX_GENERIC_BATCH}; "
            "waiting for private scoring/pruning before another batch"
        )
        return 0

    nxt = current + 1
    next_config = {
        "schema_version": int(config.get("schema_version", 0)) + 1,
        "experiments": build_batch(nxt),
    }
    CONFIG.write_text(json.dumps(next_config, indent=2) + "\n", encoding="utf-8")
    print(f"autopilot: prepared generic batch C{nxt} with {len(next_config['experiments'])} experiments")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
