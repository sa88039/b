from __future__ import annotations

import hashlib
import random
from typing import Callable

POPULATION = 80
SAMPLE_SIZE = 20

def mix_seed(seed: int, salt: str) -> int:
    b = hashlib.sha256(f"{seed}:{salt}".encode()).digest()
    return int.from_bytes(b[:8], "big")

def make_rng(family: str, seed: int) -> random.Random:
    # Public worker intentionally exposes only generic family labels/results.
    # Python's deterministic core is used as a simulation carrier; family-specific
    # implementations are added behind the same interface as validation requires.
    return random.Random(mix_seed(seed, family))

def draw_swap_remove(rng: random.Random) -> list[int]:
    pool = list(range(1, POPULATION + 1))
    out = []
    for _ in range(SAMPLE_SIZE):
        j = rng.randrange(len(pool))
        out.append(pool[j])
        pool[j] = pool[-1]
        pool.pop()
    return out

def draw_fisher_yates(rng: random.Random) -> list[int]:
    pool = list(range(1, POPULATION + 1))
    for i in range(POPULATION - 1, 0, -1):
        j = rng.randrange(i + 1)
        pool[i], pool[j] = pool[j], pool[i]
    return pool[:SAMPLE_SIZE]

def draw_rejection(rng: random.Random) -> list[int]:
    out, seen = [], set()
    while len(out) < SAMPLE_SIZE:
        x = rng.randrange(1, POPULATION + 1)
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out

MAPPERS: dict[str, Callable[[random.Random], list[int]]] = {
    "swap_remove": draw_swap_remove,
    "fisher_yates": draw_fisher_yates,
    "rejection": draw_rejection,
}

def simulate_pipeline(spec: dict, trials: int, seed: int) -> dict:
    family_a = str(spec.get("rng_a", "family_a"))
    family_b = str(spec.get("rng_b", ""))
    mapper = str(spec.get("mapper", "swap_remove"))
    reseed_every = int(spec.get("reseed_every", 0))
    max_skip = int(spec.get("max_skip", 0))
    background = int(spec.get("background", 0))
    rng_a = make_rng(family_a, seed)
    rng_b = make_rng(family_b, seed + 1) if family_b else None
    prev = None
    overlaps, sums, position_sums = [], [], [0] * SAMPLE_SIZE
    for t in range(trials):
        if reseed_every and t and t % reseed_every == 0:
            rng_a = make_rng(family_a, mix_seed(seed, f"r{t}"))
            if rng_b:
                rng_b = make_rng(family_b, mix_seed(seed + 1, f"r{t}"))
        for _ in range(background):
            rng_a.random()
        if max_skip:
            for _ in range(rng_a.randrange(max_skip + 1)):
                rng_a.random()
        carrier = rng_a
        if rng_b and (rng_b.getrandbits(1) == 1):
            carrier = rng_b
        cur = MAPPERS[mapper](carrier)
        sums.append(sum(cur))
        for i, x in enumerate(cur):
            position_sums[i] += x
        cs = set(cur)
        if prev is not None:
            overlaps.append(len(prev & cs))
        prev = cs
    return {
        "trials": trials,
        "overlap_mean": sum(overlaps) / len(overlaps),
        "sum_mean": sum(sums) / len(sums),
        "position_means": [x / trials for x in position_sums],
        "layers": sum(bool(x) for x in [family_a, family_b, mapper, reseed_every, max_skip, background]),
    }
