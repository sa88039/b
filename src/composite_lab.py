from __future__ import annotations

import hashlib
import math
import random
from typing import Callable, Protocol

POPULATION = 80
SAMPLE_SIZE = 20
MASK32 = (1 << 32) - 1
MASK64 = (1 << 64) - 1


def mix_seed(seed: int, salt: str) -> int:
    b = hashlib.sha256(f"{seed}:{salt}".encode()).digest()
    return int.from_bytes(b[:8], "big")


class RNG(Protocol):
    def random(self) -> float: ...
    def randrange(self, stop: int) -> int: ...
    def getrandbits(self, k: int) -> int: ...


class UIntRNG:
    bits = 32

    def next_uint(self) -> int:
        raise NotImplementedError

    def getrandbits(self, k: int) -> int:
        if k <= 0:
            return 0
        out = 0
        made = 0
        while made < k:
            x = self.next_uint()
            take = min(self.bits, k - made)
            piece = (x >> (self.bits - take)) & ((1 << take) - 1)
            out = (out << take) | piece
            made += take
        return out

    def random(self) -> float:
        return self.getrandbits(53) / float(1 << 53)

    def randrange(self, stop: int) -> int:
        if stop <= 0:
            raise ValueError("empty range for randrange")
        k = stop.bit_length()
        while True:
            r = self.getrandbits(k)
            if r < stop:
                return r


class JavaLCG(UIntRNG):
    def __init__(self, seed: int):
        self.state = (seed ^ 0x5DEECE66D) & ((1 << 48) - 1)

    def _next(self, bits: int) -> int:
        self.state = (self.state * 0x5DEECE66D + 0xB) & ((1 << 48) - 1)
        return self.state >> (48 - bits)

    def next_uint(self) -> int:
        return self._next(32)

    def getrandbits(self, k: int) -> int:
        if k <= 0:
            return 0
        out = 0
        made = 0
        while made < k:
            take = min(32, k - made)
            out |= self._next(take) << made
            made += take
        return out


class MSVCLCG(UIntRNG):
    def __init__(self, seed: int):
        self.state = seed & MASK32

    def next_uint(self) -> int:
        parts = []
        for _ in range(3):
            self.state = (214013 * self.state + 2531011) & MASK32
            parts.append((self.state >> 16) & 0x7FFF)
        return ((parts[0] << 17) | (parts[1] << 2) | (parts[2] & 3)) & MASK32


class ANSICLCG(UIntRNG):
    bits = 31

    def __init__(self, seed: int):
        self.state = seed & 0x7FFFFFFF

    def next_uint(self) -> int:
        self.state = (1103515245 * self.state + 12345) & 0x7FFFFFFF
        return self.state


class NumericalRecipesLCG(UIntRNG):
    def __init__(self, seed: int):
        self.state = seed & MASK32

    def next_uint(self) -> int:
        self.state = (1664525 * self.state + 1013904223) & MASK32
        return self.state


class ParkMiller(UIntRNG):
    bits = 31
    MOD = 2147483647
    MUL = 16807

    def __init__(self, seed: int):
        self.state = seed % (self.MOD - 1) + 1

    def next_uint(self) -> int:
        self.state = (self.state * self.MUL) % self.MOD
        return self.state


class XorShift32(UIntRNG):
    def __init__(self, seed: int):
        self.state = seed & MASK32 or 0x6D2B79F5

    def next_uint(self) -> int:
        x = self.state
        x ^= (x << 13) & MASK32
        x ^= (x >> 17) & MASK32
        x ^= (x << 5) & MASK32
        self.state = x & MASK32
        return self.state


class Mulberry32(UIntRNG):
    def __init__(self, seed: int):
        self.state = seed & MASK32

    def next_uint(self) -> int:
        self.state = (self.state + 0x6D2B79F5) & MASK32
        z = self.state
        z = ((z ^ (z >> 15)) * (z | 1)) & MASK32
        z ^= (z + (((z ^ (z >> 7)) * (z | 61)) & MASK32)) & MASK32
        return (z ^ (z >> 14)) & MASK32


class MTRNG:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def random(self) -> float:
        return self.rng.random()

    def randrange(self, stop: int) -> int:
        return self.rng.randrange(stop)

    def getrandbits(self, k: int) -> int:
        return self.rng.getrandbits(k)


class PCG32(UIntRNG):
    def __init__(self, seed: int):
        self.state = 0
        self.inc = ((seed << 1) | 1) & MASK64
        self.next_uint()
        self.state = (self.state + (seed & MASK64)) & MASK64
        self.next_uint()

    def next_uint(self) -> int:
        old = self.state
        self.state = (old * 6364136223846793005 + self.inc) & MASK64
        xorshifted = (((old >> 18) ^ old) >> 27) & MASK32
        rot = (old >> 59) & 31
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & MASK32


def _rotl64(x: int, k: int) -> int:
    return ((x << k) | (x >> (64 - k))) & MASK64


class Xoroshiro128Plus(UIntRNG):
    bits = 64

    def __init__(self, seed: int):
        self.s0 = mix_seed(seed, "xoroshiro-s0") & MASK64
        self.s1 = mix_seed(seed, "xoroshiro-s1") & MASK64
        if self.s0 == 0 and self.s1 == 0:
            self.s1 = 1

    def next_uint(self) -> int:
        s0, s1 = self.s0, self.s1
        result = (s0 + s1) & MASK64
        s1 ^= s0
        self.s0 = (_rotl64(s0, 55) ^ s1 ^ ((s1 << 14) & MASK64)) & MASK64
        self.s1 = _rotl64(s1, 36)
        return result


class MRG32k3a(UIntRNG):
    m1 = 4294967087
    m2 = 4294944443

    def __init__(self, seed: int):
        vals = [mix_seed(seed, f"mrg-{i}") for i in range(6)]
        self.s10 = vals[0] % (self.m1 - 1) + 1
        self.s11 = vals[1] % (self.m1 - 1) + 1
        self.s12 = vals[2] % (self.m1 - 1) + 1
        self.s20 = vals[3] % (self.m2 - 1) + 1
        self.s21 = vals[4] % (self.m2 - 1) + 1
        self.s22 = vals[5] % (self.m2 - 1) + 1

    def next_uint(self) -> int:
        p1 = (1403580 * self.s11 - 810728 * self.s10) % self.m1
        self.s10, self.s11, self.s12 = self.s11, self.s12, p1
        p2 = (527612 * self.s22 - 1370589 * self.s20) % self.m2
        self.s20, self.s21, self.s22 = self.s21, self.s22, p2
        z = p1 - p2
        if z <= 0:
            z += self.m1
        return z & MASK32


FAMILY_BUILDERS = {
    "family_01": JavaLCG,
    "family_02": MSVCLCG,
    "family_03": ANSICLCG,
    "family_04": NumericalRecipesLCG,
    "family_05": ParkMiller,
    "family_06": XorShift32,
    "family_07": Mulberry32,
    "family_08": MTRNG,
    "family_09": PCG32,
    "family_10": Xoroshiro128Plus,
    "family_11": MRG32k3a,
}


def make_rng(family: str, seed: int) -> RNG:
    try:
        builder = FAMILY_BUILDERS[family]
    except KeyError as exc:
        raise ValueError(f"unknown RNG family: {family}") from exc
    return builder(mix_seed(seed, family))


def draw_swap_remove(rng: RNG) -> list[int]:
    pool = list(range(1, POPULATION + 1))
    out = []
    for _ in range(SAMPLE_SIZE):
        j = rng.randrange(len(pool))
        out.append(pool[j])
        pool[j] = pool[-1]
        pool.pop()
    return out


def draw_fisher_yates(rng: RNG) -> list[int]:
    pool = list(range(1, POPULATION + 1))
    for i in range(POPULATION - 1, 0, -1):
        j = rng.randrange(i + 1)
        pool[i], pool[j] = pool[j], pool[i]
    return pool[:SAMPLE_SIZE]


def draw_rejection(rng: RNG) -> list[int]:
    out, seen = [], set()
    while len(out) < SAMPLE_SIZE:
        x = rng.randrange(POPULATION) + 1
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


MAPPERS: dict[str, Callable[[RNG], list[int]]] = {
    "swap_remove": draw_swap_remove,
    "fisher_yates": draw_fisher_yates,
    "rejection": draw_rejection,
}


def simulate_pipeline(spec: dict, trials: int, seed: int) -> dict:
    family_a = str(spec.get("rng_a", "family_01"))
    family_b = str(spec.get("rng_b", ""))
    mapper = str(spec.get("mapper", "swap_remove"))
    reseed_every = int(spec.get("reseed_every", 0))
    max_skip = int(spec.get("max_skip", 0))
    background = int(spec.get("background", 0))
    rng_a = make_rng(family_a, seed)
    rng_b = make_rng(family_b, seed + 1) if family_b else None

    recent_sets: list[set[int]] = []
    overlap_sums = [0.0] * 5
    overlap_counts = [0] * 5
    sums: list[int] = []
    position_sums = [0] * SAMPLE_SIZE
    freq = [0] * (POPULATION + 1)
    inversion_total = 0
    ascent_total = 0
    gap_cv_total = 0.0
    same_position_total = 0
    same_position_pairs = 0
    prev_ordered: list[int] | None = None

    flat_prev: int | None = None
    corr_n = 0
    corr_sx = corr_sy = corr_sxx = corr_syy = corr_sxy = 0.0

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
        if rng_b and rng_b.getrandbits(1) == 1:
            carrier = rng_b
        cur = MAPPERS[mapper](carrier)
        cs = set(cur)

        sums.append(sum(cur))
        for i, x in enumerate(cur):
            position_sums[i] += x
            freq[x] += 1

        for lag_idx, old_set in enumerate(recent_sets[:5]):
            overlap_sums[lag_idx] += len(cs & old_set)
            overlap_counts[lag_idx] += 1

        inversion_total += sum(
            1
            for i in range(SAMPLE_SIZE)
            for j in range(i + 1, SAMPLE_SIZE)
            if cur[i] > cur[j]
        )
        ascent_total += sum(cur[i] < cur[i + 1] for i in range(SAMPLE_SIZE - 1))

        ordered = sorted(cur)
        gaps = [ordered[i + 1] - ordered[i] for i in range(SAMPLE_SIZE - 1)]
        gap_mean = sum(gaps) / len(gaps)
        if gap_mean:
            gap_var = sum((g - gap_mean) ** 2 for g in gaps) / len(gaps)
            gap_cv_total += math.sqrt(gap_var) / gap_mean

        if prev_ordered is not None:
            same_position_total += sum(a == b for a, b in zip(prev_ordered, cur))
            same_position_pairs += 1
        prev_ordered = cur

        for x in cur:
            if flat_prev is not None:
                fx = float(flat_prev)
                fy = float(x)
                corr_n += 1
                corr_sx += fx
                corr_sy += fy
                corr_sxx += fx * fx
                corr_syy += fy * fy
                corr_sxy += fx * fy
            flat_prev = x

        recent_sets.insert(0, cs)
        if len(recent_sets) > 5:
            recent_sets.pop()

    overlap_lags = {
        str(i + 1): (
            overlap_sums[i] / overlap_counts[i] if overlap_counts[i] else None
        )
        for i in range(5)
    }
    expected_freq = trials * SAMPLE_SIZE / POPULATION
    freq_chi_square = sum(
        ((freq[x] - expected_freq) ** 2) / expected_freq
        for x in range(1, POPULATION + 1)
    )
    corr_num = corr_n * corr_sxy - corr_sx * corr_sy
    corr_den = math.sqrt(
        max(0.0, corr_n * corr_sxx - corr_sx * corr_sx)
        * max(0.0, corr_n * corr_syy - corr_sy * corr_sy)
    )
    flat_lag1_corr = corr_num / corr_den if corr_den else 0.0

    return {
        "trials": trials,
        "overlap_mean": overlap_lags["1"],
        "overlap_lags": overlap_lags,
        "sum_mean": sum(sums) / len(sums),
        "position_means": [x / trials for x in position_sums],
        "same_position_mean": (
            same_position_total / same_position_pairs if same_position_pairs else 0.0
        ),
        "flat_lag1_corr": flat_lag1_corr,
        "freq_chi_square": freq_chi_square,
        "inversion_mean": inversion_total / trials,
        "ascent_mean": ascent_total / trials,
        "gap_cv_mean": gap_cv_total / trials,
        "layers": sum(
            bool(x)
            for x in [
                family_a,
                family_b,
                mapper,
                reseed_every,
                max_skip,
                background,
            ]
        ),
    }
