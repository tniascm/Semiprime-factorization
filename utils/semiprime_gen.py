"""
Shared utilities for semiprime generation across experiments.
"""
import numpy as np
from sage.all import factor, next_prime, isqrt, randint as sage_randint


def generate_semiprimes(max_N, num_samples=60, min_N=15, seed=42):
    """
    Generate semiprimes N = p*q (p < q, both prime) distributed
    log-uniformly across [min_N, max_N].

    Parameters
    ----------
    max_N : int
        Upper bound for semiprime values.
    num_samples : int
        Number of log-uniformly spaced targets.
    min_N : int
        Lower bound for semiprime values.
    seed : int
        Random seed for reproducibility of the fallback path.

    Returns list of (N, p, q) tuples, sorted by N, deduplicated.
    """
    rng = np.random.RandomState(seed)
    targets = np.logspace(np.log10(min_N), np.log10(max_N), num_samples)
    semiprimes = []
    seen = set()

    for target in targets:
        N_approx = int(target)
        if N_approx < 6:
            continue
        found = False
        for offset in range(300):
            candidate = N_approx + offset
            if candidate < 6 or candidate in seen:
                continue
            f = factor(candidate)
            if len(f) == 2 and all(e == 1 for _, e in f):
                p, q = int(f[0][0]), int(f[1][0])
                semiprimes.append((candidate, p, q))
                seen.add(candidate)
                found = True
                break
        if not found:
            # Fallback: randomize starting point to avoid consecutive-prime bias
            sqrt_approx = int(isqrt(N_approx))
            offset = int(rng.randint(0, max(1, sqrt_approx // 4)))
            p = next_prime(sqrt_approx + offset)
            q = next_prime(p + int(rng.randint(1, max(2, sqrt_approx // 8))))
            N = int(p * q)
            if N not in seen:
                semiprimes.append((N, int(p), int(q)))
                seen.add(N)

    semiprimes.sort()
    return semiprimes


def balanced_semiprimes(bit_sizes, count_per_size=5, min_ratio=0.3, seed=42):
    """
    Generate balanced semiprimes (p/q >= min_ratio) at specified bit sizes,
    with diversity in the balance ratio.

    Parameters
    ----------
    bit_sizes : list of int
        Target bit-lengths for N.
    count_per_size : int
        Number of semiprimes per bit size.
    min_ratio : float
        Minimum p/q ratio (default 0.3 for balanced regime).
    seed : int
        Random seed for reproducibility.

    Returns list of (N, p, q) tuples.
    """
    rng = np.random.RandomState(seed)
    results = []
    for bits in bit_sizes:
        half_bits = bits // 2
        p_lo = max(3, int(2**(half_bits - 1)))
        p_hi = int(2**(half_bits + 1))
        generated = 0
        attempts = 0
        max_attempts = count_per_size * 50

        while generated < count_per_size and attempts < max_attempts:
            attempts += 1
            # Randomize starting prime across the valid range
            p_start = p_lo + int(rng.randint(0, max(1, p_hi - p_lo)))
            p = int(next_prime(p_start))
            if p >= p_hi:
                continue

            # Pick q to get the desired bit size with ratio diversity
            q_lo = max(p + 2, (2**(bits - 1) + p - 1) // p)
            q_hi = (2**bits - 1) // p
            if q_lo > q_hi:
                continue

            # Randomize q within the valid range
            q_start = q_lo + int(rng.randint(0, max(1, q_hi - q_lo)))
            q = int(next_prime(q_start))
            if q > q_hi or q == p:
                continue

            pmin, pmax = sorted([p, q])
            ratio = float(pmin) / float(pmax)
            if ratio >= min_ratio:
                N = int(p * q)
                results.append((N, int(pmin), int(pmax)))
                generated += 1

    return results
