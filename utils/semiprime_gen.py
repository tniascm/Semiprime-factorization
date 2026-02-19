"""
Shared utilities for semiprime generation across experiments.
"""
import numpy as np
from sage.all import factor, next_prime, isqrt


def generate_semiprimes(max_N, num_samples=60, min_N=15):
    """
    Generate semiprimes N = p*q (p < q, both prime) distributed
    log-uniformly across [min_N, max_N].

    Returns list of (N, p, q) tuples, sorted by N, deduplicated.
    """
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
            p = next_prime(isqrt(N_approx))
            q = next_prime(p + 1)
            N = int(p * q)
            if N not in seen:
                semiprimes.append((N, int(p), int(q)))
                seen.add(N)

    semiprimes.sort()
    return semiprimes


def balanced_semiprimes(bit_sizes, count_per_size=5):
    """
    Generate balanced semiprimes (p ~ q ~ sqrt(N)) at specified bit sizes.

    bit_sizes: list of target bit-lengths for N.
    Returns list of (N, p, q) tuples.
    """
    results = []
    for bits in bit_sizes:
        half_bits = bits // 2
        p_start = next_prime(2**(half_bits - 1))
        p = p_start
        for _ in range(count_per_size):
            q = next_prime(p + 1)
            N = int(p * q)
            results.append((N, int(p), int(q)))
            p = next_prime(q)
    return results
