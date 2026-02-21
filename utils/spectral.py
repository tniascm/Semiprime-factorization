"""
Shared DFT and spectral analysis utilities for semiprime experiments.

Standardizes the DFT convention and provides reusable spectral metrics
with Parseval verification.

DFT Convention (standard signal processing):
    hat{f}(xi) = (1/N) sum_{t=0}^{N-1} f(t) * e^{-2*pi*i*xi*t/N}

This matches BARRIER_THEOREM.md (line 85) and numpy.fft.fft (up to 1/N scaling).
For real-valued signals, |hat{f}(xi)| is identical under either sign convention.
"""
import numpy as np


def dft(sig):
    """
    Compute the normalized DFT of a signal on Z/NZ.

    Uses the standard negative-exponential convention:
        hat{f}(xi) = (1/N) sum_t f(t) e^{-2*pi*i*xi*t/N}

    This matches numpy.fft.fft divided by N.
    """
    N = len(sig)
    X = np.fft.fft(sig) / N
    return X


def verify_parseval(sig, X, rtol=1e-6):
    """
    Verify Parseval's theorem: sum|hat{f}(xi)|^2 = (1/N) sum|f(t)|^2.

    With our convention hat{f} = fft/N:
        sum|hat{f}|^2 = (1/N^2) sum|fft|^2 = (1/N^2)(N * sum|f|^2) = (1/N) sum|f|^2

    Parameters
    ----------
    sig : array-like
        Original signal f(t).
    X : array-like
        DFT coefficients hat{f}(xi).
    rtol : float
        Relative tolerance for the check.

    Returns
    -------
    bool
        True if Parseval holds within tolerance.

    Raises
    ------
    ValueError
        If Parseval check fails.
    """
    N = len(sig)
    energy_time = np.sum(np.abs(sig) ** 2) / N
    energy_freq = np.sum(np.abs(X) ** 2)

    if energy_time == 0 and energy_freq == 0:
        return True

    rel_err = abs(energy_freq - energy_time) / max(energy_time, energy_freq)
    if rel_err > rtol:
        raise ValueError(
            f"Parseval check failed: time-domain energy = {energy_time:.6e}, "
            f"freq-domain energy = {energy_freq:.6e}, relative error = {rel_err:.6e}"
        )
    return True


def precompute_gcd_classes(N, p, q):
    """
    Precompute gcd classification for all xi in [0, N).

    Returns
    -------
    gcd_class : np.array of int, length N
        0 = gcd(xi,N)==1, 1 = p|xi only, 2 = q|xi only, 3 = N|xi (i.e. xi=0)
    n_factor_modes : int
        Count of xi in [1, N) with gcd(xi, N) > 1.
    """
    N_int = int(N)
    p_int = int(p)
    q_int = int(q)
    gcd_class = np.zeros(N_int, dtype=np.int32)
    n_factor_modes = 0
    for xi in range(N_int):
        p_div = (xi % p_int == 0)
        q_div = (xi % q_int == 0)
        if p_div and q_div:
            gcd_class[xi] = 3  # xi = 0 or xi = N (only xi=0 in range)
        elif p_div:
            gcd_class[xi] = 1
            if xi > 0:
                n_factor_modes += 1
        elif q_div:
            gcd_class[xi] = 2
            if xi > 0:
                n_factor_modes += 1
        else:
            gcd_class[xi] = 0
    return gcd_class, n_factor_modes


def analyze_signal(sig, N, p, q, gcd_class=None, n_factor_modes=None):
    """
    Compute DFT and extract spectral metrics for a signal on Z/NZ.

    Parameters
    ----------
    sig : np.array
        Signal of length N.
    N, p, q : int
        Semiprime N = p*q.
    gcd_class : np.array, optional
        Precomputed gcd classification (from precompute_gcd_classes).
    n_factor_modes : int, optional
        Precomputed count of factor modes.

    Returns
    -------
    dict
        Spectral metrics including peak, energy fractions, CRT rank estimates.
    """
    N_int = int(N)
    p_int = int(p)
    q_int = int(q)

    assert len(sig) == N_int, f"Signal length {len(sig)} != N={N_int}"

    X = dft(sig)
    verify_parseval(sig, X)

    mags = np.abs(X)
    mags2 = mags ** 2

    # Skip DC component for spectral analysis
    mags_ac = mags[1:]
    mags2_ac = mags2[1:]

    # Peak-to-bulk ratio (using median for robustness against outliers)
    peak = np.max(mags_ac)
    median_val = np.median(mags_ac)
    peak_to_bulk = float(peak / median_val) if median_val > 0 else float('inf')

    # Precompute gcd classes if not provided
    if gcd_class is None or n_factor_modes is None:
        gcd_class, n_factor_modes = precompute_gcd_classes(N_int, p_int, q_int)

    # Factor-localized energy: energy at xi with gcd(xi, N) > 1
    energy_total = float(np.sum(mags2_ac))
    factor_mask = gcd_class[1:] > 0  # classes 1, 2, 3 (but 3 is only xi=0, excluded)
    p_mask = (gcd_class[1:] == 1) | (gcd_class[1:] == 3)
    q_mask = (gcd_class[1:] == 2) | (gcd_class[1:] == 3)

    energy_factor = float(np.sum(mags2_ac[factor_mask]))
    energy_p = float(np.sum(mags2_ac[p_mask]))
    energy_q = float(np.sum(mags2_ac[q_mask]))

    factor_frac = float(energy_factor / energy_total) if energy_total > 0 else 0.0
    p_frac = float(energy_p / energy_total) if energy_total > 0 else 0.0
    q_frac = float(energy_q / energy_total) if energy_total > 0 else 0.0

    # Expected factor energy under flat spectrum
    expected_factor_frac = float(n_factor_modes) / (N_int - 1) if N_int > 1 else 0.0

    # Top-mode factor extraction: do the K largest modes have gcd > 1?
    sorted_indices = np.argsort(mags_ac)[::-1]  # indices into mags_ac (0-based)
    top_10_factor = sum(1 for i in sorted_indices[:10] if gcd_class[i + 1] > 0)
    top_5_factor = sum(1 for i in sorted_indices[:5] if gcd_class[i + 1] > 0)

    # CRT rank estimate: reshape signal as p x q matrix and compute SVD
    M = np.zeros((p_int, q_int), dtype=np.float64)
    for t in range(N_int):
        M[t % p_int, t % q_int] = sig[t]
    sv = np.linalg.svd(M, compute_uv=False)
    sv_total = float(np.sum(sv))
    if sv_total > 0:
        sv_norm = sv / sv_total
        cumsum = np.cumsum(sv_norm)
        eff_rank_90 = int(np.searchsorted(cumsum, 0.90)) + 1
    else:
        eff_rank_90 = 0
    nuclear_ratio = float(sv_total / sv[0]) if sv[0] > 0 else 0.0

    return {
        'peak': float(peak),
        'median': float(median_val),
        'peak_to_bulk': peak_to_bulk,
        'energy_total': energy_total,
        'factor_energy_frac': factor_frac,
        'p_energy_frac': p_frac,
        'q_energy_frac': q_frac,
        'expected_factor_frac': expected_factor_frac,
        'factor_energy_excess': factor_frac / expected_factor_frac if expected_factor_frac > 0 else 0.0,
        'top_5_factor_count': top_5_factor,
        'top_10_factor_count': top_10_factor,
        'crt_eff_rank_90': eff_rank_90,
        'crt_nuclear_ratio': nuclear_ratio,
        'crt_top_sv': float(sv[0]),
        'crt_sv_2': float(sv[1]) if len(sv) > 1 else 0.0,
    }
