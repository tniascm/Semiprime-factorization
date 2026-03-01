# E29 Scaling Experiment: MCMC Smooth Rate vs Bit Size

## Setup

- Bit sizes: 32, 40, 48, 56, 64, 80, 96, 112, 128, 160, 200
- 10 semiprimes per size (≤128), 5 (≤160), 3 (>160)
- Candidates: 10K (≤64), 5K (65-128), 1K (129-160), 500 (>160)
- MCMC: 10 chains, simulated annealing T=10→0.1
- Lattice sieve: line sieve, capped at ≤80 bits (prohibitive beyond)
- Polynomial: base-m, degree 3/4/5 by bit size
- Smoothness: u64 trial division (norms >u64 declared not smooth)

## Results

| Bits | Degree | MCMC Unique Rate | MCMC Dup% | Smooth/SP | Uniform Rate | Lattice Rate | MCMC/Lattice | MCMC Rels/s | CADO Rels/s | MCMC/CADO |
|------|--------|------------------|-----------|-----------|--------------|--------------|--------------|-------------|-------------|-----------|
| 32   | 3      | 1.57%            | 83%       | 23.9      | 0.008%       | 0.010%       | 154x         | 3,630       | —           | —         |
| 40   | 3      | 0.67%            | 83%       | 10.5      | 0.003%       | 0.010%       | 68x          | 1,451       | —           | —         |
| 48   | 3      | 1.59%            | 80%       | 28.1      | 0.007%       | 0.004%       | 357x         | 3,500       | —           | —         |
| 56   | 3      | 1.95%            | 80%       | 35.9      | 0.003%       | 0.005%       | 392x         | 3,964       | —           | —         |
| 64   | 3      | 2.18%            | 79%       | 39.7      | 0.006%       | 0.003%       | 706x         | 3,519       | —           | —         |
| 80   | 3      | 1.82%            | 78%       | 15.8      | 0%           | 0.001%       | 1797x        | 2,109       | —           | —         |
| 96   | 3      | 1.08%            | 81%       | 8.5       | 0%           | —            | —            | 720         | 79,261      | 0.009x    |
| 112  | 4      | 0.90%            | 85%       | 4.3       | 0%           | —            | —            | 288         | 88,960      | 0.003x    |
| 128  | 4      | 0.67%            | 87%       | 2.0       | 0%           | —            | —            | 102         | 50,411      | 0.002x    |
| 160  | 5      | 0%               | —         | 0         | 0%           | —            | —            | 0           | 21,739      | 0         |
| 200  | 5      | 0%               | —         | 0         | 0%           | —            | —            | 0           | 5,358       | 0         |

CADO-NFS: production special-q lattice siever with ECM cofactorization, 4 threads.
MCMC/CADO ratio < 1 means CADO is faster. Sizes 32-80 below CADO's minimum parameter file (c30).

## Scaling Analysis (linear regression on 32-128 bit data)

- MCMC rate decay: `ln(rate) = -0.0063 * bits - 3.91` (R² = 0.21)
- Advantage ratio trend: `ratio = -3.81 * bits + 454.87`
- Predicted zero-crossing: 1581 bits (extrapolated)

## Key Findings

### 1. MCMC smooth rate decays remarkably slowly (32-128 bits)

The unique smooth rate stays in the 0.7-2.2% range across a 4x increase in bit
size (32→128). The R²=0.21 means the linear fit is weak — the rate is nearly
flat. This contrasts sharply with Dickman rho predictions for random sampling,
where smooth probability drops super-exponentially. MCMC's energy bias
concentrates on small-norm regions that remain smooth even as the overall search
space grows.

### 2. MCMC advantage grows with N

- MCMC/lattice ratio: 154x (32-bit) → 1797x (80-bit)
- MCMC/uniform: effectively infinite at 80+ bits (uniform finds zero smooth pairs)
- Energy bias becomes more valuable as the smooth fraction of the search space
  shrinks — exactly the regime where NFS operates at scale.

### 3. Uniform sampling dies at 80 bits

Uniform random sampling finds zero smooth pairs at 80+ bits with 5K candidates.
The probability of a random (a,b) pair having both norms B-smooth falls below
the sampling resolution. MCMC's energy-biased walk targets the exponentially
thin smooth tail that uniform sampling cannot reach.

### 4. u64 smoothness boundary at 160 bits

At 160+ bits (degree 5), algebraic norms exceed u64::MAX. Our trial division
smoothness check operates on u64, so all candidates are declared not smooth.
This is a code limitation, not fundamental — BigInt trial division or ECM
smoothness checking would extend the viable range. MCMC energy biasing still
works correctly (energy computed via `bits() * ln(2)` approximation for BigUint
norms).

### 5. Duplicate rate rises modestly

MCMC duplicate rate: 79% (64-bit) → 87% (128-bit). Chains converge on fewer
energy basins as N grows. Still acceptable: 13% of candidates are unique at
128 bits.

### 6. Throughput

- MCMC: 3600 rels/sec (32-bit) → 102 rels/sec (128-bit)
- Lattice sieve: 16 rels/sec (32-bit) → 1.3 rels/sec (80-bit)
- MCMC is 35-1600x faster per relation found.

### 7. CADO-NFS head-to-head (96-200 bits)

CADO-NFS (production special-q lattice siever with ECM cofactorization) was run
on 3 semiprimes per bit size from the same set used by MCMC, with 4 threads.

- **CADO is 100-500x faster** at relation collection across all tested sizes
- CADO: 79K rels/sec (96-bit) → 5.4K rels/sec (200-bit)
- MCMC: 720 rels/sec (96-bit) → 0 rels/sec (160+)
- MCMC/CADO ratio: 0.009x (96-bit) → 0.002x (128-bit) → 0 (160+)
- CADO's advantage **grows** with N: the ratio worsens from 0.009x to 0.002x
- At 160-200 bits, CADO still finds 20K-50K relations while MCMC finds zero
  (u64 smoothness boundary)

The gap has three components:
1. **Special-q lattice sieve** vs random walk: CADO exploits algebraic structure
   (lattice reduction per special-q) to enumerate smooth candidates systematically
2. **ECM cofactorization**: CADO accepts partial smoothness (large primes) and uses
   ECM to factor cofactors, dramatically increasing the yield per candidate
3. **Parallelism**: CADO's 4 threads vs MCMC's single-threaded chains

## Interpretation

MCMC energy-biased sampling beats naive uniform and line sieve approaches by
100-1800x, but is **100-500x slower than CADO-NFS** (production special-q
lattice siever). The MCMC advantage over uniform/line-sieve is real but
misleading — it compares against weak baselines. Against state-of-the-art
sieving, MCMC is not competitive.

The fundamental issue is that CADO-NFS exploits multiplicative structure
(special-q lattice reduction, per-prime sieving) that random-walk MCMC cannot
replicate. MCMC finds small norms but doesn't exploit the factorization
structure needed for efficient smoothness detection. The gap widens with N.

The L[1/3] exponent of NFS is unchanged. MCMC does not improve the constant
in a way that matters against production implementations.

## Limitations

- Lattice sieve only tested ≤80 bits (line sieve too slow beyond)
- u64 smoothness check prevents MCMC measurement at 160+ bits
- Factor bases are small (≤2^19) — real NFS uses much larger bases
- MCMC uses trial division only; CADO uses ECM cofactorization (unfair but realistic)
- CADO uses 4 threads; MCMC is single-threaded (parallelism accounts for ~4x of gap)
- 3 semiprimes per size for CADO comparison gives limited statistical power
- CADO parameter files not available below 30 digits (~100 bits)
