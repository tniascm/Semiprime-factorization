# E3: Representation Duality — Multi-Base and SSD Linearization

## Hypothesis

Re-representing a semiprime N in alternative number systems or transform domains
might expose structural patterns invisible in standard binary/decimal form,
enabling parallel divisibility testing.

## What It Tests

Two complementary approaches:

### Multi-Base Analysis (`multi-base`)
- **9 number bases**: binary, ternary, base-6, decimal, hex, primorial-3 (30),
  primorial-4 (210), prime-41, prime-43
- **RNS representation**: residues mod first 20 primes
- **Per-base statistics**: Shannon entropy, digit frequency, lag-1 autocorrelation
- **Cross-base anomaly detection**: z-score comparison of semiprime vs random
  number statistics across all bases
- **Balanced ternary**: digit weight analysis (count of +1 vs -1 digits)
- **Factorial number system**: radix-increasing representation with max-ratio stats
- **Digit transition matrices**: Markov chain model of digit sequences with
  conditional entropy computation

### SSD Linearization (`ssd-factoring`)
Three formulations of parallel divisibility testing inspired by State Space Duality:

1. **Binary lift**: represent N in binary, compute N mod d via matrix-vector
   product over the bit representation
2. **NTT domain**: embed residue computation in Number Theoretic Transform space
   (modulus 998244353) for parallel evaluation
3. **CRT parallel**: decompose N via Chinese Remainder Theorem into small-prime
   residues, then reconstruct N mod d from CRT components

## Key Finding

- **Multi-base**: Cross-base entropy variance and anomaly z-scores show no
  statistically significant difference between semiprimes and random numbers of
  the same bit length. Digit patterns in any base are CRT observables.
- **SSD**: All three linearization strategies produce correct results but offer
  no asymptotic speedup over sequential trial division. The binary lift requires
  O(log N) bits, NTT requires O(sqrt(N)) transform points, and CRT reconstruction
  requires knowing the moduli — which is equivalent to having the factor base.

## Implementation

Rust crates: [`../rust/multi-base/`](../rust/multi-base/) and
[`../rust/ssd-factoring/`](../rust/ssd-factoring/)

```bash
cd rust && cargo run -p multi-base
cd rust && cargo run -p ssd-factoring
cargo test -p multi-base -p ssd-factoring
```

## Complexity

- Multi-base feature computation: O(log^2 N) per base — poly(log N)
- Binary lift divisibility: O(log N * D) where D = number of trial divisors
- NTT-domain: O(sqrt(N) log sqrt(N)) — NOT poly(log N)
- CRT parallel: O(pi(B) * log N) where B = smoothness bound
