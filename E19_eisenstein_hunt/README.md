# E19: Eisenstein Congruence Hunt

## Hypothesis

E13 showed that Eisenstein congruences leak 63 bits of factor information
across 7 channels, but computing σ_{k-1}(N) costs O(N). Could some other
poly(log N)-computable function h(N) coincidentally equal σ_{k-1}(N) mod ℓ
for all semiprimes? If so, factoring breaks.

## What It Tests

Exhaustive fail-fast search over 23 families of poly(log N)-computable
candidate functions, tested against ground truth σ_{k-1}(N) mod ℓ for
1,000–200,000 balanced semiprimes (16-48 bit). Semiprime counts scale with ℓ
to ensure sufficient birthday collisions for mod-ℓ and mod-ℓ² consistency
checks. All semiprimes are deduplicated by N value.

### Congruence Channels (from E13)

| Weight k | ℓ (Bernoulli prime) |
|----------|---------------------|
| 12 | 691 |
| 16 | 3617 |
| 18 | 43867 |
| 20 | 283, 617 |
| 22 | 131, 593 |

### Candidate Families (23 total)

**Algebraic (15 families):**

1. **Power residues**: N^a mod ℓ, a ∈ 1..ℓ-1 (full range, uncapped)
2. **Kronecker symbols**: kronecker(d, N) for 25 discriminants
3. **Linear combinations**: c₁·N^{a₁} + c₂·N^{a₂} mod ℓ, a ∈ 0..200, c ∈ 1..10
4. **Lucas U sequences**: U_N(P,Q) mod ℓ for 220 parameter pairs
5. **Lucas V sequences**: V_N(P,Q) mod ℓ for 220 parameter pairs
6. **CF convergents**: continued fraction convergents of N/ℓ mod ℓ
7. **Discrete log residues**: ind_g(N mod ℓ) and reductions
8. **Binomial coefficients**: C(N mod ℓ, j) mod ℓ, j = 1..20
9. **Fermat/Eisenstein quotients**: (N^{ℓ-1}-1)/ℓ mod ℓ and variants
10. **Multiplicative order**: ord(N mod ℓ) and transforms
11. **Second ℓ-adic digit**: ⌊N/ℓ⌋ mod ℓ and power residues thereof
12. **Gauss sums**: Σ (t/ℓ)·t^a mod ℓ for algebraic exponents
13. **ℓ²-lifted linear combos**: c₁·(N mod ℓ²)^{a₁} + c₂·(N mod ℓ²)^{a₂} mod ℓ
14. **Order × power compositions**: ord(N)^a · N^b mod ℓ
15. **Power residue lifts**: N^a mod ℓ², reduced mod ℓ

**Bit-pattern (6 families, bridging E10-E12 carry barrier):**

16. **Popcount**: popcount(N)^a mod ℓ for a = 1..20
17. **Digit sums**: digit_sum(N, base)^a mod ℓ for bases {2,3,5,7,10,16}
18. **XOR-fold**: xor_fold(N, width)^a mod ℓ for widths {4,8,16}
19. **Byte sum**: byte_sum(N)^a mod ℓ
20. **Alternating bit sum**: alt_bit_sum(N)^a mod ℓ
21. **Bit × power compositions**: bit_func(N) · N^a mod ℓ for 5 bit primitives

**Auxiliary modulus (2 families, testing cross-modular dependence):**

22. **Auxiliary power residues**: (N mod m)^a mod ℓ for m ∈ {ℓ-1, ℓ+1}
23. **Mixed modulus compositions**: (N mod ℓ)^a · (N mod m)^b mod ℓ

### Collision Checks

Eight data-driven consistency tests subsume entire function families:

**Primary (from N mod ℓ^k):**
- **N mod ℓ**: Is σ_{k-1}(N) mod ℓ a function of (N mod ℓ) alone?
- **N mod ℓ²**: Is σ_{k-1}(N) mod ℓ a function of (N mod ℓ²) alone?

**Auxiliary modulus (genuinely different residue classes):**
- **N mod (ℓ-1)**: Tests all functions depending on multiplicative order structure
- **N mod (ℓ+1)**: Tests Frobenius trace analog
- **N mod 2(ℓ-1)**: Closes Lucas sequences with QR discriminant (period | 2(ℓ-1))
- **N mod 2(ℓ+1)**: Closes Lucas sequences with QNR discriminant (period | 2(ℓ+1))
- **N mod 2ℓ**: Tests parity combined with mod-ℓ residue
- **(N mod ℓ, N mod (ℓ-1)) joint**: Strongest single-channel test — by CRT
  (gcd(ℓ, ℓ-1) = 1), covers ALL functions of N mod ℓ(ℓ-1)

All eight fail at n(1) for all 7 channels. The very first genuine collision
is inconsistent in every case — decisive failure.

### Fail-Fast Strategy

Each candidate is tested sequentially against semiprimes. On first mismatch,
the candidate is eliminated. Expected survival after 1st semiprime: ~1/ℓ.
After 2nd: ~1/ℓ². After 3rd: essentially zero. Rayon parallelizes over
candidates. Average evaluations per candidate: ~1.0.

## Key Finding

**All 13,056,781 candidates eliminated across all 7 channels. Zero survivors.**
**All 56 auxiliary collision checks inconsistent at first collision.**

| k | ℓ | Semiprimes | col_ℓ | col_ℓ² | col_{ℓ-1} | col_{ℓ+1} | col_{2(ℓ-1)} | col_{2(ℓ+1)} | col_{2ℓ} | col_joint | Candidates | Surv |
|---|---|-----------|-------|--------|-----------|-----------|-------------|-------------|----------|-----------|------------|------|
| 12 | 691 | 5K | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | 2,024,347 | 0 |
| 16 | 3,617 | 20K | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | 2,027,263 | 0 |
| 18 | 43,867 | 200K | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | 2,067,518 | 0 |
| 20 | 283 | 1K | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | 2,023,934 | 0 |
| 20 | 617 | 5K | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | 2,024,268 | 0 |
| 22 | 131 | 1K | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | 865,212 | 0 |
| 22 | 593 | 5K | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | n(1) | 2,024,239 | 0 |

All `n(1)` means the very first genuine collision was inconsistent.

### What the Auxiliary Collision Checks Prove

The joint (N mod ℓ, N mod (ℓ-1)) check is the strongest: since
gcd(ℓ, ℓ-1) = 1, by CRT it tests whether σ_{k-1}(N) mod ℓ is a function
of N mod ℓ(ℓ-1). This subsumes:

- All polynomials over F_ℓ (already killed by N mod ℓ check)
- All functions of the multiplicative order structure (N mod (ℓ-1))
- All higher-order Gauss sums (depend on N mod (ℓ-1))
- All mixed functions combining primary and auxiliary modular residues
- All characters of order d | (ℓ-1)

The N mod (ℓ+1) check independently closes the Frobenius trace channel.

The N mod 2(ℓ-1) and N mod 2(ℓ+1) checks close the **Lucas period gap**:
Lucas sequences U_N(P,Q) and V_N(P,Q) mod ℓ have periods dividing 2(ℓ-1)
(when discriminant P²-4Q is a QR mod ℓ) or 2(ℓ+1) (when QNR). The
factor of 2 means the N mod (ℓ±1) checks alone don't fully cover them.
With both 2(ℓ-1) and 2(ℓ+1) checked, ALL Lucas sequences are subsumed
regardless of parameter choice — making the explicit 220-pair search
redundant.

## Complexity

- Candidate evaluation: O(log N · log ℓ) per candidate (modular exponentiation)
- Total search: ~13.1M candidates × ~1.0 evaluations (fail-fast) = 2.01 seconds
- Throughput: ~6.5M candidates/second
- All candidate functions are poly(log N)
- Deterministic: seeded RNG (seeds 42, 137, 271, 389) for full reproducibility
- Semiprimes deduplicated by N value to ensure genuine collision tests

## Conclusion

The Eisenstein indirect evaluation gate is empirically closed across all
tested modular channels. No poly(log N)-computable function from 23 candidate
families (algebraic + bit-pattern + auxiliary modulus) matches σ_{k-1}(N) mod ℓ
for any channel.

Eight independent collision checks confirm the target depends on the
factorization of N, not on N mod m for any tested modulus m ∈ {ℓ, ℓ²,
ℓ-1, ℓ+1, 2(ℓ-1), 2(ℓ+1), 2ℓ, ℓ(ℓ-1)}. The joint check N mod ℓ(ℓ-1) is
particularly decisive: it rules out ALL functions depending on both the
primary residue and the multiplicative group structure simultaneously.
The 2(ℓ±1) checks formally close the Lucas sequence period gap.

## Implementation

Rust crate: [`../rust/eisenstein-hunt/`](../rust/eisenstein-hunt/)

```bash
cd rust && cargo test -p eisenstein-hunt
cd rust && cargo run --release -p eisenstein-hunt
```
