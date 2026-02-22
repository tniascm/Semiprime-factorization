# E19: Eisenstein Congruence Hunt

## Hypothesis

E13 showed that Eisenstein congruences leak 63 bits of factor information
across 7 channels, but computing σ_{k-1}(N) costs O(N). Could some other
poly(log N)-computable function h(N) coincidentally equal σ_{k-1}(N) mod ℓ
for all semiprimes? If so, factoring breaks.

## What It Tests

Exhaustive fail-fast search over 21 families of poly(log N)-computable
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

### Candidate Families (21 total)

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

### Collision Checks

Two data-driven consistency tests subsume all polynomial candidates:

- **N mod ℓ**: Is σ_{k-1}(N) mod ℓ a function of (N mod ℓ) alone?
- **N mod ℓ²**: Is σ_{k-1}(N) mod ℓ a function of (N mod ℓ²) alone?

Both fail for all 7 channels. The ℓ=43867 mod-ℓ² gap (previously vacuously
consistent due to duplicate semiprimes) is now closed with 200K deduplicated
semiprimes in 24-48 bit range, providing genuine birthday collisions at
ℓ²≈1.924B.

### Fail-Fast Strategy

Each candidate is tested sequentially against semiprimes. On first mismatch,
the candidate is eliminated. Expected survival after 1st semiprime: ~1/ℓ.
After 2nd: ~1/ℓ². After 3rd: essentially zero. Rayon parallelizes over
candidates. Average evaluations per candidate: ~1.0.

## Key Finding

**All 13,050,481 candidates eliminated across all 7 channels. Zero survivors.**

| k | ℓ | col_ℓ | col_ℓ² | Candidates | Survived 1st | Survived All |
|---|---|-------|--------|-----------|-------------|-------------|
| 12 | 691 | n(1) | n(1) | 2,023,447 | 2,953 | 0 |
| 16 | 3,617 | n(1) | n(1) | 2,026,363 | 561 | 0 |
| 18 | 43,867 | n(1) | n(1) | 2,066,618 | 55 | 0 |
| 20 | 283 | n(1) | n(1) | 2,023,034 | 7,141 | 0 |
| 20 | 617 | n(1) | n(1) | 2,023,368 | 3 | 0 |
| 22 | 131 | n(1) | n(1) | 864,312 | 6,629 | 0 |
| 22 | 593 | n(1) | n(1) | 2,023,339 | 3,383 | 0 |

Column key: `col_ℓ` = N mod ℓ collision consistency (n = inconsistent, count
tested), `col_ℓ²` = N mod ℓ² collision consistency. All `n(1)` means the
very first genuine collision was inconsistent — decisive failure.

## Complexity

- Candidate evaluation: O(log N · log ℓ) per candidate (modular exponentiation)
- Total search: ~13M candidates × ~1.0 evaluations (fail-fast) = 1.97 seconds
- Throughput: ~6.6M candidates/second
- All candidate functions are poly(log N)
- Deterministic: seeded RNG (seeds 42, 137, 271, 389) for full reproducibility
- Semiprimes deduplicated by N value to ensure genuine collision tests

## Conclusion

The Eisenstein indirect evaluation gate is empirically closed. No poly(log N)-
computable function from 21 candidate families (algebraic + bit-pattern)
matches σ_{k-1}(N) mod ℓ for any channel. The collision checks independently
confirm the target depends on the factorization of N, not on N mod ℓ^k for
k = 1 or 2. The bit-pattern families bridge the E10-E12 carry barrier,
confirming that integer-representation features (popcount, digit sums, XOR
folds) also cannot reconstruct divisor sum congruences.

## Implementation

Rust crate: [`../rust/eisenstein-hunt/`](../rust/eisenstein-hunt/)

```bash
cd rust && cargo test -p eisenstein-hunt
cd rust && cargo run --release -p eisenstein-hunt
```
