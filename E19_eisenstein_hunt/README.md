# E19: Eisenstein Congruence Hunt

## Hypothesis

E13 showed that Eisenstein congruences leak 63 bits of factor information
across 7 channels, but computing σ_{k-1}(N) costs O(N). Could some other
poly(log N)-computable function h(N) coincidentally equal σ_{k-1}(N) mod ℓ
for all semiprimes? If so, factoring breaks.

## What It Tests

Exhaustive fail-fast search over 15 families of poly(log N)-computable
candidate functions, tested against ground truth σ_{k-1}(N) mod ℓ for
1,000–20,000 balanced semiprimes (16-32 bit). Semiprime counts scale with ℓ
to ensure sufficient birthday collisions for mod-ℓ and mod-ℓ² consistency
checks.

### Congruence Channels (from E13)

| Weight k | ℓ (Bernoulli prime) |
|----------|---------------------|
| 12 | 691 |
| 16 | 3617 |
| 18 | 43867 |
| 20 | 283, 617 |
| 22 | 131, 593 |

### Candidate Families (15 total)

1. **Power residues**: N^a mod ℓ, a ∈ 1..ℓ-1 (full range, uncapped)
2. **Kronecker symbols**: kronecker(d, N) for 25 discriminants
3. **Linear combinations**: c₁·N^{a₁} + c₂·N^{a₂} mod ℓ, a ∈ 0..200, c ∈ 1..10 (~200K per channel)
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

### Collision Checks

Two data-driven consistency tests subsume all polynomial candidates:

- **N mod ℓ**: Is σ_{k-1}(N) mod ℓ a function of (N mod ℓ) alone?
- **N mod ℓ²**: Is σ_{k-1}(N) mod ℓ a function of (N mod ℓ²) alone?

If either is consistent, some polynomial over F_ℓ or two-digit ℓ-adic
function could match. Both fail for 6 of 7 channels. Only ℓ=43867 shows
vacuous consistency at mod-ℓ² (ℓ²≈1.9B, too large for birthday collisions
with 20K samples).

### Fail-Fast Strategy

Each candidate is tested sequentially against semiprimes. On first mismatch,
the candidate is eliminated. Expected survival after 1st semiprime: ~1/ℓ.
After 2nd: ~1/ℓ². After 3rd: essentially zero. Rayon parallelizes over
candidates. Average evaluations per candidate: ~1.0.

## Key Finding

**All 13,048,101 candidates eliminated across all 7 channels. Zero survivors.**

| k | ℓ | col_ℓ | col_ℓ² | Candidates | Survived 1st | Survived All |
|---|---|-------|--------|-----------|-------------|-------------|
| 12 | 691 | n(2) | n(25) | 2,023,107 | 2,952 | 0 |
| 16 | 3,617 | n(1) | n(604) | 2,026,023 | 561 | 0 |
| 18 | 43,867 | n(2) | y(4620) | 2,066,278 | 71 | 0 |
| 20 | 283 | n(1) | n(13) | 2,022,694 | 7,137 | 0 |
| 20 | 617 | n(2) | n(63) | 2,023,028 | 2 | 0 |
| 22 | 131 | n(1) | n(6) | 863,972 | 6,629 | 0 |
| 22 | 593 | n(2) | n(51) | 2,022,999 | 3,383 | 0 |

Column key: `col_ℓ` = N mod ℓ collision consistency (y/n, count tested),
`col_ℓ²` = N mod ℓ² collision consistency.

Both collision checks confirm σ_{k-1}(N) mod ℓ is NOT a function of N mod ℓ
(or N mod ℓ²) alone — it depends on the factorization. The sole `y` for
ℓ=43867 at mod-ℓ² is vacuously true (ℓ²≈1.9B exceeds sample space).

## Complexity

- Candidate evaluation: O(log N · log ℓ) per candidate (modular exponentiation)
- Total search: ~13M candidates × ~1.0 evaluations (fail-fast) = 1.44 seconds
- Throughput: ~9M candidates/second
- All candidate functions are poly(log N)
- Deterministic: seeded RNG (seeds 42, 137, 271) for full reproducibility

## Conclusion

The Eisenstein indirect evaluation gate is empirically closed. No poly(log N)-
computable function from 15 candidate families matches σ_{k-1}(N) mod ℓ for
any channel. The collision checks independently confirm the target depends on
the factorization of N, not just on N mod ℓ or N mod ℓ².

## Implementation

Rust crate: [`../rust/eisenstein-hunt/`](../rust/eisenstein-hunt/)

```bash
cd rust && cargo test -p eisenstein-hunt
cd rust && cargo run --release -p eisenstein-hunt
```
