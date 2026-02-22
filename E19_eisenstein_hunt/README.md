# E19: Eisenstein Congruence Hunt

## Hypothesis

E13 showed that Eisenstein congruences leak 63 bits of factor information
across 7 channels, but computing σ_{k-1}(N) costs O(N). Could some other
poly(log N)-computable function h(N) coincidentally equal σ_{k-1}(N) mod ℓ
for all semiprimes? If so, factoring breaks.

## What It Tests

Exhaustive fail-fast search over 9 families of poly(log N)-computable
candidate functions, tested against ground truth σ_{k-1}(N) mod ℓ for
1000-5000 balanced semiprimes (16-32 bit).

### Congruence Channels (from E13)

| Weight k | ℓ (Bernoulli prime) |
|----------|---------------------|
| 12 | 691 |
| 16 | 3617 |
| 18 | 43867 |
| 20 | 283, 617 |
| 22 | 131, 593 |

### Candidate Families

1. **Power residues**: N^a mod ℓ (up to 1000 exponents per channel)
2. **Kronecker symbols**: kronecker(d, N) for 25 discriminants
3. **Linear combinations**: c₁·N^{a₁} + c₂·N^{a₂} mod ℓ (~30K per channel)
4. **Collision check**: data-driven test — is σ_{k-1}(N) a function of N mod ℓ?
5. **Lucas sequences**: U_N(P,Q) and V_N(P,Q) for 220 parameter pairs
6. **CF convergents**: continued fraction convergents of N/ℓ
7. **Discrete log residues**: ind_g(N mod ℓ) and reductions
8. **Binomial coefficients**: C(N mod ℓ, j) for j = 1..20
9. **Fermat/Eisenstein quotients**: (N^{ℓ-1}-1)/ℓ and variants

### Fail-Fast Strategy

Each candidate is tested sequentially against semiprimes. On first mismatch,
the candidate is eliminated. Expected survival after 1st semiprime: ~1/ℓ.
After 2nd: ~1/ℓ². After 3rd: essentially zero. Rayon parallelizes over
candidates.

## Key Finding

**All 230,981 candidates eliminated across all 7 channels. Zero survivors.**

| k | ℓ | Collision | Candidates | Survived 1st | Survived All |
|---|---|-----------|-----------|-------------|-------------|
| 12 | 691 | no | 33,075 | 54 | 0 |
| 16 | 3617 | no | 33,379 | 10 | 0 |
| 18 | 43867 | no | 33,382 | 1 | 0 |
| 20 | 283 | no | 32,664 | 112 | 0 |
| 20 | 617 | no | 32,998 | 55 | 0 |
| 22 | 131 | no | 32,512 | 10 | 0 |
| 22 | 593 | no | 32,971 | 56 | 0 |

The collision check confirms σ_{k-1}(N) mod ℓ is NOT a function of N mod ℓ
alone — it depends on the factorization.

## Complexity

- Candidate evaluation: O(log N · log ℓ) per candidate (modular exponentiation)
- Total search: ~231K candidates × ~1.0 evaluations (fail-fast) = 0.05 seconds
- All candidate functions are poly(log N)

## Implementation

Rust crate: [`../rust/eisenstein-hunt/`](../rust/eisenstein-hunt/)

```bash
cd rust && cargo test -p eisenstein-hunt
cd rust && cargo run --release -p eisenstein-hunt
```
