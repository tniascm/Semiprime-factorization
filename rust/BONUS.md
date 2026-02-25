# Bonus: Non-Poly(log N) Factoring Implementations

These 17 Rust crates implement factoring approaches that are NOT poly(log N).
They serve as reference implementations, comparative baselines, and explorations
of sub-exponential and heuristic methods.

## Classical Methods

| Crate | LOC | Complexity | Description |
|-------|-----|-----------|-------------|
| `classical-nfs` | 4630 | L_N[1/3] | Number Field Sieve: polynomial selection, sieving, linear algebra over GF(2), square root |
| `ecm` | 684 | L_p[1/2] | Elliptic Curve Method: Montgomery curves, stage-1/stage-2 with B1/B2 bounds |
| `cf-factor` | 2650 | O(N^1/4) | Continued fraction factoring: CFRAC, SQUFOF, regulator computation, quadratic forms |
| `lattice-reduction` | 781 | Poly(N) | Lattice-based factoring: LLL reduction on Coppersmith-style lattices |
| `cf-factor-ms` | 1962 | O(N^1/4) | Murru-Salvatori CF factoring: BSGS in real quadratic class group, regulator-guided search |
| `cado-evolve` | 8364 | L_N[1/3] | CADO-NFS wrapper with evolutionary parameter tuning, scaling protocol, batch benchmarks |

## Physics and Optimization Approaches

| Crate | LOC | Complexity | Description |
|-------|-----|-----------|-------------|
| `ising-factoring` | 744 | Heuristic | Maps factoring to Ising model energy minimization with simulated annealing |
| `quantum-inspired` | 918 | Heuristic | Quantum-inspired classical algorithms: amplitude-based search, Grover-like iteration |
| `alpha-evolve` | 6158 | Heuristic | Evolutionary search: population of factoring strategies with fitness-driven mutation, symbolic regression, novelty search |

## Spectral and Number-Theoretic Methods

| Crate | LOC | Complexity | Description |
|-------|-----|-----------|-------------|
| `spectral-factoring` | 2316 | O(N) | Modular symbols, Hecke operators, Atkin-Lehner involutions, spectral decomposition |
| `l-function-factoring` | 3327 | O(N) | L-function evaluation, Dirichlet characters, Gauss sums, class number computation |
| `class-number-oracle` | 1673 | O(sqrt(N)) | Class number computation via Eichler-Selberg trace formula, discriminant analysis |
| `trace-lattice` | 2204 | O(sqrt(N)) | Trace-based lattice construction: Frobenius traces, lattice reduction for factor recovery |
| `tnss-factoring` | 6722 | Sub-exp | Tensor Network State Sieve: tensor train decomposition, SR-pair relations, OPES optimizer |
| `smooth-pilatte` | 1925 | Sub-exp | Pilatte lattice-geometric smooth relation extraction: Fincke-Pohst enumeration, GF(2) linear algebra |

## Compression and Other

| Crate | LOC | Complexity | Description |
|-------|-----|-----------|-------------|
| `compression-factor` | 547 | Heuristic | Tests whether Kolmogorov complexity differences between semiprimes and primes are exploitable |
| `benchmarks` | 1217 | N/A | Cross-crate benchmark suite: comparative scaling analysis across all methods |

## Building and Running

All crates are part of the workspace at `rust/Cargo.toml`:

```bash
# Run a specific crate
cargo run -p classical-nfs
cargo run -p ecm

# Run all tests
cargo test

# Run benchmarks
cargo bench -p benchmarks
```

## Key Takeaway

All sub-exponential methods (NFS, ECM, CFRAC) achieve practical speedups for
specific semiprime sizes but remain fundamentally super-polynomial. Heuristic
approaches (Ising, evolutionary, compression) show no systematic advantage over
random search. Spectral methods confirm the dimension barrier (dim O(N)).
