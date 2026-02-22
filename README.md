# Semiprime Factoring Barrier: Systematic Exploration

**Author:** Andrii Potapov

A systematic computational and theoretical investigation of whether classical
poly(log N)-time integer factoring is possible, using tools from analytic number
theory, the Langlands program, algebraic extensions, group theory, and machine
learning.

## Result

**No poly(log N) classical factoring channel exists** across all tested approaches.
The barrier is computational (not information-theoretic) and is consistent with the
Quadratic Residuosity Assumption (QRA).

## Experiments

### SageMath Experiments (E4-E18)

| Experiment | What it tests | Key finding |
|-----------|--------------|-------------|
| **E4** | Hirano DW invariants | Not yet run |
| **E5** | Modular symbols dim S_k^new | O(N^1.91) scaling — killed |
| **E6** | Beilinson-Kato transform | Reduces to E7 (CRT obstruction) |
| **E7** (a-e) | Orbital integral DFT, theta functions, Kloosterman, analytic proxies | Spectral flatness: peaks decay N^{-0.25}; CRT product structure covers all pointwise Jacobi observables |
| **E8** (a-b) | L-function tomography, multi-form amplification | Root number trivial; 19 eigenforms carry ~0 bits each |
| **E9** | Hecke moments at level N | Dimension barrier: dim O(N) = exponential |
| **E10** | Integer carry signals | CRT rank grows (N^0.3) but spectra remain flat |
| **E11** | 111-feature ML sweep | All R^2_CV <= 0.025; random controls match real features |
| **E12** | Deep carry compositions | Depth HURTS: deeper = flatter spectra |
| **E13** | Eisenstein congruence channel | 63 bits of info but O(N) cost; Bach-Charles blocks poly(log N) |
| **E18** | 6 algebraic poly(log N) channels | All dead: Lucas, Frobenius, Solovay-Strassen, power residues |

E14-E17 were brainstorm gap experiments (nonlinear ML, class numbers, partial
sums, carry scaleup). All failed the poly(log N) gate and were removed.

### Rust Experiments (E1-E3)

| Experiment | What it tests | Key finding |
|-----------|--------------|-------------|
| **E1** | Group-theoretic structure of (Z/NZ)* | Smooth-order factoring requires O(sqrt(N)) samples; no poly(log N) shortcut |
| **E2** | ML feature extraction + latent space | KNN/autoencoder features are CRT observables; prediction requires labeled data (circular) |
| **E3** | Multi-base representations + SSD linearization | Cross-base anomaly z-scores insignificant; all linearizations >= O(sqrt(N)) |

### Bonus: Non-Poly(log N) Implementations

14 additional Rust implementations of classical, physics-based, spectral, and
heuristic factoring methods. See [`rust/BONUS.md`](rust/BONUS.md) for details.

## Barrier Theorem

See [BARRIER_THEOREM.md](BARRIER_THEOREM.md) for the full formalization:

1. **CRT spectral bound (proven):** Rank-r functions have DFT peaks <= r/sqrt(N)
2. **RJI model flatness (empirical):** All poly(log N)-computable observables
   (ring arithmetic + Jacobi + integer carries) produce flat spectra at factor frequencies
3. **Algebraic extension (E18):** Barrier extends to (Z/NZ)[x]/(f(x)) quotient rings
4. **QRP connection:** Spectral flatness implies QRP hardness implies factoring hardness

## Key Theoretical Results

- **Bach-Charles theorem** closes the weight-1 Edixhoven-Couveignes-Bruin gap:
  computing a_N(f) for ANY eigenform at composite N in poly(log N) implies factoring
- **Langlands ecosystem audit:** 20+ methods across functoriality, arithmetic geometry,
  and p-adic methods — all hit local decomposition, dimension barrier, or O(sqrt(N))
- **Exhaustive primitive search:** 13+ non-Langlands approaches definitively closed

## Project Structure

```
E1_group_structure/         # Group-theoretic analysis (Rust, see rust/group-structure/)
E2_ml_features/             # ML feature extraction (Rust, see rust/ai-guided/, rust/mla-number-theory/)
E3_representation_duality/  # Multi-base + SSD (Rust, see rust/multi-base/, rust/ssd-factoring/)
E4_hirano_dw_invariants/    # DW invariants (not yet run)
E5_martin_dimension/        # Modular symbols benchmark
E6_luo_ngo_kernel/          # BK transform
E7_altug_beyond_endoscopy/  # Orbital integral DFT (main line)
E8_global_projector/        # L-function tomography
E9_level_N_hecke/           # Hecke moments
E10_integer_carry/          # Carry signals
E11_feature_extraction/     # 111-feature ML sweep
E12_carry_depth/            # Deep carry compositions
E13_bach_charles/           # Eisenstein congruence channel
E18_algebraic_channels/     # Algebraic extensions
rust/                       # Rust workspace (19 crates, ~38K LOC)
  factoring-core/           # Shared library (BigUint utils, RSA gen, Pollard Rho)
  BONUS.md                  # Non-poly(log N) experiment documentation
utils/                      # Shared SageMath utilities (semiprime gen, encoding, spectral)
data/                       # JSON results and plots
research/                   # 12 annotated bibliographies
docs/plans/                 # Design documents
BARRIER_THEOREM.md          # Full barrier formalization
QRP_RESEARCH.md             # QRP literature survey
```

## Requirements

- [SageMath](https://www.sagemath.org/) >= 9.0 (for E4-E18)
- Python packages: numpy, scipy, matplotlib, scikit-learn (for E11)
- [Rust](https://www.rust-lang.org/) >= 1.70 (for E1-E3 and bonus experiments)

## Running Experiments

### SageMath experiments

```bash
cd E18_algebraic_channels
sage run_E18_algebraic_channels.sage
```

Each experiment saves JSON results to `data/` and prints a summary to stdout.

### Rust experiments

```bash
cd rust
cargo run -p group-structure     # E1
cargo run -p ai-guided           # E2
cargo run -p multi-base          # E3
cargo test                       # Run all tests
cargo bench -p benchmarks        # Comparative benchmarks
```

## Research Library

The `research/` directory contains 12 annotated bibliographies covering:

- Langlands correspondence and automorphic forms
- Quantum factoring (Shor, Regev)
- Classical factoring (NFS, ECM, CFRAC)
- AI/ML for mathematical discovery
- Compression and computation theory
- Quantum-inspired classical methods
- Physics connections (Ising, statistical mechanics)
- Post-quantum cryptography landscape
- Group theory for factoring
- Transformer architectures for number theory

## License

MIT License. See [LICENSE](LICENSE).
