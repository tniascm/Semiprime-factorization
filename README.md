# Semiprime Factoring Barrier: Systematic Exploration

**Author:** Andrii Potapov

A systematic computational and theoretical investigation of whether classical
poly(log N)-time integer factoring is possible, using tools from analytic number
theory, the Langlands program, and algebraic extensions.

## Result

**No poly(log N) classical factoring channel exists** across all tested approaches.
The barrier is computational (not information-theoretic) and is consistent with the
Quadratic Residuosity Assumption (QRA).

## Experiments

| Experiment | What it tests | Key finding |
|-----------|--------------|-------------|
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
utils/                      # Shared utilities (semiprime gen, encoding, spectral)
data/                       # JSON results and plots
BARRIER_THEOREM.md          # Full barrier formalization
QRP_RESEARCH.md             # QRP literature survey
```

## Requirements

- [SageMath](https://www.sagemath.org/) >= 9.0
- Python packages: numpy, scipy, matplotlib, scikit-learn (for E11/E14)

## Running Experiments

```bash
cd E18_algebraic_channels
sage run_E18_algebraic_channels.sage
```

Each experiment saves JSON results to `data/` and prints a summary to stdout.

## License

MIT License. See [LICENSE](LICENSE).
