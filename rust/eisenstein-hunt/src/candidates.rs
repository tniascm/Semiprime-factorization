/// Candidate function families for Eisenstein congruence hunt.
///
/// Each candidate is a poly(log N)-computable function h: u64 -> u64
/// that we test against σ_{k-1}(N) mod ℓ.

use crate::arith;

/// A candidate function identified by kind, with a human-readable name.
#[derive(Clone)]
pub struct Candidate {
    pub family: &'static str,
    pub name: String,
    pub kind: CandidateKind,
}

/// Enum-based dispatch avoids heap allocation per candidate.
#[derive(Clone)]
pub enum CandidateKind {
    /// N^exponent mod ℓ
    PowerResidue { exponent: u64 },
    /// kronecker(disc, N) mapped to F_ℓ: -1 -> ℓ-1, 0 -> 0, 1 -> 1
    KroneckerSymbol { disc: i64 },
    /// (c1 * N^a1 + c2 * N^a2) mod ℓ
    LinearCombo { a1: u64, a2: u64, c1: u64, c2: u64 },
    /// Lucas U_N(P, Q) mod ℓ
    LucasU { p: u64, q: u64 },
    /// Lucas V_N(P, Q) mod ℓ
    LucasV { p: u64, q: u64 },
    /// CF convergent numerator or denominator of N/ℓ, reduced mod ℓ
    CfConvergent { index: usize, numerator: bool },
    /// ind_g(N mod ℓ) mod divisor, where g = primitive_root(ℓ)
    DlogResidue { generator: u64, divisor: u64 },
    /// C(N mod ℓ, j) mod ℓ
    Binomial { j: u64 },
    /// Fermat quotient: (N^{ℓ-1} - 1)/ℓ mod ℓ
    FermatQuotient,
    /// Eisenstein quotient: (N^{(ℓ-1)/2} - jacobi(N,ℓ)) / ℓ mod ℓ
    EisensteinQuotient,
    /// (N^{(ℓ-1)/d} - 1) / ℓ mod ℓ for d | (ℓ-1)
    FermatQuotientDiv { divisor: u64 },
}

/// Evaluate a candidate on (n, ell).
pub fn eval(kind: &CandidateKind, n: u64, ell: u64) -> u64 {
    match kind {
        CandidateKind::PowerResidue { exponent } => {
            arith::mod_pow(n % ell, *exponent, ell)
        }
        CandidateKind::KroneckerSymbol { disc } => {
            let kr = arith::kronecker_symbol(*disc, n);
            // Map {-1, 0, 1} to F_ℓ
            if kr == -1 {
                ell - 1
            } else {
                kr as u64
            }
        }
        CandidateKind::LinearCombo { a1, a2, c1, c2 } => {
            let v1 = arith::mod_pow(n % ell, *a1, ell);
            let v2 = arith::mod_pow(n % ell, *a2, ell);
            ((*c1 as u128 * v1 as u128 + *c2 as u128 * v2 as u128) % ell as u128) as u64
        }
        CandidateKind::LucasU { p, q } => {
            arith::lucas_u(n, *p, *q, ell)
        }
        CandidateKind::LucasV { p, q } => {
            arith::lucas_v(n, *p, *q, ell)
        }
        CandidateKind::CfConvergent { index, numerator } => {
            let convs = arith::cf_convergents(n, ell, *index + 1);
            if let Some(&(h, k)) = convs.get(*index) {
                if *numerator { h % ell } else { k % ell }
            } else {
                0 // CF terminated early
            }
        }
        CandidateKind::DlogResidue { generator, divisor } => {
            let n_mod = n % ell;
            if n_mod == 0 {
                return 0;
            }
            match arith::discrete_log(n_mod, *generator, ell) {
                Some(x) => x % *divisor,
                None => 0,
            }
        }
        CandidateKind::Binomial { j } => {
            arith::binomial_mod(n, *j, ell)
        }
        CandidateKind::FermatQuotient => {
            arith::fermat_quotient(n, ell).unwrap_or(0)
        }
        CandidateKind::EisensteinQuotient => {
            if n % ell == 0 {
                return 0;
            }
            let ell_sq = ell * ell;
            let half = (ell - 1) / 2;
            let pow = arith::mod_pow(n % ell_sq, half, ell_sq);
            let jac = arith::jacobi_symbol(n as i64, ell);
            let jac_mod = if jac == -1 { ell_sq - 1 } else { jac as u64 };
            let diff = (pow + ell_sq - jac_mod) % ell_sq;
            (diff / ell) % ell
        }
        CandidateKind::FermatQuotientDiv { divisor } => {
            if n % ell == 0 {
                return 0;
            }
            let ell_sq = ell * ell;
            let exp = (ell - 1) / *divisor;
            let pow = arith::mod_pow(n % ell_sq, exp, ell_sq);
            let diff = (pow + ell_sq - 1) % ell_sq;
            (diff / ell) % ell
        }
    }
}

/// Generate all candidates for a given channel.
pub fn generate_all(ell: u64) -> Vec<Candidate> {
    let mut cands = Vec::new();

    generate_power_residues(ell, &mut cands);
    generate_kronecker(ell, &mut cands);
    generate_linear_combos(ell, &mut cands);
    generate_lucas(ell, &mut cands);
    generate_cf_convergents(&mut cands);
    generate_dlog(ell, &mut cands);
    generate_binomial(ell, &mut cands);
    generate_fermat_quotients(ell, &mut cands);

    cands
}

fn generate_power_residues(ell: u64, out: &mut Vec<Candidate>) {
    // For small ℓ, enumerate all. For large ℓ, cap at a reasonable limit.
    let max_exp = if ell <= 1000 { ell - 1 } else { 1000 };
    for a in 1..=max_exp {
        out.push(Candidate {
            family: "power_residue",
            name: format!("N^{} mod {}", a, ell),
            kind: CandidateKind::PowerResidue { exponent: a },
        });
    }
}

fn generate_kronecker(ell: u64, out: &mut Vec<Candidate>) {
    let discriminants: &[i64] = &[
        -1, -3, -4, 5, -7, 8, -8, -11, 12, 13, -15, -19, -20, 21, -23,
        -24, 28, -31, -35, -39, -40, -43, -47, -51, -52,
    ];
    for &d in discriminants {
        out.push(Candidate {
            family: "kronecker",
            name: format!("kronecker({}, N) in F_{}", d, ell),
            kind: CandidateKind::KroneckerSymbol { disc: d },
        });
    }
}

fn generate_linear_combos(ell: u64, out: &mut Vec<Candidate>) {
    let max_a = 50u64.min(ell - 1);
    let max_c = 5u64;

    for a1 in 0..max_a {
        for a2 in (a1 + 1)..=max_a {
            for c1 in 1..=max_c {
                for c2 in 1..=max_c {
                    out.push(Candidate {
                        family: "linear_combo",
                        name: format!("{}*N^{} + {}*N^{} mod {}", c1, a1, c2, a2, ell),
                        kind: CandidateKind::LinearCombo { a1, a2, c1, c2 },
                    });
                }
            }
        }
    }
}

fn generate_lucas(ell: u64, out: &mut Vec<Candidate>) {
    for p in 1..=20u64 {
        for q_offset in 0..=10u64 {
            // Q ranges from 1 to 11 (avoiding negative for simplicity in u64)
            // Also test Q = ell - 1, ..., ell - 5 (equivalent to -1, ..., -5 mod ell)
            let q_values: Vec<u64> = if q_offset <= 5 {
                vec![q_offset + 1]
            } else {
                vec![ell - (q_offset - 5)]
            };
            for q in q_values {
                out.push(Candidate {
                    family: "lucas_u",
                    name: format!("U_N({}, {}) mod {}", p, q, ell),
                    kind: CandidateKind::LucasU { p, q },
                });
                out.push(Candidate {
                    family: "lucas_v",
                    name: format!("V_N({}, {}) mod {}", p, q, ell),
                    kind: CandidateKind::LucasV { p, q },
                });
            }
        }
    }
}

fn generate_cf_convergents(out: &mut Vec<Candidate>) {
    for i in 0..5 {
        out.push(Candidate {
            family: "cf_convergent",
            name: format!("CF_h_{}", i),
            kind: CandidateKind::CfConvergent { index: i, numerator: true },
        });
        out.push(Candidate {
            family: "cf_convergent",
            name: format!("CF_k_{}", i),
            kind: CandidateKind::CfConvergent { index: i, numerator: false },
        });
    }
}

fn generate_dlog(ell: u64, out: &mut Vec<Candidate>) {
    let g = arith::primitive_root(ell);

    // Raw discrete log
    out.push(Candidate {
        family: "dlog",
        name: format!("ind_{}(N) mod {}", g, ell - 1),
        kind: CandidateKind::DlogResidue { generator: g, divisor: ell - 1 },
    });

    // Discrete log mod small divisors of ℓ-1
    let mut d = 2u64;
    let mut n = ell - 1;
    let mut divisors = Vec::new();
    while d * d <= n {
        if n % d == 0 {
            divisors.push(d);
            while n % d == 0 {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        divisors.push(n);
    }

    for &div in &divisors {
        out.push(Candidate {
            family: "dlog",
            name: format!("ind_{}(N) mod {}", g, div),
            kind: CandidateKind::DlogResidue { generator: g, divisor: div },
        });
    }

    // Also test ℓ-1 divided by each prime factor
    for &div in &divisors {
        let big_div = (ell - 1) / div;
        out.push(Candidate {
            family: "dlog",
            name: format!("ind_{}(N) mod {}", g, big_div),
            kind: CandidateKind::DlogResidue { generator: g, divisor: big_div },
        });
    }
}

fn generate_binomial(ell: u64, out: &mut Vec<Candidate>) {
    let max_j = 20u64.min(ell - 1);
    for j in 1..=max_j {
        out.push(Candidate {
            family: "binomial",
            name: format!("C(N mod {}, {}) mod {}", ell, j, ell),
            kind: CandidateKind::Binomial { j },
        });
    }
}

fn generate_fermat_quotients(ell: u64, out: &mut Vec<Candidate>) {
    // Standard Fermat quotient
    out.push(Candidate {
        family: "fermat_quotient",
        name: format!("(N^{}-1)/{} mod {}", ell - 1, ell, ell),
        kind: CandidateKind::FermatQuotient,
    });

    // Eisenstein quotient
    out.push(Candidate {
        family: "eisenstein_quotient",
        name: format!("(N^{}-J(N,{}))/{} mod {}", (ell - 1) / 2, ell, ell, ell),
        kind: CandidateKind::EisensteinQuotient,
    });

    // Fermat quotient with divisors of ℓ-1
    let mut d = 2u64;
    let mut n = ell - 1;
    let mut divisors = Vec::new();
    while d * d <= n {
        if n % d == 0 {
            divisors.push(d);
            while n % d == 0 {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        divisors.push(n);
    }

    for &div in &divisors {
        out.push(Candidate {
            family: "fermat_quotient_div",
            name: format!("(N^{}-1)/{} mod {}", (ell - 1) / div, ell, ell),
            kind: CandidateKind::FermatQuotientDiv { divisor: div },
        });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_power_residue_eval() {
        let kind = CandidateKind::PowerResidue { exponent: 3 };
        // 10^3 mod 691 = 1000 mod 691 = 309
        assert_eq!(eval(&kind, 10, 691), 309);
    }

    #[test]
    fn test_kronecker_eval() {
        let kind = CandidateKind::KroneckerSymbol { disc: -1 };
        // (-1/15) where 15 = 3*5: (-1/3) = -1, (-1/5) = 1, product = -1 -> ℓ-1
        assert_eq!(eval(&kind, 15, 691), 690);
    }

    #[test]
    fn test_linear_combo_eval() {
        let kind = CandidateKind::LinearCombo { a1: 1, a2: 2, c1: 1, c2: 1 };
        // (1*10 + 1*100) mod 691 = 110
        assert_eq!(eval(&kind, 10, 691), 110);
    }

    #[test]
    fn test_generate_all_count() {
        let cands = generate_all(691);
        // Should have many candidates
        assert!(cands.len() > 1000, "expected > 1000 candidates, got {}", cands.len());
        // Check families are present
        let families: std::collections::HashSet<&str> = cands.iter().map(|c| c.family).collect();
        assert!(families.contains("power_residue"));
        assert!(families.contains("kronecker"));
        assert!(families.contains("linear_combo"));
        assert!(families.contains("lucas_u"));
        assert!(families.contains("cf_convergent"));
        assert!(families.contains("binomial"));
        assert!(families.contains("fermat_quotient"));
    }

    #[test]
    fn test_generate_all_large_ell() {
        let cands = generate_all(43867);
        // Power residues capped at 1000, plus other families
        assert!(cands.len() > 1000);
        let power_count = cands.iter().filter(|c| c.family == "power_residue").count();
        assert_eq!(power_count, 1000);
    }
}
