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

/// Which bit-pattern primitive to use in compositions.
#[derive(Clone, Copy)]
pub enum BitPatternKind {
    Popcount,
    DigitSum2,
    DigitSum10,
    XorFold8,
    ByteSum,
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
    /// Multiplicative order ord_ℓ(N) mod divisor
    MultOrder { divisor: u64 },
    /// Second ℓ-adic digit: (N mod ℓ²) / ℓ
    SecondLadicDigit,
    /// Polynomial over F_ℓ evaluated at (N mod ℓ²) / ℓ (second digit)
    SecondDigitPower { exponent: u64 },
    /// Gauss sum residue: Σ (t/ℓ) * t^{N mod (ℓ-1)} mod ℓ
    GaussSum,
    /// c1 * N^a1 + c2 * N^a2 mod ℓ² reduced: ((c1*N^a1+c2*N^a2) mod ℓ² ) / ℓ
    LinearComboLift { a1: u64, a2: u64, c1: u64, c2: u64 },
    /// Composition: ord_ℓ(N) * N^a mod ℓ
    OrderTimesPower { exponent: u64 },
    /// N^a mod ℓ² (full lift, not just quotient)
    PowerResidueLift { exponent: u64 },
    /// popcount(N) mod ℓ, optionally raised to a power
    Popcount { exponent: u64 },
    /// Digit sum of N in given base, mod ℓ
    DigitSum { base: u64, exponent: u64 },
    /// XOR-fold of N at given bit width, mod ℓ
    XorFold { width: u32, exponent: u64 },
    /// Byte sum of N, mod ℓ
    ByteSum { exponent: u64 },
    /// Alternating bit sum of N, mod ℓ
    AlternatingBitSum { exponent: u64 },
    /// Composition: bit-pattern function * power residue mod ℓ
    BitTimesPower { bit_kind: BitPatternKind, power_exp: u64 },
    /// Auxiliary power residue: (N mod aux_mod)^exponent mod ℓ
    /// Tests functions depending on N through a different modulus (e.g., ℓ-1)
    AuxPowerResidue { aux_mod: u64, exponent: u64 },
    /// Mixed modulus combo: (N mod ℓ)^exp1 * (N mod aux_mod)^exp2 mod ℓ
    /// Mixes primary and auxiliary modular residues
    MixedModCombo { exp1: u64, exp2: u64, aux_mod: u64 },
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
        CandidateKind::MultOrder { divisor } => {
            match arith::mult_order(n % ell, ell) {
                Some(ord) => ord % *divisor,
                None => 0,
            }
        }
        CandidateKind::SecondLadicDigit => {
            let ell_sq = ell * ell;
            (n % ell_sq) / ell
        }
        CandidateKind::SecondDigitPower { exponent } => {
            let ell_sq = ell * ell;
            let digit2 = (n % ell_sq) / ell;
            arith::mod_pow(digit2, *exponent, ell)
        }
        CandidateKind::GaussSum => {
            let a = n % (ell - 1);
            arith::gauss_sum_algebraic(a, ell)
        }
        CandidateKind::LinearComboLift { a1, a2, c1, c2 } => {
            let ell_sq = ell * ell;
            let v1 = arith::mod_pow(n % ell_sq, *a1, ell_sq);
            let v2 = arith::mod_pow(n % ell_sq, *a2, ell_sq);
            let combo = (*c1 as u128 * v1 as u128 + *c2 as u128 * v2 as u128) % ell_sq as u128;
            ((combo as u64) / ell) % ell
        }
        CandidateKind::OrderTimesPower { exponent } => {
            let ord = match arith::mult_order(n % ell, ell) {
                Some(o) => o,
                None => return 0,
            };
            let pw = arith::mod_pow(n % ell, *exponent, ell);
            (ord as u128 * pw as u128 % ell as u128) as u64
        }
        CandidateKind::PowerResidueLift { exponent } => {
            let ell_sq = ell * ell;
            let val = arith::mod_pow(n % ell_sq, *exponent, ell_sq);
            (val / ell) % ell
        }
        CandidateKind::Popcount { exponent } => {
            let pc = arith::popcount(n);
            arith::mod_pow(pc % ell, *exponent, ell)
        }
        CandidateKind::DigitSum { base, exponent } => {
            let ds = arith::digit_sum(n, *base);
            arith::mod_pow(ds % ell, *exponent, ell)
        }
        CandidateKind::XorFold { width, exponent } => {
            let xf = arith::xor_fold(n, *width);
            arith::mod_pow(xf % ell, *exponent, ell)
        }
        CandidateKind::ByteSum { exponent } => {
            let bs = arith::byte_sum(n);
            arith::mod_pow(bs % ell, *exponent, ell)
        }
        CandidateKind::AlternatingBitSum { exponent } => {
            let abs = arith::alternating_bit_sum(n, ell);
            arith::mod_pow(abs, *exponent, ell)
        }
        CandidateKind::BitTimesPower { bit_kind, power_exp } => {
            let bit_val = match bit_kind {
                BitPatternKind::Popcount => arith::popcount(n) % ell,
                BitPatternKind::DigitSum2 => arith::digit_sum(n, 2) % ell,
                BitPatternKind::DigitSum10 => arith::digit_sum(n, 10) % ell,
                BitPatternKind::XorFold8 => arith::xor_fold(n, 8) % ell,
                BitPatternKind::ByteSum => arith::byte_sum(n) % ell,
            };
            let pw = arith::mod_pow(n % ell, *power_exp, ell);
            (bit_val as u128 * pw as u128 % ell as u128) as u64
        }
        CandidateKind::AuxPowerResidue { aux_mod, exponent } => {
            let n_aux = n % *aux_mod;
            arith::mod_pow(n_aux % ell, *exponent, ell)
        }
        CandidateKind::MixedModCombo { exp1, exp2, aux_mod } => {
            let v1 = arith::mod_pow(n % ell, *exp1, ell);
            let v2 = arith::mod_pow((n % *aux_mod) % ell, *exp2, ell);
            (v1 as u128 * v2 as u128 % ell as u128) as u64
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
    generate_mult_order(ell, &mut cands);
    generate_ladic_digits(ell, &mut cands);
    generate_gauss_sum(ell, &mut cands);
    generate_linear_combo_lifts(ell, &mut cands);
    generate_order_compositions(ell, &mut cands);
    generate_power_residue_lifts(ell, &mut cands);
    generate_bit_patterns(ell, &mut cands);
    generate_aux_candidates(ell, &mut cands);

    cands
}

fn generate_power_residues(ell: u64, out: &mut Vec<Candidate>) {
    // Enumerate all exponents mod (ℓ-1) since N^a ≡ N^{a mod ord} by Fermat.
    // Full coverage: test every exponent in 1..ℓ-1.
    for a in 1..ell {
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
    // Expand exponent range: up to 200 or ℓ-1, whichever is smaller.
    // Coefficients 1..10 for broader coverage.
    let max_a = 200u64.min(ell - 1);
    let max_c = 10u64;

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

fn generate_mult_order(ell: u64, out: &mut Vec<Candidate>) {
    // Raw multiplicative order
    out.push(Candidate {
        family: "mult_order",
        name: format!("ord_{}(N)", ell),
        kind: CandidateKind::MultOrder { divisor: ell - 1 },
    });

    // Order mod small divisors of ℓ-1
    let divisors = small_prime_factors(ell - 1);
    for &div in &divisors {
        out.push(Candidate {
            family: "mult_order",
            name: format!("ord_{}(N) mod {}", ell, div),
            kind: CandidateKind::MultOrder { divisor: div },
        });
        let big_div = (ell - 1) / div;
        out.push(Candidate {
            family: "mult_order",
            name: format!("ord_{}(N) mod {}", ell, big_div),
            kind: CandidateKind::MultOrder { divisor: big_div },
        });
    }
}

fn generate_ladic_digits(ell: u64, out: &mut Vec<Candidate>) {
    // Second ℓ-adic digit: (N mod ℓ²) / ℓ
    out.push(Candidate {
        family: "ladic_digit",
        name: format!("(N mod {}^2) / {}", ell, ell),
        kind: CandidateKind::SecondLadicDigit,
    });

    // Powers of the second digit
    let max_exp = 20u64.min(ell - 1);
    for a in 1..=max_exp {
        out.push(Candidate {
            family: "ladic_digit_power",
            name: format!("((N mod {}^2)/{}))^{} mod {}", ell, ell, a, ell),
            kind: CandidateKind::SecondDigitPower { exponent: a },
        });
    }
}

fn generate_gauss_sum(ell: u64, out: &mut Vec<Candidate>) {
    // Gauss sum is O(ℓ) to evaluate but ℓ ≤ 43867 — acceptable cost.
    out.push(Candidate {
        family: "gauss_sum",
        name: format!("gauss_sum(N mod {}, {})", ell - 1, ell),
        kind: CandidateKind::GaussSum,
    });
}

fn generate_linear_combo_lifts(ell: u64, out: &mut Vec<Candidate>) {
    // Linear combos evaluated mod ℓ², then extract the ℓ-adic "carry" digit.
    // Smaller search space than base linear combos since this is more exotic.
    let max_a = 30u64.min(ell - 1);
    let max_c = 5u64;

    for a1 in 0..max_a {
        for a2 in (a1 + 1)..=max_a {
            for c1 in 1..=max_c {
                for c2 in 1..=max_c {
                    out.push(Candidate {
                        family: "linear_combo_lift",
                        name: format!("lift({}*N^{}+{}*N^{}) mod {}", c1, a1, c2, a2, ell),
                        kind: CandidateKind::LinearComboLift { a1, a2, c1, c2 },
                    });
                }
            }
        }
    }
}

fn generate_order_compositions(ell: u64, out: &mut Vec<Candidate>) {
    // ord_ℓ(N) * N^a mod ℓ: mixes multiplicative order with power residue
    let max_exp = 50u64.min(ell - 1);
    for a in 0..=max_exp {
        out.push(Candidate {
            family: "order_x_power",
            name: format!("ord*N^{} mod {}", a, ell),
            kind: CandidateKind::OrderTimesPower { exponent: a },
        });
    }
}

fn generate_power_residue_lifts(ell: u64, out: &mut Vec<Candidate>) {
    // N^a mod ℓ² → extract the "carry" (second ℓ-adic digit of the power)
    // Tests whether ℓ²-level structure helps.
    let max_exp = 200u64.min(ell - 1);
    for a in 1..=max_exp {
        out.push(Candidate {
            family: "power_lift",
            name: format!("(N^{} mod {}^2)/{} mod {}", a, ell, ell, ell),
            kind: CandidateKind::PowerResidueLift { exponent: a },
        });
    }
}

fn generate_bit_patterns(ell: u64, out: &mut Vec<Candidate>) {
    let max_exp = 20u64.min(ell - 1);

    // Popcount and powers
    for a in 1..=max_exp {
        out.push(Candidate {
            family: "popcount",
            name: format!("popcount(N)^{} mod {}", a, ell),
            kind: CandidateKind::Popcount { exponent: a },
        });
    }

    // Digit sums in bases 2, 3, 5, 7, 10, 16
    let bases: &[u64] = &[2, 3, 5, 7, 10, 16];
    for &base in bases {
        for a in 1..=max_exp {
            out.push(Candidate {
                family: "digit_sum",
                name: format!("digitsum_{}(N)^{} mod {}", base, a, ell),
                kind: CandidateKind::DigitSum { base, exponent: a },
            });
        }
    }

    // XOR-fold at widths 4, 8, 16
    let widths: &[u32] = &[4, 8, 16];
    for &w in widths {
        for a in 1..=max_exp {
            out.push(Candidate {
                family: "xor_fold",
                name: format!("xorfold_{}_N^{} mod {}", w, a, ell),
                kind: CandidateKind::XorFold { width: w, exponent: a },
            });
        }
    }

    // Byte sum and powers
    for a in 1..=max_exp {
        out.push(Candidate {
            family: "byte_sum",
            name: format!("bytesum(N)^{} mod {}", a, ell),
            kind: CandidateKind::ByteSum { exponent: a },
        });
    }

    // Alternating bit sum and powers
    for a in 1..=max_exp {
        out.push(Candidate {
            family: "alt_bit_sum",
            name: format!("altbitsum(N)^{} mod {}", a, ell),
            kind: CandidateKind::AlternatingBitSum { exponent: a },
        });
    }

    // Compositions: bit-pattern × power residue
    let bit_kinds = [
        (BitPatternKind::Popcount, "popcount"),
        (BitPatternKind::DigitSum2, "ds2"),
        (BitPatternKind::DigitSum10, "ds10"),
        (BitPatternKind::XorFold8, "xor8"),
        (BitPatternKind::ByteSum, "bsum"),
    ];
    let max_power = 20u64.min(ell - 1);
    for &(kind, label) in &bit_kinds {
        for a in 1..=max_power {
            out.push(Candidate {
                family: "bit_x_power",
                name: format!("{}*N^{} mod {}", label, a, ell),
                kind: CandidateKind::BitTimesPower { bit_kind: kind, power_exp: a },
            });
        }
    }
}

fn generate_aux_candidates(ell: u64, out: &mut Vec<Candidate>) {
    // Auxiliary moduli: ℓ-1, ℓ+1 — these are NOT covered by the N mod ℓ collision check
    let aux_moduli: Vec<(u64, &str)> = vec![
        (ell - 1, "ℓ-1"),
        (ell + 1, "ℓ+1"),
    ];
    let max_exp = 50u64.min(ell - 1);

    for &(aux_mod, label) in &aux_moduli {
        // Auxiliary power residues: (N mod aux_mod)^a mod ℓ
        for a in 1..=max_exp {
            out.push(Candidate {
                family: "aux_power",
                name: format!("(N mod {})^{} mod {}", label, a, ell),
                kind: CandidateKind::AuxPowerResidue { aux_mod, exponent: a },
            });
        }

        // Mixed modulus combos: (N mod ℓ)^a * (N mod aux_mod)^b mod ℓ
        let max_mixed = 20u64.min(ell - 1);
        for a in 1..=max_mixed {
            for b in 1..=max_mixed {
                out.push(Candidate {
                    family: "mixed_mod",
                    name: format!("N^{}*(N mod {})^{} mod {}", a, label, b, ell),
                    kind: CandidateKind::MixedModCombo { exp1: a, exp2: b, aux_mod },
                });
            }
        }
    }
}

/// Helper: extract small prime factors of n.
fn small_prime_factors(mut n: u64) -> Vec<u64> {
    let mut factors = Vec::new();
    let mut d = 2u64;
    while d * d <= n {
        if n % d == 0 {
            factors.push(d);
            while n % d == 0 {
                n /= d;
            }
        }
        d += 1;
    }
    if n > 1 {
        factors.push(n);
    }
    factors
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
        // Power residues now uncapped: ℓ-1 = 43866 candidates
        let power_count = cands.iter().filter(|c| c.family == "power_residue").count();
        assert_eq!(power_count, 43866);
        // Should have new families
        let families: std::collections::HashSet<&str> = cands.iter().map(|c| c.family).collect();
        assert!(families.contains("mult_order"));
        assert!(families.contains("ladic_digit"));
        assert!(families.contains("gauss_sum"));
        assert!(families.contains("power_lift"));
    }
}
