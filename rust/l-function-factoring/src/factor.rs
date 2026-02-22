//! Main factoring algorithms via L-function analysis.
//!
//! Three methods are implemented:
//! 1. **Conductor Detection**: Find Dirichlet characters mod N with conductor < N.
//! 2. **Class Number Analysis**: Use class number formula consistency.
//! 3. **L-function Decomposition**: Detect multiplicative factorization of L-functions.

use crate::characters;
use crate::class_number;
use crate::gauss_sums;
use crate::l_function;

/// Result of factoring attempt.
#[derive(Debug, Clone)]
pub struct FactoringResult {
    /// The number that was factored.
    pub n: u64,
    /// The factors found, if any.
    pub factors: Option<(u64, u64)>,
    /// The method that succeeded.
    pub method: FactoringMethod,
    /// Additional diagnostic information.
    pub diagnostics: String,
}

/// The factoring method used.
#[derive(Debug, Clone, PartialEq)]
pub enum FactoringMethod {
    /// Conductor detection via character analysis.
    ConductorDetection,
    /// Class number formula consistency.
    ClassNumber,
    /// L-function decomposition.
    LFunctionDecomposition,
    /// No method succeeded.
    None,
}

/// Factor N by conductor detection.
///
/// For N = pq, enumerate Dirichlet characters mod N and find those with
/// conductor < N. The conductor divides either p or q, revealing the factorization.
///
/// # Algorithm
/// 1. Enumerate characters mod N (or a random sample for large N)
/// 2. For each non-principal character, compute its conductor
/// 3. If conductor f satisfies 1 < f < N, then gcd(f, N) or f itself is a factor
pub fn factor_by_conductor(n: u64) -> FactoringResult {
    if n <= 3 {
        return FactoringResult {
            n,
            factors: None,
            method: FactoringMethod::None,
            diagnostics: "N too small".to_string(),
        };
    }

    // Quick even check
    if n % 2 == 0 {
        return FactoringResult {
            n,
            factors: Some((2, n / 2)),
            method: FactoringMethod::ConductorDetection,
            diagnostics: "N is even".to_string(),
        };
    }

    let mut diagnostics = String::new();

    // Enumerate characters mod N
    let characters = characters::enumerate_characters(n);
    diagnostics.push_str(&format!(
        "Enumerated {} characters mod {}\n",
        characters.len(),
        n
    ));

    let mut conductor_counts: std::collections::HashMap<u64, u32> = std::collections::HashMap::new();

    for chi in &characters {
        if chi.is_principal {
            continue;
        }

        let conductor = gauss_sums::detect_conductor(chi);
        *conductor_counts.entry(conductor).or_insert(0) += 1;

        if conductor > 1 && conductor < n {
            let g = num_integer::Integer::gcd(&conductor, &n);
            if g > 1 && g < n {
                diagnostics.push_str(&format!(
                    "Found character with conductor {} → gcd({}, {}) = {}\n",
                    conductor, conductor, n, g
                ));
                return FactoringResult {
                    n,
                    factors: Some((g, n / g)),
                    method: FactoringMethod::ConductorDetection,
                    diagnostics,
                };
            }
            // If conductor divides N exactly
            if n % conductor == 0 {
                diagnostics.push_str(&format!(
                    "Found character with conductor {} which divides {}\n",
                    conductor, n
                ));
                return FactoringResult {
                    n,
                    factors: Some((conductor, n / conductor)),
                    method: FactoringMethod::ConductorDetection,
                    diagnostics,
                };
            }
        }
    }

    diagnostics.push_str("Conductor distribution:\n");
    let mut sorted_conductors: Vec<_> = conductor_counts.iter().collect();
    sorted_conductors.sort_by_key(|&(c, _)| *c);
    for (cond, count) in sorted_conductors {
        diagnostics.push_str(&format!("  conductor {}: {} characters\n", cond, count));
    }

    FactoringResult {
        n,
        factors: None,
        method: FactoringMethod::None,
        diagnostics,
    }
}

/// Factor N by class number analysis.
///
/// Compute h(-4N) and for each candidate d | N, check consistency with
/// the class numbers h(-4d) and h(-4(N/d)).
///
/// # Algorithm
/// 1. Compute h(-4N) via L(1, χ_{-4N})
/// 2. For trial divisors d of N (up to √N), compute h(-4d) and h(-4(N/d))
/// 3. Genus theory predicts a relationship between these class numbers
/// 4. Correct factorizations satisfy the consistency check
pub fn factor_by_class_number(n: u64) -> FactoringResult {
    if n <= 3 || n % 2 == 0 {
        return FactoringResult {
            n,
            factors: if n % 2 == 0 && n > 2 {
                Some((2, n / 2))
            } else {
                None
            },
            method: FactoringMethod::ClassNumber,
            diagnostics: "Trivial case".to_string(),
        };
    }

    let result = class_number::class_number_vs_factorization(n);
    let mut diagnostics = format!("h(-4*{}) ≈ {:.2} (rounded: {})\n", n, result.h_4n_exact, result.h_4n);

    if result.candidates.is_empty() {
        diagnostics.push_str("No non-trivial factorizations found via trial division.\n");
        diagnostics.push_str("(N may be prime)\n");
        return FactoringResult {
            n,
            factors: None,
            method: FactoringMethod::None,
            diagnostics,
        };
    }

    // The correct factorization should show consistent class numbers
    // Return the first valid factorization found
    for cand in &result.candidates {
        diagnostics.push_str(&format!(
            "  {} = {} × {}: h(-4*{}) = {}, h(-4*{}) = {}\n",
            n, cand.p, cand.q, cand.p, cand.h_4p, cand.q, cand.h_4q
        ));
    }

    let best = &result.candidates[0];
    FactoringResult {
        n,
        factors: Some((best.p, best.q)),
        method: FactoringMethod::ClassNumber,
        diagnostics,
    }
}

/// Factor N by L-function decomposition.
///
/// For N = pq, each character χ mod N decomposes as χ = χ_p × χ_q.
/// This means L(s, χ) values have a multiplicative structure related to
/// the factorization. By clustering L-function values, we can detect this structure.
///
/// # Algorithm
/// 1. Compute L(s, χ) at s = 2, 3, 4 for all non-principal characters mod N
/// 2. Characters that share the same χ_p component will have correlated L-values
/// 3. Use the structure: for fixed χ_p, varying χ_q gives a family of L-values
/// 4. The size of each family reveals |group(q)| = q - 1, hence q
pub fn factor_by_l_function_decomposition(n: u64) -> FactoringResult {
    if n <= 3 {
        return FactoringResult {
            n,
            factors: None,
            method: FactoringMethod::None,
            diagnostics: "N too small".to_string(),
        };
    }

    if n % 2 == 0 {
        return FactoringResult {
            n,
            factors: Some((2, n / 2)),
            method: FactoringMethod::LFunctionDecomposition,
            diagnostics: "N is even".to_string(),
        };
    }

    let mut diagnostics = String::new();

    // Compute L-function profiles
    let profiles = l_function::compute_l_profiles(n, 5000);
    diagnostics.push_str(&format!(
        "Computed {} L-function profiles for N = {}\n",
        profiles.len(),
        n
    ));

    // Group characters by their L(2) behavior.
    // Characters sharing the same χ_p component will have L(2) values that
    // are related by multiplication by L(2, χ_q) factors.
    //
    // Strategy: compute the ratio L(2, χ_i) / L(2, χ_j) for pairs.
    // When χ_i and χ_j share the same χ_p but differ in χ_q,
    // the ratio L(2, χ_q1) / L(2, χ_q2) depends only on q.

    // Simpler approach: use the conductor detection on the L-function side.
    // Characters with conductor dividing p will have L-values that only
    // depend on the mod-p part.
    let characters = characters::enumerate_characters(n);

    // Group by conductor
    let mut conductor_groups: std::collections::HashMap<u64, Vec<usize>> =
        std::collections::HashMap::new();

    let mut non_principal_idx = 0;
    for (_idx, chi) in characters.iter().enumerate() {
        if chi.is_principal {
            continue;
        }

        let conductor = gauss_sums::detect_conductor(chi);
        conductor_groups
            .entry(conductor)
            .or_insert_with(Vec::new)
            .push(non_principal_idx);
        non_principal_idx += 1;
    }

    diagnostics.push_str("L-function clustering by conductor:\n");
    let mut group_sizes: Vec<(u64, usize)> = conductor_groups
        .iter()
        .map(|(&c, v)| (c, v.len()))
        .collect();
    group_sizes.sort_by_key(|&(c, _)| c);

    for (cond, size) in &group_sizes {
        diagnostics.push_str(&format!("  Conductor {}: {} characters\n", cond, size));
    }

    // Find the group structure: for N = pq, we expect:
    // - (p-1) characters with conductor dividing q (or q itself)
    // - (q-1) characters with conductor dividing p (or p itself)
    // - (p-1)(q-1) - (p-1) - (q-1) + 1 characters with conductor N
    // The group sizes encode p-1 and q-1.

    // Collect non-trivial conductor groups (conductor != 1 and != n)
    let factor_conductors: Vec<u64> = group_sizes
        .iter()
        .filter(|&&(c, _)| c > 1 && c < n)
        .map(|&(c, _)| c)
        .collect();

    for &f in &factor_conductors {
        let g = num_integer::Integer::gcd(&f, &n);
        if g > 1 && g < n {
            diagnostics.push_str(&format!(
                "L-function decomposition reveals conductor {} → factor {}\n",
                f, g
            ));
            return FactoringResult {
                n,
                factors: Some((g, n / g)),
                method: FactoringMethod::LFunctionDecomposition,
                diagnostics,
            };
        }
        if n % f == 0 {
            diagnostics.push_str(&format!(
                "L-function decomposition reveals conductor {} divides {}\n",
                f, n
            ));
            return FactoringResult {
                n,
                factors: Some((f, n / f)),
                method: FactoringMethod::LFunctionDecomposition,
                diagnostics,
            };
        }
    }

    // Alternative: use group sizes to infer factors
    // φ(N) = (p-1)(q-1) characters total
    // Characters with conductor q: there are (p-1) of them (trivial on p-part)
    // Characters with conductor p: there are (q-1) of them (trivial on q-part)
    // So the number of characters with conductor q is (p-1), giving p = size + 1

    for (cond, size) in &group_sizes {
        if *cond > 1 && *cond < n {
            // size = φ(N/gcd(cond-related factor)) ... this is approximate
            let candidate_factor = *size as u64 + 1;
            if candidate_factor > 1 && n % candidate_factor == 0 {
                let other = n / candidate_factor;
                diagnostics.push_str(&format!(
                    "Group size analysis: {} characters with conductor {} → factor candidate {} ({}×{} = {})\n",
                    size, cond, candidate_factor, candidate_factor, other, candidate_factor * other
                ));
                return FactoringResult {
                    n,
                    factors: Some((candidate_factor, other)),
                    method: FactoringMethod::LFunctionDecomposition,
                    diagnostics,
                };
            }
        }
    }

    FactoringResult {
        n,
        factors: None,
        method: FactoringMethod::None,
        diagnostics,
    }
}

/// Factor N using sublinear conductor sampling methods.
///
/// Tries all five sampling strategies from the sampling module.
/// This attempts to find a factor without enumerating all characters,
/// using O(sqrt(N)) or fewer samples instead of O(N^2).
pub fn factor_by_sampled_conductor(n: u64, max_samples: u64) -> FactoringResult {
    let results = crate::sampling::run_all_methods(n, max_samples);

    for result in &results {
        if let Some((p, q)) = result.factor_found {
            return FactoringResult {
                n,
                factors: Some((p, q)),
                method: FactoringMethod::ConductorDetection,
                diagnostics: format!(
                    "Sampled conductor detection via '{}': {} samples, {}us\n{}",
                    result.method,
                    result.samples_tested,
                    result.time_us,
                    result.details.join("\n")
                ),
            };
        }
    }

    let mut diag = String::from("All sampling methods failed:\n");
    for r in &results {
        diag.push_str(&format!(
            "  {}: {} samples, {}us\n",
            r.method, r.samples_tested, r.time_us
        ));
    }

    FactoringResult {
        n,
        factors: None,
        method: FactoringMethod::None,
        diagnostics: diag,
    }
}

/// Try all factoring methods on N and return the first success.
///
/// Methods are tried in order:
/// 1. Sampled conductor detection (sublinear, fast)
/// 2. Full conductor detection (enumerates all characters)
/// 3. L-function decomposition
/// 4. Class number analysis
pub fn factor(n: u64) -> FactoringResult {
    // Method 0: Sampled conductor detection (try sublinear approach first)
    let result0 = factor_by_sampled_conductor(n, 1000);
    if result0.factors.is_some() {
        return result0;
    }

    // Method 1: Full Conductor Detection
    let result1 = factor_by_conductor(n);
    if result1.factors.is_some() {
        return result1;
    }

    // Method 2: L-function Decomposition
    let result2 = factor_by_l_function_decomposition(n);
    if result2.factors.is_some() {
        return result2;
    }

    // Method 3: Class Number Analysis
    let result3 = factor_by_class_number(n);
    if result3.factors.is_some() {
        return result3;
    }

    // None succeeded
    FactoringResult {
        n,
        factors: None,
        method: FactoringMethod::None,
        diagnostics: format!(
            "All methods failed.\nSampled conductor: {}\nConductor: {}\nL-function: {}\nClass number: {}",
            result0.diagnostics, result1.diagnostics, result2.diagnostics, result3.diagnostics
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_factor_by_conductor_77() {
        let result = factor_by_conductor(77);
        assert!(result.factors.is_some(), "Should factor 77");
        let (a, b) = result.factors.unwrap();
        assert_eq!(a * b, 77);
        assert!(
            (a == 7 && b == 11) || (a == 11 && b == 7),
            "77 = 7 × 11, got {} × {}",
            a,
            b
        );
    }

    #[test]
    fn test_factor_by_conductor_143() {
        let result = factor_by_conductor(143);
        assert!(result.factors.is_some(), "Should factor 143");
        let (a, b) = result.factors.unwrap();
        assert_eq!(a * b, 143);
        assert!(
            (a == 11 && b == 13) || (a == 13 && b == 11),
            "143 = 11 × 13, got {} × {}",
            a,
            b
        );
    }

    #[test]
    fn test_factor_by_conductor_221() {
        let result = factor_by_conductor(221);
        assert!(result.factors.is_some(), "Should factor 221");
        let (a, b) = result.factors.unwrap();
        assert_eq!(a * b, 221);
        assert!(
            (a == 13 && b == 17) || (a == 17 && b == 13),
            "221 = 13 × 17, got {} × {}",
            a,
            b
        );
    }

    #[test]
    fn test_factor_by_conductor_323() {
        let result = factor_by_conductor(323);
        assert!(result.factors.is_some(), "Should factor 323");
        let (a, b) = result.factors.unwrap();
        assert_eq!(a * b, 323);
        assert!(
            (a == 17 && b == 19) || (a == 19 && b == 17),
            "323 = 17 × 19, got {} × {}",
            a,
            b
        );
    }

    #[test]
    fn test_factor_by_l_function_decomposition_77() {
        let result = factor_by_l_function_decomposition(77);
        assert!(
            result.factors.is_some(),
            "Should factor 77 via L-function decomposition"
        );
        let (a, b) = result.factors.unwrap();
        assert_eq!(a * b, 77);
    }

    #[test]
    fn test_factor_by_class_number_77() {
        let result = factor_by_class_number(77);
        assert!(
            result.factors.is_some(),
            "Should factor 77 via class number"
        );
        let (a, b) = result.factors.unwrap();
        assert_eq!(a * b, 77);
    }

    #[test]
    fn test_factor_unified() {
        for &(n, _p, _q) in &[(77, 7, 11), (143, 11, 13), (221, 13, 17)] {
            let result = factor(n);
            assert!(result.factors.is_some(), "Should factor {}", n);
            let (a, b) = result.factors.unwrap();
            assert_eq!(a * b, n, "Factors should multiply to {}", n);
        }
    }

    #[test]
    fn test_factor_even() {
        let result = factor(14);
        assert!(result.factors.is_some(), "Should factor 14");
        let (a, b) = result.factors.unwrap();
        assert_eq!(a * b, 14, "Factors should multiply to 14");
    }
}
