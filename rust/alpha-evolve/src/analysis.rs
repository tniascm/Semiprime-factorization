//! Post-evolution analysis, reporting, and interpretation.
//!
//! Analyzes evolved programs to extract insights about:
//! - Which primitives appear most in high-fitness programs
//! - Recurring structural motifs (subtree patterns)
//! - Scaling curves: success rate vs bit-size
//! - Primitive co-occurrence: which primitives appear together
//! - Comparison to known baselines (Pollard rho, SQUFOF, ECM, Fermat)
//! - Novel compositions not seen in any seed or known algorithm

use std::collections::HashMap;
use std::time::Instant;

use num_bigint::BigUint;
use num_traits::Zero;
use serde::Serialize;

use crate::macros::MacroKind;
use crate::novelty::NoveltyArchive;
use crate::{PrimitiveOp, Program, ProgramNode};

// ---------------------------------------------------------------------------
// Report structures
// ---------------------------------------------------------------------------

/// Complete analysis report, serializable to JSON.
#[derive(Debug, Serialize)]
pub struct AnalysisReport {
    /// Primitive frequency counts in top programs.
    pub primitive_frequency: Vec<PrimitiveFrequency>,
    /// Macro block frequency counts in top programs.
    pub macro_frequency: Vec<MacroFrequency>,
    /// Structural motifs: recurring subtree patterns.
    pub structural_motifs: Vec<StructuralMotif>,
    /// Scaling curves for top programs.
    pub scaling_curves: Vec<ScalingCurve>,
    /// Primitive co-occurrence matrix (upper triangle, sparse).
    pub co_occurrence: Vec<CoOccurrenceEntry>,
    /// Baseline comparison results.
    pub baseline_comparison: Vec<BaselineComparison>,
    /// Novel compositions not matching any known algorithm pattern.
    pub novel_compositions: Vec<NovelComposition>,
    /// Novelty archive statistics.
    pub novelty_stats: NoveltyStats,
}

/// Frequency of a primitive operation in high-fitness programs.
#[derive(Debug, Serialize)]
pub struct PrimitiveFrequency {
    pub primitive: String,
    pub count: usize,
    pub fraction: f64,
}

/// Frequency of a macro block kind in high-fitness programs.
#[derive(Debug, Serialize)]
pub struct MacroFrequency {
    pub macro_kind: String,
    pub count: usize,
    pub fraction: f64,
}

/// A recurring structural motif (subtree pattern).
#[derive(Debug, Serialize)]
pub struct StructuralMotif {
    /// String representation of the subtree pattern.
    pub pattern: String,
    /// Number of times this pattern appears across top programs.
    pub count: usize,
    /// Average fitness of programs containing this pattern.
    pub avg_fitness: f64,
}

/// Scaling curve for a single program.
#[derive(Debug, Serialize)]
pub struct ScalingCurve {
    /// Program description.
    pub program: String,
    /// Program fitness score.
    pub fitness: f64,
    /// Success rate at each bit size: (bit_size, successes, attempts, avg_time_ms).
    pub bit_results: Vec<BitSizeResult>,
}

/// Result at a specific bit size.
#[derive(Debug, Serialize)]
pub struct BitSizeResult {
    pub bit_size: u32,
    pub successes: u32,
    pub attempts: u32,
    pub success_rate: f64,
    pub avg_time_ms: f64,
}

/// Co-occurrence entry: two primitives appearing together in successful programs.
#[derive(Debug, Serialize)]
pub struct CoOccurrenceEntry {
    pub primitive_a: String,
    pub primitive_b: String,
    pub count: usize,
}

/// Comparison of an evolved program against a known baseline algorithm.
#[derive(Debug, Serialize)]
pub struct BaselineComparison {
    pub bit_size: u32,
    pub baseline_name: String,
    pub baseline_success_rate: f64,
    pub baseline_avg_time_ms: f64,
    pub evolved_success_rate: f64,
    pub evolved_avg_time_ms: f64,
    pub evolved_is_better: bool,
}

/// A novel composition — a primitive combination not seen in seeds or known algorithms.
#[derive(Debug, Serialize)]
pub struct NovelComposition {
    pub program: String,
    pub fitness: f64,
    pub novel_primitives: Vec<String>,
    pub description: String,
}

/// Novelty archive summary statistics.
#[derive(Debug, Serialize)]
pub struct NoveltyStats {
    pub archive_size: usize,
    pub unique_behaviors: usize,
}

// ---------------------------------------------------------------------------
// Primitive extraction
// ---------------------------------------------------------------------------

/// Collect all primitive operations from a program tree.
fn collect_primitives(node: &ProgramNode) -> Vec<String> {
    let mut result = Vec::new();
    collect_primitives_recursive(node, &mut result);
    result
}

fn collect_primitives_recursive(node: &ProgramNode, out: &mut Vec<String>) {
    match node {
        ProgramNode::Leaf(op) => {
            out.push(primitive_name(op));
        }
        ProgramNode::Sequence(children) => {
            for child in children {
                collect_primitives_recursive(child, out);
            }
        }
        ProgramNode::IterateNode { body, .. } => {
            collect_primitives_recursive(body, out);
        }
        ProgramNode::GcdCheck { setup } => {
            collect_primitives_recursive(setup, out);
        }
        ProgramNode::ConditionalGt {
            if_true, if_false, ..
        } => {
            collect_primitives_recursive(if_true, out);
            collect_primitives_recursive(if_false, out);
        }
        ProgramNode::MemoryOp { store, .. } => {
            if *store {
                out.push("MemoryStore".to_string());
            } else {
                out.push("MemoryLoad".to_string());
            }
        }
        ProgramNode::MacroBlock { kind, .. } => {
            out.push(macro_kind_name(kind));
        }
        ProgramNode::Hybrid { first, second } => {
            collect_primitives_recursive(first, out);
            collect_primitives_recursive(second, out);
        }
    }
}

/// Get a canonical name for a primitive operation (ignoring parameters).
fn primitive_name(op: &PrimitiveOp) -> String {
    match op {
        PrimitiveOp::ModPow => "ModPow".to_string(),
        PrimitiveOp::Gcd => "Gcd".to_string(),
        PrimitiveOp::RandomElement => "RandomElement".to_string(),
        PrimitiveOp::Iterate { .. } => "Iterate".to_string(),
        PrimitiveOp::AccumulateGcd { .. } => "AccumulateGcd".to_string(),
        PrimitiveOp::SubtractGcd => "SubtractGcd".to_string(),
        PrimitiveOp::Square => "Square".to_string(),
        PrimitiveOp::AddConst { .. } => "AddConst".to_string(),
        PrimitiveOp::MultiplyMod => "MultiplyMod".to_string(),
        PrimitiveOp::FermatStep { .. } => "FermatStep".to_string(),
        PrimitiveOp::HartStep => "HartStep".to_string(),
        PrimitiveOp::WilliamsStep { .. } => "WilliamsStep".to_string(),
        PrimitiveOp::ISqrt => "ISqrt".to_string(),
        PrimitiveOp::IsPerfectSquare => "IsPerfectSquare".to_string(),
        PrimitiveOp::CfConvergent { .. } => "CfConvergent".to_string(),
        PrimitiveOp::SqufofStep => "SqufofStep".to_string(),
        PrimitiveOp::RhoFormStep => "RhoFormStep".to_string(),
        PrimitiveOp::EcmCurve { .. } => "EcmCurve".to_string(),
        PrimitiveOp::LllShortVector => "LllShortVector".to_string(),
        PrimitiveOp::SmoothTest { .. } => "SmoothTest".to_string(),
        PrimitiveOp::PilatteVector => "PilatteVector".to_string(),
        PrimitiveOp::QuadraticResidue => "QuadraticResidue".to_string(),
        PrimitiveOp::PollardPm1 { .. } => "PollardPm1".to_string(),
        PrimitiveOp::DixonAccumulate => "DixonAccumulate".to_string(),
        PrimitiveOp::DixonCombine => "DixonCombine".to_string(),
    }
}

/// Get a canonical name for a macro kind.
fn macro_kind_name(kind: &MacroKind) -> String {
    match kind {
        MacroKind::Squfof => "MacroSqufof".to_string(),
        MacroKind::Ecm => "MacroEcm".to_string(),
        MacroKind::PollardRho => "MacroPollardRho".to_string(),
        MacroKind::FermatScan => "MacroFermatScan".to_string(),
        MacroKind::LatticeSmooth => "MacroLatticeSmooth".to_string(),
        MacroKind::ClassWalk => "MacroClassWalk".to_string(),
    }
}

// ---------------------------------------------------------------------------
// Subtree motif extraction
// ---------------------------------------------------------------------------

/// Collect all subtree patterns (depth ≤ 3) from a program node.
fn collect_subtree_patterns(node: &ProgramNode, max_depth: u32) -> Vec<String> {
    let mut patterns = Vec::new();
    collect_patterns_recursive(node, max_depth, &mut patterns);
    patterns
}

fn collect_patterns_recursive(node: &ProgramNode, max_depth: u32, out: &mut Vec<String>) {
    if max_depth == 0 {
        return;
    }

    // Record this subtree's pattern (truncated to depth)
    let pattern = format_node_pattern(node, max_depth);
    if pattern.len() <= 200 {
        // Avoid pathologically long patterns
        out.push(pattern);
    }

    // Recurse into children
    match node {
        ProgramNode::Leaf(_) | ProgramNode::MemoryOp { .. } => {}
        ProgramNode::Sequence(children) => {
            for child in children {
                collect_patterns_recursive(child, max_depth - 1, out);
            }
        }
        ProgramNode::IterateNode { body, .. } => {
            collect_patterns_recursive(body, max_depth - 1, out);
        }
        ProgramNode::GcdCheck { setup } => {
            collect_patterns_recursive(setup, max_depth - 1, out);
        }
        ProgramNode::ConditionalGt {
            if_true, if_false, ..
        } => {
            collect_patterns_recursive(if_true, max_depth - 1, out);
            collect_patterns_recursive(if_false, max_depth - 1, out);
        }
        ProgramNode::MacroBlock { .. } => {}
        ProgramNode::Hybrid { first, second } => {
            collect_patterns_recursive(first, max_depth - 1, out);
            collect_patterns_recursive(second, max_depth - 1, out);
        }
    }
}

/// Format a program node as a structural pattern (ignoring exact parameter values).
fn format_node_pattern(node: &ProgramNode, depth: u32) -> String {
    if depth == 0 {
        return "...".to_string();
    }
    match node {
        ProgramNode::Leaf(op) => primitive_name(op),
        ProgramNode::Sequence(children) => {
            let child_strs: Vec<String> = children
                .iter()
                .map(|c| format_node_pattern(c, depth - 1))
                .collect();
            format!("Seq[{}]", child_strs.join(", "))
        }
        ProgramNode::IterateNode { body, .. } => {
            format!("Loop(_, {})", format_node_pattern(body, depth - 1))
        }
        ProgramNode::GcdCheck { setup } => {
            format!("GcdChk({})", format_node_pattern(setup, depth - 1))
        }
        ProgramNode::ConditionalGt {
            if_true, if_false, ..
        } => {
            format!(
                "If(_, {}, {})",
                format_node_pattern(if_true, depth - 1),
                format_node_pattern(if_false, depth - 1)
            )
        }
        ProgramNode::MemoryOp { store, .. } => {
            if *store {
                "Store(_)".to_string()
            } else {
                "Load(_)".to_string()
            }
        }
        ProgramNode::MacroBlock { kind, .. } => macro_kind_name(kind),
        ProgramNode::Hybrid { first, second } => {
            format!(
                "Hybrid({}, {})",
                format_node_pattern(first, depth - 1),
                format_node_pattern(second, depth - 1)
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Known algorithm patterns (for novel composition detection)
// ---------------------------------------------------------------------------

/// Known algorithm primitive sets (from seeds and classical algorithms).
fn known_algorithm_patterns() -> Vec<(&'static str, Vec<&'static str>)> {
    vec![
        (
            "Pollard Rho",
            vec![
                "RandomElement",
                "Square",
                "AddConst",
                "SubtractGcd",
                "Gcd",
            ],
        ),
        (
            "Fermat's Method",
            vec!["ISqrt", "FermatStep", "IsPerfectSquare"],
        ),
        ("Hart's Method", vec!["HartStep", "ISqrt", "IsPerfectSquare"]),
        (
            "Lehman's Method",
            vec!["FermatStep", "IsPerfectSquare", "Gcd"],
        ),
        ("Trial Division", vec!["AddConst", "AccumulateGcd"]),
        (
            "Dixon's Method",
            vec!["DixonAccumulate", "DixonCombine", "SmoothTest"],
        ),
        (
            "CF-Based",
            vec!["CfConvergent", "RhoFormStep", "SqufofStep"],
        ),
        ("ECM", vec!["EcmCurve"]),
        ("Pollard p-1", vec!["PollardPm1"]),
        ("Lattice-Based", vec!["LllShortVector", "PilatteVector"]),
        (
            "Williams p+1",
            vec!["WilliamsStep"],
        ),
    ]
}

/// Determine if a program's primitive set matches any known algorithm pattern.
fn matches_known_pattern(primitives: &[String]) -> Vec<String> {
    let mut matched = Vec::new();
    let prim_set: std::collections::HashSet<&str> =
        primitives.iter().map(|s| s.as_str()).collect();

    for (name, pattern_prims) in known_algorithm_patterns() {
        let pattern_count = pattern_prims
            .iter()
            .filter(|p| prim_set.contains(*p))
            .count();
        // Consider it a match if more than half the pattern's primitives are present
        if pattern_count > 0 && pattern_count * 2 >= pattern_prims.len() {
            matched.push(name.to_string());
        }
    }
    matched
}

// ---------------------------------------------------------------------------
// Core analysis functions
// ---------------------------------------------------------------------------

/// Analyze primitive frequency across top programs.
pub fn analyze_primitive_frequency(programs: &[(Program, f64)]) -> Vec<PrimitiveFrequency> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut total = 0usize;

    for (program, _fitness) in programs {
        let prims = collect_primitives(&program.root);
        for prim in &prims {
            *counts.entry(prim.clone()).or_insert(0) += 1;
            total += 1;
        }
    }

    let mut freq: Vec<PrimitiveFrequency> = counts
        .into_iter()
        .map(|(primitive, count)| PrimitiveFrequency {
            primitive,
            count,
            fraction: if total > 0 {
                count as f64 / total as f64
            } else {
                0.0
            },
        })
        .collect();

    freq.sort_by(|a, b| b.count.cmp(&a.count));
    freq
}

/// Analyze macro block frequency across top programs.
pub fn analyze_macro_frequency(programs: &[(Program, f64)]) -> Vec<MacroFrequency> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut total = 0usize;

    for (program, _fitness) in programs {
        collect_macros_recursive(&program.root, &mut counts, &mut total);
    }

    let mut freq: Vec<MacroFrequency> = counts
        .into_iter()
        .map(|(macro_kind, count)| MacroFrequency {
            macro_kind,
            count,
            fraction: if total > 0 {
                count as f64 / total as f64
            } else {
                0.0
            },
        })
        .collect();

    freq.sort_by(|a, b| b.count.cmp(&a.count));
    freq
}

fn collect_macros_recursive(
    node: &ProgramNode,
    counts: &mut HashMap<String, usize>,
    total: &mut usize,
) {
    match node {
        ProgramNode::MacroBlock { kind, .. } => {
            *counts.entry(macro_kind_name(kind)).or_insert(0) += 1;
            *total += 1;
        }
        ProgramNode::Sequence(children) => {
            for child in children {
                collect_macros_recursive(child, counts, total);
            }
        }
        ProgramNode::IterateNode { body, .. } => {
            collect_macros_recursive(body, counts, total);
        }
        ProgramNode::GcdCheck { setup } => {
            collect_macros_recursive(setup, counts, total);
        }
        ProgramNode::ConditionalGt {
            if_true, if_false, ..
        } => {
            collect_macros_recursive(if_true, counts, total);
            collect_macros_recursive(if_false, counts, total);
        }
        ProgramNode::Hybrid { first, second } => {
            collect_macros_recursive(first, counts, total);
            collect_macros_recursive(second, counts, total);
        }
        _ => {}
    }
}

/// Find recurring structural motifs across top programs.
pub fn analyze_structural_motifs(
    programs: &[(Program, f64)],
    min_count: usize,
) -> Vec<StructuralMotif> {
    let mut pattern_counts: HashMap<String, (usize, f64)> = HashMap::new();

    for (program, fitness) in programs {
        // Extract subtree patterns up to depth 3
        let patterns = collect_subtree_patterns(&program.root, 3);

        // Deduplicate within a single program
        let unique_patterns: std::collections::HashSet<String> = patterns.into_iter().collect();

        for pattern in unique_patterns {
            let entry = pattern_counts.entry(pattern).or_insert((0, 0.0));
            entry.0 += 1;
            entry.1 += fitness;
        }
    }

    let mut motifs: Vec<StructuralMotif> = pattern_counts
        .into_iter()
        .filter(|(_, (count, _))| *count >= min_count)
        .map(|(pattern, (count, fitness_sum))| StructuralMotif {
            pattern,
            count,
            avg_fitness: fitness_sum / count as f64,
        })
        .collect();

    motifs.sort_by(|a, b| b.count.cmp(&a.count));
    motifs.truncate(50); // Keep top 50 motifs
    motifs
}

/// Generate scaling curves for the top programs.
pub fn analyze_scaling_curves(
    programs: &[(Program, f64)],
    bit_sizes: &[u32],
    samples_per_size: usize,
) -> Vec<ScalingCurve> {
    let mut rng = rand::thread_rng();

    programs
        .iter()
        .take(10) // Top 10 programs
        .map(|(program, fitness)| {
            let mut bit_results = Vec::new();

            for &bits in bit_sizes {
                let mut successes = 0u32;
                let mut total_time = 0u128;

                for _ in 0..samples_per_size {
                    let target = factoring_core::generate_rsa_target(bits, &mut rng);
                    let start = Instant::now();
                    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        program.evaluate(&target.n)
                    }));
                    let elapsed = start.elapsed();
                    total_time += elapsed.as_millis();

                    if let Ok(Some(ref factor)) = result {
                        if !factor.is_zero()
                            && *factor > BigUint::from(1u32)
                            && *factor < target.n
                            && (&target.n % factor).is_zero()
                        {
                            successes += 1;
                        }
                    }
                }

                bit_results.push(BitSizeResult {
                    bit_size: bits,
                    successes,
                    attempts: samples_per_size as u32,
                    success_rate: successes as f64 / samples_per_size as f64,
                    avg_time_ms: total_time as f64 / samples_per_size as f64,
                });
            }

            ScalingCurve {
                program: format!("{}", program),
                fitness: *fitness,
                bit_results,
            }
        })
        .collect()
}

/// Build the primitive co-occurrence matrix.
pub fn analyze_co_occurrence(programs: &[(Program, f64)]) -> Vec<CoOccurrenceEntry> {
    let mut pair_counts: HashMap<(String, String), usize> = HashMap::new();

    for (program, _) in programs {
        let prims = collect_primitives(&program.root);
        let unique: Vec<String> = {
            let mut u: Vec<String> = prims.into_iter().collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();
            u.sort();
            u
        };

        // Count co-occurrences (upper triangle)
        for i in 0..unique.len() {
            for j in (i + 1)..unique.len() {
                let key = (unique[i].clone(), unique[j].clone());
                *pair_counts.entry(key).or_insert(0) += 1;
            }
        }
    }

    let mut entries: Vec<CoOccurrenceEntry> = pair_counts
        .into_iter()
        .map(|((a, b), count)| CoOccurrenceEntry {
            primitive_a: a,
            primitive_b: b,
            count,
        })
        .collect();

    entries.sort_by(|a, b| b.count.cmp(&a.count));
    entries.truncate(50); // Top 50 pairs
    entries
}

/// Compare the best evolved program against known baseline algorithms.
pub fn analyze_baseline_comparison(
    best_program: &Program,
    baselines: &[(&str, Program)],
    bit_sizes: &[u32],
    samples: usize,
) -> Vec<BaselineComparison> {
    let mut rng = rand::thread_rng();
    let mut results = Vec::new();

    for &bits in bit_sizes {
        // Generate test semiprimes (shared across all comparisons for fairness)
        let targets: Vec<factoring_core::RsaTarget> = (0..samples)
            .map(|_| factoring_core::generate_rsa_target(bits, &mut rng))
            .collect();

        // Test evolved program
        let (evolved_successes, evolved_time) = test_program_on_targets(best_program, &targets);
        let evolved_success_rate = evolved_successes as f64 / samples as f64;
        let evolved_avg_time = evolved_time as f64 / samples as f64;

        // Test each baseline
        for (name, baseline) in baselines {
            let (baseline_successes, baseline_time) =
                test_program_on_targets(baseline, &targets);
            let baseline_success_rate = baseline_successes as f64 / samples as f64;
            let baseline_avg_time = baseline_time as f64 / samples as f64;

            results.push(BaselineComparison {
                bit_size: bits,
                baseline_name: name.to_string(),
                baseline_success_rate,
                baseline_avg_time_ms: baseline_avg_time,
                evolved_success_rate,
                evolved_avg_time_ms: evolved_avg_time,
                evolved_is_better: evolved_success_rate > baseline_success_rate
                    || (evolved_success_rate == baseline_success_rate
                        && evolved_avg_time < baseline_avg_time),
            });
        }
    }

    results
}

/// Test a program on a set of pre-generated targets.
fn test_program_on_targets(
    program: &Program,
    targets: &[factoring_core::RsaTarget],
) -> (u32, u128) {
    let mut successes = 0u32;
    let mut total_time = 0u128;

    for target in targets {
        let start = Instant::now();
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            program.evaluate(&target.n)
        }));
        let elapsed = start.elapsed();
        total_time += elapsed.as_millis();

        if let Ok(Some(ref factor)) = result {
            if !factor.is_zero()
                && *factor > BigUint::from(1u32)
                && *factor < target.n
                && (&target.n % factor).is_zero()
            {
                successes += 1;
            }
        }
    }

    (successes, total_time)
}

/// Identify novel compositions — programs using primitive combinations
/// not seen in any seed or known algorithm.
pub fn analyze_novel_compositions(programs: &[(Program, f64)]) -> Vec<NovelComposition> {
    let mut novel = Vec::new();

    for (program, fitness) in programs {
        let prims = collect_primitives(&program.root);
        let matched = matches_known_pattern(&prims);

        // A composition is novel if it doesn't fully match any single known pattern
        // AND it uses primitives from at least 2 different algorithm families
        let prim_set: std::collections::HashSet<&str> =
            prims.iter().map(|s| s.as_str()).collect();

        let families_present = count_algorithm_families(&prim_set);

        if families_present >= 2 {
            let novel_prims: Vec<String> = prims
                .iter()
                .cloned()
                .collect::<std::collections::HashSet<_>>()
                .into_iter()
                .collect();

            let description = if matched.is_empty() {
                format!(
                    "Entirely novel combination spanning {} algorithm families",
                    families_present
                )
            } else {
                format!(
                    "Hybrid of {} with novel elements from {} families",
                    matched.join(" + "),
                    families_present
                )
            };

            novel.push(NovelComposition {
                program: format!("{}", program),
                fitness: *fitness,
                novel_primitives: novel_prims,
                description,
            });
        }
    }

    // Sort by fitness descending
    novel.sort_by(|a, b| {
        b.fitness
            .partial_cmp(&a.fitness)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    novel.truncate(20);
    novel
}

/// Count how many distinct algorithm families are represented by the primitives.
fn count_algorithm_families(prims: &std::collections::HashSet<&str>) -> usize {
    let families: Vec<(&str, &[&str])> = vec![
        ("Pollard Rho", &["SubtractGcd", "Square", "AddConst"]),
        ("Fermat", &["FermatStep", "IsPerfectSquare"]),
        ("Hart", &["HartStep"]),
        ("CF", &["CfConvergent", "SqufofStep", "RhoFormStep"]),
        ("ECM", &["EcmCurve"]),
        ("Lattice", &["LllShortVector", "PilatteVector"]),
        ("Dixon", &["DixonAccumulate", "DixonCombine"]),
        ("Pollard p-1", &["PollardPm1"]),
        ("Williams", &["WilliamsStep"]),
        ("Smooth", &["SmoothTest"]),
    ];

    families
        .iter()
        .filter(|(_, family_prims)| family_prims.iter().any(|p| prims.contains(p)))
        .count()
}

// ---------------------------------------------------------------------------
// Full report generation
// ---------------------------------------------------------------------------

/// Generate a complete analysis report from the top programs.
///
/// Takes the top N programs (sorted by fitness descending) and the novelty
/// archive. Runs all analysis functions and compiles a comprehensive report.
pub fn generate_report(
    top_programs: &[(Program, f64)],
    baselines: &[(&str, Program)],
    novelty_archive: &NoveltyArchive,
) -> AnalysisReport {
    let bit_sizes: Vec<u32> = vec![16, 20, 24, 28, 32, 36, 40, 48];
    let samples_per_size = 5;

    // 1. Primitive frequency
    let primitive_frequency = analyze_primitive_frequency(top_programs);

    // 2. Macro frequency
    let macro_frequency = analyze_macro_frequency(top_programs);

    // 3. Structural motifs (min 2 occurrences)
    let structural_motifs = analyze_structural_motifs(top_programs, 2);

    // 4. Scaling curves for top 10
    let scaling_curves = analyze_scaling_curves(top_programs, &bit_sizes, samples_per_size);

    // 5. Co-occurrence matrix
    let co_occurrence = analyze_co_occurrence(top_programs);

    // 6. Baseline comparison (best evolved vs baselines)
    let baseline_comparison = if let Some((best_prog, _)) = top_programs.first() {
        analyze_baseline_comparison(
            best_prog,
            baselines,
            &[16, 24, 32, 40, 48],
            samples_per_size,
        )
    } else {
        Vec::new()
    };

    // 7. Novel compositions
    let novel_compositions = analyze_novel_compositions(top_programs);

    // 8. Novelty stats
    let novelty_stats = NoveltyStats {
        archive_size: novelty_archive.size(),
        unique_behaviors: novelty_archive.unique_count(),
    };

    AnalysisReport {
        primitive_frequency,
        macro_frequency,
        structural_motifs,
        scaling_curves,
        co_occurrence,
        baseline_comparison,
        novel_compositions,
        novelty_stats,
    }
}

/// Print a human-readable summary of the analysis report.
pub fn print_report_summary(report: &AnalysisReport) {
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║          Alpha-Evolve v3 — Post-Evolution Analysis         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();

    // Primitive frequency
    println!("─── Primitive Frequency (top 15) ───");
    for (i, pf) in report.primitive_frequency.iter().take(15).enumerate() {
        println!(
            "  {:>2}. {:<20} {:>5}× ({:>5.1}%)",
            i + 1,
            pf.primitive,
            pf.count,
            pf.fraction * 100.0
        );
    }
    println!();

    // Macro frequency
    if !report.macro_frequency.is_empty() {
        println!("─── Macro Block Frequency ───");
        for mf in &report.macro_frequency {
            println!(
                "      {:<20} {:>5}× ({:>5.1}%)",
                mf.macro_kind,
                mf.count,
                mf.fraction * 100.0
            );
        }
        println!();
    }

    // Structural motifs
    println!("─── Top Structural Motifs (≥2 occurrences) ───");
    for (i, motif) in report.structural_motifs.iter().take(10).enumerate() {
        println!(
            "  {:>2}. {:>3}× (avg fit: {:>8.1}) — {}",
            i + 1,
            motif.count,
            motif.avg_fitness,
            motif.pattern
        );
    }
    println!();

    // Scaling curves
    println!("─── Scaling Curves (top programs) ───");
    println!(
        "  {:>25} | {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "Program", "16b", "20b", "24b", "28b", "32b", "36b", "40b", "48b"
    );
    println!("  {}+{}",
        "-".repeat(25),
        "-".repeat(57)
    );
    for curve in report.scaling_curves.iter().take(5) {
        let program_short = if curve.program.len() > 25 {
            format!("{}...", &curve.program[..22])
        } else {
            curve.program.clone()
        };
        let rates: Vec<String> = curve
            .bit_results
            .iter()
            .map(|r| format!("{:>5.0}%", r.success_rate * 100.0))
            .collect();
        println!("  {:>25} | {}", program_short, rates.join(" "));
    }
    println!();

    // Co-occurrence
    println!("─── Top Primitive Co-occurrences ───");
    for (i, co) in report.co_occurrence.iter().take(10).enumerate() {
        println!(
            "  {:>2}. {} + {} — {:>3} programs",
            i + 1,
            co.primitive_a,
            co.primitive_b,
            co.count
        );
    }
    println!();

    // Baseline comparison
    if !report.baseline_comparison.is_empty() {
        println!("─── Baseline Comparison ───");
        let mut current_bits = 0u32;
        for bc in &report.baseline_comparison {
            if bc.bit_size != current_bits {
                current_bits = bc.bit_size;
                println!("  {}−bit semiprimes:", bc.bit_size);
            }
            let status = if bc.evolved_is_better { "✓" } else { "✗" };
            println!(
                "    {} vs {:<15}: evolved {:.0}% / {:.1}ms vs baseline {:.0}% / {:.1}ms",
                status,
                bc.baseline_name,
                bc.evolved_success_rate * 100.0,
                bc.evolved_avg_time_ms,
                bc.baseline_success_rate * 100.0,
                bc.baseline_avg_time_ms,
            );
        }
        println!();
    }

    // Novel compositions
    if !report.novel_compositions.is_empty() {
        println!("─── Novel Compositions ───");
        for (i, nc) in report.novel_compositions.iter().take(5).enumerate() {
            println!("  {:>2}. fitness={:.1} — {}", i + 1, nc.fitness, nc.description);
            let prog_short = if nc.program.len() > 70 {
                format!("{}...", &nc.program[..67])
            } else {
                nc.program.clone()
            };
            println!("      {}", prog_short);
        }
        println!();
    }

    // Novelty stats
    println!("─── Novelty Archive ───");
    println!(
        "  Archive size: {} | Unique behaviors: {}",
        report.novelty_stats.archive_size, report.novelty_stats.unique_behaviors
    );
    println!();
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{seed_fermat_like, seed_pollard_rho, seed_trial_like};

    #[test]
    fn test_collect_primitives() {
        let program = seed_pollard_rho();
        let prims = collect_primitives(&program.root);
        assert!(!prims.is_empty(), "Should collect primitives from rho seed");
        assert!(
            prims.contains(&"RandomElement".to_string()),
            "Pollard rho uses RandomElement"
        );
    }

    #[test]
    fn test_primitive_frequency() {
        let programs = vec![
            (seed_pollard_rho(), 100.0),
            (seed_fermat_like(), 80.0),
            (seed_trial_like(), 60.0),
        ];
        let freq = analyze_primitive_frequency(&programs);
        assert!(!freq.is_empty(), "Should produce frequency data");
        // Frequencies should sum to 1.0
        let total_frac: f64 = freq.iter().map(|f| f.fraction).sum();
        assert!(
            (total_frac - 1.0).abs() < 0.01,
            "Fractions should sum to ~1.0, got {}",
            total_frac
        );
    }

    #[test]
    fn test_structural_motifs() {
        let programs = vec![
            (seed_pollard_rho(), 100.0),
            (seed_pollard_rho(), 90.0), // Same structure, should produce motifs
            (seed_fermat_like(), 80.0),
        ];
        let motifs = analyze_structural_motifs(&programs, 2);
        assert!(
            !motifs.is_empty(),
            "Two identical programs should produce motifs with count ≥ 2"
        );
    }

    #[test]
    fn test_co_occurrence() {
        let programs = vec![
            (seed_pollard_rho(), 100.0),
            (seed_fermat_like(), 80.0),
        ];
        let co = analyze_co_occurrence(&programs);
        assert!(!co.is_empty(), "Should find co-occurring primitives");
    }

    #[test]
    fn test_novel_compositions() {
        // A program that combines Pollard rho + CF + ECM should be novel
        let hybrid = Program {
            root: ProgramNode::Sequence(vec![
                ProgramNode::Leaf(PrimitiveOp::RandomElement),
                ProgramNode::Leaf(PrimitiveOp::Square),
                ProgramNode::Leaf(PrimitiveOp::SubtractGcd),
                ProgramNode::Leaf(PrimitiveOp::CfConvergent { k: 10 }),
                ProgramNode::Leaf(PrimitiveOp::EcmCurve { b1: 1000 }),
            ]),
        };

        let programs = vec![(hybrid, 50.0)];
        let novel = analyze_novel_compositions(&programs);
        assert!(
            !novel.is_empty(),
            "A program combining rho + CF + ECM should be flagged as novel"
        );
    }

    #[test]
    fn test_known_pattern_matching() {
        let rho_prims = vec![
            "RandomElement".to_string(),
            "Square".to_string(),
            "AddConst".to_string(),
            "SubtractGcd".to_string(),
        ];
        let matched = matches_known_pattern(&rho_prims);
        assert!(
            matched.contains(&"Pollard Rho".to_string()),
            "Should match Pollard Rho pattern, got {:?}",
            matched
        );
    }

    #[test]
    fn test_count_algorithm_families() {
        let prims: std::collections::HashSet<&str> =
            ["SubtractGcd", "CfConvergent", "EcmCurve"]
                .iter()
                .copied()
                .collect();
        let families = count_algorithm_families(&prims);
        assert_eq!(
            families, 3,
            "Should detect 3 families (Rho, CF, ECM), got {}",
            families
        );
    }

    #[test]
    fn test_format_node_pattern() {
        let node = ProgramNode::GcdCheck {
            setup: Box::new(ProgramNode::Leaf(PrimitiveOp::Square)),
        };
        let pattern = format_node_pattern(&node, 2);
        assert_eq!(pattern, "GcdChk(Square)");
    }

    #[test]
    fn test_scaling_curves_no_panic() {
        let programs = vec![(seed_pollard_rho(), 100.0)];
        let curves = analyze_scaling_curves(&programs, &[16], 2);
        assert_eq!(curves.len(), 1);
        assert_eq!(curves[0].bit_results.len(), 1);
        assert_eq!(curves[0].bit_results[0].attempts, 2);
    }
}
