//! Scaling protocol: classical baseline collection at multiple sizes.
//!
//! Provides infrastructure for the "slope vs intercept" experiment:
//! does batch smoothness give a size-dependent speedup (slope change)
//! or just a fixed offset (intercept shift)?
//!
//! # Schema versioning
//!
//! All output JSON files carry `schema_version`, `git_commit`, and
//! `crate_version` fields. Bump `SCHEMA_VERSION` whenever telemetry
//! field semantics change to prevent cross-version comparisons.
//!
//! # Telemetry semantics (pinned definitions)
//!
//! These definitions are normative. Future code comparing "classical vs batch"
//! MUST use the same quantities, or increment `SCHEMA_VERSION`.
//!
//! - **`relations_found`**: The LAST "Total number of relations" or
//!   "rels found" count reported by CADO-NFS during sieving. This is the
//!   raw relation count BEFORE duplicate removal and filtering. If CADO-NFS
//!   reports multiple relation counts, the last one wins (most complete).
//!
//! - **`unique_relations`**: The "unique relations" or "unique rels" count
//!   reported after duplicate removal but BEFORE singleton/clique filtering.
//!
//! - **`matrix_size`**: (rows, cols) of the matrix passed to linear algebra
//!   (post-filter, post-merge). Parsed from "matrix is R x C" lines.
//!
//! - **`sieve_throughput`**: `relations_found / sieve_wall_secs` where
//!   `sieve_wall_secs` is the wall-clock time for the "Lattice Sieving"
//!   phase only (not including filtering or overhead). Units: relations/sec.
//!
//! - **`overhead_secs`**: `trial_wall_clock - sum(all stage wall times)`,
//!   clipped to zero. Represents CADO-NFS orchestration time (server startup,
//!   task dispatch, I/O). May be slightly negative due to rounding; we clip
//!   rather than allowing negative to avoid confusion.
//!
//! - **`stage_sum_secs`**: `sum(all stage wall times)`. The sum of all
//!   parsed phase wall-clock times. `overhead = wall_clock - stage_sum`.
//!
//! - **`cpu_wall_ratio`**: For phases that report CPU time, the ratio
//!   `sum(cpu) / sum(real)` indicating parallelism utilization.

use std::collections::HashMap;
use std::time::Duration;

use serde::{Deserialize, Serialize};

/// Current schema version. Bump when telemetry field semantics change.
pub const SCHEMA_VERSION: &str = "scaling-v1";

/// Crate version from Cargo.toml, embedded at compile time.
pub const CRATE_VERSION: &str = env!("CARGO_PKG_VERSION");

// ---------------------------------------------------------------------------
// Size specification
// ---------------------------------------------------------------------------

/// Specification for one size class in the scaling protocol.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeSpec {
    /// Number of decimal digits (e.g., 30 for c30).
    pub digits: u32,
    /// Approximate bit size (e.g., 100 for c30).
    pub bits: u32,
    /// Number of composites to generate and factor.
    pub num_composites: usize,
    /// Human-readable expected time per trial (informational only).
    pub expected_time_description: String,
    /// Hard timeout per trial.
    pub timeout: Duration,
}

/// Default size specifications for the scaling protocol.
///
/// These cover the range from trivial (c30, ~1s) to substantial (c100, ~hours),
/// providing enough dynamic range to detect slope changes in the scaling curve.
pub fn default_size_specs() -> Vec<SizeSpec> {
    vec![
        SizeSpec {
            digits: 30,
            bits: 100,
            num_composites: 20,
            expected_time_description: "~1s".to_string(),
            timeout: Duration::from_secs(60),
        },
        SizeSpec {
            digits: 40,
            bits: 133,
            num_composites: 15,
            expected_time_description: "~5s".to_string(),
            timeout: Duration::from_secs(120),
        },
        SizeSpec {
            digits: 50,
            bits: 166,
            num_composites: 10,
            expected_time_description: "~30s".to_string(),
            timeout: Duration::from_secs(300),
        },
        SizeSpec {
            digits: 60,
            bits: 199,
            num_composites: 8,
            expected_time_description: "~2min".to_string(),
            timeout: Duration::from_secs(600),
        },
        SizeSpec {
            digits: 80,
            bits: 266,
            num_composites: 5,
            expected_time_description: "~20min".to_string(),
            timeout: Duration::from_secs(3600),
        },
        SizeSpec {
            digits: 100,
            bits: 332,
            num_composites: 3,
            expected_time_description: "~2hr".to_string(),
            timeout: Duration::from_secs(14400),
        },
    ]
}

// ---------------------------------------------------------------------------
// Composite persistence
// ---------------------------------------------------------------------------

/// A stored composite with known factorization for verification.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredComposite {
    /// Sequential ID within the size class (0-indexed).
    pub id: usize,
    /// The composite N = p × q as a decimal string.
    pub n: String,
    /// First prime factor as a decimal string.
    pub p: String,
    /// Second prime factor as a decimal string.
    pub q: String,
    /// Bit length of N.
    pub n_bits: u32,
    /// Decimal digit count of N.
    pub n_digits: u32,
}

/// JSON file storing composites for one size class.
///
/// Composites are reusable across methods (classical, batch, etc.) to ensure
/// fair comparison on identical inputs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompositesFile {
    /// Schema version for forwards-compatibility checking.
    pub schema_version: String,
    /// Size class (decimal digits).
    pub size_digits: u32,
    /// Approximate bit size used for generation.
    pub size_bits: u32,
    /// The stored composites.
    pub composites: Vec<StoredComposite>,
    /// ISO 8601 timestamp of generation.
    pub generated_at: String,
    /// Git commit hash at generation time.
    pub git_commit: String,
}

// ---------------------------------------------------------------------------
// Per-trial telemetry
// ---------------------------------------------------------------------------

/// Telemetry for a single NFS stage within a trial.
///
/// Wall and CPU times come from CADO-NFS log parsing. See module-level
/// doc for precise semantic definitions of each field.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageTelemetry {
    /// Wall-clock seconds for this stage.
    pub wall_secs: f64,
    /// CPU seconds for this stage (None if CADO-NFS only reported wall time).
    /// Only available for stages using "Total cpu/real time" format.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_secs: Option<f64>,
    /// Relations found during this stage (only meaningful for sieve).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relations_found: Option<u64>,
    /// Sieve throughput: relations_found / wall_secs (only for sieve stage).
    /// See module doc for precise definition.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub relations_per_sec: Option<f64>,
    /// Matrix dimensions (rows, cols) — only for linalg stage.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub matrix_dims: Option<(u64, u64)>,
}

/// Debug information from CADO-NFS log parsing.
///
/// Stores raw parsed lines to help diagnose parser drift without
/// rerunning long factorization jobs.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ParseDebug {
    /// Raw timing lines that were successfully parsed.
    pub parsed_timing_lines: Vec<String>,
    /// Raw relation-count source lines.
    pub parsed_relation_lines: Vec<String>,
    /// Raw matrix-size source lines.
    pub parsed_matrix_lines: Vec<String>,
    /// Process exit status (0 = success, None = timeout/killed).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub exit_status: Option<i32>,
    /// Whether the process was killed due to timeout.
    pub timed_out: bool,
}

/// Result of a single factorization trial within the scaling protocol.
///
/// All time fields are in seconds for JSON readability.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrialResult {
    /// Which composite was factored (index into CompositesFile).
    pub composite_id: usize,
    /// Whether factorization completed successfully.
    pub success: bool,
    /// Whether factors were verified against known p, q.
    pub verified: bool,
    /// Total wall-clock seconds for the entire CADO-NFS run.
    pub wall_clock_secs: f64,
    /// Per-stage telemetry, keyed by CADO-NFS phase name.
    pub stages: HashMap<String, StageTelemetry>,
    /// Orchestration overhead: wall_clock - sum(stage wall times), clipped ≥ 0.
    /// See module doc for precise definition.
    pub overhead_secs: f64,
    /// Sum of all stage wall-clock times.
    pub stage_sum_secs: f64,
    /// Ratio of total CPU time to total wall-clock time across all stages
    /// that report CPU time. Indicates parallelism utilization.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cpu_wall_ratio: Option<f64>,
    /// Debug info: raw parsed lines from CADO-NFS output.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub parse_debug: Option<ParseDebug>,
}

// ---------------------------------------------------------------------------
// Per-size summary statistics
// ---------------------------------------------------------------------------

/// Sieve throughput statistics across trials.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputStats {
    /// Mean relations per second across successful trials.
    pub mean_relations_per_sec: f64,
    /// Standard deviation of relations per second.
    pub std_relations_per_sec: f64,
    /// Median relations per second.
    pub median_relations_per_sec: f64,
}

/// Summary statistics for one size class.
///
/// Computed from successful trials only.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SizeSummary {
    /// Number of trials attempted.
    pub n_trials: usize,
    /// Number of successful trials.
    pub n_success: usize,
    /// Success rate (n_success / n_trials).
    pub success_rate: f64,
    /// Percentile statistics for wall-clock times of successful trials.
    pub wall_clock_percentiles: crate::validation::Percentiles,
    /// Mean fraction of total wall-clock time per stage.
    /// Keys are stage names, values are fractions in [0, 1].
    pub stage_fractions: HashMap<String, f64>,
    /// Sieve throughput statistics.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sieve_throughput: Option<ThroughputStats>,
    /// Mean overhead fraction (overhead / wall_clock).
    pub mean_overhead_fraction: f64,
    /// Mean CPU/wall ratio across trials (parallelism utilization).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mean_cpu_wall_ratio: Option<f64>,
}

// ---------------------------------------------------------------------------
// Hardware and environment
// ---------------------------------------------------------------------------

/// Hardware and software environment for reproducibility.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareInfo {
    /// CPU model string (e.g., "Apple M4 Pro").
    pub cpu_model: String,
    /// Number of logical cores.
    pub cores: usize,
    /// Total RAM in bytes.
    pub ram_bytes: u64,
    /// Operating system description.
    pub os: String,
    /// Path to CADO-NFS installation.
    pub cado_nfs_dir: String,
}

// ---------------------------------------------------------------------------
// Top-level result types
// ---------------------------------------------------------------------------

/// Complete result for one size class.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingResult {
    /// Schema version for forwards-compatibility.
    pub schema_version: String,
    /// Git commit hash at execution time.
    pub git_commit: String,
    /// Crate version (from Cargo.toml).
    pub crate_version: String,
    /// Size class specification.
    pub size: SizeSpec,
    /// Method name (e.g., "classical-cado-nfs").
    pub method: String,
    /// CADO-NFS parameters used (from default_for_bits).
    pub params_description: String,
    /// Path to the composites file used.
    pub composites_file: String,
    /// Individual trial results.
    pub trials: Vec<TrialResult>,
    /// Aggregated summary statistics.
    pub summary: SizeSummary,
    /// Hardware and environment info.
    pub hardware: HardwareInfo,
    /// ISO 8601 timestamp of execution start.
    pub started_at: String,
    /// ISO 8601 timestamp of execution end.
    pub finished_at: String,
}

/// Combined result across all size classes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScalingProtocolResult {
    /// Schema version for forwards-compatibility.
    pub schema_version: String,
    /// Git commit hash at execution time.
    pub git_commit: String,
    /// Crate version (from Cargo.toml).
    pub crate_version: String,
    /// Method name (e.g., "classical-cado-nfs").
    pub method: String,
    /// Per-size results, ordered by increasing digits.
    pub sizes: Vec<ScalingResult>,
    /// Hardware and environment info.
    pub hardware: HardwareInfo,
    /// ISO 8601 timestamp of protocol start.
    pub started_at: String,
    /// ISO 8601 timestamp of protocol end.
    pub finished_at: String,
}

// ---------------------------------------------------------------------------
// Utility: ISO 8601 timestamp without chrono dependency
// ---------------------------------------------------------------------------

/// Return the current time as an ISO 8601 string (UTC).
///
/// Uses `date` command on Unix. Falls back to epoch on failure.
pub fn iso_now() -> String {
    #[cfg(unix)]
    {
        let output = std::process::Command::new("date")
            .args(["-u", "+%Y-%m-%dT%H:%M:%SZ"])
            .output();
        if let Ok(out) = output {
            if out.status.success() {
                return String::from_utf8_lossy(&out.stdout).trim().to_string();
            }
        }
    }
    // Fallback: use SystemTime
    let d = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    format!("epoch:{}", d.as_secs())
}

/// Return the current git commit short hash, or "unknown".
pub fn git_commit_hash() -> String {
    let output = std::process::Command::new("git")
        .args(["rev-parse", "--short", "HEAD"])
        .output();
    if let Ok(out) = output {
        if out.status.success() {
            return String::from_utf8_lossy(&out.stdout).trim().to_string();
        }
    }
    "unknown".to_string()
}

// ---------------------------------------------------------------------------
// Summary computation
// ---------------------------------------------------------------------------

/// Compute summary statistics from a set of trial results.
///
/// Only successful trials contribute to timing percentiles and throughput.
/// All trials count toward n_trials and success_rate.
pub fn compute_size_summary(trials: &[TrialResult]) -> SizeSummary {
    let n_trials = trials.len();
    let successful: Vec<&TrialResult> = trials.iter().filter(|t| t.success).collect();
    let n_success = successful.len();
    let success_rate = if n_trials > 0 {
        n_success as f64 / n_trials as f64
    } else {
        0.0
    };

    // Wall-clock percentiles
    let wall_times: Vec<f64> = successful.iter().map(|t| t.wall_clock_secs).collect();
    let wall_clock_percentiles = crate::validation::Percentiles::from_times(&wall_times);

    // Stage fractions: mean of (stage_wall / trial_wall) across successful trials
    let mut stage_fraction_sums: HashMap<String, f64> = HashMap::new();
    let mut stage_counts: HashMap<String, usize> = HashMap::new();
    for trial in &successful {
        if trial.wall_clock_secs > 0.0 {
            for (stage, telemetry) in &trial.stages {
                let frac = telemetry.wall_secs / trial.wall_clock_secs;
                *stage_fraction_sums.entry(stage.clone()).or_insert(0.0) += frac;
                *stage_counts.entry(stage.clone()).or_insert(0) += 1;
            }
        }
    }
    let stage_fractions: HashMap<String, f64> = stage_fraction_sums
        .into_iter()
        .map(|(stage, sum)| {
            let count = stage_counts[&stage] as f64;
            (stage, sum / count)
        })
        .collect();

    // Sieve throughput
    let throughputs: Vec<f64> = successful
        .iter()
        .filter_map(|t| {
            t.stages
                .iter()
                .find(|(k, _)| {
                    let kl = k.to_lowercase();
                    kl.contains("sieve") || kl.contains("lattice sieving")
                })
                .and_then(|(_, s)| s.relations_per_sec)
        })
        .collect();

    let sieve_throughput = if throughputs.is_empty() {
        None
    } else {
        let mean = throughputs.iter().sum::<f64>() / throughputs.len() as f64;
        let variance = if throughputs.len() > 1 {
            throughputs.iter().map(|x| (x - mean).powi(2)).sum::<f64>()
                / (throughputs.len() - 1) as f64
        } else {
            0.0
        };
        let std_dev = variance.sqrt();
        let mut sorted = throughputs.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let median = if sorted.len() % 2 == 0 {
            (sorted[sorted.len() / 2 - 1] + sorted[sorted.len() / 2]) / 2.0
        } else {
            sorted[sorted.len() / 2]
        };
        Some(ThroughputStats {
            mean_relations_per_sec: mean,
            std_relations_per_sec: std_dev,
            median_relations_per_sec: median,
        })
    };

    // Mean overhead fraction
    let mean_overhead_fraction = if successful.is_empty() {
        0.0
    } else {
        let sum: f64 = successful
            .iter()
            .map(|t| {
                if t.wall_clock_secs > 0.0 {
                    t.overhead_secs / t.wall_clock_secs
                } else {
                    0.0
                }
            })
            .sum();
        sum / successful.len() as f64
    };

    // Mean CPU/wall ratio
    let cpu_wall_ratios: Vec<f64> = successful
        .iter()
        .filter_map(|t| t.cpu_wall_ratio)
        .collect();
    let mean_cpu_wall_ratio = if cpu_wall_ratios.is_empty() {
        None
    } else {
        Some(cpu_wall_ratios.iter().sum::<f64>() / cpu_wall_ratios.len() as f64)
    };

    SizeSummary {
        n_trials,
        n_success,
        success_rate,
        wall_clock_percentiles,
        stage_fractions,
        sieve_throughput,
        mean_overhead_fraction,
        mean_cpu_wall_ratio,
    }
}

// ---------------------------------------------------------------------------
// Hardware detection
// ---------------------------------------------------------------------------

/// Detect hardware info for the current machine.
pub fn detect_hardware(cado_nfs_dir: &str) -> HardwareInfo {
    let cpu_model = detect_cpu_model();
    let cores = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);
    let ram_bytes = detect_ram_bytes();
    let os = detect_os();

    HardwareInfo {
        cpu_model,
        cores,
        ram_bytes,
        os,
        cado_nfs_dir: cado_nfs_dir.to_string(),
    }
}

fn detect_cpu_model() -> String {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sysctl")
            .args(["-n", "machdep.cpu.brand_string"])
            .output();
        if let Ok(out) = output {
            if out.status.success() {
                return String::from_utf8_lossy(&out.stdout).trim().to_string();
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/cpuinfo") {
            for line in contents.lines() {
                if line.starts_with("model name") {
                    if let Some(val) = line.split(':').nth(1) {
                        return val.trim().to_string();
                    }
                }
            }
        }
    }
    "unknown".to_string()
}

fn detect_ram_bytes() -> u64 {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sysctl")
            .args(["-n", "hw.memsize"])
            .output();
        if let Ok(out) = output {
            if out.status.success() {
                let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
                if let Ok(bytes) = s.parse::<u64>() {
                    return bytes;
                }
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/proc/meminfo") {
            for line in contents.lines() {
                if line.starts_with("MemTotal:") {
                    // Format: "MemTotal:       16384000 kB"
                    let parts: Vec<&str> = line.split_whitespace().collect();
                    if parts.len() >= 2 {
                        if let Ok(kb) = parts[1].parse::<u64>() {
                            return kb * 1024;
                        }
                    }
                }
            }
        }
    }
    0
}

fn detect_os() -> String {
    #[cfg(target_os = "macos")]
    {
        let output = std::process::Command::new("sw_vers")
            .args(["-productVersion"])
            .output();
        if let Ok(out) = output {
            if out.status.success() {
                let version = String::from_utf8_lossy(&out.stdout).trim().to_string();
                return format!("macOS {}", version);
            }
        }
    }
    #[cfg(target_os = "linux")]
    {
        if let Ok(contents) = std::fs::read_to_string("/etc/os-release") {
            for line in contents.lines() {
                if line.starts_with("PRETTY_NAME=") {
                    return line
                        .trim_start_matches("PRETTY_NAME=")
                        .trim_matches('"')
                        .to_string();
                }
            }
        }
    }
    std::env::consts::OS.to_string()
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_size_specs() {
        let specs = default_size_specs();
        assert_eq!(specs.len(), 6);

        // Verify all sizes present in expected order
        let expected_digits = [30, 40, 50, 60, 80, 100];
        let expected_bits = [100, 133, 166, 199, 266, 332];
        for (i, spec) in specs.iter().enumerate() {
            assert_eq!(spec.digits, expected_digits[i]);
            assert_eq!(spec.bits, expected_bits[i]);
            assert!(spec.num_composites > 0);
            assert!(spec.timeout.as_secs() > 0);
        }

        // Timeouts should be monotonically increasing
        for i in 1..specs.len() {
            assert!(specs[i].timeout >= specs[i - 1].timeout);
        }

        // Composites count should be monotonically decreasing
        for i in 1..specs.len() {
            assert!(specs[i].num_composites <= specs[i - 1].num_composites);
        }
    }

    #[test]
    fn test_stored_composite_roundtrip() {
        let composite = StoredComposite {
            id: 0,
            n: "851137823547".to_string(),
            p: "924637".to_string(),
            q: "920291".to_string(),
            n_bits: 40,
            n_digits: 12,
        };

        let json = serde_json::to_string_pretty(&composite).unwrap();
        let parsed: StoredComposite = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.id, 0);
        assert_eq!(parsed.n, "851137823547");
        assert_eq!(parsed.p, "924637");
        assert_eq!(parsed.q, "920291");
        assert_eq!(parsed.n_bits, 40);
        assert_eq!(parsed.n_digits, 12);
    }

    #[test]
    fn test_composites_file_roundtrip() {
        let file = CompositesFile {
            schema_version: SCHEMA_VERSION.to_string(),
            size_digits: 30,
            size_bits: 100,
            composites: vec![StoredComposite {
                id: 0,
                n: "123456789".to_string(),
                p: "12347".to_string(),
                q: "10001".to_string(),
                n_bits: 27,
                n_digits: 9,
            }],
            generated_at: "2026-02-25T00:00:00Z".to_string(),
            git_commit: "abc1234".to_string(),
        };

        let json = serde_json::to_string_pretty(&file).unwrap();
        let parsed: CompositesFile = serde_json::from_str(&json).unwrap();

        assert_eq!(parsed.schema_version, SCHEMA_VERSION);
        assert_eq!(parsed.size_digits, 30);
        assert_eq!(parsed.composites.len(), 1);
        assert_eq!(parsed.composites[0].n, "123456789");
    }

    #[test]
    fn test_trial_result_serialization() {
        let mut stages = HashMap::new();
        stages.insert(
            "Lattice Sieving".to_string(),
            StageTelemetry {
                wall_secs: 3.02,
                cpu_secs: None,
                relations_found: Some(52609),
                relations_per_sec: Some(17420.2),
                matrix_dims: None,
            },
        );
        stages.insert(
            "Linear Algebra".to_string(),
            StageTelemetry {
                wall_secs: 1.22,
                cpu_secs: Some(0.47),
                relations_found: None,
                relations_per_sec: None,
                matrix_dims: Some((3000, 3200)),
            },
        );

        let trial = TrialResult {
            composite_id: 0,
            success: true,
            verified: true,
            wall_clock_secs: 5.5,
            stages,
            overhead_secs: 1.26,
            stage_sum_secs: 4.24,
            cpu_wall_ratio: Some(0.85),
            parse_debug: Some(ParseDebug {
                parsed_timing_lines: vec![
                    "Info:Lattice Sieving: Total time: 3.02s".to_string(),
                ],
                parsed_relation_lines: vec![
                    "found 52609 relations".to_string(),
                ],
                parsed_matrix_lines: vec![
                    "matrix is 3000 x 3200".to_string(),
                ],
                exit_status: Some(0),
                timed_out: false,
            }),
        };

        let json = serde_json::to_string_pretty(&trial).unwrap();
        let parsed: TrialResult = serde_json::from_str(&json).unwrap();

        assert!(parsed.success);
        assert!(parsed.verified);
        assert!((parsed.wall_clock_secs - 5.5).abs() < 0.01);
        assert_eq!(parsed.stages.len(), 2);
        assert!(parsed.stages.contains_key("Lattice Sieving"));
        assert!(parsed.stages.contains_key("Linear Algebra"));
        assert!(parsed.parse_debug.is_some());
        let debug = parsed.parse_debug.unwrap();
        assert_eq!(debug.parsed_timing_lines.len(), 1);
        assert!(!debug.timed_out);
    }

    #[test]
    fn test_compute_summary_empty() {
        let summary = compute_size_summary(&[]);
        assert_eq!(summary.n_trials, 0);
        assert_eq!(summary.n_success, 0);
        assert!((summary.success_rate - 0.0).abs() < f64::EPSILON);
        assert_eq!(summary.wall_clock_percentiles.count, 0);
        assert!(summary.sieve_throughput.is_none());
    }

    #[test]
    fn test_compute_summary_with_data() {
        let make_trial = |id: usize, wall: f64, sieve_wall: f64, rels: u64| -> TrialResult {
            let mut stages = HashMap::new();
            stages.insert(
                "Lattice Sieving".to_string(),
                StageTelemetry {
                    wall_secs: sieve_wall,
                    cpu_secs: Some(sieve_wall * 2.0),
                    relations_found: Some(rels),
                    relations_per_sec: Some(rels as f64 / sieve_wall),
                    matrix_dims: None,
                },
            );
            stages.insert(
                "Linear Algebra".to_string(),
                StageTelemetry {
                    wall_secs: wall - sieve_wall - 0.5,
                    cpu_secs: Some((wall - sieve_wall - 0.5) * 1.5),
                    relations_found: None,
                    relations_per_sec: None,
                    matrix_dims: Some((1000, 1100)),
                },
            );
            let stage_sum = sieve_wall + (wall - sieve_wall - 0.5);
            TrialResult {
                composite_id: id,
                success: true,
                verified: true,
                wall_clock_secs: wall,
                stages,
                overhead_secs: (wall - stage_sum).max(0.0),
                stage_sum_secs: stage_sum,
                cpu_wall_ratio: Some(1.5),
                parse_debug: None,
            }
        };

        let trials = vec![
            make_trial(0, 10.0, 6.0, 50000),
            make_trial(1, 12.0, 7.0, 60000),
            make_trial(2, 11.0, 6.5, 55000),
        ];

        let summary = compute_size_summary(&trials);
        assert_eq!(summary.n_trials, 3);
        assert_eq!(summary.n_success, 3);
        assert!((summary.success_rate - 1.0).abs() < f64::EPSILON);
        assert_eq!(summary.wall_clock_percentiles.count, 3);
        // Mean wall time should be (10 + 12 + 11) / 3 = 11.0
        assert!((summary.wall_clock_percentiles.mean - 11.0).abs() < 0.01);
        // Sieve throughput should exist
        assert!(summary.sieve_throughput.is_some());
        let tp = summary.sieve_throughput.unwrap();
        // Mean throughput: (50000/6 + 60000/7 + 55000/6.5) / 3
        assert!(tp.mean_relations_per_sec > 8000.0);
        assert!(tp.mean_relations_per_sec < 9000.0);
        // Stage fractions should exist
        assert!(summary.stage_fractions.contains_key("Lattice Sieving"));
        assert!(summary.stage_fractions.contains_key("Linear Algebra"));
        // Sieve should be the dominant stage
        let sieve_frac = summary.stage_fractions["Lattice Sieving"];
        assert!(sieve_frac > 0.4, "Sieve should be >40% of wall time");
        // CPU/wall ratio should be populated
        assert!(summary.mean_cpu_wall_ratio.is_some());
        assert!((summary.mean_cpu_wall_ratio.unwrap() - 1.5).abs() < 0.01);
    }

    #[test]
    fn test_compute_summary_with_failures() {
        let success_trial = TrialResult {
            composite_id: 0,
            success: true,
            verified: true,
            wall_clock_secs: 10.0,
            stages: HashMap::new(),
            overhead_secs: 10.0,
            stage_sum_secs: 0.0,
            cpu_wall_ratio: None,
            parse_debug: None,
        };
        let failure_trial = TrialResult {
            composite_id: 1,
            success: false,
            verified: false,
            wall_clock_secs: 60.0,
            stages: HashMap::new(),
            overhead_secs: 60.0,
            stage_sum_secs: 0.0,
            cpu_wall_ratio: None,
            parse_debug: None,
        };

        let summary = compute_size_summary(&[success_trial, failure_trial]);
        assert_eq!(summary.n_trials, 2);
        assert_eq!(summary.n_success, 1);
        assert!((summary.success_rate - 0.5).abs() < f64::EPSILON);
        // Only the successful trial should contribute to percentiles
        assert_eq!(summary.wall_clock_percentiles.count, 1);
        assert!((summary.wall_clock_percentiles.mean - 10.0).abs() < 0.01);
    }

    #[test]
    fn test_schema_version_constant() {
        assert_eq!(SCHEMA_VERSION, "scaling-v1");
        assert!(!CRATE_VERSION.is_empty());
    }

    #[test]
    fn test_hardware_detection() {
        let hw = detect_hardware("/tmp/fake-cado");
        assert!(hw.cores > 0);
        assert!(!hw.os.is_empty());
        assert_eq!(hw.cado_nfs_dir, "/tmp/fake-cado");
        // CPU model should be detected on macOS/Linux
        #[cfg(any(target_os = "macos", target_os = "linux"))]
        assert_ne!(hw.cpu_model, "unknown");
    }

    #[test]
    fn test_parse_debug_serialization() {
        let debug = ParseDebug {
            parsed_timing_lines: vec!["Info:Sieve: Total time: 3.0s".to_string()],
            parsed_relation_lines: vec![],
            parsed_matrix_lines: vec![],
            exit_status: None,
            timed_out: true,
        };
        let json = serde_json::to_string(&debug).unwrap();
        let parsed: ParseDebug = serde_json::from_str(&json).unwrap();
        assert!(parsed.timed_out);
        assert!(parsed.exit_status.is_none());
        assert_eq!(parsed.parsed_timing_lines.len(), 1);
    }

    #[test]
    fn test_iso_now_format() {
        let ts = iso_now();
        // Should be non-empty and look like a timestamp
        assert!(!ts.is_empty());
        // On Unix, should contain 'T' for ISO 8601
        #[cfg(unix)]
        assert!(ts.contains('T') || ts.starts_with("epoch:"));
    }

    #[test]
    fn test_git_commit_hash_format() {
        let hash = git_commit_hash();
        // Should be non-empty
        assert!(!hash.is_empty());
        // If in a git repo, should be a short hex hash
        if hash != "unknown" {
            assert!(hash.len() >= 7);
            assert!(hash.chars().all(|c| c.is_ascii_hexdigit()));
        }
    }
}
