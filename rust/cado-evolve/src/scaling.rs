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
                    kl.contains("siev")
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
// Composite persistence
// ---------------------------------------------------------------------------

/// Ensure composites exist for a size class, loading from disk or generating.
///
/// Composites are stored as `scaling_composites_c{digits}.json` in `output_dir`.
/// If the file exists and has at least `spec.num_composites` entries, it is
/// loaded and returned. Otherwise, new composites are generated and saved.
///
/// # Arguments
/// - `spec`: Size specification (digits, bits, num_composites)
/// - `output_dir`: Directory for JSON files
/// - `rng`: Random number generator for composite generation
pub fn ensure_composites(
    spec: &SizeSpec,
    output_dir: &std::path::Path,
    rng: &mut impl rand::Rng,
) -> Result<Vec<StoredComposite>, String> {
    let filename = format!("scaling_composites_c{}.json", spec.digits);
    let filepath = output_dir.join(&filename);

    // Try loading existing composites
    if filepath.exists() {
        match std::fs::read_to_string(&filepath) {
            Ok(contents) => {
                match serde_json::from_str::<CompositesFile>(&contents) {
                    Ok(file) => {
                        if file.composites.len() >= spec.num_composites {
                            log::info!(
                                "Loaded {} composites for c{} from {}",
                                file.composites.len(),
                                spec.digits,
                                filepath.display()
                            );
                            return Ok(file.composites[..spec.num_composites].to_vec());
                        }
                        log::warn!(
                            "File {} has {} composites but need {}, regenerating",
                            filepath.display(),
                            file.composites.len(),
                            spec.num_composites
                        );
                    }
                    Err(e) => {
                        log::warn!("Failed to parse {}: {}, regenerating", filepath.display(), e);
                    }
                }
            }
            Err(e) => {
                log::warn!("Failed to read {}: {}, regenerating", filepath.display(), e);
            }
        }
    }

    // Generate new composites
    log::info!(
        "Generating {} composites for c{} ({}-bit)",
        spec.num_composites,
        spec.digits,
        spec.bits
    );

    let mut composites = Vec::with_capacity(spec.num_composites);
    for id in 0..spec.num_composites {
        let target = factoring_core::generate_rsa_target(spec.bits, rng);
        let n_str = target.n.to_string();
        let n_digits = n_str.len() as u32;
        composites.push(StoredComposite {
            id,
            n: n_str,
            p: target.p.to_string(),
            q: target.q.to_string(),
            n_bits: target.n.bits() as u32,
            n_digits,
        });
    }

    // Save to disk
    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("Failed to create output dir {}: {}", output_dir.display(), e))?;

    let file = CompositesFile {
        schema_version: SCHEMA_VERSION.to_string(),
        size_digits: spec.digits,
        size_bits: spec.bits,
        composites: composites.clone(),
        generated_at: iso_now(),
        git_commit: git_commit_hash(),
    };

    let json = serde_json::to_string_pretty(&file)
        .map_err(|e| format!("Failed to serialize composites: {}", e))?;
    std::fs::write(&filepath, &json)
        .map_err(|e| format!("Failed to write {}: {}", filepath.display(), e))?;

    log::info!(
        "Saved {} composites for c{} to {}",
        composites.len(),
        spec.digits,
        filepath.display()
    );

    Ok(composites)
}

// ---------------------------------------------------------------------------
// Trial execution
// ---------------------------------------------------------------------------

/// Build a TrialResult from a CadoResult and a StoredComposite.
///
/// Extracts per-stage telemetry, computes overhead and CPU/wall ratio,
/// and verifies factors against known p × q.
pub fn build_trial_result(
    composite: &StoredComposite,
    cado_result: &crate::cado::CadoResult,
    parse_debug: Option<ParseDebug>,
) -> TrialResult {
    let wall_clock_secs = cado_result.total_time.as_secs_f64();

    // Build per-stage telemetry
    let mut stages = HashMap::new();
    for (phase, wall_dur) in &cado_result.phase_times {
        // Skip the "Complete Factorization" total — it's a sum, not a stage
        let phase_lower = phase.to_lowercase();
        if phase_lower.contains("complete factorization") {
            continue;
        }

        let wall_secs = wall_dur.as_secs_f64();
        let cpu_secs = cado_result
            .phase_cpu_times
            .get(phase)
            .map(|d| d.as_secs_f64());

        // Sieve-specific: attach relations_found and throughput
        let is_sieve = phase_lower.contains("siev");
        let relations_found = if is_sieve {
            cado_result.relations_found
        } else {
            None
        };
        let relations_per_sec = if is_sieve && wall_secs > 0.0 {
            cado_result
                .relations_found
                .map(|r| r as f64 / wall_secs)
        } else {
            None
        };

        // Linalg-specific: attach matrix dims
        let is_linalg = phase_lower.contains("linalg")
            || phase_lower.contains("linear algebra")
            || phase_lower.contains("bwc");
        let matrix_dims = if is_linalg {
            cado_result.matrix_size
        } else {
            None
        };

        stages.insert(
            phase.clone(),
            StageTelemetry {
                wall_secs,
                cpu_secs,
                relations_found,
                relations_per_sec,
                matrix_dims,
            },
        );
    }

    // Fix aggregate-time phases. CADO-NFS reports "Total time: Xs" for
    // client-dispatched phases (sieve, polyselect) where the value is
    // aggregate CPU time across all workers, NOT wall time. We detect these
    // by checking if stage_sum exceeds wall_clock (impossible if all values
    // are true wall times) and adjust aggregate-time phases accordingly.
    let raw_stage_sum: f64 = stages.values().map(|s| s.wall_secs).sum();
    if raw_stage_sum > wall_clock_secs * 1.05 {
        // Some phases reported aggregate CPU time. Identify them:
        // aggregate-time phases have cpu_secs == Some(x) where x == wall_secs
        // (set by Pattern 3 in parse_phase_timing).
        let mut known_wall_sum = 0.0;
        let mut aggregate_phases: Vec<String> = Vec::new();
        let mut aggregate_cpu_sum = 0.0;

        for (name, stage) in &stages {
            if let Some(cpu) = stage.cpu_secs {
                if (cpu - stage.wall_secs).abs() < 0.001 && stage.wall_secs > 1.0 {
                    // This is an aggregate-time phase (cpu == wall means
                    // it came from "Total time:" format)
                    aggregate_phases.push(name.clone());
                    aggregate_cpu_sum += cpu;
                    continue;
                }
            }
            known_wall_sum += stage.wall_secs;
        }

        if !aggregate_phases.is_empty() {
            let remaining_wall = (wall_clock_secs - known_wall_sum).max(0.0);
            // Distribute remaining wall time proportionally by CPU time
            for phase_name in &aggregate_phases {
                if let Some(stage) = stages.get_mut(phase_name) {
                    let fraction = if aggregate_cpu_sum > 0.0 {
                        stage.cpu_secs.unwrap_or(0.0) / aggregate_cpu_sum
                    } else {
                        1.0 / aggregate_phases.len() as f64
                    };
                    stage.wall_secs = remaining_wall * fraction;
                    // Recompute sieve throughput with corrected wall time
                    if let Some(rels) = stage.relations_found {
                        if stage.wall_secs > 0.0 {
                            stage.relations_per_sec = Some(rels as f64 / stage.wall_secs);
                        }
                    }
                }
            }
        }
    }

    // Compute stage_sum and overhead
    let stage_sum_secs: f64 = stages.values().map(|s| s.wall_secs).sum();
    let overhead_secs = (wall_clock_secs - stage_sum_secs).max(0.0);

    // Compute CPU/wall ratio across stages with CPU time
    let (total_cpu, total_real_with_cpu) = stages.values().fold((0.0, 0.0), |(cpu, real), s| {
        if let Some(c) = s.cpu_secs {
            (cpu + c, real + s.wall_secs)
        } else {
            (cpu, real)
        }
    });
    let cpu_wall_ratio = if total_real_with_cpu > 0.0 {
        Some(total_cpu / total_real_with_cpu)
    } else {
        None
    };

    // Verify factors
    let verified = if cado_result.success && cado_result.factors.len() >= 2 {
        verify_factors(composite, &cado_result.factors)
    } else {
        false
    };

    TrialResult {
        composite_id: composite.id,
        success: cado_result.success,
        verified,
        wall_clock_secs,
        stages,
        overhead_secs,
        stage_sum_secs,
        cpu_wall_ratio,
        parse_debug,
    }
}

/// Verify that the found factors match the known p × q.
///
/// Requires at least 2 non-trivial factors whose product equals N.
fn verify_factors(composite: &StoredComposite, factors: &[String]) -> bool {
    use num_bigint::BigUint;
    use num_traits::One;

    if factors.len() < 2 {
        return false;
    }

    let known_n: BigUint = match composite.n.parse() {
        Ok(n) => n,
        Err(_) => return false,
    };

    let parsed: Vec<BigUint> = factors
        .iter()
        .filter_map(|f| f.parse::<BigUint>().ok())
        .collect();

    if parsed.len() < 2 {
        return false;
    }

    // All factors must be > 1 (non-trivial)
    if parsed.iter().any(|f| *f <= BigUint::one()) {
        return false;
    }

    let product: BigUint = parsed.into_iter().fold(BigUint::one(), |acc, f| acc * f);
    product == known_n
}

/// Extract ParseDebug from a CADO-NFS log and CadoResult.
///
/// Collects the raw lines that were used for timing, relation, and matrix
/// parsing so that parser drift can be diagnosed without rerunning.
pub fn extract_parse_debug(log: &str, timed_out: bool) -> ParseDebug {
    let mut debug = ParseDebug {
        parsed_timing_lines: Vec::new(),
        parsed_relation_lines: Vec::new(),
        parsed_matrix_lines: Vec::new(),
        exit_status: None,
        timed_out,
    };

    for line in log.lines() {
        let trimmed = line.trim();
        let clean = crate::cado::strip_ansi(trimmed);

        // Timing lines
        if clean.contains("Total cpu/real time for ")
            || clean.contains("Total cpu/elapsed time for ")
            || clean.contains("Total time:")
        {
            debug.parsed_timing_lines.push(clean.clone());
        }

        // Relation lines
        if trimmed.contains("rels found")
            || trimmed.contains("relations found")
            || trimmed.contains("number of relations")
            || (trimmed.contains("found") && trimmed.contains("relation"))
        {
            debug.parsed_relation_lines.push(clean.clone());
        }

        // Matrix lines
        if trimmed.contains("matrix is") || trimmed.contains("Matrix has") {
            debug.parsed_matrix_lines.push(clean.clone());
        }
    }

    debug
}

/// Run a single factorization trial with full telemetry.
///
/// # Arguments
/// - `install`: Validated CADO-NFS installation
/// - `composite`: The composite to factor (with known p, q for verification)
/// - `params`: CADO-NFS parameter configuration
/// - `timeout`: Maximum wall-clock time
///
/// # Returns
/// A `TrialResult` with success/failure, per-stage telemetry, and parse debug info.
pub fn run_trial(
    install: &crate::cado::CadoInstallation,
    composite: &StoredComposite,
    params: &crate::params::CadoParams,
    timeout: Duration,
) -> TrialResult {
    let n: num_bigint::BigUint = match composite.n.parse() {
        Ok(n) => n,
        Err(e) => {
            log::error!("Failed to parse composite N='{}': {}", composite.n, e);
            return TrialResult {
                composite_id: composite.id,
                success: false,
                verified: false,
                wall_clock_secs: 0.0,
                stages: HashMap::new(),
                overhead_secs: 0.0,
                stage_sum_secs: 0.0,
                cpu_wall_ratio: None,
                parse_debug: Some(ParseDebug {
                    parsed_timing_lines: vec![],
                    parsed_relation_lines: vec![],
                    parsed_matrix_lines: vec![],
                    exit_status: None,
                    timed_out: false,
                }),
            };
        }
    };

    match install.run_with_kill_timeout(&n, params, timeout) {
        Ok(cado_result) => {
            let parse_debug = extract_parse_debug(&cado_result.log_tail, false);
            build_trial_result(composite, &cado_result, Some(parse_debug))
        }
        Err(crate::cado::CadoError::Timeout(_)) => {
            log::warn!(
                "Trial {} timed out after {:?}",
                composite.id,
                timeout
            );
            TrialResult {
                composite_id: composite.id,
                success: false,
                verified: false,
                wall_clock_secs: timeout.as_secs_f64(),
                stages: HashMap::new(),
                overhead_secs: 0.0,
                stage_sum_secs: 0.0,
                cpu_wall_ratio: None,
                parse_debug: Some(ParseDebug {
                    parsed_timing_lines: vec![],
                    parsed_relation_lines: vec![],
                    parsed_matrix_lines: vec![],
                    exit_status: None,
                    timed_out: true,
                }),
            }
        }
        Err(e) => {
            log::error!("Trial {} failed: {}", composite.id, e);
            TrialResult {
                composite_id: composite.id,
                success: false,
                verified: false,
                wall_clock_secs: 0.0,
                stages: HashMap::new(),
                overhead_secs: 0.0,
                stage_sum_secs: 0.0,
                cpu_wall_ratio: None,
                parse_debug: Some(ParseDebug {
                    parsed_timing_lines: vec![],
                    parsed_relation_lines: vec![],
                    parsed_matrix_lines: vec![],
                    exit_status: None,
                    timed_out: false,
                }),
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Per-size execution
// ---------------------------------------------------------------------------

/// Run the scaling protocol for a single size class.
///
/// Executes `run_trial()` for each composite, saves results incrementally
/// to `scaling_classical_c{digits}.json`, and prints per-trial progress.
///
/// # Arguments
/// - `install`: Validated CADO-NFS installation
/// - `spec`: Size specification
/// - `composites`: Pre-generated composites with known factors
/// - `output_dir`: Directory for JSON output
///
/// # Returns
/// A `ScalingResult` with all trials and summary statistics.
pub fn run_scaling_for_size(
    install: &crate::cado::CadoInstallation,
    spec: &SizeSpec,
    composites: &[StoredComposite],
    output_dir: &std::path::Path,
) -> ScalingResult {
    let started_at = iso_now();
    let params = crate::params::CadoParams::default_for_bits(spec.bits);
    let hardware = detect_hardware(&install.root_dir.to_string_lossy());
    let result_filename = format!("scaling_classical_c{}.json", spec.digits);
    let result_filepath = output_dir.join(&result_filename);
    let composites_filename = format!("scaling_composites_c{}.json", spec.digits);

    eprintln!(
        "\n━━━ c{} ({}-bit) — {} trials, timeout {:?} ━━━",
        spec.digits,
        spec.bits,
        composites.len(),
        spec.timeout
    );

    let mut trials = Vec::with_capacity(composites.len());

    for (i, composite) in composites.iter().enumerate() {
        eprint!(
            "  Trial {}/{} (N={}...): ",
            i + 1,
            composites.len(),
            &composite.n[..composite.n.len().min(20)]
        );

        let trial = run_trial(install, composite, &params, spec.timeout);

        if trial.success {
            let verified_str = if trial.verified { "✓" } else { "⚠" };
            eprintln!(
                "{} {:.2}s (sieve {:.0}%, overhead {:.0}%)",
                verified_str,
                trial.wall_clock_secs,
                trial
                    .stages
                    .iter()
                    .find(|(k, _)| k.to_lowercase().contains("siev"))
                    .map(|(_, s)| s.wall_secs / trial.wall_clock_secs * 100.0)
                    .unwrap_or(0.0),
                if trial.wall_clock_secs > 0.0 {
                    trial.overhead_secs / trial.wall_clock_secs * 100.0
                } else {
                    0.0
                }
            );
        } else {
            eprintln!(
                "✗ FAILED (wall {:.2}s, timeout={})",
                trial.wall_clock_secs,
                trial
                    .parse_debug
                    .as_ref()
                    .map(|d| d.timed_out)
                    .unwrap_or(false)
            );
        }

        trials.push(trial);

        // Incremental save after each trial
        let summary = compute_size_summary(&trials);
        let partial_result = ScalingResult {
            schema_version: SCHEMA_VERSION.to_string(),
            git_commit: git_commit_hash(),
            crate_version: CRATE_VERSION.to_string(),
            size: spec.clone(),
            method: "classical-cado-nfs".to_string(),
            params_description: params.summary(),
            composites_file: composites_filename.clone(),
            trials: trials.clone(),
            summary,
            hardware: hardware.clone(),
            started_at: started_at.clone(),
            finished_at: iso_now(),
        };

        if let Ok(json) = serde_json::to_string_pretty(&partial_result) {
            let _ = std::fs::write(&result_filepath, &json);
        }
    }

    let finished_at = iso_now();
    let summary = compute_size_summary(&trials);

    // Print size summary
    let success_count = trials.iter().filter(|t| t.success).count();
    eprintln!(
        "  Summary: {}/{} succeeded, median {:.2}s, mean {:.2}s, p90 {:.2}s",
        success_count,
        trials.len(),
        summary.wall_clock_percentiles.p50,
        summary.wall_clock_percentiles.mean,
        summary.wall_clock_percentiles.p90
    );

    ScalingResult {
        schema_version: SCHEMA_VERSION.to_string(),
        git_commit: git_commit_hash(),
        crate_version: CRATE_VERSION.to_string(),
        size: spec.clone(),
        method: "classical-cado-nfs".to_string(),
        params_description: params.summary(),
        composites_file: composites_filename,
        trials,
        summary,
        hardware,
        started_at,
        finished_at,
    }
}

// ---------------------------------------------------------------------------
// Top-level orchestrator
// ---------------------------------------------------------------------------

/// Run the complete scaling protocol across multiple size classes.
///
/// For each size in `size_filter` (or all defaults if empty):
/// 1. Generate or load composites
/// 2. Check if results already exist (skip if so — incremental resume)
/// 3. Run trials and save per-size JSON
/// 4. Combine all sizes into a protocol-level summary
/// 5. Print summary table
///
/// # Arguments
/// - `install`: Validated CADO-NFS installation
/// - `output_dir`: Base directory for all output files
/// - `size_filter`: If non-empty, only run these digit counts. Empty = all defaults.
/// - `rng`: Random number generator for composite generation
pub fn run_scaling_protocol(
    install: &crate::cado::CadoInstallation,
    output_dir: &std::path::Path,
    size_filter: &[u32],
    rng: &mut impl rand::Rng,
) -> Result<ScalingProtocolResult, String> {
    let started_at = iso_now();
    let all_specs = default_size_specs();

    let specs: Vec<&SizeSpec> = if size_filter.is_empty() {
        all_specs.iter().collect()
    } else {
        all_specs
            .iter()
            .filter(|s| size_filter.contains(&s.digits))
            .collect()
    };

    if specs.is_empty() {
        return Err(format!(
            "No matching size specs for filter {:?}. Available: {:?}",
            size_filter,
            all_specs.iter().map(|s| s.digits).collect::<Vec<_>>()
        ));
    }

    std::fs::create_dir_all(output_dir)
        .map_err(|e| format!("Failed to create output dir: {}", e))?;

    let hardware = detect_hardware(&install.root_dir.to_string_lossy());

    eprintln!("╔══════════════════════════════════════════════════╗");
    eprintln!("║       Scaling Protocol — Classical Baseline      ║");
    eprintln!("╚══════════════════════════════════════════════════╝");
    eprintln!(
        "  Sizes: {}",
        specs
            .iter()
            .map(|s| format!("c{}", s.digits))
            .collect::<Vec<_>>()
            .join(", ")
    );
    eprintln!("  Hardware: {} ({} cores)", hardware.cpu_model, hardware.cores);
    eprintln!("  Output: {}", output_dir.display());
    eprintln!();

    let mut size_results = Vec::new();

    for spec in &specs {
        // Step 1: Ensure composites exist
        let composites = ensure_composites(spec, output_dir, rng)?;

        // Step 2: Check for existing results (incremental resume)
        let result_filename = format!("scaling_classical_c{}.json", spec.digits);
        let result_filepath = output_dir.join(&result_filename);

        if result_filepath.exists() {
            match std::fs::read_to_string(&result_filepath) {
                Ok(contents) => {
                    match serde_json::from_str::<ScalingResult>(&contents) {
                        Ok(existing) => {
                            if existing.trials.len() >= spec.num_composites {
                                eprintln!(
                                    "\n━━━ c{} — SKIPPED (already complete: {} trials) ━━━",
                                    spec.digits,
                                    existing.trials.len()
                                );
                                size_results.push(existing);
                                continue;
                            }
                            eprintln!(
                                "\n━━━ c{} — resuming ({}/{} trials done) ━━━",
                                spec.digits,
                                existing.trials.len(),
                                spec.num_composites
                            );
                        }
                        Err(e) => {
                            log::warn!(
                                "Failed to parse existing result {}: {}, re-running",
                                result_filepath.display(),
                                e
                            );
                        }
                    }
                }
                Err(e) => {
                    log::warn!(
                        "Failed to read {}: {}, re-running",
                        result_filepath.display(),
                        e
                    );
                }
            }
        }

        // Step 3: Run trials
        let result = run_scaling_for_size(install, spec, &composites, output_dir);
        size_results.push(result);
    }

    // Save combined protocol result
    let finished_at = iso_now();
    let protocol_result = ScalingProtocolResult {
        schema_version: SCHEMA_VERSION.to_string(),
        git_commit: git_commit_hash(),
        crate_version: CRATE_VERSION.to_string(),
        method: "classical-cado-nfs".to_string(),
        sizes: size_results,
        hardware: hardware.clone(),
        started_at,
        finished_at,
    };

    let combined_filepath = output_dir.join("scaling_protocol_classical.json");
    let json = serde_json::to_string_pretty(&protocol_result)
        .map_err(|e| format!("Failed to serialize protocol result: {}", e))?;
    std::fs::write(&combined_filepath, &json)
        .map_err(|e| format!("Failed to write {}: {}", combined_filepath.display(), e))?;

    // Print summary table
    print_summary_table(&protocol_result);

    eprintln!(
        "\nResults saved to: {}",
        combined_filepath.display()
    );

    Ok(protocol_result)
}

/// Print a summary table of scaling protocol results.
fn print_summary_table(result: &ScalingProtocolResult) {
    eprintln!();
    eprintln!("┌──────┬──────┬─────────┬─────────┬─────────┬─────────┬────────┬──────┐");
    eprintln!("│  cN  │ bits │  N (ok) │  median │   mean  │   p90   │ sieve% │  ok% │");
    eprintln!("├──────┼──────┼─────────┼─────────┼─────────┼─────────┼────────┼──────┤");

    for size in &result.sizes {
        let s = &size.summary;
        let sieve_pct = s
            .stage_fractions
            .iter()
            .find(|(k, _)| k.to_lowercase().contains("siev"))
            .map(|(_, v)| v * 100.0)
            .unwrap_or(0.0);

        eprintln!(
            "│ c{:<3} │ {:>4} │ {:>3}/{:<3} │ {:>6.1}s │ {:>6.1}s │ {:>6.1}s │ {:>5.1}% │ {:>3.0}% │",
            size.size.digits,
            size.size.bits,
            s.n_success,
            s.n_trials,
            s.wall_clock_percentiles.p50,
            s.wall_clock_percentiles.mean,
            s.wall_clock_percentiles.p90,
            sieve_pct,
            s.success_rate * 100.0,
        );
    }

    eprintln!("└──────┴──────┴─────────┴─────────┴─────────┴─────────┴────────┴──────┘");
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

    #[test]
    fn test_ensure_composites_generate_and_load() {
        use rand::SeedableRng;

        let tmpdir = tempfile::tempdir().unwrap();
        let spec = SizeSpec {
            digits: 30,
            bits: 100,
            num_composites: 3,
            expected_time_description: "~1s".to_string(),
            timeout: Duration::from_secs(60),
        };

        // Generate composites
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);
        let composites = ensure_composites(&spec, tmpdir.path(), &mut rng).unwrap();
        assert_eq!(composites.len(), 3);

        // Each composite should have valid p × q = N
        for c in &composites {
            let n: num_bigint::BigUint = c.n.parse().unwrap();
            let p: num_bigint::BigUint = c.p.parse().unwrap();
            let q: num_bigint::BigUint = c.q.parse().unwrap();
            assert_eq!(p * q, n, "p × q should equal N for composite {}", c.id);
            assert!(c.n_bits >= 90, "Should be ~100 bits, got {}", c.n_bits);
        }

        // File should exist
        let filepath = tmpdir.path().join("scaling_composites_c30.json");
        assert!(filepath.exists());

        // Load should return the same composites
        let mut rng2 = rand::rngs::StdRng::seed_from_u64(99);
        let loaded = ensure_composites(&spec, tmpdir.path(), &mut rng2).unwrap();
        assert_eq!(loaded.len(), 3);
        assert_eq!(loaded[0].n, composites[0].n);
        assert_eq!(loaded[1].n, composites[1].n);
        assert_eq!(loaded[2].n, composites[2].n);
    }

    #[test]
    fn test_verify_factors_correct() {
        let composite = StoredComposite {
            id: 0,
            n: "143".to_string(),
            p: "11".to_string(),
            q: "13".to_string(),
            n_bits: 8,
            n_digits: 3,
        };
        assert!(verify_factors(&composite, &["11".to_string(), "13".to_string()]));
        assert!(verify_factors(&composite, &["13".to_string(), "11".to_string()]));
    }

    #[test]
    fn test_verify_factors_incorrect() {
        let composite = StoredComposite {
            id: 0,
            n: "143".to_string(),
            p: "11".to_string(),
            q: "13".to_string(),
            n_bits: 8,
            n_digits: 3,
        };
        assert!(!verify_factors(&composite, &["7".to_string(), "11".to_string()]));
        assert!(!verify_factors(&composite, &["143".to_string()]));
        assert!(!verify_factors(&composite, &[]));
    }

    #[test]
    fn test_build_trial_result_success() {
        use crate::cado::CadoResult;

        let composite = StoredComposite {
            id: 0,
            n: "143".to_string(),
            p: "11".to_string(),
            q: "13".to_string(),
            n_bits: 8,
            n_digits: 3,
        };

        let mut phase_times = HashMap::new();
        phase_times.insert("Lattice Sieving".to_string(), Duration::from_secs_f64(3.0));
        phase_times.insert("Linear Algebra".to_string(), Duration::from_secs_f64(1.0));
        phase_times.insert("Square Root".to_string(), Duration::from_secs_f64(0.5));

        let mut phase_cpu_times = HashMap::new();
        phase_cpu_times.insert("Linear Algebra".to_string(), Duration::from_secs_f64(2.0));
        phase_cpu_times.insert("Square Root".to_string(), Duration::from_secs_f64(0.4));

        let cado_result = CadoResult {
            success: true,
            n: "143".to_string(),
            factors: vec!["11".to_string(), "13".to_string()],
            total_time: Duration::from_secs_f64(6.0),
            phase_times,
            phase_cpu_times,
            relations_found: Some(50000),
            matrix_size: Some((3000, 3200)),
            unique_relations: Some(42000),
            log_tail: String::new(),
        };

        let trial = build_trial_result(&composite, &cado_result, None);

        assert!(trial.success);
        assert!(trial.verified);
        assert!((trial.wall_clock_secs - 6.0).abs() < 0.01);
        assert_eq!(trial.stages.len(), 3);

        // Verify sieve stage has relations
        let sieve = &trial.stages["Lattice Sieving"];
        assert!((sieve.wall_secs - 3.0).abs() < 0.01);
        assert_eq!(sieve.relations_found, Some(50000));
        assert!(sieve.relations_per_sec.is_some());
        assert!((sieve.relations_per_sec.unwrap() - 50000.0 / 3.0).abs() < 1.0);
        assert!(sieve.cpu_secs.is_none()); // Sieving uses Pattern 3

        // Verify linalg stage has matrix dims and CPU time
        let linalg = &trial.stages["Linear Algebra"];
        assert_eq!(linalg.matrix_dims, Some((3000, 3200)));
        assert_eq!(linalg.cpu_secs, Some(2.0));

        // Verify overhead
        let stage_sum = 3.0 + 1.0 + 0.5;
        assert!((trial.stage_sum_secs - stage_sum).abs() < 0.01);
        assert!((trial.overhead_secs - (6.0 - stage_sum)).abs() < 0.01);

        // Verify CPU/wall ratio (only for stages with CPU: LA=2.0/1.0, sqrt=0.4/0.5)
        // total_cpu = 2.4, total_real_with_cpu = 1.5
        assert!(trial.cpu_wall_ratio.is_some());
        assert!((trial.cpu_wall_ratio.unwrap() - 2.4 / 1.5).abs() < 0.01);
    }

    #[test]
    fn test_build_trial_result_filters_complete_factorization() {
        use crate::cado::CadoResult;

        let composite = StoredComposite {
            id: 0,
            n: "143".to_string(),
            p: "11".to_string(),
            q: "13".to_string(),
            n_bits: 8,
            n_digits: 3,
        };

        let mut phase_times = HashMap::new();
        phase_times.insert("Lattice Sieving".to_string(), Duration::from_secs_f64(3.0));
        // This should be filtered out (it's a sum, not a stage)
        phase_times.insert(
            "Complete Factorization / Discrete logarithm".to_string(),
            Duration::from_secs_f64(10.0),
        );

        let cado_result = CadoResult {
            success: true,
            n: "143".to_string(),
            factors: vec!["11".to_string(), "13".to_string()],
            total_time: Duration::from_secs_f64(10.0),
            phase_times,
            phase_cpu_times: HashMap::new(),
            relations_found: None,
            matrix_size: None,
            unique_relations: None,
            log_tail: String::new(),
        };

        let trial = build_trial_result(&composite, &cado_result, None);

        // Should only have Lattice Sieving, not Complete Factorization
        assert_eq!(trial.stages.len(), 1);
        assert!(trial.stages.contains_key("Lattice Sieving"));
        assert!(!trial.stages.contains_key("Complete Factorization / Discrete logarithm"));
    }

    #[test]
    fn test_extract_parse_debug() {
        let log = "\x1b[32;1mInfo\x1b[0m:Lattice Sieving: Total number of relations: 52609\n\
                   \x1b[32;1mInfo\x1b[0m:Lattice Sieving: Total time: 0.98s\n\
                   \x1b[32;1mInfo\x1b[0m:Linear Algebra: Total cpu/real time for bwc: 0.38/1.19\n\
                   matrix is 3000 x 3200\n\
                   \x1b[32;1mInfo\x1b[0m:Complete Factorization: Total cpu/elapsed time for entire factorization 2.68/13.47\n";

        let debug = extract_parse_debug(log, false);

        assert!(!debug.timed_out);
        // Should capture timing lines
        assert_eq!(debug.parsed_timing_lines.len(), 3);
        assert!(debug.parsed_timing_lines[0].contains("Total time: 0.98s"));
        assert!(debug.parsed_timing_lines[1].contains("Total cpu/real time for bwc"));
        assert!(debug.parsed_timing_lines[2].contains("Total cpu/elapsed time"));
        // Should capture relation lines
        assert_eq!(debug.parsed_relation_lines.len(), 1);
        assert!(debug.parsed_relation_lines[0].contains("52609"));
        // Should capture matrix lines
        assert_eq!(debug.parsed_matrix_lines.len(), 1);
        assert!(debug.parsed_matrix_lines[0].contains("3000 x 3200"));
    }
}
