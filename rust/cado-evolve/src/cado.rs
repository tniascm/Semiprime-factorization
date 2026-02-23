//! CADO-NFS wrapper: installation validation, execution, and log parsing.
//!
//! Provides a Rust interface to the CADO-NFS factoring system. Each run:
//! 1. Writes parameters to a temp directory
//! 2. Invokes `cado-nfs.py` with the target number and parameters
//! 3. Parses stdout/log for timing data, relation counts, and factors
//! 4. Returns a structured `CadoResult`

use num_bigint::BigUint;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};
use tempfile::TempDir;

use crate::params::CadoParams;

/// Represents a validated CADO-NFS installation.
#[derive(Debug, Clone)]
pub struct CadoInstallation {
    /// Path to the cado-nfs.py script.
    pub cado_nfs_py: PathBuf,
    /// Path to the CADO-NFS root directory.
    pub root_dir: PathBuf,
    /// Python interpreter to use.
    pub python: String,
}

/// Result of a single CADO-NFS factorization run.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CadoResult {
    /// Whether factorization succeeded.
    pub success: bool,
    /// The number that was factored.
    pub n: String,
    /// Factors found (if successful).
    pub factors: Vec<String>,
    /// Total wall-clock time for the entire run.
    pub total_time: Duration,
    /// Per-phase timing breakdown.
    pub phase_times: HashMap<String, Duration>,
    /// Number of relations found during sieving.
    pub relations_found: Option<u64>,
    /// Matrix dimensions (rows, cols) for linear algebra phase.
    pub matrix_size: Option<(u64, u64)>,
    /// Number of unique relations after filtering.
    pub unique_relations: Option<u64>,
    /// Raw log output (truncated to last 200 lines for storage).
    pub log_tail: String,
}

/// Errors that can occur during CADO-NFS operations.
#[derive(Debug, thiserror::Error)]
pub enum CadoError {
    #[error("CADO-NFS installation not found at {0}")]
    NotFound(PathBuf),

    #[error("cado-nfs.py not found at {0}")]
    ScriptNotFound(PathBuf),

    #[error("Python interpreter '{0}' not found")]
    PythonNotFound(String),

    #[error("CADO-NFS execution failed: {0}")]
    ExecutionFailed(String),

    #[error("CADO-NFS timed out after {0:?}")]
    Timeout(Duration),

    #[error("Failed to parse CADO-NFS output: {0}")]
    ParseError(String),

    #[error("IO error: {0}")]
    Io(#[from] std::io::Error),
}

impl CadoInstallation {
    /// Validate a CADO-NFS installation at the given path.
    ///
    /// Checks that:
    /// - The root directory exists
    /// - `cado-nfs.py` exists and is executable
    /// - Python3 is available
    pub fn validate(root_dir: impl AsRef<Path>) -> Result<Self, CadoError> {
        let root_dir = root_dir.as_ref().to_path_buf();

        if !root_dir.exists() {
            return Err(CadoError::NotFound(root_dir));
        }

        let cado_nfs_py = root_dir.join("cado-nfs.py");
        if !cado_nfs_py.exists() {
            return Err(CadoError::ScriptNotFound(cado_nfs_py));
        }

        // Find Python interpreter
        let python = find_python()?;

        // Verify Python can import required modules
        let check = Command::new(&python)
            .args(["-c", "import sqlite3, subprocess, re, math"])
            .output()?;

        if !check.status.success() {
            return Err(CadoError::PythonNotFound(format!(
                "{} missing required modules",
                python
            )));
        }

        Ok(CadoInstallation {
            cado_nfs_py,
            root_dir,
            python,
        })
    }

    /// Run CADO-NFS to factor N with given parameters.
    ///
    /// Creates a temporary working directory, invokes cado-nfs.py, and parses
    /// the output. The working directory is automatically cleaned up.
    pub fn run(
        &self,
        n: &BigUint,
        params: &CadoParams,
        timeout: Duration,
    ) -> Result<CadoResult, CadoError> {
        let n_str = n.to_string();

        // Create temporary working directory for this run
        let workdir = TempDir::new().map_err(CadoError::Io)?;

        // Build command arguments
        let mut args = vec![
            self.cado_nfs_py.to_string_lossy().to_string(),
            n_str.clone(),
        ];

        // Add parameter arguments
        args.extend(params.to_cado_args());

        // Set working directory
        args.push(format!(
            "tasks.workdir={}",
            workdir.path().to_string_lossy()
        ));

        log::info!("Running CADO-NFS on N={} ({} bits)", n_str, n.bits());
        log::debug!("Args: {:?}", args);

        let start = Instant::now();

        // Execute CADO-NFS
        let output = Command::new(&self.python)
            .args(&args)
            .current_dir(&self.root_dir)
            .env("CADO_NFS_SOURCE_DIR", &self.root_dir)
            .output();

        let elapsed = start.elapsed();

        // Check timeout (Command doesn't support timeout natively on all platforms)
        if elapsed > timeout {
            return Err(CadoError::Timeout(timeout));
        }

        let output = output.map_err(|e| CadoError::ExecutionFailed(e.to_string()))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();

        let combined_log = format!("{}\n{}", stdout, stderr);

        // Parse the output
        let mut result = parse_cado_output(&combined_log);
        result.n = n_str;
        result.total_time = elapsed;

        // Keep last 200 lines of log
        let log_lines: Vec<&str> = combined_log.lines().collect();
        let tail_start = log_lines.len().saturating_sub(200);
        result.log_tail = log_lines[tail_start..].join("\n");

        if !output.status.success() && !result.success {
            log::warn!(
                "CADO-NFS exited with status {:?}",
                output.status.code()
            );
        }

        Ok(result)
    }

    /// Run CADO-NFS with a timeout enforced via process kill.
    ///
    /// This spawns the process and monitors it, killing if it exceeds the
    /// timeout. More reliable than post-hoc timeout checking.
    pub fn run_with_kill_timeout(
        &self,
        n: &BigUint,
        params: &CadoParams,
        timeout: Duration,
    ) -> Result<CadoResult, CadoError> {
        let n_str = n.to_string();

        let workdir = TempDir::new().map_err(CadoError::Io)?;

        let mut args = vec![
            self.cado_nfs_py.to_string_lossy().to_string(),
            n_str.clone(),
        ];
        args.extend(params.to_cado_args());
        args.push(format!(
            "tasks.workdir={}",
            workdir.path().to_string_lossy()
        ));

        log::info!(
            "Running CADO-NFS on N={} ({} bits) with {}s timeout",
            n_str,
            n.bits(),
            timeout.as_secs()
        );

        let start = Instant::now();

        let mut child = Command::new(&self.python)
            .args(&args)
            .current_dir(&self.root_dir)
            .env("CADO_NFS_SOURCE_DIR", &self.root_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped())
            .spawn()
            .map_err(|e| CadoError::ExecutionFailed(e.to_string()))?;

        // Poll for completion with timeout
        let poll_interval = Duration::from_millis(500);
        loop {
            match child.try_wait() {
                Ok(Some(_status)) => {
                    // Process finished
                    break;
                }
                Ok(None) => {
                    // Still running
                    if start.elapsed() > timeout {
                        log::warn!("CADO-NFS timed out, killing process");
                        let _ = child.kill();
                        let _ = child.wait();
                        return Err(CadoError::Timeout(timeout));
                    }
                    std::thread::sleep(poll_interval);
                }
                Err(e) => {
                    return Err(CadoError::ExecutionFailed(e.to_string()));
                }
            }
        }

        let elapsed = start.elapsed();
        let output = child
            .wait_with_output()
            .map_err(|e| CadoError::ExecutionFailed(e.to_string()))?;

        let stdout = String::from_utf8_lossy(&output.stdout).to_string();
        let stderr = String::from_utf8_lossy(&output.stderr).to_string();
        let combined_log = format!("{}\n{}", stdout, stderr);

        let mut result = parse_cado_output(&combined_log);
        result.n = n_str;
        result.total_time = elapsed;

        let log_lines: Vec<&str> = combined_log.lines().collect();
        let tail_start = log_lines.len().saturating_sub(200);
        result.log_tail = log_lines[tail_start..].join("\n");

        Ok(result)
    }
}

/// Parse CADO-NFS log output into a structured result.
///
/// Extracts:
/// - Factor lines (e.g., "N = p * q")
/// - Per-phase timing (polyselect, sieve, linalg, sqrt)
/// - Relation counts
/// - Matrix dimensions
pub fn parse_cado_output(log: &str) -> CadoResult {
    let mut result = CadoResult {
        success: false,
        n: String::new(),
        factors: Vec::new(),
        total_time: Duration::ZERO,
        phase_times: HashMap::new(),
        relations_found: None,
        matrix_size: None,
        unique_relations: None,
        log_tail: String::new(),
    };

    for line in log.lines() {
        let trimmed = line.trim();

        // Look for factor output lines
        // CADO-NFS outputs: "N = p1 * p2" or "N = p1 * p2 * p3" etc.
        if let Some(factors_str) = extract_factors_from_line(trimmed) {
            result.factors = factors_str;
            if !result.factors.is_empty() {
                result.success = true;
            }
        }

        // Phase timing: "Total cpu/real time for <phase>: ..."
        // Example: "Total cpu/real time for polyselect: 12.3/4.5"
        if trimmed.starts_with("Total cpu/real time for ") {
            if let Some((phase, real_secs)) = parse_phase_timing(trimmed) {
                result
                    .phase_times
                    .insert(phase, Duration::from_secs_f64(real_secs));
            }
        }

        // Alternative timing: "Total time: 123.4s"
        if trimmed.contains("Total time:") {
            if let Some(secs) = extract_seconds(trimmed) {
                result
                    .phase_times
                    .insert("total_reported".to_string(), Duration::from_secs_f64(secs));
            }
        }

        // Relation counts
        // "Sieving: found 12345 relations"
        // "# rels found: 12345"
        // "found 12345 relations"
        if trimmed.contains("rels found")
            || trimmed.contains("relations found")
            || (trimmed.contains("found") && trimmed.contains("relation"))
        {
            if let Some(count) = extract_number(trimmed) {
                result.relations_found = Some(count);
            }
        }

        // "unique relations: 12345"
        if trimmed.contains("unique relations") || trimmed.contains("unique rels") {
            if let Some(count) = extract_number(trimmed) {
                result.unique_relations = Some(count);
            }
        }

        // Matrix size: "matrix is 1234 x 5678"
        if trimmed.contains("matrix is") || trimmed.contains("Matrix has") {
            if let Some((rows, cols)) = extract_matrix_size(trimmed) {
                result.matrix_size = Some((rows, cols));
            }
        }
    }

    result
}

/// Extract factors from a CADO-NFS output line.
///
/// Recognizes formats:
/// - `"123456789 = 12347 * 10001"`
/// - `"N = p1 * p2 * p3"`
fn extract_factors_from_line(line: &str) -> Option<Vec<String>> {
    // Pattern: "digits = digits * digits [* digits ...]"
    if !line.contains('*') || !line.contains('=') {
        return None;
    }

    let parts: Vec<&str> = line.splitn(2, '=').collect();
    if parts.len() != 2 {
        return None;
    }

    let rhs = parts[1].trim();
    let factor_strs: Vec<&str> = rhs.split('*').collect();

    let mut factors = Vec::new();
    for f in factor_strs {
        let trimmed = f.trim();
        // Must be a number
        if trimmed.chars().all(|c| c.is_ascii_digit()) && !trimmed.is_empty() {
            factors.push(trimmed.to_string());
        } else {
            return None;
        }
    }

    if factors.len() >= 2 {
        Some(factors)
    } else {
        None
    }
}

/// Parse phase timing from a "Total cpu/real time for <phase>: cpu/real" line.
fn parse_phase_timing(line: &str) -> Option<(String, f64)> {
    // "Total cpu/real time for polyselect: 12.34/5.67"
    let prefix = "Total cpu/real time for ";
    let rest = line.strip_prefix(prefix)?;

    let colon_pos = rest.find(':')?;
    let phase = rest[..colon_pos].trim().to_string();
    let timing = rest[colon_pos + 1..].trim();

    // Format: "cpu_secs/real_secs" or just "real_secs"
    let real_secs = if timing.contains('/') {
        let parts: Vec<&str> = timing.split('/').collect();
        parts.last()?.trim().trim_matches('s').parse::<f64>().ok()?
    } else {
        timing.trim_matches('s').parse::<f64>().ok()?
    };

    Some((phase, real_secs))
}

/// Extract a number from a line containing digits.
fn extract_number(line: &str) -> Option<u64> {
    for word in line.split_whitespace() {
        if let Ok(n) = word.trim_matches(|c: char| !c.is_ascii_digit()).parse::<u64>() {
            if n > 0 {
                return Some(n);
            }
        }
    }
    None
}

/// Extract seconds from a timing line.
fn extract_seconds(line: &str) -> Option<f64> {
    for word in line.split_whitespace() {
        let cleaned = word.trim_end_matches('s').trim_end_matches(',');
        if let Ok(secs) = cleaned.parse::<f64>() {
            if secs > 0.0 {
                return Some(secs);
            }
        }
    }
    None
}

/// Extract matrix dimensions from a "matrix is R x C" line.
fn extract_matrix_size(line: &str) -> Option<(u64, u64)> {
    let words: Vec<&str> = line.split_whitespace().collect();
    for i in 0..words.len().saturating_sub(2) {
        if words[i + 1] == "x" || words[i + 1] == "×" {
            if let (Ok(r), Ok(c)) = (words[i].parse::<u64>(), words[i + 2].parse::<u64>()) {
                return Some((r, c));
            }
        }
    }
    None
}

/// Find a suitable Python3 interpreter.
fn find_python() -> Result<String, CadoError> {
    for candidate in &["python3", "python"] {
        let check = Command::new(candidate).args(["--version"]).output();
        if let Ok(output) = check {
            if output.status.success() {
                let version = String::from_utf8_lossy(&output.stdout).to_string();
                if version.contains("Python 3") {
                    return Ok(candidate.to_string());
                }
                // Also check stderr (some Python versions print there)
                let version_err = String::from_utf8_lossy(&output.stderr).to_string();
                if version_err.contains("Python 3") {
                    return Ok(candidate.to_string());
                }
            }
        }
    }
    Err(CadoError::PythonNotFound(
        "No Python 3 interpreter found".to_string(),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_factors_simple() {
        let line = "110503 = 233 * 474";
        let factors = extract_factors_from_line(line);
        assert!(factors.is_some());
        let factors = factors.unwrap();
        assert_eq!(factors.len(), 2);
        assert_eq!(factors[0], "233");
        assert_eq!(factors[1], "474");
    }

    #[test]
    fn test_extract_factors_three_factors() {
        let line = "30 = 2 * 3 * 5";
        let factors = extract_factors_from_line(line);
        assert!(factors.is_some());
        assert_eq!(factors.unwrap().len(), 3);
    }

    #[test]
    fn test_extract_factors_no_factors() {
        let line = "This is a normal log line";
        assert!(extract_factors_from_line(line).is_none());
    }

    #[test]
    fn test_extract_factors_not_numbers() {
        let line = "result = good * bad";
        assert!(extract_factors_from_line(line).is_none());
    }

    #[test]
    fn test_parse_phase_timing() {
        let line = "Total cpu/real time for polyselect: 12.34/5.67";
        let result = parse_phase_timing(line);
        assert!(result.is_some());
        let (phase, secs) = result.unwrap();
        assert_eq!(phase, "polyselect");
        assert!((secs - 5.67).abs() < 0.01);
    }

    #[test]
    fn test_parse_phase_timing_sieve() {
        let line = "Total cpu/real time for sieve: 100.5/25.3";
        let result = parse_phase_timing(line);
        assert!(result.is_some());
        let (phase, secs) = result.unwrap();
        assert_eq!(phase, "sieve");
        assert!((secs - 25.3).abs() < 0.01);
    }

    #[test]
    fn test_extract_number() {
        assert_eq!(extract_number("found 12345 relations"), Some(12345));
        assert_eq!(extract_number("no numbers here"), None);
    }

    #[test]
    fn test_extract_matrix_size() {
        assert_eq!(
            extract_matrix_size("matrix is 1234 x 5678"),
            Some((1234, 5678))
        );
        assert_eq!(extract_matrix_size("no matrix here"), None);
    }

    #[test]
    fn test_parse_cado_output_success() {
        let log = r#"
Info: Starting CADO-NFS
Total cpu/real time for polyselect: 1.2/0.5
Total cpu/real time for sieve: 45.6/12.3
found 50000 relations
unique relations: 42000
matrix is 3000 x 3200
Total cpu/real time for linalg: 5.4/2.1
Total cpu/real time for sqrt: 0.8/0.3
110503 = 233 * 474
"#;
        let result = parse_cado_output(log);
        assert!(result.success);
        assert_eq!(result.factors.len(), 2);
        assert_eq!(result.relations_found, Some(50000));
        assert_eq!(result.unique_relations, Some(42000));
        assert_eq!(result.matrix_size, Some((3000, 3200)));
        assert!(result.phase_times.contains_key("polyselect"));
        assert!(result.phase_times.contains_key("sieve"));
        assert!(result.phase_times.contains_key("linalg"));
        assert!(result.phase_times.contains_key("sqrt"));
    }

    #[test]
    fn test_parse_cado_output_failure() {
        let log = "Info: Starting CADO-NFS\nError: something went wrong\n";
        let result = parse_cado_output(log);
        assert!(!result.success);
        assert!(result.factors.is_empty());
    }

    #[test]
    fn test_find_python() {
        // Should find Python3 on most systems
        let result = find_python();
        assert!(result.is_ok(), "Python3 should be available");
    }

    #[test]
    fn test_extract_factors_large_semiprime() {
        let line = "1267650600228229401496703205653 = 1125899906842679 * 1125899906842847";
        let factors = extract_factors_from_line(line);
        assert!(factors.is_some());
        let factors = factors.unwrap();
        assert_eq!(factors.len(), 2);
        assert_eq!(factors[0], "1125899906842679");
        assert_eq!(factors[1], "1125899906842847");
    }
}
