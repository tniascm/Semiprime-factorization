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
    /// Per-phase wall-clock timing breakdown.
    pub phase_times: HashMap<String, Duration>,
    /// Per-phase CPU timing breakdown (from "cpu/real" format lines).
    /// Only populated for phases that report CPU time (Patterns 1 & 2).
    /// Pattern 3 ("Total time: Xs") does not include CPU time.
    pub phase_cpu_times: HashMap<String, Duration>,
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

    /// Get the number of worker threads for CADO-NFS based on available CPUs.
    fn num_worker_threads(&self) -> usize {
        std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
    }

    /// Return common server/client arguments needed when using --parameters.
    ///
    /// CADO-NFS uses a client-server architecture. When run without --parameters,
    /// it auto-detects and starts local clients. When --parameters is used, we
    /// must explicitly configure the server whitelist and client setup.
    fn server_client_args(&self) -> Vec<String> {
        let nrclients = (self.num_worker_threads() / 2).max(2);
        vec![
            "server.whitelist=0.0.0.0/0".to_string(),
            "slaves.hostnames=localhost".to_string(),
            format!("slaves.nrclients={}", nrclients),
        ]
    }

    /// Return server address argument to avoid hostname resolution issues.
    ///
    /// CADO-NFS resolves the machine's hostname for its certificate/SSL setup.
    /// On macOS, .local hostnames require mDNS (Bonjour) which may not always
    /// be available. Using `server.address=localhost` bypasses this issue.
    fn server_address_args(&self) -> Vec<String> {
        vec!["server.address=localhost".to_string()]
    }

    /// Find the nearest available CADO-NFS parameter file for a number with
    /// the given number of decimal digits.
    ///
    /// CADO-NFS only looks ±3 digits from the target, and has gaps (e.g., c30
    /// then c60). This function finds the closest available file.
    pub fn find_param_file(&self, num_digits: usize) -> Option<PathBuf> {
        let params_dir = self.root_dir.join("parameters").join("factor");

        // Known CADO-NFS parameter file digit counts
        let available: Vec<usize> = {
            let mut avail = Vec::new();
            // Scan the directory for params.cN files
            if let Ok(entries) = std::fs::read_dir(&params_dir) {
                for entry in entries.flatten() {
                    let name = entry.file_name().to_string_lossy().to_string();
                    if let Some(rest) = name.strip_prefix("params.c") {
                        if let Ok(digits) = rest.parse::<usize>() {
                            avail.push(digits);
                        }
                    }
                }
            }
            avail.sort();
            avail
        };

        if available.is_empty() {
            return None;
        }

        // Find the nearest available digit count, strongly preferring
        // files for LARGER numbers (using params for too-small numbers means
        // insufficient sieving and factorization failure)
        let nearest = available
            .iter()
            .min_by_key(|&&d| {
                let diff = (d as i64 - num_digits as i64).unsigned_abs();
                if d >= num_digits {
                    // File for same or larger number: ideal
                    diff * 2
                } else {
                    // File for smaller number: penalize heavily
                    // (will likely fail due to insufficient relations)
                    diff * 10 + 100
                }
            })
            .copied()?;

        let path = params_dir.join(format!("params.c{}", nearest));
        if path.exists() {
            Some(path)
        } else {
            None
        }
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
        let num_digits = n_str.len();

        // Create temporary working directory for this run
        let workdir = TempDir::new().map_err(CadoError::Io)?;

        // Build command arguments
        let mut args = vec![
            self.cado_nfs_py.to_string_lossy().to_string(),
        ];

        // Add --parameters to specify the nearest parameter file
        // This is needed because CADO-NFS only looks ±3 digits from target
        let using_explicit_params = if let Some(param_file) = self.find_param_file(num_digits) {
            args.push("--parameters".to_string());
            args.push(param_file.to_string_lossy().to_string());
            true
        } else {
            false
        };

        // Number to factor (must come after --parameters but before key=value args)
        args.push(n_str.clone());

        // Add parameter overrides
        args.extend(params.to_cado_args());

        // When using --parameters, CADO-NFS doesn't auto-start local worker
        // clients. We must explicitly configure server whitelist and clients.
        if using_explicit_params {
            args.extend(self.server_client_args());
        }

        // Always use localhost to avoid .local hostname resolution issues on macOS
        args.extend(self.server_address_args());

        // Set working directory
        args.push(format!(
            "tasks.workdir={}",
            workdir.path().to_string_lossy()
        ));

        log::info!("Running CADO-NFS on N={} ({} bits, c{})", n_str, n.bits(), num_digits);
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

    /// Run CADO-NFS with its built-in default parameters (no overrides).
    ///
    /// Uses `--parameters` to specify the nearest available parameter file,
    /// but does NOT pass any parameter overrides. This is the correct way
    /// to run a baseline measurement.
    pub fn run_default(
        &self,
        n: &BigUint,
        timeout: Duration,
    ) -> Result<CadoResult, CadoError> {
        let n_str = n.to_string();
        let num_digits = n_str.len();

        let workdir = TempDir::new().map_err(CadoError::Io)?;

        let mut args = vec![
            self.cado_nfs_py.to_string_lossy().to_string(),
        ];

        // Add --parameters for the nearest parameter file.
        // When using --parameters, we must also configure server/client
        // because CADO-NFS doesn't auto-start local workers in that mode.
        let using_explicit_params = if let Some(param_file) = self.find_param_file(num_digits) {
            args.push("--parameters".to_string());
            args.push(param_file.to_string_lossy().to_string());
            true
        } else {
            false
        };

        args.push(n_str.clone());

        if using_explicit_params {
            args.extend(self.server_client_args());
        }

        // Always use localhost to avoid .local hostname resolution issues on macOS
        args.extend(self.server_address_args());

        args.push(format!(
            "tasks.workdir={}",
            workdir.path().to_string_lossy()
        ));

        log::info!(
            "Running CADO-NFS (default params) on N={} ({} bits, c{}) with {}s timeout",
            n_str, n.bits(), num_digits, timeout.as_secs()
        );

        self.spawn_and_wait(&args, &n_str, timeout, workdir)
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
        let num_digits = n_str.len();

        let workdir = TempDir::new().map_err(CadoError::Io)?;

        let mut args = vec![
            self.cado_nfs_py.to_string_lossy().to_string(),
        ];

        // Add --parameters for the nearest parameter file.
        // When using --parameters, we must also configure server/client.
        let using_explicit_params = if let Some(param_file) = self.find_param_file(num_digits) {
            args.push("--parameters".to_string());
            args.push(param_file.to_string_lossy().to_string());
            true
        } else {
            false
        };

        args.push(n_str.clone());
        args.extend(params.to_cado_args());

        if using_explicit_params {
            args.extend(self.server_client_args());
        }

        // Always use localhost to avoid .local hostname resolution issues on macOS
        args.extend(self.server_address_args());

        args.push(format!(
            "tasks.workdir={}",
            workdir.path().to_string_lossy()
        ));

        log::info!(
            "Running CADO-NFS on N={} ({} bits, c{}) with {}s timeout",
            n_str,
            n.bits(),
            num_digits,
            timeout.as_secs()
        );

        self.spawn_and_wait(&args, &n_str, timeout, workdir)
    }

    /// Spawn CADO-NFS process, wait with timeout, parse output.
    ///
    /// Uses process group signalling on Unix to ensure all child processes
    /// (server, clients, workers) are killed on timeout.
    fn spawn_and_wait(
        &self,
        args: &[String],
        n_str: &str,
        timeout: Duration,
        workdir: TempDir,
    ) -> Result<CadoResult, CadoError> {
        let _ = workdir; // keep alive for process duration

        let start = Instant::now();

        // Create a new process group so we can kill all CADO-NFS subprocesses.
        let mut cmd = Command::new(&self.python);
        cmd.args(args)
            .current_dir(&self.root_dir)
            .env("CADO_NFS_SOURCE_DIR", &self.root_dir)
            .stdout(std::process::Stdio::piped())
            .stderr(std::process::Stdio::piped());

        // On Unix, start in a new process group for clean kill
        #[cfg(unix)]
        {
            use std::os::unix::process::CommandExt;
            cmd.process_group(0);
        }

        let mut child = cmd
            .spawn()
            .map_err(|e| CadoError::ExecutionFailed(e.to_string()))?;

        // Take stdout/stderr handles before polling, so we can read them
        // after the process exits. wait_with_output() can fail if try_wait()
        // already consumed the exit status.
        let child_stdout = child.stdout.take();
        let child_stderr = child.stderr.take();

        // Poll for completion with timeout
        let poll_interval = Duration::from_millis(500);
        let child_pid = child.id() as i32;
        loop {
            match child.try_wait() {
                Ok(Some(_status)) => {
                    // Process finished — kill the process group to clean up
                    // daemon client processes that CADO-NFS leaves behind.
                    #[cfg(unix)]
                    {
                        unsafe {
                            libc::kill(-child_pid, libc::SIGTERM);
                        }
                        std::thread::sleep(Duration::from_millis(200));
                        unsafe {
                            libc::kill(-child_pid, libc::SIGKILL);
                        }
                    }
                    break;
                }
                Ok(None) => {
                    // Still running
                    if start.elapsed() > timeout {
                        log::warn!("CADO-NFS timed out, killing process group");
                        // Kill the entire process group (server + clients)
                        #[cfg(unix)]
                        {
                            unsafe {
                                libc::kill(-child_pid, libc::SIGTERM);
                            }
                            std::thread::sleep(Duration::from_millis(500));
                            unsafe {
                                libc::kill(-child_pid, libc::SIGKILL);
                            }
                        }
                        #[cfg(not(unix))]
                        {
                            let _ = child.kill();
                        }
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

        // Read stdout/stderr from the handles we took earlier
        use std::io::Read;
        let stdout = if let Some(mut out) = child_stdout {
            let mut s = String::new();
            let _ = out.read_to_string(&mut s);
            s
        } else {
            String::new()
        };
        let stderr = if let Some(mut err) = child_stderr {
            let mut s = String::new();
            let _ = err.read_to_string(&mut s);
            s
        } else {
            String::new()
        };
        let combined_log = format!("{}\n{}", stdout, stderr);

        log::debug!(
            "CADO-NFS stdout ({} bytes), stderr ({} bytes)",
            stdout.len(),
            stderr.len()
        );
        log::debug!("CADO-NFS stdout: {:?}", stdout.trim());

        let mut result = parse_cado_output(&combined_log);
        result.n = n_str.to_string();
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
        phase_cpu_times: HashMap::new(),
        relations_found: None,
        matrix_size: None,
        unique_relations: None,
        log_tail: String::new(),
    };

    // CADO-NFS outputs factors on stdout as space-separated numbers.
    // When combined with stderr, factors may be at the BEGINNING of the
    // combined log (stdout first, then stderr) or at the END (if stderr
    // comes first). Check both the first 5 and last 5 non-empty lines.
    let lines: Vec<&str> = log.lines().collect();
    let check_lines: Vec<&str> = lines
        .iter()
        .rev()
        .take(5)
        .chain(lines.iter().take(5))
        .copied()
        .collect();

    for line in &check_lines {
        let trimmed = line.trim();
        if let Some(factors_str) = extract_factors_space_separated(trimmed) {
            result.factors = factors_str;
            result.success = true;
            break;
        }
    }

    for line in log.lines() {
        let trimmed = line.trim();

        // Look for factor output lines (format: "N = p1 * p2")
        if !result.success {
            if let Some(factors_str) = extract_factors_from_line(trimmed) {
                result.factors = factors_str;
                if !result.factors.is_empty() {
                    result.success = true;
                }
            }
        }

        // Phase timing from CADO-NFS log lines.
        // CADO-NFS wraps these in ANSI color codes, so strip them.
        // Real format after stripping:
        //   "Info:Lattice Sieving: Total time: 3.02s"
        //   "Info:Filtering - Singleton removal: Total cpu/real time for purge: 0.29/0.157198"
        //   "Info:Linear Algebra: Total cpu/real time for bwc: 0.47/1.22"
        //   "Info:Complete Factorization / ...: Total cpu/elapsed time for entire ... 9.21/12.4287"
        let clean_line = strip_ansi(trimmed);
        if clean_line.contains("Total cpu/real time for ")
            || clean_line.contains("Total cpu/elapsed time for ")
            || clean_line.contains("Total time:")
        {
            if let Some((phase, real_secs, cpu_secs)) = parse_phase_timing(&clean_line) {
                result
                    .phase_times
                    .insert(phase.clone(), Duration::from_secs_f64(real_secs));
                if let Some(cpu) = cpu_secs {
                    result
                        .phase_cpu_times
                        .insert(phase, Duration::from_secs_f64(cpu));
                }
            }
        }

        // Relation counts
        // "Sieving: found 12345 relations"
        // "Total number of relations: 52609"
        // "found 12345 relations"
        if trimmed.contains("rels found")
            || trimmed.contains("relations found")
            || trimmed.contains("number of relations")
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

/// Extract factors from CADO-NFS's space-separated output.
///
/// CADO-NFS outputs factors on the last line as: "p1 p2 [p3 ...]"
/// where each token is a large integer.
fn extract_factors_space_separated(line: &str) -> Option<Vec<String>> {
    // Strip ANSI escape codes
    let clean = strip_ansi(line);
    let trimmed = clean.trim();

    // Must be 2+ space-separated numbers, each at least 2 digits
    let parts: Vec<&str> = trimmed.split_whitespace().collect();
    if parts.len() < 2 {
        return None;
    }

    let mut factors = Vec::new();
    for part in &parts {
        if part.len() >= 2 && part.chars().all(|c| c.is_ascii_digit()) {
            factors.push(part.to_string());
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

/// Parse phase timing from CADO-NFS timing lines.
///
/// Handles real CADO-NFS output formats (after ANSI stripping):
///
/// 1. `Info:<Phase>: Total cpu/real time for <sub>: cpu/real`
///    e.g. "Info:Linear Algebra: Total cpu/real time for bwc: 0.47/1.22"
///
/// 2. `Info:<Phase>: Total cpu/elapsed time for ... cpu/real`
///    e.g. "Info:Complete Factorization ...: Total cpu/elapsed time for entire ... 9.21/12.4287"
///
/// 3. `Info:<Phase>: Total time: <secs>s`
///    e.g. "Info:Lattice Sieving: Total time: 3.02s"
///    e.g. "Info:Polynomial Selection (root optimized): Total time: 1.9"
///
/// Returns `(phase_name, real_seconds, Option<cpu_seconds>)`.
/// Patterns 1 & 2 report both CPU and real time (cpu_seconds is `Some`).
/// Pattern 3 reports only wall-clock time (cpu_seconds is `None`).
///
/// The phase name is extracted from the `Info:` prefix (the part between
/// first "Info:" and the "Total" keyword), which is more reliable than
/// parsing the sub-component name after "Total cpu/real time for".
fn parse_phase_timing(line: &str) -> Option<(String, f64, Option<f64>)> {
    // Extract phase name from the Info: prefix if present.
    // Format: "Info:<Phase>: Total ..."
    let phase_name = extract_cado_phase_name(line);

    // Pattern 1: "Total cpu/real time for <sub>: cpu/real"
    if let Some(pos) = line.find("Total cpu/real time for ") {
        let after = &line[pos + "Total cpu/real time for ".len()..];
        // Find the timing value: look for "cpu/real" format after the sub-component
        // Sub-component may be followed by ": cpu/real" or just "cpu/real"
        if let Some(colon_pos) = after.find(':') {
            let timing = after[colon_pos + 1..].trim();
            if let Some((cpu, real)) = extract_cpu_and_real_time(timing) {
                return Some((phase_name.unwrap_or_else(|| {
                    after[..colon_pos].trim().to_string()
                }), real, Some(cpu)));
            }
        }
        // Fallback: look for last token with '/'
        let parts: Vec<&str> = after.split_whitespace().collect();
        for part in parts.iter().rev() {
            if part.contains('/') {
                if let Some((cpu, real)) = extract_cpu_and_real_time(part) {
                    return Some((phase_name.unwrap_or_else(|| "unknown".to_string()), real, Some(cpu)));
                }
            }
        }
    }

    // Pattern 2: "Total cpu/elapsed time for ... cpu/real"
    if let Some(pos) = line.find("Total cpu/elapsed time for ") {
        let after = &line[pos + "Total cpu/elapsed time for ".len()..];
        let parts: Vec<&str> = after.split_whitespace().collect();
        for part in parts.iter().rev() {
            if part.contains('/') {
                if let Some((cpu, real)) = extract_cpu_and_real_time(part) {
                    let phase = phase_name.unwrap_or_else(|| "Complete Factorization".to_string());
                    return Some((phase, real, Some(cpu)));
                }
            }
        }
    }

    // Pattern 3: "Total time: <secs>s" — no CPU time available
    if let Some(pos) = line.find("Total time:") {
        let after = &line[pos + "Total time:".len()..];
        if let Some(secs) = extract_seconds(after) {
            return Some((phase_name.unwrap_or_else(|| "unknown".to_string()), secs, None));
        }
    }

    None
}

/// Extract the phase name from a CADO-NFS log line.
///
/// CADO-NFS prefixes lines with `Info:<PhaseGroup>:` or just the phase
/// group directly. This extracts the phase group name.
///
/// Examples:
/// - "Info:Lattice Sieving: Total time: 3.02s" → Some("Lattice Sieving")
/// - "Info:Linear Algebra: Total cpu/real time for bwc: ..." → Some("Linear Algebra")
/// - "Info:Filtering - Singleton removal: Total ..." → Some("Filtering - Singleton removal")
fn extract_cado_phase_name(line: &str) -> Option<String> {
    // Strip "Info:" prefix if present
    let rest = line.strip_prefix("Info:").unwrap_or(line);

    // Find where "Total" appears — phase name is everything before it
    let total_pos = rest.find("Total ")?;
    if total_pos == 0 {
        return None;
    }

    // Phase name is the text before "Total", trimmed of trailing ": "
    let phase = rest[..total_pos].trim().trim_end_matches(':').trim();
    if phase.is_empty() {
        return None;
    }

    Some(phase.to_string())
}

/// Extract both CPU and real/wall-clock time from a "cpu/real" timing string.
///
/// Returns `(cpu_seconds, real_seconds)`.
/// Input format: "0.47/1.22" or "9.21/12.4287"
fn extract_cpu_and_real_time(timing: &str) -> Option<(f64, f64)> {
    let trimmed = timing.trim().trim_matches('s');
    if !trimmed.contains('/') {
        return None;
    }
    let parts: Vec<&str> = trimmed.split('/').collect();
    if parts.len() != 2 {
        return None;
    }
    let cpu = parts[0].parse::<f64>().ok()?;
    let real = parts[1].parse::<f64>().ok()?;
    Some((cpu, real))
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

/// Strip ANSI escape codes from a string.
pub fn strip_ansi(s: &str) -> String {
    let mut result = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c == '\x1b' {
            // Skip until we find 'm' (end of ANSI sequence)
            for c2 in chars.by_ref() {
                if c2 == 'm' {
                    break;
                }
            }
        } else {
            result.push(c);
        }
    }
    result
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
    fn test_parse_phase_timing_bare() {
        // Old format: bare "Total cpu/real time for ..." without Info: prefix
        let line = "Total cpu/real time for polyselect: 12.34/5.67";
        let result = parse_phase_timing(line);
        assert!(result.is_some());
        let (phase, real, cpu) = result.unwrap();
        assert_eq!(phase, "polyselect");
        assert!((real - 5.67).abs() < 0.01);
        assert!(cpu.is_some());
        assert!((cpu.unwrap() - 12.34).abs() < 0.01);
    }

    #[test]
    fn test_parse_phase_timing_sieve_bare() {
        let line = "Total cpu/real time for sieve: 100.5/25.3";
        let result = parse_phase_timing(line);
        assert!(result.is_some());
        let (phase, real, cpu) = result.unwrap();
        assert_eq!(phase, "sieve");
        assert!((real - 25.3).abs() < 0.01);
        assert!(cpu.is_some());
        assert!((cpu.unwrap() - 100.5).abs() < 0.01);
    }

    #[test]
    fn test_parse_phase_timing_real_linalg() {
        // Real CADO-NFS format with Info: prefix
        let line = "Info:Linear Algebra: Total cpu/real time for bwc: 0.47/1.22";
        let result = parse_phase_timing(line);
        assert!(result.is_some());
        let (phase, real, cpu) = result.unwrap();
        assert_eq!(phase, "Linear Algebra");
        assert!((real - 1.22).abs() < 0.01);
        assert!(cpu.is_some());
        assert!((cpu.unwrap() - 0.47).abs() < 0.01);
    }

    #[test]
    fn test_parse_phase_timing_real_sqrt() {
        let line = "Info:Square Root: Total cpu/real time for sqrt: 0.11/0.102183";
        let result = parse_phase_timing(line);
        assert!(result.is_some());
        let (phase, real, cpu) = result.unwrap();
        assert_eq!(phase, "Square Root");
        assert!((real - 0.102).abs() < 0.01);
        assert!(cpu.is_some());
        assert!((cpu.unwrap() - 0.11).abs() < 0.01);
    }

    #[test]
    fn test_parse_phase_timing_real_sieving_total_time() {
        // "Total time:" format (no cpu/real) — cpu_secs should be None
        let line = "Info:Lattice Sieving: Total time: 3.02s";
        let result = parse_phase_timing(line);
        assert!(result.is_some());
        let (phase, real, cpu) = result.unwrap();
        assert_eq!(phase, "Lattice Sieving");
        assert!((real - 3.02).abs() < 0.01);
        assert!(cpu.is_none(), "Pattern 3 should not report CPU time");
    }

    #[test]
    fn test_parse_phase_timing_real_polyselect() {
        let line = "Info:Polynomial Selection (root optimized): Total time: 1.9";
        let result = parse_phase_timing(line);
        assert!(result.is_some());
        let (phase, real, cpu) = result.unwrap();
        assert_eq!(phase, "Polynomial Selection (root optimized)");
        assert!((real - 1.9).abs() < 0.01);
        assert!(cpu.is_none(), "Pattern 3 should not report CPU time");
    }

    #[test]
    fn test_parse_phase_timing_real_filtering() {
        let line = "Info:Filtering - Singleton removal: Total cpu/real time for purge: 0.29/0.157198";
        let result = parse_phase_timing(line);
        assert!(result.is_some());
        let (phase, real, cpu) = result.unwrap();
        assert_eq!(phase, "Filtering - Singleton removal");
        assert!((real - 0.157).abs() < 0.01);
        assert!(cpu.is_some());
        assert!((cpu.unwrap() - 0.29).abs() < 0.01);
    }

    #[test]
    fn test_parse_phase_timing_real_complete() {
        let line = "Info:Complete Factorization / Discrete logarithm: Total cpu/elapsed time for entire Complete Factorization 9.21/12.4287";
        let result = parse_phase_timing(line);
        assert!(result.is_some());
        let (phase, real, cpu) = result.unwrap();
        assert!(phase.contains("Complete Factorization"));
        assert!((real - 12.43).abs() < 0.01);
        assert!(cpu.is_some());
        assert!((cpu.unwrap() - 9.21).abs() < 0.01);
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
Info:Polynomial Selection (root optimized): Total time: 0.5
Info:Lattice Sieving: Total time: 12.3s
found 50000 relations
unique relations: 42000
matrix is 3000 x 3200
Info:Linear Algebra: Total cpu/real time for bwc: 5.4/2.1
Info:Square Root: Total cpu/real time for sqrt: 0.8/0.3
110503 = 233 * 474
"#;
        let result = parse_cado_output(log);
        assert!(result.success);
        assert_eq!(result.factors.len(), 2);
        assert_eq!(result.relations_found, Some(50000));
        assert_eq!(result.unique_relations, Some(42000));
        assert_eq!(result.matrix_size, Some((3000, 3200)));
        // Wall-clock times populated for all phases
        assert!(result.phase_times.contains_key("Polynomial Selection (root optimized)"));
        assert!(result.phase_times.contains_key("Lattice Sieving"));
        assert!(result.phase_times.contains_key("Linear Algebra"));
        assert!(result.phase_times.contains_key("Square Root"));
        // CPU times only populated for Patterns 1 & 2 (cpu/real format)
        assert!(!result.phase_cpu_times.contains_key("Polynomial Selection (root optimized)"),
            "Pattern 3 phases should not have CPU time");
        assert!(!result.phase_cpu_times.contains_key("Lattice Sieving"),
            "Pattern 3 phases should not have CPU time");
        assert!(result.phase_cpu_times.contains_key("Linear Algebra"),
            "Pattern 1 phases should have CPU time");
        assert!(result.phase_cpu_times.contains_key("Square Root"),
            "Pattern 1 phases should have CPU time");
        // Verify CPU time values
        let linalg_cpu = result.phase_cpu_times["Linear Algebra"].as_secs_f64();
        assert!((linalg_cpu - 5.4).abs() < 0.01);
        let sqrt_cpu = result.phase_cpu_times["Square Root"].as_secs_f64();
        assert!((sqrt_cpu - 0.8).abs() < 0.01);
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

    #[test]
    fn test_extract_factors_space_separated() {
        let line = "824640114685687 1030343599805129";
        let factors = extract_factors_space_separated(line);
        assert!(factors.is_some());
        let factors = factors.unwrap();
        assert_eq!(factors.len(), 2);
        assert_eq!(factors[0], "824640114685687");
        assert_eq!(factors[1], "1030343599805129");
    }

    #[test]
    fn test_extract_factors_space_separated_rejects_words() {
        let line = "Info: Starting factorization";
        assert!(extract_factors_space_separated(line).is_none());
    }

    #[test]
    fn test_strip_ansi() {
        let ansi = "\x1b[32;1mInfo\x1b[0m:Total cpu/real time for sqrt: 0.1/0.39";
        let clean = strip_ansi(ansi);
        assert_eq!(clean, "Info:Total cpu/real time for sqrt: 0.1/0.39");
    }

    #[test]
    fn test_parse_phase_timing_elapsed() {
        let line = "Total cpu/elapsed time for entire Complete Factorization 2.68/13.47";
        let result = parse_phase_timing(line);
        assert!(result.is_some());
        let (phase, real, cpu) = result.unwrap();
        assert!(phase.contains("Complete Factorization"));
        assert!((real - 13.47).abs() < 0.01);
        assert!(cpu.is_some());
        assert!((cpu.unwrap() - 2.68).abs() < 0.01);
    }

    #[test]
    fn test_parse_real_cado_output() {
        // Actual CADO-NFS output from a 100-bit factorization
        let log = "\x1b[32;1mInfo\x1b[0m:Lattice Sieving: Total number of relations: 52609\n\
                   \x1b[32;1mInfo\x1b[0m:Lattice Sieving: Total time: 0.98s\n\
                   \x1b[32;1mInfo\x1b[0m:Linear Algebra: Total cpu/real time for bwc: 0.38/1.19\n\
                   \x1b[32;1mInfo\x1b[0m:Square Root: Total cpu/real time for sqrt: 0.1/0.39\n\
                   \x1b[32;1mInfo\x1b[0m:Complete Factorization / Discrete logarithm: Total cpu/elapsed time for entire Complete Factorization 2.68/13.47\n\
                   824640114685687 1030343599805129\n";

        let result = parse_cado_output(log);
        assert!(result.success);
        assert_eq!(result.factors.len(), 2);
        assert_eq!(result.factors[0], "824640114685687");
        assert_eq!(result.factors[1], "1030343599805129");
        assert_eq!(result.relations_found, Some(52609));

        // Verify CPU times are populated for cpu/real format phases
        assert!(result.phase_cpu_times.contains_key("Linear Algebra"));
        assert!(result.phase_cpu_times.contains_key("Square Root"));
        assert!(result.phase_cpu_times.contains_key("Complete Factorization / Discrete logarithm"));
        // Lattice Sieving uses "Total time:" format — no CPU time
        assert!(!result.phase_cpu_times.contains_key("Lattice Sieving"));

        let linalg_cpu = result.phase_cpu_times["Linear Algebra"].as_secs_f64();
        assert!((linalg_cpu - 0.38).abs() < 0.01);
        let complete_cpu = result.phase_cpu_times["Complete Factorization / Discrete logarithm"].as_secs_f64();
        assert!((complete_cpu - 2.68).abs() < 0.01);
    }

    #[test]
    fn test_extract_cpu_and_real_time() {
        // Standard cpu/real format
        assert_eq!(extract_cpu_and_real_time("0.47/1.22"), Some((0.47, 1.22)));
        assert_eq!(extract_cpu_and_real_time("9.21/12.4287"), Some((9.21, 12.4287)));
        // Bare seconds — not cpu/real format
        assert_eq!(extract_cpu_and_real_time("3.02"), None);
        assert_eq!(extract_cpu_and_real_time("3.02s"), None);
        // Empty
        assert_eq!(extract_cpu_and_real_time(""), None);
        // Malformed
        assert_eq!(extract_cpu_and_real_time("abc/def"), None);
    }
}
