//! Lightweight observability for the NFS pipeline.
//!
//! Provides [`StageTimer`] for tracking individual stage durations (with
//! optional timeout), [`StageResult`] for finished stages, and
//! [`PipelineTimings`] for collecting all stage results with JSON and
//! human-readable summary output.

use std::fmt;
use std::time::{Duration, Instant};

/// Timer for a single pipeline stage with optional timeout.
pub struct StageTimer {
    name: String,
    start: Instant,
    timeout: Option<Duration>,
    sub_stages: Vec<(String, Duration)>,
}

/// Error returned when a stage exceeds its timeout.
#[derive(Debug)]
pub struct StageTimeoutError {
    pub stage: String,
    pub elapsed_ms: f64,
    pub timeout_ms: f64,
}

impl fmt::Display for StageTimeoutError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "stage '{}' timed out: {:.0}ms elapsed, {:.0}ms limit",
            self.stage, self.elapsed_ms, self.timeout_ms
        )
    }
}

impl std::error::Error for StageTimeoutError {}

/// Result of a completed stage.
pub struct StageResult {
    pub name: String,
    pub total_ms: f64,
    pub sub_stages: Vec<(String, f64)>,
    pub timed_out: bool,
}

/// Collection of all stage timings for a pipeline run.
pub struct PipelineTimings {
    pub stages: Vec<StageResult>,
    pub total_ms: f64,
}

impl StageTimer {
    /// Create a new timer that starts immediately.
    ///
    /// If `timeout_ms` is `Some`, [`check_timeout`](Self::check_timeout) will
    /// return an error once the elapsed time exceeds the given limit.
    pub fn new(name: &str, timeout_ms: Option<u64>) -> Self {
        Self {
            name: name.to_string(),
            start: Instant::now(),
            timeout: timeout_ms.map(Duration::from_millis),
            sub_stages: Vec::new(),
        }
    }

    /// Milliseconds elapsed since the timer was created.
    pub fn elapsed_ms(&self) -> f64 {
        self.start.elapsed().as_secs_f64() * 1000.0
    }

    /// Returns `Err` if the elapsed time exceeds the configured timeout.
    ///
    /// Always returns `Ok(())` when no timeout was set.
    pub fn check_timeout(&self) -> Result<(), StageTimeoutError> {
        if let Some(limit) = self.timeout {
            let elapsed = self.start.elapsed();
            if elapsed > limit {
                return Err(StageTimeoutError {
                    stage: self.name.clone(),
                    elapsed_ms: elapsed.as_secs_f64() * 1000.0,
                    timeout_ms: limit.as_secs_f64() * 1000.0,
                });
            }
        }
        Ok(())
    }

    /// Record a sub-stage timing.
    pub fn sub_stage(&mut self, name: &str, duration: Duration) {
        self.sub_stages.push((name.to_string(), duration));
    }

    /// Consume the timer and produce a [`StageResult`].
    pub fn finish(self) -> StageResult {
        StageResult {
            name: self.name,
            total_ms: self.start.elapsed().as_secs_f64() * 1000.0,
            sub_stages: self
                .sub_stages
                .into_iter()
                .map(|(n, d)| (n, d.as_secs_f64() * 1000.0))
                .collect(),
            timed_out: false,
        }
    }

    /// Like [`finish`](Self::finish) but marks the result as timed out.
    pub fn finish_timed_out(self) -> StageResult {
        StageResult {
            name: self.name,
            total_ms: self.start.elapsed().as_secs_f64() * 1000.0,
            sub_stages: self
                .sub_stages
                .into_iter()
                .map(|(n, d)| (n, d.as_secs_f64() * 1000.0))
                .collect(),
            timed_out: true,
        }
    }
}

impl PipelineTimings {
    /// Create an empty collection.
    pub fn new() -> Self {
        Self {
            stages: Vec::new(),
            total_ms: 0.0,
        }
    }

    /// Push a completed stage result.
    pub fn add(&mut self, stage: StageResult) {
        self.stages.push(stage);
    }

    /// Set the overall pipeline total (typically from `NfsResult.total_ms`).
    pub fn set_total(&mut self, total_ms: f64) {
        self.total_ms = total_ms;
    }

    /// One-line human-readable summary.
    ///
    /// Example: `"poly=176ms sieve=256ms filter=22ms la=249ms sqrt=69ms total=772ms"`
    pub fn summary_line(&self) -> String {
        let mut parts: Vec<String> = self
            .stages
            .iter()
            .map(|s| format!("{}={:.0}ms", s.name, s.total_ms))
            .collect();
        parts.push(format!("total={:.0}ms", self.total_ms));
        parts.join(" ")
    }

    /// Manually-built JSON representation (no serde_json dependency).
    ///
    /// Output matches:
    /// ```json
    /// {"stages":[{"name":"polyselect","total_ms":176.0,"sub_stages":[],"timed_out":false}],"total_ms":772.0}
    /// ```
    pub fn to_json(&self) -> String {
        let mut out = String::from("{\"stages\":[");
        for (i, stage) in self.stages.iter().enumerate() {
            if i > 0 {
                out.push(',');
            }
            out.push_str("{\"name\":\"");
            out.push_str(&escape_json_string(&stage.name));
            out.push_str("\",\"total_ms\":");
            out.push_str(&format_f64(stage.total_ms));
            out.push_str(",\"sub_stages\":[");
            for (j, (sub_name, sub_ms)) in stage.sub_stages.iter().enumerate() {
                if j > 0 {
                    out.push(',');
                }
                out.push_str("[\"");
                out.push_str(&escape_json_string(sub_name));
                out.push_str("\",");
                out.push_str(&format_f64(*sub_ms));
                out.push(']');
            }
            out.push_str("],\"timed_out\":");
            out.push_str(if stage.timed_out { "true" } else { "false" });
            out.push('}');
        }
        out.push_str("],\"total_ms\":");
        out.push_str(&format_f64(self.total_ms));
        out.push('}');
        out
    }
}

impl Default for PipelineTimings {
    fn default() -> Self {
        Self::new()
    }
}

/// Escape a string for safe embedding in JSON.
fn escape_json_string(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out
}

/// Format an f64 for JSON output — integer values get a `.0` suffix.
fn format_f64(v: f64) -> String {
    if v.fract() == 0.0 && v.is_finite() {
        format!("{:.1}", v)
    } else {
        // Use default Display which avoids unnecessary trailing zeros
        // but always includes a decimal point for non-integer values.
        let s = format!("{}", v);
        if s.contains('.') || s.contains('e') || s.contains('E') {
            s
        } else {
            format!("{}.0", s)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::time::Duration;

    #[test]
    fn stage_timer_basic() {
        let timer = StageTimer::new("test", None);
        assert!(timer.elapsed_ms() >= 0.0);
        let result = timer.finish();
        assert_eq!(result.name, "test");
        assert!(!result.timed_out);
        assert!(result.sub_stages.is_empty());
    }

    #[test]
    fn stage_timer_with_sub_stages() {
        let mut timer = StageTimer::new("sieve", None);
        timer.sub_stage("bucket", Duration::from_millis(10));
        timer.sub_stage("scan", Duration::from_millis(20));
        let result = timer.finish();
        assert_eq!(result.sub_stages.len(), 2);
        assert_eq!(result.sub_stages[0].0, "bucket");
        assert!((result.sub_stages[0].1 - 10.0).abs() < 0.5);
        assert_eq!(result.sub_stages[1].0, "scan");
        assert!((result.sub_stages[1].1 - 20.0).abs() < 0.5);
    }

    #[test]
    fn stage_timer_no_timeout() {
        let timer = StageTimer::new("test", None);
        assert!(timer.check_timeout().is_ok());
    }

    #[test]
    fn stage_timer_within_timeout() {
        let timer = StageTimer::new("test", Some(60_000));
        assert!(timer.check_timeout().is_ok());
    }

    #[test]
    fn stage_timer_finish_timed_out() {
        let timer = StageTimer::new("test", None);
        let result = timer.finish_timed_out();
        assert!(result.timed_out);
    }

    #[test]
    fn pipeline_timings_summary() {
        let mut timings = PipelineTimings::new();
        timings.add(StageResult {
            name: "poly".to_string(),
            total_ms: 100.0,
            sub_stages: vec![],
            timed_out: false,
        });
        timings.add(StageResult {
            name: "sieve".to_string(),
            total_ms: 200.0,
            sub_stages: vec![],
            timed_out: false,
        });
        timings.set_total(300.0);
        assert_eq!(timings.summary_line(), "poly=100ms sieve=200ms total=300ms");
    }

    #[test]
    fn pipeline_timings_json() {
        let mut timings = PipelineTimings::new();
        timings.add(StageResult {
            name: "poly".to_string(),
            total_ms: 176.0,
            sub_stages: vec![],
            timed_out: false,
        });
        timings.set_total(176.0);
        let json = timings.to_json();
        assert!(json.contains("\"name\":\"poly\""));
        assert!(json.contains("\"total_ms\":176.0"));
        assert!(json.contains("\"timed_out\":false"));
        assert!(json.contains("\"sub_stages\":[]"));
    }

    #[test]
    fn pipeline_timings_json_with_sub_stages() {
        let mut timings = PipelineTimings::new();
        timings.add(StageResult {
            name: "sieve".to_string(),
            total_ms: 256.0,
            sub_stages: vec![
                ("bucket".to_string(), 50.0),
                ("scan".to_string(), 100.0),
            ],
            timed_out: false,
        });
        timings.set_total(256.0);
        let json = timings.to_json();
        assert!(json.contains("[\"bucket\",50.0]"));
        assert!(json.contains("[\"scan\",100.0]"));
    }

    #[test]
    fn pipeline_timings_empty() {
        let timings = PipelineTimings::new();
        assert_eq!(timings.summary_line(), "total=0ms");
        assert_eq!(timings.to_json(), "{\"stages\":[],\"total_ms\":0.0}");
    }

    #[test]
    fn escape_json_string_special_chars() {
        assert_eq!(escape_json_string("hello"), "hello");
        assert_eq!(escape_json_string("a\"b"), "a\\\"b");
        assert_eq!(escape_json_string("a\\b"), "a\\\\b");
        assert_eq!(escape_json_string("a\nb"), "a\\nb");
    }

    #[test]
    fn format_f64_values() {
        assert_eq!(format_f64(176.0), "176.0");
        assert_eq!(format_f64(0.0), "0.0");
        assert_eq!(format_f64(3.14), "3.14");
    }

    #[test]
    fn stage_timeout_error_display() {
        let err = StageTimeoutError {
            stage: "sieve".to_string(),
            elapsed_ms: 5000.0,
            timeout_ms: 3000.0,
        };
        let msg = format!("{}", err);
        assert!(msg.contains("sieve"));
        assert!(msg.contains("5000"));
        assert!(msg.contains("3000"));
    }
}
