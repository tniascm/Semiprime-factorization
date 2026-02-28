//! Structured JSONL event logging with UTC timestamps.

use std::fs::{File, OpenOptions};
use std::io::Write;
use std::path::PathBuf;

use chrono::Utc;
use serde_json::Value;

/// JSONL event logger with UTC timestamps.
pub struct EventLogger {
    file: Option<File>,
}

impl EventLogger {
    /// Create a new logger writing to the given path.
    /// Creates parent directories if needed.
    pub fn new(path: &PathBuf) -> std::io::Result<Self> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok(Self { file: Some(file) })
    }

    /// Create a no-op logger (for tests or when no output path is set).
    pub fn noop() -> Self {
        Self { file: None }
    }

    /// Log a structured event with UTC timestamp.
    pub fn log(&mut self, event: &str, data: Value) {
        let ts = Utc::now().to_rfc3339_opts(chrono::SecondsFormat::Millis, true);
        let mut obj = serde_json::Map::new();
        obj.insert("ts".to_string(), Value::String(ts.clone()));
        obj.insert("event".to_string(), Value::String(event.to_string()));

        if let Value::Object(map) = data {
            for (k, v) in map {
                obj.insert(k, v);
            }
        }

        let line = serde_json::to_string(&Value::Object(obj)).unwrap_or_default();

        if let Some(ref mut f) = self.file {
            let _ = writeln!(f, "{}", line);
            let _ = f.flush();
        }

        // Also print errors to stderr for visibility
        if event == "error" {
            eprintln!("[ERROR] {}: {}", ts, line);
        }
    }
}

/// Write a checkpoint file (just the semiprime index).
pub fn write_checkpoint(output_dir: &PathBuf, index: usize) -> std::io::Result<()> {
    let path = output_dir.join("checkpoint.txt");
    std::fs::write(path, format!("{}\n", index))
}

/// Read checkpoint file, returning the next index to process.
pub fn read_checkpoint(output_dir: &PathBuf) -> Option<usize> {
    let path = output_dir.join("checkpoint.txt");
    std::fs::read_to_string(path)
        .ok()
        .and_then(|s| s.trim().parse::<usize>().ok())
}
