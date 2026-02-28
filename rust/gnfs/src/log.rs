use serde::Serialize;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{Instant, SystemTime, UNIX_EPOCH};

/// Structured logger for a GNFS pipeline stage.
pub struct StageLogger {
    stage: String,
    start: Instant,
    log_file: Option<BufWriter<File>>,
}

/// Format current time as HH:MM:SS UTC.
pub fn utc_timestamp() -> String {
    let secs = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let h = (secs / 3600) % 24;
    let m = (secs / 60) % 60;
    let s = secs % 60;
    format!("{:02}:{:02}:{:02} UTC", h, m, s)
}

/// Format current time as ISO 8601 for JSON logs.
fn iso_timestamp() -> String {
    let d = SystemTime::now().duration_since(UNIX_EPOCH).unwrap();
    let secs = d.as_secs();
    let day = secs / 86400;
    let time_of_day = secs % 86400;
    let y = 1970 + day / 365;
    let remaining = day % 365;
    let mo = remaining / 30 + 1;
    let d = remaining % 30 + 1;
    let h = time_of_day / 3600;
    let m = (time_of_day % 3600) / 60;
    let s = time_of_day % 60;
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}Z", y, mo, d, h, m, s)
}

impl StageLogger {
    /// Create a new logger. If output_dir is Some, writes log.jsonl there.
    pub fn new(stage: &str, output_dir: Option<&Path>) -> Self {
        let log_file = output_dir.map(|dir| {
            let stage_dir = dir.join(format!("stage_{}", stage));
            fs::create_dir_all(&stage_dir).ok();
            let path = stage_dir.join("log.jsonl");
            BufWriter::new(File::create(path).expect("Failed to create log file"))
        });
        Self {
            stage: stage.to_string(),
            start: Instant::now(),
            log_file,
        }
    }

    /// Log a console message with timestamp.
    pub fn log(&mut self, msg: &str) {
        let elapsed = self.start.elapsed().as_secs_f64();
        println!("[{}] [{}] {}", utc_timestamp(), self.stage, msg);

        if let Some(ref mut f) = self.log_file {
            let entry = serde_json::json!({
                "ts": iso_timestamp(),
                "elapsed_s": elapsed,
                "stage": self.stage,
                "event": "log",
                "msg": msg,
            });
            writeln!(f, "{}", entry).ok();
            f.flush().ok();
        }
    }

    /// Log stage start with parameters.
    pub fn start(&mut self, params: &impl Serialize) {
        let elapsed = self.start.elapsed().as_secs_f64();
        println!("[{}] [{}] START", utc_timestamp(), self.stage);

        if let Some(ref mut f) = self.log_file {
            let entry = serde_json::json!({
                "ts": iso_timestamp(),
                "elapsed_s": elapsed,
                "stage": self.stage,
                "event": "start",
                "params": serde_json::to_value(params).unwrap_or_default(),
            });
            writeln!(f, "{}", entry).ok();
            f.flush().ok();
        }
    }

    /// Log stage completion with summary.
    pub fn finish(&mut self, summary: &impl Serialize) {
        let elapsed = self.start.elapsed().as_secs_f64();
        println!(
            "[{}] [{}] DONE — {:.1}s",
            utc_timestamp(),
            self.stage,
            elapsed
        );

        if let Some(ref mut f) = self.log_file {
            let entry = serde_json::json!({
                "ts": iso_timestamp(),
                "elapsed_s": elapsed,
                "stage": self.stage,
                "event": "finish",
                "summary": serde_json::to_value(summary).unwrap_or_default(),
            });
            writeln!(f, "{}", entry).ok();
            f.flush().ok();
        }
    }

    /// Write a checkpoint file and log the event.
    pub fn checkpoint(&mut self, state: &impl Serialize, output_dir: &Path, filename: &str) {
        let stage_dir = output_dir.join(format!("stage_{}", self.stage));
        fs::create_dir_all(&stage_dir).ok();
        let path = stage_dir.join(filename);
        let json = serde_json::to_string_pretty(state).unwrap();
        fs::write(&path, json).expect("Failed to write checkpoint");

        self.log(&format!("Checkpoint written: {}", filename));
    }

    /// Elapsed seconds since stage start.
    pub fn elapsed_secs(&self) -> f64 {
        self.start.elapsed().as_secs_f64()
    }
}

/// Set up the run directory and write run_config.json.
pub fn setup_run_dir(base_dir: &str, n_digits: u32, seed: u64) -> PathBuf {
    let epoch = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs();
    let dir_name = format!("c{}_seed{}_{}", n_digits, seed, epoch);
    let run_dir = PathBuf::from(base_dir).join(dir_name);
    fs::create_dir_all(&run_dir).expect("Failed to create run directory");
    run_dir
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_utc_timestamp_format() {
        let ts = utc_timestamp();
        assert!(ts.ends_with(" UTC"));
        assert_eq!(ts.len(), 12);
    }

    #[test]
    fn test_stage_logger_no_file() {
        let mut logger = StageLogger::new("test", None);
        logger.log("hello world");
    }
}
