use std::io::Write;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const MAX_LOG_BYTES: u64 = 1_048_576;
const ROTATED_LOGS: usize = 3;

pub fn log_default(message: impl AsRef<str>) {
    let data_dir = default_data_dir();
    let _ = log(&data_dir, message);
}

pub fn log(data_dir: &Path, message: impl AsRef<str>) -> std::io::Result<()> {
    let log_dir = data_dir.join("logs");
    std::fs::create_dir_all(&log_dir)?;
    let path = log_dir.join("bitty.log");
    rotate_if_needed(&path)?;
    let mut file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)?;
    writeln!(file, "{} {}", timestamp(), message.as_ref())
}

pub fn log_path(data_dir: &Path) -> PathBuf {
    data_dir.join("logs").join("bitty.log")
}

pub fn rotate_if_needed(path: &Path) -> std::io::Result<()> {
    let Ok(metadata) = std::fs::metadata(path) else {
        return Ok(());
    };
    if metadata.len() < MAX_LOG_BYTES {
        return Ok(());
    }
    for index in (1..=ROTATED_LOGS).rev() {
        let from = if index == 1 {
            path.to_path_buf()
        } else {
            rotated_path(path, index - 1)
        };
        let to = rotated_path(path, index);
        if from.exists() {
            let _ = std::fs::remove_file(&to);
            let _ = std::fs::rename(from, to);
        }
    }
    Ok(())
}

pub fn read_last_lines(path: &Path, count: usize) -> std::io::Result<String> {
    let contents = std::fs::read_to_string(path)?;
    let lines = contents.lines().collect::<Vec<_>>();
    let start = lines.len().saturating_sub(count);
    Ok(lines[start..].join("\n"))
}

fn rotated_path(path: &Path, index: usize) -> PathBuf {
    PathBuf::from(format!("{}.{}", path.display(), index))
}

fn timestamp() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or_default();
    format!("[{seconds}]")
}

fn default_data_dir() -> PathBuf {
    std::env::var("BITTY_DATA_DIR")
        .ok()
        .map(PathBuf::from)
        .or_else(|| {
            std::env::var("HOME")
                .ok()
                .map(|home| PathBuf::from(home).join(".bitty"))
        })
        .unwrap_or_else(|| PathBuf::from(".bitty"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn last_lines_returns_requested_tail() {
        let dir = std::env::temp_dir().join(format!("bitty-log-test-{}", timestamp()));
        let path = log_path(&dir);
        log(&dir, "one").unwrap();
        log(&dir, "two").unwrap();
        let tail = read_last_lines(&path, 1).unwrap();
        assert!(tail.contains("two"));
        assert!(!tail.contains("one"));
        let _ = std::fs::remove_dir_all(dir);
    }
}
