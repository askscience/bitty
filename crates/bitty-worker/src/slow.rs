//! Says why a step of the chain took far longer than the usual one: what the machine did
//! in the meantime. A node that pages its layers in from the disk, or shares its processor
//! with something else, shows up in its own log, which the leader can ask for.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

/// A step is slow when it took this many times the fastest one of the node.
const SLOW_FACTOR: u64 = 3;
/// Shorter steps are never worth a line.
const SLOW_FLOOR_MICROS: u64 = 100_000;
/// Lines one node writes before it stops: the first ones tell the story.
const MAX_LINES: u64 = 500;

#[derive(Clone, Default)]
pub struct SlowSteps {
    fastest_micros: Arc<AtomicU64>,
    lines: Arc<AtomicU64>,
}

/// What the kernel counted so far.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct MachineCounters {
    /// Pages that had to be read from the disk before a program could go on.
    major_faults: u64,
    /// KiB read from the disks.
    read_kib: u64,
    /// Pages taken back from the swap.
    swapped_in: u64,
}

impl MachineCounters {
    pub fn now() -> Option<Self> {
        parse_vmstat(&std::fs::read_to_string("/proc/vmstat").ok()?)
    }
}

fn parse_vmstat(text: &str) -> Option<MachineCounters> {
    let value = |name: &str| {
        text.lines().find_map(|line| {
            let (key, value) = line.split_once(' ')?;
            (key == name).then(|| value.trim().parse::<u64>().ok())?
        })
    };
    Some(MachineCounters {
        major_faults: value("pgmajfault")?,
        read_kib: value("pgpgin")?,
        swapped_in: value("pswpin").unwrap_or(0),
    })
}

/// The load of the last minute and the memory programs can still get, in MB.
fn load_and_memory() -> Option<(String, u64)> {
    let load = std::fs::read_to_string("/proc/loadavg").ok()?;
    let memory = std::fs::read_to_string("/proc/meminfo").ok()?;
    let available_kib = memory.lines().find_map(|line| {
        line.strip_prefix("MemAvailable:")?
            .trim()
            .trim_end_matches("kB")
            .trim()
            .parse::<u64>()
            .ok()
    })?;
    Some((load.split(' ').next()?.to_string(), available_kib / 1024))
}

impl SlowSteps {
    /// Takes the time of a step of one token. Returns the line to log when it was slow.
    pub fn step(&self, micros: u64, before: Option<MachineCounters>) -> Option<String> {
        let fastest = self.fastest_micros.load(Ordering::Relaxed);
        if fastest == 0 || micros < fastest {
            self.fastest_micros.store(micros.max(1), Ordering::Relaxed);
            return None;
        }
        if micros < SLOW_FLOOR_MICROS || micros < fastest * SLOW_FACTOR {
            return None;
        }
        if self.lines.fetch_add(1, Ordering::Relaxed) >= MAX_LINES {
            return None;
        }
        Some(describe(
            micros,
            fastest,
            before,
            MachineCounters::now(),
            load_and_memory(),
        ))
    }
}

fn describe(
    micros: u64,
    fastest: u64,
    before: Option<MachineCounters>,
    after: Option<MachineCounters>,
    machine: Option<(String, u64)>,
) -> String {
    let mut line = format!(
        "slow step: {} ms, the fastest took {} ms",
        micros / 1000,
        fastest / 1000
    );
    if let (Some(before), Some(after)) = (before, after) {
        line.push_str(&format!(
            "; meanwhile {} pages waited for the disk, {} MiB read from it, {} pages back from swap",
            after.major_faults.saturating_sub(before.major_faults),
            after.read_kib.saturating_sub(before.read_kib) / 1024,
            after.swapped_in.saturating_sub(before.swapped_in),
        ));
    }
    if let Some((load, available_mb)) = machine {
        line.push_str(&format!(
            "; load {load}, {available_mb} MB of memory available"
        ));
    }
    line
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_the_counters_of_the_kernel() {
        let counters = parse_vmstat("nr_free_pages 10\npgpgin 2048\npswpin 3\npgmajfault 7\n");
        assert_eq!(
            counters,
            Some(MachineCounters {
                major_faults: 7,
                read_kib: 2048,
                swapped_in: 3
            })
        );
        assert_eq!(parse_vmstat("nr_free_pages 10\n"), None);
    }

    #[test]
    fn only_a_step_far_slower_than_the_fastest_gets_a_line() {
        let steps = SlowSteps::default();
        assert!(steps.step(40_000, None).is_none());
        assert!(steps.step(90_000, None).is_none());
        assert!(steps.step(110_000, None).is_none());
        let line = steps.step(700_000, None).unwrap();
        assert!(
            line.starts_with("slow step: 700 ms, the fastest took 40 ms"),
            "{line}"
        );
    }

    #[test]
    fn the_line_says_what_the_machine_did() {
        let before = MachineCounters {
            major_faults: 10,
            read_kib: 1024,
            swapped_in: 0,
        };
        let after = MachineCounters {
            major_faults: 310,
            read_kib: 51 * 1024,
            swapped_in: 4,
        };
        let line = describe(
            700_000,
            40_000,
            Some(before),
            Some(after),
            Some(("3.20".into(), 512)),
        );
        assert_eq!(
            line,
            "slow step: 700 ms, the fastest took 40 ms; meanwhile 300 pages waited for the disk, 50 MiB read from it, 4 pages back from swap; load 3.20, 512 MB of memory available"
        );
    }
}
