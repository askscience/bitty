use std::io::{self, Write};
use std::path::Path;
use std::time::Duration;

pub const C: &str = "\x1b[36m";
pub const G: &str = "\x1b[32m";
pub const Y: &str = "\x1b[33m";
pub const R: &str = "\x1b[31m";
pub const D: &str = "\x1b[2m";
pub const B: &str = "\x1b[1m";
pub const N: &str = "\x1b[0m";

const BRAILLE: &[&str] = &["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];

pub fn ok() -> String {
    format!("{G}ok{N}")
}
pub fn fail() -> String {
    format!("{R}fail{N}")
}
pub fn ready() -> String {
    format!("{G}ready{N}")
}
pub fn not_ready() -> String {
    format!("{Y}not ready{N}")
}

pub fn label(text: &str) -> String {
    format!("  {D}{}{N}", text)
}
pub fn dim(text: &str) -> String {
    format!("{D}{}{N}", text)
}
pub fn bold(text: &str) -> String {
    format!("{B}{}{N}", text)
}
pub fn green(text: &str) -> String {
    format!("{G}{}{N}", text)
}
pub fn yellow(text: &str) -> String {
    format!("{Y}{}{N}", text)
}
pub fn cyan(text: &str) -> String {
    format!("{C}{}{N}", text)
}
pub fn red(text: &str) -> String {
    format!("{R}{}{N}", text)
}
pub fn rule() {
    println!("  {}", "─".repeat(50));
}

pub fn header(title: &str, data_dir: &Path, version: &str) {
    println!();
    println!(
        "  {B}{}{N}  {D}v{}  ·  {}{N}",
        title,
        version,
        data_dir.display()
    );
    rule();
    println!();
}

pub async fn spinner<F, T>(label: &str, f: F) -> T
where
    F: std::future::Future<Output = T>,
{
    let frames = BRAILLE;
    let mut tick = 0usize;
    let label_owned = label.to_string();
    let handle = tokio::spawn(async move {
        loop {
            print!(
                "\r  {} {} {}...{}",
                frames[tick % frames.len()],
                D,
                label_owned,
                N
            );
            io::stdout().flush().ok();
            tokio::time::sleep(Duration::from_millis(80)).await;
            tick += 1;
        }
    });
    let result = f.await;
    handle.abort();
    print!("\r{}\r", " ".repeat(label.len() + 20));
    result
}

pub fn prompt_line(prompt: &str) -> String {
    print!("  {C}›{N} {} ", prompt);
    io::stdout().flush().ok();
    let mut line = String::new();
    io::stdin().read_line(&mut line).ok();
    line.trim().to_string()
}

pub fn prompt_bool(prompt: &str, default: bool) -> bool {
    let hint = if default { "Y/n" } else { "y/N" };
    let input = prompt_line(&format!("{} [{}]", prompt, hint));
    match input.to_lowercase().as_str() {
        "y" | "yes" => true,
        "n" | "no" => false,
        _ => default,
    }
}

pub fn confirm_destructive(msg: &str) -> bool {
    println!("  {Y}{}{N}", msg);
    println!();
    prompt_bool("type yes to confirm", false)
}
