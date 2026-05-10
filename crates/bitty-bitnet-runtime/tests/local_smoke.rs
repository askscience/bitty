use std::path::PathBuf;

fn model_dir() -> Option<PathBuf> {
    let home = PathBuf::from(
        std::env::var("BITTY_MODELS_DIR")
            .unwrap_or_else(|_| {
                let home = std::env::var("HOME").unwrap_or_default();
                format!("{home}/.bitty/models")
            }),
    );
    if home.exists() {
        Some(home)
    } else {
        None
    }
}

fn model_path(filename: &str) -> Option<PathBuf> {
    model_dir().map(|d| d.join(filename))
}

fn run_greedy(model_name: &str, prompt: &str) -> Option<String> {
    let path = model_path(model_name)?;
    if !path.exists() {
        return None;
    }
    let cpu = bitty_bitnet_runtime::cpu_backend::CpuModel::load(&path, None).ok()?;
    let tokens = cpu
        .tokenizer()
        .encode(prompt, true)
        .ok()?;
    cpu.generate_from_ids(&tokens, 64, 0.0, 40, 1.0, Some(42), |_| {})
        .ok()
}

fn ascii_ratio(s: &str) -> f64 {
    let printable = s.chars().filter(|c| c.is_ascii_graphic() || c.is_whitespace()).count();
    if s.is_empty() {
        0.0
    } else {
        printable as f64 / s.chars().count() as f64
    }
}

#[test]
fn tinyllama_generated_output_is_mostly_ascii() {
    let Some(text) = run_greedy("tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf", "The capital of France is") else {
        eprintln!("skipping: model file not found");
        return;
    };
    assert!(!text.is_empty(), "model produced empty output");
    assert!(text.len() > 5, "output too short: {text:?}");
    assert!(ascii_ratio(&text) > 0.8, "low ASCII ratio ({:.2}): {text:?}", ascii_ratio(&text));
}

#[test]
fn smollm2_generated_output_is_mostly_ascii() {
    let Some(text) = run_greedy("smollm2-1.7b-instruct.Q4_K_M.gguf", "The capital of France is") else {
        eprintln!("skipping: model file not found");
        return;
    };
    assert!(!text.is_empty(), "model produced empty output");
    assert!(text.len() > 5, "output too short: {text:?}");
    assert!(ascii_ratio(&text) > 0.8, "low ASCII ratio ({:.2}): {text:?}", ascii_ratio(&text));
}

#[test]
fn gemma3_generated_output_is_mostly_ascii() {
    let Some(text) = run_greedy("gemma-3-4b-it-Q4_K_M.gguf", "The capital of France is") else {
        eprintln!("skipping: model file not found");
        return;
    };
    assert!(!text.is_empty(), "model produced empty output");
    assert!(text.len() > 5, "output too short: {text:?}");
    assert!(ascii_ratio(&text) > 0.8, "low ASCII ratio ({:.2}): {text:?}", ascii_ratio(&text));
}

#[test]
fn deepseek_r1_generated_output_is_mostly_ascii() {
    let Some(text) = run_greedy("deepseek-r1-distill-qwen-1.5b-q4_k_m.gguf", "2+2=") else {
        eprintln!("skipping: model file not found");
        return;
    };
    assert!(!text.is_empty(), "model produced empty output");
    assert!(text.len() > 3, "output too short: {text:?}");
    assert!(ascii_ratio(&text) > 0.8, "low ASCII ratio ({:.2}): {text:?}", ascii_ratio(&text));
}
