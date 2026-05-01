use std::path::PathBuf;
use std::process::Command;

const DEFAULT_RUNTIME_DIR: &str = "external/BitNet";
const DEFAULT_MODEL_PATH: &str = "external/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf";

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = BitNetConfig::from_env();
    let runtime_dir = PathBuf::from(&config.runtime_dir);
    let script = runtime_dir.join("run_inference.py");
    let model = PathBuf::from(&config.model);
    let python = PathBuf::from(&config.python);

    if !script.exists() {
        return Err(format!(
            "BitNet runtime not found at {}. Run scripts/setup_bitnet.sh first, or pass --runtime-dir.",
            script.display()
        )
        .into());
    }

    if !model.exists() {
        return Err(format!(
            "BitNet model not found at {}. Run scripts/setup_bitnet.sh first, or pass --model.",
            model.display()
        )
        .into());
    }

    let runtime_dir = runtime_dir.canonicalize()?;
    let model = model.canonicalize()?;
    let python = if python.components().count() > 1 {
        python.canonicalize()?
    } else {
        python
    };

    let status = Command::new(python)
        .current_dir(runtime_dir)
        .arg("run_inference.py")
        .arg("-m")
        .arg(model)
        .arg("-p")
        .arg(config.prompt)
        .arg("-n")
        .arg(config.n_predict.to_string())
        .arg("-t")
        .arg(config.threads.to_string())
        .arg("-c")
        .arg(config.ctx_size.to_string())
        .arg("-temp")
        .arg(config.temperature.to_string())
        .args(config.conversation.then_some("-cnv"))
        .status()?;

    if !status.success() {
        return Err(format!("BitNet inference failed with status {status}").into());
    }

    Ok(())
}

#[derive(Clone, Debug)]
struct BitNetConfig {
    runtime_dir: String,
    model: String,
    python: String,
    prompt: String,
    n_predict: u32,
    threads: u32,
    ctx_size: u32,
    temperature: f32,
    conversation: bool,
}

impl BitNetConfig {
    fn from_env() -> Self {
        let mut config = Self {
            runtime_dir: DEFAULT_RUNTIME_DIR.into(),
            model: DEFAULT_MODEL_PATH.into(),
            python: "python".into(),
            prompt: "You are a helpful assistant. Explain BitNet in one paragraph.".into(),
            n_predict: 128,
            threads: std::thread::available_parallelism()
                .map(|threads| threads.get() as u32)
                .unwrap_or(4),
            ctx_size: 2048,
            temperature: 0.7,
            conversation: false,
        };

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--runtime-dir" => config.runtime_dir = required_next(&mut args, "--runtime-dir"),
                "--model" => config.model = required_next(&mut args, "--model"),
                "--python" => config.python = required_next(&mut args, "--python"),
                "--prompt" => config.prompt = required_next(&mut args, "--prompt"),
                "--n-predict" => config.n_predict = parse_next(&mut args, "--n-predict"),
                "--threads" => config.threads = parse_next(&mut args, "--threads"),
                "--ctx-size" => config.ctx_size = parse_next(&mut args, "--ctx-size"),
                "--temperature" => config.temperature = parse_next(&mut args, "--temperature"),
                "--conversation" => config.conversation = true,
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                unknown => {
                    eprintln!("unknown argument: {unknown}");
                    print_help();
                    std::process::exit(2);
                }
            }
        }

        config
    }
}

fn required_next(args: &mut impl Iterator<Item = String>, name: &str) -> String {
    args.next().unwrap_or_else(|| {
        eprintln!("missing value for {name}");
        std::process::exit(2);
    })
}

fn parse_next<T>(args: &mut impl Iterator<Item = String>, name: &str) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = required_next(args, name);
    value.parse().unwrap_or_else(|err| {
        eprintln!("invalid value for {name}: {err}");
        std::process::exit(2);
    })
}

fn print_help() {
    println!(
        "Usage: cargo run -p bitty-inference --bin bitty-bitnet -- [--prompt TEXT] [--n-predict N] [--threads N]"
    );
    println!();
    println!("Defaults:");
    println!("  runtime: {DEFAULT_RUNTIME_DIR}");
    println!("  model:   {DEFAULT_MODEL_PATH}");
    println!();
    println!("Run scripts/setup_bitnet.sh first to install the official bitnet.cpp runtime and GGUF model.");
}
