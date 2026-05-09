use bitty_bitnet_runtime::BitNetRuntime;
use std::path::Path;

const DEFAULT_MODEL_PATH: &str = "external/BitNet/models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf";

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = RustBitNetConfig::from_env();
    let mut runtime = BitNetRuntime::load(Path::new(&config.model)).await?;
    let output = runtime
        .generate_full(&config.prompt, config.max_tokens, config.temperature)
        .await?;
    println!("{output}");
    Ok(())
}

#[derive(Clone, Debug)]
struct RustBitNetConfig {
    model: String,
    prompt: String,
    max_tokens: usize,
    temperature: f32,
}

impl RustBitNetConfig {
    fn from_env() -> Self {
        let mut config = Self {
            model: DEFAULT_MODEL_PATH.into(),
            prompt: "Say hello in five words".into(),
            max_tokens: 32,
            temperature: 0.7,
        };

        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--model" => config.model = required_next(&mut args, "--model"),
                "--prompt" => config.prompt = required_next(&mut args, "--prompt"),
                "--max-tokens" => config.max_tokens = parse_next(&mut args, "--max-tokens"),
                "--temperature" => config.temperature = parse_next(&mut args, "--temperature"),
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
        "Usage: cargo run -p bitty-inference --bin bitty-rust-bitnet -- [--model PATH] [--prompt TEXT] [--max-tokens N]"
    );
    println!("Default model: {DEFAULT_MODEL_PATH}");
}
