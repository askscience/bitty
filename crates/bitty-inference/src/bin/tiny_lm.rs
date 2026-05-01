use bitty_inference::TinyLanguageModel;

fn main() {
    let config = TinyLmConfig::from_env();
    let model = TinyLanguageModel::default();
    let generated = model.generate(&config.prompt, config.chars, config.seed);

    println!("{generated}");
}

#[derive(Clone, Debug)]
struct TinyLmConfig {
    prompt: String,
    chars: usize,
    seed: u64,
}

impl TinyLmConfig {
    fn from_env() -> Self {
        let mut config = Self {
            prompt: "The coordinator".into(),
            chars: 240,
            seed: 7,
        };
        let mut args = std::env::args().skip(1);

        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--prompt" => {
                    config.prompt = args.next().unwrap_or_else(|| {
                        eprintln!("missing value for --prompt");
                        std::process::exit(2);
                    })
                }
                "--chars" => config.chars = parse_next(&mut args, "--chars"),
                "--seed" => config.seed = parse_next(&mut args, "--seed"),
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

fn parse_next<T>(args: &mut impl Iterator<Item = String>, name: &str) -> T
where
    T: std::str::FromStr,
    T::Err: std::fmt::Display,
{
    let value = args.next().unwrap_or_else(|| {
        eprintln!("missing value for {name}");
        std::process::exit(2);
    });
    value.parse().unwrap_or_else(|err| {
        eprintln!("invalid value for {name}: {err}");
        std::process::exit(2);
    })
}

fn print_help() {
    println!("Usage: cargo run -p bitty-inference --bin bitty-tiny-lm -- [--prompt TEXT] [--chars N] [--seed N]");
}
