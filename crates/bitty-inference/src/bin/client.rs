use bitty_protocol::pb::coordinator_service_client::CoordinatorServiceClient;
use bitty_protocol::pb::GenerateRequest;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ClientConfig::from_env();
    let mut client =
        CoordinatorServiceClient::connect(normalize_endpoint(&config.coordinator)).await?;
    let mut stream = client
        .generate(GenerateRequest {
            request_id: config.request_id,
            prompt_tokens: Vec::new(),
            prompt: config.prompt,
            max_new_tokens: config.max_tokens,
            temperature: config.temperature,
        })
        .await?
        .into_inner();

    while let Some(token) = stream.message().await? {
        print!("{}", token.text);
        if token.finished {
            println!();
        }
    }
    Ok(())
}

#[derive(Clone, Debug)]
struct ClientConfig {
    coordinator: String,
    prompt: String,
    max_tokens: u32,
    temperature: f32,
    request_id: String,
}

impl ClientConfig {
    fn from_env() -> Self {
        let mut config = Self {
            coordinator: "127.0.0.1:50051".into(),
            prompt: "Hello".into(),
            max_tokens: 32,
            temperature: 0.0,
            request_id: uuid_like_request_id(),
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--coordinator" => config.coordinator = required_next(&mut args, "--coordinator"),
                "--prompt" => config.prompt = required_next(&mut args, "--prompt"),
                "--max-tokens" => config.max_tokens = parse_next(&mut args, "--max-tokens"),
                "--temperature" => config.temperature = parse_next(&mut args, "--temperature"),
                "--request-id" => config.request_id = required_next(&mut args, "--request-id"),
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

fn normalize_endpoint(endpoint: &str) -> String {
    if endpoint.starts_with("http://") || endpoint.starts_with("https://") {
        endpoint.into()
    } else {
        format!("http://{endpoint}")
    }
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

fn required_next(args: &mut impl Iterator<Item = String>, name: &str) -> String {
    args.next().unwrap_or_else(|| {
        eprintln!("missing value for {name}");
        std::process::exit(2);
    })
}

fn uuid_like_request_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("client-{nanos:x}")
}

fn print_help() {
    println!("Usage: cargo run -p bitty-inference --bin bitty-client -- --coordinator HOST:PORT --prompt TEXT --max-tokens N");
}
