use bitty_protocol::cli::{parse_next_or_exit, required_next_or_exit};
use bitty_protocol::endpoint::normalize_endpoint;
use bitty_protocol::pb::coordinator_service_client::CoordinatorServiceClient;
use bitty_protocol::pb::GenerateRequest;
use bitty_protocol::security::BITTY_TOKEN_HEADER;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let config = ClientConfig::from_env();
    let mut client =
        CoordinatorServiceClient::connect(normalize_endpoint(&config.coordinator)).await?;
    let mut stream = client
        .generate(request_with_token(
            GenerateRequest {
                request_id: config.request_id,
                prompt_tokens: Vec::new(),
                prompt: config.prompt,
                max_new_tokens: config.max_tokens,
                temperature: config.temperature,
            },
            config.token.as_deref(),
        )?)
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
    token: Option<String>,
}

impl ClientConfig {
    fn from_env() -> Self {
        let mut config = Self {
            coordinator: "127.0.0.1:50051".into(),
            prompt: "Hello".into(),
            max_tokens: 32,
            temperature: 0.0,
            request_id: uuid_like_request_id(),
            token: None,
        };
        let mut args = std::env::args().skip(1);
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--coordinator" => {
                    config.coordinator = required_next_or_exit(&mut args, "--coordinator")
                }
                "--prompt" => config.prompt = required_next_or_exit(&mut args, "--prompt"),
                "--max-tokens" => config.max_tokens = parse_next_or_exit(&mut args, "--max-tokens"),
                "--temperature" => {
                    config.temperature = parse_next_or_exit(&mut args, "--temperature")
                }
                "--request-id" => {
                    config.request_id = required_next_or_exit(&mut args, "--request-id")
                }
                "--token" | "--cluster-token" => {
                    config.token = Some(required_next_or_exit(&mut args, "--token"))
                }
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

fn request_with_token<T>(
    message: T,
    token: Option<&str>,
) -> Result<tonic::Request<T>, Box<dyn std::error::Error>> {
    let mut request = tonic::Request::new(message);
    if let Some(token) = token.filter(|token| !token.is_empty()) {
        request
            .metadata_mut()
            .insert(BITTY_TOKEN_HEADER, token.parse()?);
    }
    Ok(request)
}

fn uuid_like_request_id() -> String {
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    format!("client-{nanos:x}")
}

fn print_help() {
    println!("Usage: cargo run -p bitty-inference --bin bitty-client -- --coordinator HOST:PORT --prompt TEXT --max-tokens N [--token TOKEN]");
}
