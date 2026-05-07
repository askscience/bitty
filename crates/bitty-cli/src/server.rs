use crate::logger;
use crate::model_store::{installed_models, resolve_model};
use crate::settings::BittySettings;
use serde_json::json;
use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};

pub fn serve(settings: BittySettings) -> Result<(), Box<dyn std::error::Error>> {
    let listener = TcpListener::bind(&settings.api_host)?;
    logger::log(
        &settings.data_dir,
        format!("api server listening on {}", settings.api_host),
    )?;
    println!("bitty serve listening on http://{}", settings.api_host);
    for stream in listener.incoming() {
        match stream {
            Ok(stream) => handle_stream(stream, &settings)?,
            Err(err) => {
                let _ = logger::log(&settings.data_dir, format!("api connection failed: {err}"));
                eprintln!("bitty serve connection failed: {err}");
            }
        }
    }
    Ok(())
}

fn handle_stream(
    mut stream: TcpStream,
    settings: &BittySettings,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut buffer = vec![0_u8; 64 * 1024];
    let read = stream.read(&mut buffer)?;
    let request = String::from_utf8_lossy(&buffer[..read]);
    let first = request.lines().next().unwrap_or_default();
    let body = request.split("\r\n\r\n").nth(1).unwrap_or_default();
    let response = if first.starts_with("GET /v1/models ") || first.starts_with("GET /api/tags ") {
        models_response(settings)
    } else if first.starts_with("POST /api/show ") {
        show_response(settings, body)
    } else if first.starts_with("POST /v1/chat/completions ")
        || first.starts_with("POST /v1/completions ")
        || first.starts_with("POST /api/generate ")
        || first.starts_with("POST /api/chat ")
    {
        generate_response(settings, body)
    } else if first.starts_with("POST /api/pull ") {
        json!({"status": "pull through API is not implemented yet; use bitty pull"}).to_string()
    } else {
        json!({"error": "unknown endpoint"}).to_string()
    };
    write_json(&mut stream, &response)?;
    Ok(())
}

fn models_response(settings: &BittySettings) -> String {
    let models = installed_models(settings);
    let data = models
        .iter()
        .map(|model| json!({"id": model.id(), "object": "model", "owned_by": "bitty"}))
        .collect::<Vec<_>>();
    json!({"object": "list", "data": data, "models": data}).to_string()
}

fn show_response(settings: &BittySettings, body: &str) -> String {
    let model_name = json_field(body, "model").unwrap_or_else(|| settings.default_model.clone());
    match resolve_model(settings, &model_name) {
        Some(model) => json!({
            "model": model.id(),
            "path": model.model_path(settings),
            "backend": model.backend,
            "quantization": model.quantization,
            "layers": model.layers,
            "parameters": {
                "temperature": model.temperature,
                "num_predict": model.num_predict,
                "num_ctx": model.num_ctx
            }
        })
        .to_string(),
        None => json!({"error": format!("model not found: {model_name}")}).to_string(),
    }
}

fn generate_response(settings: &BittySettings, body: &str) -> String {
    let model = json_field(body, "model").unwrap_or_else(|| settings.default_model.clone());
    let prompt = json_field(body, "prompt")
        .unwrap_or_else(|| json_field(body, "content").unwrap_or_else(|| "Hello".into()));
    let raw = std::env::current_exe()
        .ok()
        .and_then(|exe| {
            std::process::Command::new(exe)
                .arg("run")
                .arg(&model)
                .arg(&prompt)
                .arg("--local")
                .arg("--no-daemon")
                .arg("--data-dir")
                .arg(settings.data_dir.as_os_str())
                .arg("--num-predict")
                .arg(settings.default_num_predict.to_string())
                .output()
                .ok()
        })
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).to_string());
    let text = clean_server_output(raw);
    json!({
        "model": model,
        "created_at": "",
        "response": text,
        "message": {
            "role": "assistant",
            "content": text
        },
        "done": true,
        "choices": [{
            "index": 0,
            "message": {"role": "assistant", "content": text},
            "finish_reason": "stop"
        }]
    })
    .to_string()
}

/// Strip ANSI escape codes, spinner characters, and status lines from CLI output.
fn clean_server_output(raw: Option<String>) -> String {
    let text = match raw {
        Some(s) => s.trim().to_string(),
        None => return "Bitty generation failed; check that the model is pulled and supported.".into(),
    };
    if text.is_empty() {
        return "Bitty generation failed; check that the model is pulled and supported.".into();
    }
    // Strip ANSI escape sequences (\x1b[...m)
    let text = ansi_strip(&text);
    // Strip spinner frames and status lines
    let lines: Vec<&str> = text
        .lines()
        .filter(|line| {
            let trimmed = line.trim();
            !trimmed.is_empty()
                && !trimmed.starts_with("cluster:")
                && !trimmed.starts_with("⠋")
                && !trimmed.starts_with("⠙")
                && !trimmed.starts_with("⠹")
                && !trimmed.starts_with("⠸")
                && !trimmed.starts_with("⠼")
                && !trimmed.starts_with("⠴")
                && !trimmed.starts_with("⠦")
                && !trimmed.starts_with("⠧")
                && !trimmed.starts_with("⠇")
                && !trimmed.starts_with("⠏")
                && !trimmed.starts_with("connecting")
                && !trimmed.starts_with("loading")
                && !trimmed.starts_with("generating")
                && !trimmed.starts_with("\u{1b}[")
        })
        .collect();
    let result = lines.join("").trim().to_string();
    if result.is_empty() {
        "Bitty generation completed but produced no output.".into()
    } else {
        result
    }
}

/// Remove ANSI escape codes (CSI sequences: ESC [ ... m).
fn ansi_strip(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '\x1b' && chars.peek() == Some(&'[') {
            chars.next(); // consume '['
            while let Some(&ch) = chars.peek() {
                if ch == 'm' {
                    chars.next();
                    break;
                }
                chars.next();
            }
        } else {
            result.push(c);
        }
    }
    result
}

fn write_json(stream: &mut TcpStream, body: &str) -> std::io::Result<()> {
    write!(
        stream,
        "HTTP/1.1 200 OK\r\ncontent-type: application/json\r\ncontent-length: {}\r\nconnection: close\r\n\r\n{}",
        body.len(),
        body
    )
}

fn json_field(body: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\"");
    let after_key = body.split(&needle).nth(1)?;
    let after_colon = after_key.split_once(':')?.1.trim_start();
    let value = after_colon.strip_prefix('"')?;
    let end = value.find('"')?;
    Some(value[..end].to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::model_store::{write_manifest, ModelSpec};
    use std::time::{SystemTime, UNIX_EPOCH};

    #[test]
    fn v1_models_response_lists_installed_models() {
        let temp = std::env::temp_dir().join(format!(
            "bitty-server-test-{}",
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let settings = BittySettings::defaults(temp.clone());
        let spec = ModelSpec {
            name: "demo".into(),
            tag: "latest".into(),
            filename: "demo.gguf".into(),
            backend: "bitnet-i2s".into(),
            quantization: "i2_s".into(),
            ..Default::default()
        };
        write_manifest(&settings, &spec, &temp.join("demo.gguf")).unwrap();
        let response = models_response(&settings);
        assert!(response.contains("\"id\":\"demo\""));
        let _ = std::fs::remove_dir_all(temp);
    }
}
