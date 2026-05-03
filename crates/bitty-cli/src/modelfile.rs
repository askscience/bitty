use crate::model_store::{write_manifest, ModelSpec};
use crate::settings::BittySettings;
use std::path::Path;

#[derive(Clone, Debug, Default, PartialEq)]
pub struct BittyModelfile {
    pub from: String,
    pub parameters: Vec<(String, String)>,
    pub system: Option<String>,
    pub template: Option<String>,
    pub messages: Vec<(String, String)>,
    pub license: Option<String>,
    pub unsupported: Vec<String>,
}

impl BittyModelfile {
    pub fn to_model_spec(&self, name: &str) -> ModelSpec {
        let mut spec = ModelSpec {
            name: name.into(),
            tag: "latest".into(),
            display_name: name.into(),
            backend: "bitnet-i2s".into(),
            quantization: "i2_s".into(),
            filename: Path::new(&self.from)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("model.gguf")
                .into(),
            layers: 30,
            temperature: 0.7,
            num_predict: 128,
            num_ctx: 2048,
            path: Some(self.from.clone().into()),
            ..Default::default()
        };
        for (key, value) in &self.parameters {
            match key.as_str() {
                "temperature" => spec.temperature = value.parse().unwrap_or(spec.temperature),
                "num_predict" => spec.num_predict = value.parse().unwrap_or(spec.num_predict),
                "num_ctx" => spec.num_ctx = value.parse().unwrap_or(spec.num_ctx),
                _ => {}
            }
        }
        spec
    }
}

pub fn parse_modelfile(contents: &str) -> BittyModelfile {
    let mut parsed = BittyModelfile::default();
    let mut lines = contents.lines().peekable();
    while let Some(line) = lines.next() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let (instruction, rest) = line
            .split_once(char::is_whitespace)
            .map(|(left, right)| (left.to_ascii_uppercase(), right.trim().to_string()))
            .unwrap_or_else(|| (line.to_ascii_uppercase(), String::new()));
        let value = if rest.starts_with("\"\"\"") {
            read_multiline(rest, &mut lines)
        } else {
            trim_quotes(&rest).to_string()
        };
        match instruction.as_str() {
            "FROM" => parsed.from = value,
            "PARAMETER" => {
                if let Some((key, value)) = value.split_once(char::is_whitespace) {
                    parsed
                        .parameters
                        .push((key.into(), trim_quotes(value.trim()).into()));
                }
            }
            "SYSTEM" => parsed.system = Some(value),
            "TEMPLATE" => parsed.template = Some(value),
            "MESSAGE" => {
                if let Some((role, message)) = value.split_once(char::is_whitespace) {
                    parsed
                        .messages
                        .push((role.into(), trim_quotes(message.trim()).into()));
                }
            }
            "LICENSE" => parsed.license = Some(value),
            other => parsed.unsupported.push(other.into()),
        }
    }
    parsed
}

pub fn create_profile(
    settings: &BittySettings,
    name: &str,
    modelfile_path: &Path,
) -> Result<ModelSpec, Box<dyn std::error::Error>> {
    let contents = std::fs::read_to_string(modelfile_path)?;
    let parsed = parse_modelfile(&contents);
    if parsed.from.is_empty() {
        return Err("Modelfile requires FROM".into());
    }
    let spec = parsed.to_model_spec(name);
    let model_path = spec
        .path
        .clone()
        .unwrap_or_else(|| spec.filename.clone().into());
    write_manifest(settings, &spec, &model_path)?;
    Ok(spec)
}

fn read_multiline<'a>(
    first: String,
    lines: &mut std::iter::Peekable<impl Iterator<Item = &'a str>>,
) -> String {
    let mut value = first.trim_start_matches("\"\"\"").to_string();
    if let Some((before, _)) = value.split_once("\"\"\"") {
        return before.to_string();
    }
    for line in lines.by_ref() {
        if let Some((before, _)) = line.split_once("\"\"\"") {
            if !value.is_empty() {
                value.push('\n');
            }
            value.push_str(before);
            break;
        }
        if !value.is_empty() {
            value.push('\n');
        }
        value.push_str(line);
    }
    value
}

fn trim_quotes(value: &str) -> &str {
    value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or(value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_modelfile() {
        let parsed = parse_modelfile(
            r#"FROM ./model.gguf
PARAMETER temperature 0.2
SYSTEM """hello"""
"#,
        );
        assert_eq!(parsed.from, "./model.gguf");
        assert_eq!(parsed.parameters[0], ("temperature".into(), "0.2".into()));
        assert_eq!(parsed.system.as_deref(), Some("hello"));
    }
}
