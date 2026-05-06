use crate::settings::{ensure_parent, parse_assignment, unquote, BittySettings};
use std::path::{Path, PathBuf};

const BUILTIN_REGISTRY: &str = include_str!("../../../models/registry.toml");

#[derive(Clone, Debug, Default, PartialEq)]
pub struct ModelSpec {
    pub name: String,
    pub tag: String,
    pub display_name: String,
    pub backend: String,
    pub quantization: String,
    pub filename: String,
    pub layers: u32,
    pub url: String,
    pub source: String,
    pub temperature: f32,
    pub num_predict: u32,
    pub num_ctx: u32,
    pub path: Option<PathBuf>,
}

impl ModelSpec {
    pub fn id(&self) -> String {
        if self.tag.is_empty() || self.tag == "latest" {
            self.name.clone()
        } else {
            format!("{}:{}", self.name, self.tag)
        }
    }

    pub fn model_dir(&self, settings: &BittySettings) -> PathBuf {
        settings
            .models_dir
            .join(&self.name)
            .join(if self.tag.is_empty() {
                "latest"
            } else {
                &self.tag
            })
    }

    pub fn model_path(&self, settings: &BittySettings) -> PathBuf {
        self.path
            .clone()
            .unwrap_or_else(|| self.model_dir(settings).join(&self.filename))
    }

    pub fn manifest_path(&self, settings: &BittySettings) -> PathBuf {
        self.model_dir(settings).join("manifest.toml")
    }

    pub fn to_manifest(&self, model_path: &Path) -> String {
        format!(
            "name = \"{}\"\ntag = \"{}\"\ndisplay_name = \"{}\"\nbackend = \"{}\"\nquantization = \"{}\"\nfilename = \"{}\"\nlayers = {}\nurl = \"{}\"\nsource = \"{}\"\ntemperature = {}\nnum_predict = {}\nnum_ctx = {}\npath = \"{}\"\n",
            esc(&self.name),
            esc(if self.tag.is_empty() { "latest" } else { &self.tag }),
            esc(&self.display_name),
            esc(&self.backend),
            esc(&self.quantization),
            esc(&self.filename),
            self.layers,
            esc(&self.url),
            esc(&self.source),
            self.temperature,
            self.num_predict,
            self.num_ctx,
            esc(&model_path.display().to_string())
        )
    }
}

pub fn registry_models() -> Vec<ModelSpec> {
    parse_registry(BUILTIN_REGISTRY)
}

pub fn find_registry_model(name: &str) -> Option<ModelSpec> {
    let (model, tag) = split_model_tag(name);
    let model_lower = model.to_lowercase();
    registry_models()
        .into_iter()
        .find(|spec| {
            spec.name.to_lowercase() == model_lower
                && (tag.is_none() || tag == Some(spec.tag.as_str()))
        })
}

pub fn installed_models(settings: &BittySettings) -> Vec<ModelSpec> {
    let mut specs = Vec::new();
    let Ok(names) = std::fs::read_dir(&settings.models_dir) else {
        return specs;
    };
    for name in names.flatten() {
        let Ok(tags) = std::fs::read_dir(name.path()) else {
            continue;
        };
        for tag in tags.flatten() {
            let manifest = tag.path().join("manifest.toml");
            if let Ok(contents) = std::fs::read_to_string(manifest) {
                specs.push(parse_manifest(&contents));
            }
        }
    }
    specs
}

pub fn resolve_model(settings: &BittySettings, name_or_path: &str) -> Option<ModelSpec> {
    let path = PathBuf::from(name_or_path);
    if path.exists() {
        return Some(ModelSpec {
            name: path
                .file_stem()
                .and_then(|name| name.to_str())
                .unwrap_or("local-model")
                .into(),
            tag: "local".into(),
            display_name: "Local GGUF model".into(),
            backend: "bitnet-i2s".into(),
            quantization: "i2_s".into(),
            filename: path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("model.gguf")
                .into(),
            layers: 30,
            temperature: settings.default_temperature,
            num_predict: settings.default_num_predict,
            num_ctx: settings.default_num_ctx,
            path: Some(path),
            ..Default::default()
        });
    }
    let name_lower = name_or_path.to_lowercase();
    installed_models(settings)
        .into_iter()
        .find(|spec| {
            spec.id().to_lowercase() == name_lower
                || spec.name.to_lowercase() == name_lower
        })
        .or_else(|| find_registry_model(name_or_path))
}

pub fn pull_model(
    settings: &BittySettings,
    model: &str,
) -> Result<ModelSpec, Box<dyn std::error::Error>> {
    let spec = find_registry_model(model).ok_or_else(|| format!("unknown model: {model}"))?;
    let model_path = spec.model_path(settings);
    ensure_parent(&model_path)?;
    if model_path.exists() {
        write_manifest(settings, &spec, &model_path)?;
        return Ok(spec);
    }
    if spec.url.is_empty() {
        return Err(format!("model {} has no download URL", spec.id()).into());
    }
    let status = std::process::Command::new("curl")
        .arg("-L")
        .arg("--fail")
        .arg("--progress-bar")
        .arg("-o")
        .arg(&model_path)
        .arg(&spec.url)
        .status()?;
    if !status.success() {
        return Err(format!("download failed for {}", spec.id()).into());
    }
    write_manifest(settings, &spec, &model_path)?;
    Ok(spec)
}

pub fn write_manifest(
    settings: &BittySettings,
    spec: &ModelSpec,
    model_path: &Path,
) -> std::io::Result<()> {
    let manifest = spec.manifest_path(settings);
    ensure_parent(&manifest)?;
    std::fs::write(manifest, spec.to_manifest(model_path))
}

pub fn remove_model(settings: &BittySettings, model: &str) -> std::io::Result<()> {
    if let Some(spec) = installed_models(settings)
        .into_iter()
        .find(|spec| spec.id() == model || spec.name == model)
    {
        std::fs::remove_dir_all(spec.model_dir(settings))?;
    }
    Ok(())
}

pub fn copy_model(settings: &BittySettings, source: &str, dest: &str) -> std::io::Result<()> {
    let Some(mut spec) = resolve_model(settings, source) else {
        return Err(std::io::Error::new(
            std::io::ErrorKind::NotFound,
            "source model not found",
        ));
    };
    let source_path = spec.model_path(settings);
    spec.name = dest.into();
    spec.tag = "latest".into();
    spec.path = Some(source_path.clone());
    write_manifest(settings, &spec, &source_path)
}

pub fn parse_registry(contents: &str) -> Vec<ModelSpec> {
    let mut models = Vec::new();
    let mut current = ModelSpec::default();
    let mut in_model = false;
    for line in contents.lines() {
        let line = line.trim();
        if line == "[[model]]" {
            if in_model {
                models.push(current);
            }
            current = ModelSpec::default();
            current.tag = "latest".into();
            in_model = true;
            continue;
        }
        if let Some((key, value)) = parse_assignment(line) {
            set_spec_value(&mut current, key, value);
        }
    }
    if in_model {
        models.push(current);
    }
    models
}

pub fn parse_manifest(contents: &str) -> ModelSpec {
    let mut spec = ModelSpec::default();
    for line in contents.lines() {
        if let Some((key, value)) = parse_assignment(line) {
            set_spec_value(&mut spec, key, value);
        }
    }
    spec
}

fn set_spec_value(spec: &mut ModelSpec, key: &str, value: &str) {
    match key {
        "name" => spec.name = value.into(),
        "tag" => spec.tag = value.into(),
        "display_name" => spec.display_name = value.into(),
        "backend" => spec.backend = value.into(),
        "quantization" => spec.quantization = value.into(),
        "filename" => spec.filename = value.into(),
        "layers" => spec.layers = value.parse().unwrap_or_default(),
        "url" => spec.url = value.into(),
        "source" => spec.source = value.into(),
        "temperature" => spec.temperature = value.parse().unwrap_or(0.7),
        "num_predict" => spec.num_predict = value.parse().unwrap_or(128),
        "num_ctx" => spec.num_ctx = value.parse().unwrap_or(2048),
        "path" => spec.path = Some(PathBuf::from(unquote(value))),
        _ => {}
    }
}

fn split_model_tag(name: &str) -> (&str, Option<&str>) {
    name.split_once(':')
        .map(|(name, tag)| (name, Some(tag)))
        .unwrap_or((name, None))
}

fn esc(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_contains_bitnet() {
        let model = find_registry_model("bitnet-b1.58").unwrap();
        assert_eq!(model.filename, "ggml-model-i2_s.gguf");
    }
}
