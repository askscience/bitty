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
    /// A small model with the same vocabulary: on the leader it guesses the next tokens.
    pub draft_filename: String,
    pub draft_url: String,
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
    registry_models()
        .into_iter()
        .find(|spec| spec.name == model && (tag.is_none() || tag == Some(spec.tag.as_str())))
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
            backend: "llama-cpp".into(),
            quantization: "q4_k_m".into(),
            filename: path
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or("model.gguf")
                .into(),
            layers: 40,
            temperature: settings.default_temperature,
            num_predict: settings.default_num_predict,
            num_ctx: settings.default_num_ctx,
            path: Some(path),
            ..Default::default()
        });
    }
    installed_models(settings)
        .into_iter()
        .find(|spec| spec.id() == name_or_path || spec.name == name_or_path)
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

/// Where the stage file for layers `[start, end)` of a model lives.
pub fn stage_path(settings: &BittySettings, spec: &ModelSpec, start: u32, end: u32) -> PathBuf {
    let stem = Path::new(&spec.filename)
        .file_stem()
        .and_then(|stem| stem.to_str())
        .unwrap_or("model");
    spec.model_dir(settings)
        .join("stages")
        .join(format!("{stem}.layers-{start}-{end}.gguf"))
}

/// The model a stage comes from: a registry name, a local GGUF or an http(s) URL.
fn stage_spec(settings: &BittySettings, model: &str) -> Result<(ModelSpec, bool), String> {
    let is_url = model.starts_with("https://") || model.starts_with("http://");
    if !is_url {
        return resolve_model(settings, model)
            .map(|spec| (spec, false))
            .ok_or_else(|| format!("unknown model: {model}"));
    }
    let filename = model
        .split(['?', '#'])
        .next()
        .and_then(|path| path.rsplit('/').next())
        .filter(|name| !name.is_empty())
        .unwrap_or("model.gguf");
    let spec = ModelSpec {
        name: Path::new(filename)
            .file_stem()
            .and_then(|stem| stem.to_str())
            .unwrap_or("model")
            .into(),
        tag: "remote".into(),
        filename: filename.into(),
        url: model.into(),
        ..Default::default()
    };
    Ok((spec, true))
}

/// Fetches only the layers `[start, end)` of a model into a stage file.
///
/// `model` is a registry name, a local GGUF or an http(s) URL. A whole copy already on disk is
/// used as the source, otherwise the tensors come from the model URL with range requests.
/// The range `0..0` gives the vocabulary file: the metadata and no weights.
pub fn pull_stage(
    settings: &BittySettings,
    model: &str,
    start: u32,
    end: u32,
    output: Option<&Path>,
    cancel: Option<bitty_model::CancelCheck>,
) -> Result<(PathBuf, bitty_model::StageFileReport), Box<dyn std::error::Error>> {
    let (spec, is_url) = stage_spec(settings, model)?;
    let dest = match output {
        Some(path) => path.to_path_buf(),
        None => stage_path(settings, &spec, start, end),
    };
    ensure_parent(&dest)?;
    let whole = spec.model_path(settings);
    let report = if !is_url && whole.exists() {
        bitty_model::write_stage_file(&bitty_model::FileSource::new(whole), start, end, &dest)?
    } else if spec.url.is_empty() {
        return Err(format!("model {} has no download URL", spec.id()).into());
    } else {
        let source = bitty_model::HttpSource::new(&spec.url).with_cancel(cancel);
        bitty_model::write_stage_file(&source, start, end, &dest)?
    };
    Ok((dest, report))
}

/// The stage file for `[start, end)`, fetched only when it is not on disk yet.
pub fn ensure_stage(
    settings: &BittySettings,
    model: &str,
    start: u32,
    end: u32,
    cancel: Option<bitty_model::CancelCheck>,
) -> Result<PathBuf, String> {
    let (spec, _) = stage_spec(settings, model)?;
    let dest = stage_path(settings, &spec, start, end);
    if dest.is_file() {
        return Ok(dest);
    }
    eprintln!("bitty: fetching layers {start}..{end} of {}", spec.id());
    let mut fetched = pull_stage(settings, model, start, end, None, cancel.clone());
    let no_space = |err: &Box<dyn std::error::Error>| {
        matches!(
            err.downcast_ref::<bitty_model::StageFileError>(),
            Some(bitty_model::StageFileError::NoSpace { .. })
        )
    };
    // Layers of an older plan are of no use once the plan has changed: they make room first.
    if fetched.as_ref().is_err_and(no_space) && remove_other_stages(settings, &spec, &dest) > 0 {
        fetched = pull_stage(settings, model, start, end, None, cancel);
    }
    let (dest, report) = fetched.map_err(|err| {
        eprintln!("bitty: layers {start}..{end} not fetched: {err}");
        err.to_string()
    })?;
    eprintln!(
        "bitty: layers {start}..{end} ready, {} MiB in {}",
        report.bytes / (1024 * 1024),
        dest.display()
    );
    Ok(dest)
}

/// The stage files of a model other than `keep` and the vocabulary, with their sizes.
fn other_stages(settings: &BittySettings, spec: &ModelSpec, keep: &Path) -> Vec<(PathBuf, u64)> {
    let vocabulary = stage_path(settings, spec, 0, 0);
    let Ok(entries) = std::fs::read_dir(spec.model_dir(settings).join("stages")) else {
        return Vec::new();
    };
    entries
        .flatten()
        .map(|entry| entry.path())
        // the partial file of `keep` is a download that goes on, not an older plan
        .filter(|path| path != keep && path != &vocabulary && path.with_extension("") != keep)
        .filter(|path| {
            path.extension()
                .is_some_and(|extension| extension == "gguf" || extension == "partial")
        })
        .filter_map(|path| {
            let len = std::fs::metadata(&path).ok()?.len();
            Some((path, len))
        })
        .collect()
}

/// Deletes the stage files of other layer ranges. Returns how many went.
fn remove_other_stages(settings: &BittySettings, spec: &ModelSpec, keep: &Path) -> usize {
    other_stages(settings, spec, keep)
        .into_iter()
        .filter(|(path, len)| {
            let removed = std::fs::remove_file(path).is_ok();
            if removed {
                eprintln!(
                    "bitty: removed {} ({} MiB) to make room",
                    path.display(),
                    len / (1024 * 1024)
                );
            }
            removed
        })
        .count()
}

/// The disk a node can use for the layers of a model, in MB: what is free plus the layers of
/// older plans, which go when room is needed. `None` when the node holds the whole model or the
/// disk cannot be read.
pub fn disk_for_stages_mb(settings: &BittySettings, model: &str) -> Option<u64> {
    if whole_model_path(settings, model).is_some() {
        return None;
    }
    let (spec, _) = stage_spec(settings, model).ok()?;
    let dir = spec.model_dir(settings).join("stages");
    let free = bitty_model::usable_disk_bytes(&dir)?;
    let held: u64 = other_stages(settings, &spec, Path::new(""))
        .iter()
        .map(|(_, len)| len)
        .sum();
    Some((free + held) / (1024 * 1024))
}

/// The model as any node of the network can fetch it: its name in the registry, also when
/// it was given as the path of the installed file, or the URL of a GGUF. `None` for a file
/// of this machine alone.
pub fn network_name(settings: &BittySettings, model: &str) -> Option<String> {
    if model.starts_with("https://") || model.starts_with("http://") {
        return Some(model.into());
    }
    let name = match Path::new(model).strip_prefix(&settings.models_dir) {
        Ok(inside) => inside
            .components()
            .next()?
            .as_os_str()
            .to_str()?
            .to_string(),
        Err(_) => model.to_string(),
    };
    find_registry_model(&name)
        .filter(|spec| !spec.url.is_empty())
        .map(|spec| spec.name)
}

/// The name of a model that lies in the store of this machine, whether the registry still
/// lists it or not. `None` for a file anywhere else.
pub fn stored_name(settings: &BittySettings, model: &str) -> Option<String> {
    let inside = Path::new(model).strip_prefix(&settings.models_dir).ok()?;
    Some(
        inside
            .components()
            .next()?
            .as_os_str()
            .to_str()?
            .to_string(),
    )
}

/// What a node holds of a model and what it is still fetching, in words: the leader logs it,
/// so the state of a far machine can be read without going there.
pub fn holdings(settings: &BittySettings, model: &str) -> String {
    if whole_model_path(settings, model).is_some() {
        return "holds the whole model".into();
    }
    let Ok((spec, _)) = stage_spec(settings, model) else {
        return String::new();
    };
    let Ok(entries) = std::fs::read_dir(spec.model_dir(settings).join("stages")) else {
        return "holds no layers".into();
    };
    let mut parts = Vec::new();
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let (name, partial) = match name.strip_suffix(".partial") {
            Some(name) => (name, true),
            None => (name.as_str(), false),
        };
        let Some((start, end)) = name
            .strip_suffix(".gguf")
            .and_then(|name| name.rsplit_once(".layers-"))
            .and_then(|(_, range)| range.split_once('-'))
            .and_then(|(start, end)| Some((start.parse::<u32>().ok()?, end.parse::<u32>().ok()?)))
        else {
            continue;
        };
        // the stage with no layers holds the embeddings and the head only
        if start == end {
            continue;
        }
        let mb = entry.metadata().map(|meta| meta.len()).unwrap_or(0) / (1024 * 1024);
        parts.push((start, partial, end, mb));
    }
    if parts.is_empty() {
        return "holds no layers".into();
    }
    parts.sort();
    parts
        .iter()
        .map(|(start, partial, end, mb)| {
            if *partial {
                format!("fetching {start}..{end}, {mb} MiB so far")
            } else {
                format!("holds {start}..{end}")
            }
        })
        .collect::<Vec<_>>()
        .join("; ")
}

/// Whether a node can fetch stage files of this model: it is in the registry or it is a URL.
pub fn has_stage_source(settings: &BittySettings, model: &str) -> bool {
    stage_spec(settings, model).is_ok_and(|(spec, _)| !spec.url.is_empty())
}

/// The draft model of a model: where its file is kept and where it comes from.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DraftFile {
    pub path: PathBuf,
    pub url: String,
}

impl DraftFile {
    /// Downloads the file. A download that was cut goes on from where it stopped.
    pub fn fetch(&self) -> Result<(), String> {
        if self.url.is_empty() {
            return Err(format!("{} is not there", self.path.display()));
        }
        ensure_parent(&self.path).map_err(|err| err.to_string())?;
        let mut part = self.path.clone().into_os_string();
        part.push(".part");
        let status = std::process::Command::new("curl")
            .args(["-L", "--fail", "--silent", "--show-error", "--retry", "5"])
            .args(["-C", "-", "-o"])
            .arg(&part)
            .arg(&self.url)
            .status()
            .map_err(|err| format!("curl: {err}"))?;
        if !status.success() {
            return Err(format!("download failed: {}", self.url));
        }
        std::fs::rename(&part, &self.path).map_err(|err| err.to_string())
    }
}

/// The draft model the leader of `model` uses, if there is one: the file named by
/// `BITTY_DRAFT_MODEL`, otherwise the one of the registry, kept next to the model.
/// `BITTY_DRAFT_MODEL=off` gives none.
pub fn draft_model(settings: &BittySettings, model: &str) -> Option<DraftFile> {
    if let Some(path) = std::env::var_os("BITTY_DRAFT_MODEL").filter(|path| !path.is_empty()) {
        return (path != "off").then(|| DraftFile {
            path: PathBuf::from(path),
            url: String::new(),
        });
    }
    let (spec, is_url) = stage_spec(settings, model).ok()?;
    if is_url {
        return None;
    }
    // a model installed by an older build has a manifest without the draft
    let spec = if spec.draft_filename.is_empty() {
        find_registry_model(&spec.name).filter(|known| !known.draft_filename.is_empty())?
    } else {
        spec
    };
    Some(DraftFile {
        path: spec.model_dir(settings).join(&spec.draft_filename),
        url: spec.draft_url,
    })
}

pub fn whole_model_path(settings: &BittySettings, model: &str) -> Option<PathBuf> {
    let direct = Path::new(model);
    if direct.is_file() {
        return Some(direct.to_path_buf());
    }
    let (spec, is_url) = stage_spec(settings, model).ok()?;
    let path = spec.model_path(settings);
    (!is_url && path.is_file()).then_some(path)
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
        "draft_filename" => spec.draft_filename = value.into(),
        "draft_url" => spec.draft_url = value.into(),
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
    fn registry_contains_qwen() {
        let model = find_registry_model("qwen3.5-35b-a3b").unwrap();
        assert_eq!(model.filename, "Qwen3.5-35B-A3B-Q4_K_M.gguf");
    }
}
