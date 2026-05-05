use std::path::{Path, PathBuf};

#[derive(Clone, Debug, PartialEq)]
pub struct BittySettings {
    pub data_dir: PathBuf,
    pub models_dir: PathBuf,
    pub default_model: String,
    pub api_host: String,
    pub auto_pull: bool,
    pub auto_start_node: bool,
    pub default_temperature: f32,
    pub default_num_predict: u32,
    pub default_num_ctx: u32,
    pub iroh_relays: String,
    pub cluster_mode: String,
    pub cluster_name: String,
    pub cluster_description: String,
    pub active_cluster: String,
}

impl BittySettings {
    pub fn load(data_dir: PathBuf) -> Self {
        let mut settings = Self::defaults(data_dir);
        let path = settings.path();
        if let Ok(contents) = std::fs::read_to_string(path) {
            for line in contents.lines() {
                let Some((key, value)) = parse_assignment(line) else {
                    continue;
                };
                settings.set_value(key, value);
            }
        }
        settings
    }

    pub fn defaults(data_dir: PathBuf) -> Self {
        let models_dir = data_dir.join("models");
        Self {
            data_dir,
            models_dir,
            default_model: "bitnet-b1.58".into(),
            api_host: "127.0.0.1:11435".into(),
            auto_pull: true,
            auto_start_node: true,
            default_temperature: 0.7,
            default_num_predict: 128,
            default_num_ctx: 2048,
            iroh_relays: "public".into(),
            cluster_mode: "private".into(),
            cluster_name: String::new(),
            cluster_description: String::new(),
            active_cluster: String::new(),
        }
    }

    pub fn path(&self) -> PathBuf {
        self.data_dir.join("config.toml")
    }

    pub fn save(&self) -> std::io::Result<()> {
        std::fs::create_dir_all(&self.data_dir)?;
        std::fs::write(self.path(), self.to_toml())
    }

    pub fn get(&self, key: &str) -> Option<String> {
        match key {
            "data_dir" => Some(self.data_dir.display().to_string()),
            "models_dir" => Some(self.models_dir.display().to_string()),
            "default_model" => Some(self.default_model.clone()),
            "api_host" => Some(self.api_host.clone()),
            "auto_pull" => Some(self.auto_pull.to_string()),
            "auto_start_node" => Some(self.auto_start_node.to_string()),
            "default_temperature" => Some(self.default_temperature.to_string()),
            "default_num_predict" => Some(self.default_num_predict.to_string()),
            "default_num_ctx" => Some(self.default_num_ctx.to_string()),
            "iroh_relays" => Some(self.iroh_relays.clone()),
            "cluster_mode" => Some(self.cluster_mode.clone()),
            "cluster_name" => Some(self.cluster_name.clone()),
            "cluster_description" => Some(self.cluster_description.clone()),
            "active_cluster" => Some(self.active_cluster.clone()),
            _ => None,
        }
    }

    pub fn set_value(&mut self, key: &str, value: &str) -> bool {
        match key {
            "data_dir" => self.data_dir = PathBuf::from(value),
            "models_dir" => self.models_dir = PathBuf::from(value),
            "default_model" => self.default_model = value.into(),
            "api_host" => self.api_host = value.into(),
            "auto_pull" => self.auto_pull = parse_bool(value),
            "auto_start_node" => self.auto_start_node = parse_bool(value),
            "default_temperature" => {
                self.default_temperature = value.parse().unwrap_or(self.default_temperature)
            }
            "default_num_predict" => {
                self.default_num_predict = value.parse().unwrap_or(self.default_num_predict)
            }
            "default_num_ctx" => {
                self.default_num_ctx = value.parse().unwrap_or(self.default_num_ctx)
            }
            "iroh_relays" => self.iroh_relays = value.into(),
            "cluster_mode" => self.cluster_mode = value.into(),
            "cluster_name" => self.cluster_name = value.into(),
            "cluster_description" => self.cluster_description = value.into(),
            "active_cluster" => self.active_cluster = value.into(),
            _ => return false,
        }
        true
    }

    pub fn to_toml(&self) -> String {
        format!(
            "data_dir = \"{}\"\nmodels_dir = \"{}\"\ndefault_model = \"{}\"\napi_host = \"{}\"\nauto_pull = {}\nauto_start_node = {}\ndefault_temperature = {}\ndefault_num_predict = {}\ndefault_num_ctx = {}\niroh_relays = \"{}\"\ncluster_mode = \"{}\"\ncluster_name = \"{}\"\ncluster_description = \"{}\"\nactive_cluster = \"{}\"\n",
            escape(&self.data_dir.display().to_string()),
            escape(&self.models_dir.display().to_string()),
            escape(&self.default_model),
            escape(&self.api_host),
            self.auto_pull,
            self.auto_start_node,
            self.default_temperature,
            self.default_num_predict,
            self.default_num_ctx,
            escape(&self.iroh_relays),
            escape(&self.cluster_mode),
            escape(&self.cluster_name),
            escape(&self.cluster_description),
            escape(&self.active_cluster)
        )
    }
}

pub fn parse_assignment(line: &str) -> Option<(&str, &str)> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') || line.starts_with('[') {
        return None;
    }
    let (key, value) = line.split_once('=')?;
    Some((key.trim(), unquote(value.trim())))
}

pub fn unquote(value: &str) -> &str {
    value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or(value)
}

fn parse_bool(value: &str) -> bool {
    matches!(value, "true" | "1" | "yes" | "on")
}

fn escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

pub fn ensure_parent(path: &Path) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn settings_set_and_get_values() {
        let mut settings = BittySettings::defaults(PathBuf::from("/tmp/bitty"));
        assert!(settings.set_value("default_temperature", "0.2"));
        assert_eq!(settings.get("default_temperature").as_deref(), Some("0.2"));
    }
}
