use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ClusterStore {
    aliases: BTreeMap<String, String>,
}

impl ClusterStore {
    pub fn load(data_dir: &Path) -> Self {
        let path = path(data_dir);
        let mut store = Self::default();
        if let Ok(contents) = std::fs::read_to_string(path) {
            for line in contents.lines() {
                let line = line.trim();
                if line.is_empty() || line.starts_with('#') {
                    continue;
                }
                let Some((name, invite)) = line.split_once('=') else {
                    continue;
                };
                store
                    .aliases
                    .insert(name.trim().to_string(), unquote(invite.trim()).to_string());
            }
        }
        store
    }

    pub fn save(&self, data_dir: &Path) -> std::io::Result<()> {
        std::fs::create_dir_all(data_dir)?;
        let mut contents = String::new();
        for (name, invite) in &self.aliases {
            contents.push_str(name);
            contents.push_str(" = \"");
            contents.push_str(&escape(invite));
            contents.push_str("\"\n");
        }
        std::fs::write(path(data_dir), contents)
    }

    pub fn get(&self, name: &str) -> Option<&str> {
        self.aliases.get(name).map(String::as_str)
    }

    pub fn insert(
        &mut self,
        name: Option<&str>,
        invite: &str,
        replace: bool,
    ) -> Result<String, String> {
        let name = name
            .map(normalize_name)
            .filter(|name| !name.is_empty())
            .unwrap_or_else(|| self.generated_name(invite));
        if let Some(existing) = self.aliases.get(&name) {
            if existing == invite {
                return Ok(name);
            }
            if !replace {
                return Err(format!(
                    "cluster name `{name}` already exists for a different invite. Use --replace or choose another name."
                ));
            }
        }
        self.aliases.insert(name.clone(), invite.to_string());
        Ok(name)
    }

    pub fn aliases(&self) -> impl Iterator<Item = (&str, &str)> {
        self.aliases
            .iter()
            .map(|(name, invite)| (name.as_str(), invite.as_str()))
    }

    fn generated_name(&self, invite: &str) -> String {
        let suffix = short_hash(invite);
        let base = format!("cluster-{suffix}");
        if !self.aliases.contains_key(&base) {
            return base;
        }
        for index in 2..1000 {
            let candidate = format!("{base}-{index}");
            if !self.aliases.contains_key(&candidate) {
                return candidate;
            }
        }
        format!("{base}-{}", self.aliases.len() + 1)
    }
}

pub fn path(data_dir: &Path) -> PathBuf {
    data_dir.join("clusters.toml")
}

pub fn looks_like_invite(value: &str) -> bool {
    value.starts_with("iroh://")
}

fn normalize_name(value: &str) -> String {
    value
        .trim()
        .chars()
        .map(|ch| {
            if ch.is_ascii_alphanumeric() || ch == '-' || ch == '_' {
                ch.to_ascii_lowercase()
            } else {
                '-'
            }
        })
        .collect::<String>()
        .trim_matches('-')
        .to_string()
}

fn short_hash(value: &str) -> String {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in value.as_bytes() {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    format!("{:08x}", (hash & 0xffff_ffff) as u32)
}

fn unquote(value: &str) -> &str {
    value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .unwrap_or(value)
}

fn escape(value: &str) -> String {
    value.replace('\\', "\\\\").replace('"', "\\\"")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn refuses_alias_conflicts() {
        let mut store = ClusterStore::default();
        assert_eq!(
            store.insert(Some("Home"), "iroh://a", false).unwrap(),
            "home"
        );
        assert!(store.insert(Some("home"), "iroh://b", false).is_err());
        assert_eq!(
            store.insert(Some("home"), "iroh://b", true).unwrap(),
            "home"
        );
        assert_eq!(store.get("home"), Some("iroh://b"));
    }
}
