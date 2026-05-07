use std::collections::HashMap;
use std::sync::Mutex;

/// Bind Group Cache — caches wgpu::BindGroup objects by string key.
///
/// Uses interior mutability so `insert` and `get_or_insert_with` work
/// through `&self`, enabling caching during forward passes.
pub struct BgCache {
    entries: Mutex<HashMap<String, wgpu::BindGroup>>,
}

impl Default for BgCache {
    fn default() -> Self {
        Self::new()
    }
}

impl BgCache {
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    pub fn get(&self, key: &str) -> Option<wgpu::BindGroup> {
        self.entries.lock().unwrap().get(key).cloned()
    }

    pub fn insert(&self, key: impl Into<String>, bind_group: wgpu::BindGroup) {
        self.entries.lock().unwrap().insert(key.into(), bind_group);
    }

    pub fn get_or_insert_with(
        &self,
        key: impl Into<String>,
        f: impl FnOnce() -> wgpu::BindGroup,
    ) -> wgpu::BindGroup {
        let key = key.into();
        let mut entries = self.entries.lock().unwrap();
        if let Some(bg) = entries.get(&key) {
            return bg.clone();
        }
        let bg = f();
        entries.insert(key, bg.clone());
        bg
    }

    pub fn clear(&mut self) {
        self.entries.get_mut().unwrap().clear();
    }
}
