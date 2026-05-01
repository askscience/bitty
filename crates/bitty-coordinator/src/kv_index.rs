use bitty_inference::PrefixCacheKey;
use bitty_protocol::NodeId;
use std::collections::HashMap;
use std::time::{Duration, Instant};

#[derive(Clone, Debug)]
pub struct CacheEntry {
    pub owner: NodeId,
    pub expires_at: Instant,
}

#[derive(Clone, Debug, Default)]
pub struct KvIndex {
    entries: HashMap<PrefixCacheKey, CacheEntry>,
}

impl KvIndex {
    pub fn put(&mut self, key: PrefixCacheKey, owner: NodeId, ttl: Duration) {
        self.entries.insert(
            key,
            CacheEntry {
                owner,
                expires_at: Instant::now() + ttl,
            },
        );
    }

    pub fn get(&mut self, key: &PrefixCacheKey) -> Option<NodeId> {
        let entry = self.entries.get(key)?;
        if Instant::now() > entry.expires_at {
            self.entries.remove(key);
            return None;
        }
        Some(entry.owner.clone())
    }

    pub fn flush(&mut self) {
        self.entries.clear();
    }
}
