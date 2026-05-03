use bitty_model::{MappedWeightShard, ShardError, WeightShard, WeightShardManifest};
use bitty_protocol::{AssignedLayerRange, NodeId};
use std::collections::HashMap;
use std::path::Path;

#[derive(Debug, Default)]
pub struct ShardStore {
    shards: HashMap<NodeId, WeightShard>,
    mapped: HashMap<NodeId, MappedWeightShard>,
}

impl ShardStore {
    pub fn insert_verified(&mut self, shard: WeightShard) -> Result<(), ShardError> {
        shard.verify()?;
        self.shards.insert(shard.manifest.node_id.clone(), shard);
        Ok(())
    }

    pub fn mmap_verified(
        &mut self,
        manifest: WeightShardManifest,
        path: impl AsRef<Path>,
    ) -> Result<(), ShardError> {
        let shard = WeightShard::mmap(manifest, path)?;
        self.mapped.insert(shard.manifest.node_id.clone(), shard);
        Ok(())
    }

    pub fn range_for(&self, node_id: &NodeId) -> Option<&AssignedLayerRange> {
        self.shards
            .get(node_id)
            .map(|shard| &shard.manifest.range)
            .or_else(|| self.mapped.get(node_id).map(|shard| &shard.manifest.range))
    }

    pub fn touch(&self, node_id: &NodeId) -> Option<u8> {
        self.shards
            .get(node_id)
            .and_then(|shard| shard.bytes.first().copied())
            .or_else(|| {
                self.mapped
                    .get(node_id)
                    .and_then(|shard| shard.bytes().first().copied())
            })
    }

    pub fn is_ready(&self, node_id: &NodeId) -> bool {
        self.range_for(node_id).is_some() && self.touch(node_id).is_some()
    }

    pub fn loaded_nodes(&self) -> Vec<NodeId> {
        self.shards
            .keys()
            .chain(self.mapped.keys())
            .cloned()
            .collect()
    }
}
