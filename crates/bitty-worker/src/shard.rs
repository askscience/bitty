use bitty_model::{ShardError, WeightShard};
use bitty_protocol::{AssignedLayerRange, NodeId};
use std::collections::HashMap;

#[derive(Clone, Debug, Default)]
pub struct ShardStore {
    shards: HashMap<NodeId, WeightShard>,
}

impl ShardStore {
    pub fn insert_verified(&mut self, shard: WeightShard) -> Result<(), ShardError> {
        shard.verify()?;
        self.shards.insert(shard.manifest.node_id.clone(), shard);
        Ok(())
    }

    pub fn range_for(&self, node_id: &NodeId) -> Option<&AssignedLayerRange> {
        self.shards.get(node_id).map(|shard| &shard.manifest.range)
    }
}
