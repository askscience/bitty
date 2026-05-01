use bitty_protocol::{LayerAssignment, NodeId};
use thiserror::Error;

#[derive(Clone, Debug, PartialEq)]
pub struct RingTopology {
    pub epoch: String,
    pub assignments: Vec<LayerAssignment>,
}

impl RingTopology {
    pub fn new(epoch: impl Into<String>, mut assignments: Vec<LayerAssignment>) -> Self {
        let node_ids = assignments
            .iter()
            .map(|assignment| assignment.node_id.clone())
            .collect::<Vec<_>>();
        let len = assignments.len();

        if len > 0 {
            for (index, assignment) in assignments.iter_mut().enumerate() {
                assignment.next_node_id = Some(node_ids[(index + 1) % len].clone());
            }
        }

        Self {
            epoch: epoch.into(),
            assignments,
        }
    }

    pub fn first_node(&self) -> Option<&NodeId> {
        self.assignments
            .first()
            .map(|assignment| &assignment.node_id)
    }

    pub fn next_after(&self, node_id: &NodeId) -> Result<&NodeId, TopologyError> {
        self.assignments
            .iter()
            .find(|assignment| &assignment.node_id == node_id)
            .and_then(|assignment| assignment.next_node_id.as_ref())
            .ok_or_else(|| TopologyError::UnknownNode(node_id.clone()))
    }
}

#[derive(Debug, Error)]
pub enum TopologyError {
    #[error("unknown topology node {0}")]
    UnknownNode(NodeId),
}
