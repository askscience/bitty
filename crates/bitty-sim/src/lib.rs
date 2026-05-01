use bitty_coordinator::{Halda, RingTopology, SchedulerConfig};
use bitty_inference::FakeLayerExecutor;
use bitty_protocol::{
    ActivationDType, ActivationTensor, HardwareProfile, LayerMetadata, NodeId, NodeTier,
};
use bitty_worker::{RingWorker, RingWorkerError};
use std::sync::Arc;
use thiserror::Error;

#[derive(Clone, Debug, Default)]
pub struct ChaosProfile {
    pub corrupt_node: Option<NodeId>,
    pub drop_node: Option<NodeId>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct HopLatency {
    pub node_id: NodeId,
    pub simulated_micros: u64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct SimulationReport {
    pub final_activation: ActivationTensor,
    pub hops: Vec<HopLatency>,
}

pub struct SimulatedCluster {
    topology: RingTopology,
    workers: Vec<RingWorker<FakeLayerExecutor>>,
    chaos: ChaosProfile,
}

impl SimulatedCluster {
    pub fn build(
        profiles: Vec<HardwareProfile>,
        layers: Vec<LayerMetadata>,
    ) -> Result<Self, SimulationError> {
        let assignments = Halda::new(SchedulerConfig::default()).assign(&profiles, &layers)?;
        let topology = RingTopology::new("sim-epoch-0", assignments.clone());
        let executor = Arc::new(FakeLayerExecutor);
        let workers = assignments
            .into_iter()
            .map(|assignment| {
                RingWorker::new(assignment.node_id.clone(), assignment, executor.clone())
            })
            .collect();

        Ok(Self {
            topology,
            workers,
            chaos: ChaosProfile::default(),
        })
    }

    pub fn with_chaos(mut self, chaos: ChaosProfile) -> Self {
        self.chaos = chaos;
        self
    }

    pub async fn run_token(&self, request_id: &str) -> Result<SimulationReport, SimulationError> {
        let mut activation = ActivationTensor::new(
            request_id,
            0,
            0,
            0,
            vec![4],
            ActivationDType::Fp16,
            vec![1, 2, 3, 4],
        );
        let mut hops = Vec::new();

        for worker in &self.workers {
            if self.chaos.drop_node.as_ref() == Some(&worker.node_id) {
                return Err(SimulationError::DroppedNode(worker.node_id.clone()));
            }

            if self.chaos.corrupt_node.as_ref() == Some(&worker.node_id) {
                activation.payload.push(255);
            }

            let before = activation.payload.len() as u64;
            activation = worker.forward(activation).await?;
            hops.push(HopLatency {
                node_id: worker.node_id.clone(),
                simulated_micros: before * 10,
            });
        }

        Ok(SimulationReport {
            final_activation: activation,
            hops,
        })
    }

    pub fn topology(&self) -> &RingTopology {
        &self.topology
    }
}

#[derive(Debug, Error)]
pub enum SimulationError {
    #[error(transparent)]
    Scheduler(#[from] bitty_coordinator::HaldaError),
    #[error(transparent)]
    Worker(#[from] RingWorkerError),
    #[error("simulated node drop: {0}")]
    DroppedNode(NodeId),
}

pub fn demo_profiles(count: usize) -> Vec<HardwareProfile> {
    (0..count)
        .map(|index| HardwareProfile {
            node_id: NodeId::new(format!("sim-{index}")),
            cpu_tflops: 0.5 + index as f64,
            gpu_tflops: if index == 0 { 20.0 } else { 0.0 },
            memory_gb: 4.0,
            memory_bandwidth_gbps: 20.0,
            disk_bandwidth_mbps: 400.0,
            network_rtt_ms: 10.0 + index as f64,
            uplink_mbps: 100.0,
            os: "linux".into(),
            tier: if index == 0 { NodeTier::S } else { NodeTier::D },
        })
        .collect()
}

pub fn demo_layers(count: u32) -> Vec<LayerMetadata> {
    (0..count)
        .map(|layer_id| LayerMetadata {
            layer_id,
            weight_bytes: 512_000,
            activation_bytes: 4096,
            estimated_flops: 1e9,
            precision_critical: layer_id == 0 || layer_id + 1 == count,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn simulated_cluster_runs_ring() {
        let cluster = SimulatedCluster::build(demo_profiles(4), demo_layers(8)).unwrap();
        let report = cluster.run_token("req").await.unwrap();

        assert_eq!(report.hops.len(), cluster.topology().assignments.len());
        assert!(report.final_activation.verify_checksum());
    }

    #[tokio::test]
    async fn chaos_drop_reports_node() {
        let cluster = SimulatedCluster::build(demo_profiles(4), demo_layers(8))
            .unwrap()
            .with_chaos(ChaosProfile {
                drop_node: Some(NodeId::new("sim-0")),
                corrupt_node: None,
            });

        assert!(matches!(
            cluster.run_token("req").await,
            Err(SimulationError::DroppedNode(_))
        ));
    }
}
