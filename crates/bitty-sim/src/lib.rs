use bitty_coordinator::{Halda, RingTopology, SchedulerConfig};
use bitty_inference::FakeLayerExecutor;
use bitty_protocol::{
    ActivationDType, ActivationTensor, HardwareProfile, LayerMetadata, NodeId, NodeTier,
    TokenOutput,
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

#[derive(Clone, Debug, PartialEq)]
pub struct StreamedSimulation {
    pub tokens: Vec<TokenOutput>,
    pub reports: Vec<SimulationReport>,
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
        self.run_token_at(request_id, 0, vec![1, 2, 3, 4]).await
    }

    pub async fn run_token_at(
        &self,
        request_id: &str,
        token_position: u32,
        payload: Vec<u8>,
    ) -> Result<SimulationReport, SimulationError> {
        let mut activation = ActivationTensor::new(
            request_id,
            token_position,
            0,
            0,
            vec![payload.len() as u32],
            ActivationDType::Fp16,
            payload,
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

    pub async fn stream_tokens(
        &self,
        request_id: &str,
        token_count: u32,
    ) -> Result<StreamedSimulation, SimulationError> {
        let mut reports = Vec::new();
        let mut tokens = Vec::new();

        for position in 0..token_count {
            let seed = position.to_le_bytes().to_vec();
            let report = self.run_token_at(request_id, position, seed).await?;
            let token_id = report
                .final_activation
                .payload
                .chunks_exact(4)
                .last()
                .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
                .unwrap_or(position);
            let gen_latency_us = report
                .hops
                .iter()
                .map(|hop| hop.simulated_micros)
                .sum::<u64>();
            tokens.push(TokenOutput {
                request_id: request_id.into(),
                token_position: position,
                token_id,
                text: format!("<tok:{token_id}>"),
                finished: position + 1 == token_count,
                log_prob: 0.0,
                gen_latency_us,
            });
            reports.push(report);
        }

        Ok(StreamedSimulation { tokens, reports })
    }

    pub async fn run_batch(
        &self,
        request_ids: &[String],
        token_count: u32,
    ) -> Result<Vec<StreamedSimulation>, SimulationError> {
        let mut outputs = Vec::with_capacity(request_ids.len());
        for request_id in request_ids {
            outputs.push(self.stream_tokens(request_id, token_count).await?);
        }
        Ok(outputs)
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
            ram_mb: 4096,
            vram_mb: if index == 0 { 8192 } else { 0 },
            architecture: "x86_64".into(),
            gpus: Vec::new(),
            os_reclaim_score: 0.0,
            worker_endpoint: String::new(),
            model_path: String::new(),
            backend_type: if index == 0 { "gpu" } else { "cpu" }.into(),
            layer_eligible: true,
            max_layers: u32::MAX,
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

    #[tokio::test]
    async fn simulated_cluster_streams_tokens() {
        let cluster = SimulatedCluster::build(demo_profiles(4), demo_layers(8)).unwrap();
        let output = cluster.stream_tokens("req", 3).await.unwrap();

        assert_eq!(output.tokens.len(), 3);
        assert!(output.tokens.last().unwrap().finished);
        assert_eq!(output.reports.len(), 3);
    }
}
