use crate::{metrics, RingWorker};
use bitty_inference::{FakeLayerExecutor, LayerExecutor};
use bitty_protocol::pb::worker_service_server::{WorkerService, WorkerServiceServer};
use bitty_protocol::pb::{
    ActivationTensor as ProtoActivationTensor, BitNetLogits as ProtoBitNetLogits, CleanupRequest,
    CleanupResponse, HeartbeatResponse, LoadShardRequest, LoadShardResponse, TopologyUpdate,
};
use bitty_protocol::{ActivationTensor, LayerAssignment, NodeId, ShardManifestMessage};
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

#[derive(Clone)]
pub struct NetworkWorker<E = FakeLayerExecutor> {
    node_id: NodeId,
    assignment: Arc<Mutex<Option<LayerAssignment>>>,
    executor: Arc<E>,
    topology_epoch: Arc<Mutex<String>>,
    loaded_shard: Arc<Mutex<Option<ShardManifestMessage>>>,
}

impl NetworkWorker<FakeLayerExecutor> {
    pub fn with_fake_executor(node_id: impl Into<String>) -> Self {
        Self::new(NodeId::new(node_id), Arc::new(FakeLayerExecutor))
    }
}

impl<E> NetworkWorker<E>
where
    E: LayerExecutor + 'static,
{
    pub fn new(node_id: NodeId, executor: Arc<E>) -> Self {
        Self {
            node_id,
            assignment: Arc::new(Mutex::new(None)),
            executor,
            topology_epoch: Arc::new(Mutex::new("worker-epoch-0".into())),
            loaded_shard: Arc::new(Mutex::new(None)),
        }
    }

    pub async fn serve(self, listen_addr: &str) -> Result<(), Box<dyn std::error::Error>> {
        let addr = listen_addr.parse()?;
        println!("bitty-worker: serving activation RPCs on {listen_addr}");
        Server::builder()
            .add_service(WorkerServiceServer::new(self))
            .serve(addr)
            .await?;
        Ok(())
    }
}

#[tonic::async_trait]
impl<E> WorkerService for NetworkWorker<E>
where
    E: LayerExecutor + 'static,
{
    async fn forward_activation(
        &self,
        request: Request<ProtoActivationTensor>,
    ) -> Result<Response<ProtoActivationTensor>, Status> {
        let activation = ActivationTensor::try_from(request.into_inner())
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
        let assignment = self
            .assignment
            .lock()
            .await
            .clone()
            .ok_or_else(|| Status::failed_precondition("worker has no layer assignment"))?;
        let loaded_shard = self
            .loaded_shard
            .lock()
            .await
            .clone()
            .ok_or_else(|| Status::failed_precondition("worker shard is not loaded"))?;
        if loaded_shard.range != assignment.range {
            return Err(Status::failed_precondition("loaded shard is stale"));
        }

        let worker = RingWorker::new(self.node_id.clone(), assignment, self.executor.clone());
        let payload_len = activation.payload.len() as u64;
        let output = worker
            .forward(activation)
            .await
            .map_err(|err| Status::internal(err.to_string()))?;
        metrics::record_activation_bytes("rx", payload_len);
        metrics::record_activation_bytes("tx", output.payload.len() as u64);
        Ok(Response::new((&output).into()))
    }

    async fn final_logits(
        &self,
        request: Request<ProtoActivationTensor>,
    ) -> Result<Response<ProtoBitNetLogits>, Status> {
        let activation = ActivationTensor::try_from(request.into_inner())
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
        if !activation.verify_checksum() {
            return Err(Status::invalid_argument("activation checksum failed"));
        }
        self.loaded_shard
            .lock()
            .await
            .clone()
            .ok_or_else(|| Status::failed_precondition("worker shard is not loaded"))?;
        let logits = self
            .executor
            .final_logits(activation)
            .await
            .map_err(|err| Status::internal(err.to_string()))?;
        Ok(Response::new((&logits).into()))
    }

    async fn apply_topology(
        &self,
        request: Request<TopologyUpdate>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let update = request.into_inner();
        let assignment = update
            .assignments
            .into_iter()
            .find(|assignment| assignment.node_id == self.node_id.0)
            .map(LayerAssignment::try_from)
            .transpose()
            .map_err(|err| Status::invalid_argument(err.to_string()))?;

        *self.assignment.lock().await = assignment;
        *self.topology_epoch.lock().await = update.topology_epoch.clone();

        Ok(Response::new(HeartbeatResponse {
            accepted: true,
            topology_epoch: update.topology_epoch,
        }))
    }

    async fn load_shard(
        &self,
        request: Request<LoadShardRequest>,
    ) -> Result<Response<LoadShardResponse>, Status> {
        let manifest = request
            .into_inner()
            .manifest
            .ok_or_else(|| Status::invalid_argument("missing shard manifest"))?;
        let manifest = ShardManifestMessage::try_from(manifest)
            .map_err(|err| Status::invalid_argument(err.to_string()))?;

        if manifest.node_id != self.node_id {
            return Err(Status::permission_denied(
                "shard assigned to a different node",
            ));
        }
        if !manifest.path.is_empty() && !std::path::Path::new(&manifest.path).exists() {
            return Err(Status::not_found(format!(
                "shard path does not exist: {}",
                manifest.path
            )));
        }

        *self.loaded_shard.lock().await = Some(manifest);
        Ok(Response::new(LoadShardResponse {
            loaded: true,
            message: "shard manifest accepted; worker is ready for this topology".into(),
        }))
    }

    async fn cleanup(
        &self,
        request: Request<CleanupRequest>,
    ) -> Result<Response<CleanupResponse>, Status> {
        let request_id = request.into_inner().request_id;
        if request_id.is_empty() {
            return Err(Status::invalid_argument("missing request_id"));
        }
        Ok(Response::new(CleanupResponse { cleaned: true }))
    }
}
