use crate::{Halda, Registry, SchedulerConfig};
use bitty_protocol::pb::coordinator_service_server::{
    CoordinatorService, CoordinatorServiceServer,
};
use bitty_protocol::pb::{
    ActivationTensor as ProtoActivationTensor, HeartbeatRequest, HeartbeatResponse,
    RegisterWorkerRequest, RegisterWorkerResponse, TokenOutput as ProtoTokenOutput,
};
use bitty_protocol::{HardwareProfile, Heartbeat, LayerMetadata};
use futures::stream;
use std::pin::Pin;
use std::sync::Arc;
use tokio::sync::Mutex;
use tonic::transport::Server;
use tonic::{Request, Response, Status};

#[derive(Clone, Debug)]
pub struct NetworkCoordinator {
    registry: Arc<Mutex<Registry>>,
    layers: Arc<Vec<LayerMetadata>>,
    topology_epoch: Arc<Mutex<String>>,
}

impl NetworkCoordinator {
    pub fn new(layers: Vec<LayerMetadata>) -> Self {
        Self {
            registry: Arc::new(Mutex::new(Registry::default())),
            layers: Arc::new(layers),
            topology_epoch: Arc::new(Mutex::new("network-epoch-0".into())),
        }
    }

    pub async fn serve(self, listen_addr: &str) -> Result<(), Box<dyn std::error::Error>> {
        let addr = listen_addr.parse()?;
        println!("bitty-coordinator: listening on {listen_addr}");
        Server::builder()
            .add_service(CoordinatorServiceServer::new(self))
            .serve(addr)
            .await?;
        Ok(())
    }

    async fn current_assignments(&self) -> Result<Vec<bitty_protocol::LayerAssignment>, Status> {
        let profiles = self.registry.lock().await.profiles();
        if profiles.is_empty() {
            return Ok(Vec::new());
        }

        Halda::new(SchedulerConfig::default())
            .assign(&profiles, &self.layers)
            .map_err(|err| Status::failed_precondition(err.to_string()))
    }

    async fn advance_epoch(&self) -> String {
        let mut epoch = self.topology_epoch.lock().await;
        let next = epoch
            .rsplit_once('-')
            .and_then(|(_, suffix)| suffix.parse::<u64>().ok())
            .unwrap_or(0)
            + 1;
        *epoch = format!("network-epoch-{next}");
        epoch.clone()
    }

    async fn epoch(&self) -> String {
        self.topology_epoch.lock().await.clone()
    }
}

#[tonic::async_trait]
impl CoordinatorService for NetworkCoordinator {
    type StreamTokensStream =
        Pin<Box<dyn futures::Stream<Item = Result<ProtoTokenOutput, Status>> + Send + 'static>>;

    async fn register_worker(
        &self,
        request: Request<RegisterWorkerRequest>,
    ) -> Result<Response<RegisterWorkerResponse>, Status> {
        let profile = request
            .into_inner()
            .profile
            .ok_or_else(|| Status::invalid_argument("missing worker profile"))?;
        let profile = HardwareProfile::try_from(profile)
            .map_err(|err| Status::invalid_argument(err.to_string()))?;
        let node_id = profile.node_id.clone();

        self.registry.lock().await.register(profile);
        let topology_epoch = self.advance_epoch().await;
        let assignments = self.current_assignments().await?;

        println!(
            "registered worker {node_id}; active_nodes={} assignments={}",
            self.registry.lock().await.profiles().len(),
            assignments.len()
        );

        Ok(Response::new(RegisterWorkerResponse {
            assignments: assignments.iter().map(Into::into).collect(),
            topology_epoch,
        }))
    }

    async fn heartbeat(
        &self,
        request: Request<HeartbeatRequest>,
    ) -> Result<Response<HeartbeatResponse>, Status> {
        let heartbeat = Heartbeat::from(request.into_inner());
        let accepted = self.registry.lock().await.heartbeat(heartbeat);

        Ok(Response::new(HeartbeatResponse {
            accepted,
            topology_epoch: self.epoch().await,
        }))
    }

    async fn stream_tokens(
        &self,
        request: Request<ProtoActivationTensor>,
    ) -> Result<Response<Self::StreamTokensStream>, Status> {
        let activation = request.into_inner();
        let token = ProtoTokenOutput {
            request_id: activation.request_id,
            token_position: activation.token_position,
            token_id: 0,
            text: "<network-token-placeholder>".into(),
            finished: true,
        };

        Ok(Response::new(Box::pin(stream::iter([Ok(token)]))))
    }
}
