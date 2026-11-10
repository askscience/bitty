use bitty_protocol::endpoint::{normalize_endpoint, validate_worker_endpoint_for_dial};
use bitty_protocol::iroh_transport::{
    self, IrohFrame, BITTY_WORKER_ALPN, DEFAULT_FRAME_LIMIT, WORKER_CLEANUP,
    WORKER_FORWARD_ACTIVATION, WORKER_LOAD_SHARD,
};
use bitty_protocol::pb::worker_service_client::WorkerServiceClient;
use bitty_protocol::pb::{
    ActivationTensor as ProtoActivationTensor, LoadShardRequest,
};
use bitty_protocol::security::BITTY_TOKEN_HEADER;
use bitty_protocol::ActivationTensor;
use iroh::{Endpoint, EndpointAddr};
use tonic::{Request, Status};

pub async fn connect_worker(
    endpoint: &str,
    iroh_endpoint: Option<Endpoint>,
    cluster_token: Option<String>,
) -> Result<WorkerRpcClient, Status> {
    validate_worker_endpoint_for_dial(endpoint)
        .map_err(|err| Status::invalid_argument(err.to_string()))?;
    if let Some(target) = bitty_protocol::iroh_transport::parse_iroh_target(endpoint) {
        let iroh_endpoint = iroh_endpoint
            .ok_or_else(|| Status::failed_precondition("coordinator has no iroh endpoint"))?;
        return Ok(WorkerRpcClient::Iroh(IrohWorkerClient {
            endpoint: iroh_endpoint,
            endpoint_addr: target.endpoint_addr,
            token: target.token.or(cluster_token).unwrap_or_default(),
            connection: None,
        }));
    }
    let client = WorkerServiceClient::connect(normalize_endpoint(endpoint))
        .await
        .map_err(|err| Status::unavailable(err.to_string()))?;
    Ok(WorkerRpcClient::Tcp {
        client,
        token: cluster_token,
    })
}

pub enum WorkerRpcClient {
    Tcp {
        client: WorkerServiceClient<tonic::transport::Channel>,
        token: Option<String>,
    },
    Iroh(IrohWorkerClient),
}

impl WorkerRpcClient {
    pub async fn load_shard(&mut self, request: LoadShardRequest) -> Result<(), Status> {
        match self {
            Self::Tcp { client, token } => {
                client
                    .load_shard(request_with_token(request, token.as_deref()))
                    .await
                    .map_err(|err| Status::failed_precondition(err.to_string()))?;
            }
            Self::Iroh(client) => {
                let _: bitty_protocol::pb::LoadShardResponse = client
                    .request(WORKER_LOAD_SHARD, &request)
                    .await
                    .map_err(|err| Status::failed_precondition(err.to_string()))?;
            }
        }
        Ok(())
    }

    pub async fn forward_activation(
        &mut self,
        activation: &ActivationTensor,
    ) -> Result<ActivationTensor, Status> {
        match self {
            Self::Tcp { client, token } => ActivationTensor::try_from(
                client
                    .forward_activation(request_with_token(
                        ProtoActivationTensor::from(activation),
                        token.as_deref(),
                    ))
                    .await
                    .map_err(|err| Status::internal(err.to_string()))?
                    .into_inner(),
            )
            .map_err(|err| Status::internal(err.to_string())),
            Self::Iroh(client) => {
                let response = client
                    .request::<_, ProtoActivationTensor>(
                        WORKER_FORWARD_ACTIVATION,
                        &ProtoActivationTensor::from(activation),
                    )
                    .await
                    .map_err(|err| Status::internal(err.to_string()))?;
                ActivationTensor::try_from(response)
                    .map_err(|err| Status::internal(err.to_string()))
            }
        }
    }

    pub async fn cleanup(
        &mut self,
        request: bitty_protocol::pb::CleanupRequest,
    ) -> Result<(), Status> {
        match self {
            Self::Tcp { client, token } => {
                let _ = client
                    .cleanup(request_with_token(request, token.as_deref()))
                    .await;
            }
            Self::Iroh(client) => {
                let _ = client
                    .request::<_, bitty_protocol::pb::CleanupResponse>(WORKER_CLEANUP, &request)
                    .await;
            }
        }
        Ok(())
    }
}

pub struct IrohWorkerClient {
    endpoint: Endpoint,
    endpoint_addr: EndpointAddr,
    token: String,
    /// Kept between requests: a token is one request to every worker, and a handshake for
    /// each of them would cost more than the layers.
    connection: Option<iroh::endpoint::Connection>,
}

impl IrohWorkerClient {
    async fn request<M, R>(
        &mut self,
        op: u8,
        message: &M,
    ) -> Result<R, iroh_transport::IrohTransportError>
    where
        M: prost::Message,
        R: prost::Message + Default,
    {
        let frame = IrohFrame::message(op, self.token.clone(), message);
        // a connection that died while idle fails here, before anything was sent, so
        // connecting again cannot run a request twice
        let open = match &self.connection {
            Some(connection) => connection.open_bi().await.ok(),
            None => None,
        };
        let (send, recv) = match open {
            Some(streams) => streams,
            None => {
                let connection = self
                    .endpoint
                    .connect(self.endpoint_addr.clone(), BITTY_WORKER_ALPN)
                    .await?;
                let streams = connection.open_bi().await?;
                self.connection = Some(connection);
                streams
            }
        };
        let response =
            iroh_transport::request_on_stream(send, recv, frame, DEFAULT_FRAME_LIMIT).await?;
        response.decode_message(op)
    }
}

fn request_with_token<T>(message: T, token: Option<&str>) -> Request<T> {
    let mut request = Request::new(message);
    if let Some(token) = token.filter(|token| !token.is_empty()) {
        if let Ok(value) = token.parse() {
            request.metadata_mut().insert(BITTY_TOKEN_HEADER, value);
        }
    }
    request
}
