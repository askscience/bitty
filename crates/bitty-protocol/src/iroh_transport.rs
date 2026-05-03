use iroh::endpoint::{RecvStream, SendStream};
use iroh::{Endpoint, EndpointAddr, EndpointId, RelayUrl, TransportAddr};
use prost::Message;
use std::net::SocketAddr;
use thiserror::Error;

pub const BITTY_SCHEDULER_ALPN: &[u8] = b"bitty/scheduler/0";
pub const BITTY_WORKER_ALPN: &[u8] = b"bitty/worker/0";

pub const SCHEDULER_REGISTER_WORKER: u8 = 1;
pub const SCHEDULER_HEARTBEAT: u8 = 2;
pub const SCHEDULER_GENERATE: u8 = 3;
pub const SCHEDULER_CLUSTER_STATUS: u8 = 4;

pub const WORKER_FORWARD_ACTIVATION: u8 = 1;
pub const WORKER_FINAL_LOGITS: u8 = 2;
pub const WORKER_APPLY_TOPOLOGY: u8 = 3;
pub const WORKER_LOAD_SHARD: u8 = 4;
pub const WORKER_CLEANUP: u8 = 5;

pub const DEFAULT_FRAME_LIMIT: usize = 64 * 1024 * 1024;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrohFrame {
    pub op: u8,
    pub token: String,
    pub payload: Vec<u8>,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct IrohTarget {
    pub endpoint_addr: EndpointAddr,
    pub token: Option<String>,
}

impl IrohFrame {
    pub fn message<M>(op: u8, token: impl Into<String>, message: &M) -> Self
    where
        M: Message,
    {
        Self {
            op,
            token: token.into(),
            payload: message.encode_to_vec(),
        }
    }

    pub fn decode_message<M>(&self, expected_op: u8) -> Result<M, IrohTransportError>
    where
        M: Message + Default,
    {
        if self.op != expected_op {
            return Err(IrohTransportError::UnexpectedOp {
                expected: expected_op,
                actual: self.op,
            });
        }
        M::decode(self.payload.as_slice()).map_err(IrohTransportError::Decode)
    }
}

#[derive(Debug, Error)]
pub enum IrohTransportError {
    #[error("iroh connect failed: {0}")]
    Connect(#[from] iroh::endpoint::ConnectError),
    #[error("iroh connection failed: {0}")]
    Connection(#[from] iroh::endpoint::ConnectionError),
    #[error("iroh write failed: {0}")]
    Write(#[from] iroh::endpoint::WriteError),
    #[error("iroh stream is closed: {0}")]
    ClosedStream(#[from] iroh::endpoint::ClosedStream),
    #[error("iroh stream stopped: {0}")]
    Stopped(#[from] iroh::endpoint::StoppedError),
    #[error("iroh read failed: {0}")]
    Read(#[from] iroh::endpoint::ReadToEndError),
    #[error("protobuf decode failed: {0}")]
    Decode(#[from] prost::DecodeError),
    #[error("frame too large: {0} bytes")]
    FrameTooLarge(usize),
    #[error("frame is truncated")]
    Truncated,
    #[error("invalid token length")]
    InvalidTokenLength,
    #[error("invalid utf-8 token")]
    InvalidToken,
    #[error("unexpected operation: expected {expected}, got {actual}")]
    UnexpectedOp { expected: u8, actual: u8 },
    #[error("cluster token was rejected")]
    Unauthorized,
}

pub async fn request(
    endpoint: &Endpoint,
    remote: EndpointId,
    alpn: &[u8],
    frame: IrohFrame,
    response_limit: usize,
) -> Result<IrohFrame, IrohTransportError> {
    request_addr(endpoint, remote.into(), alpn, frame, response_limit).await
}

pub async fn request_addr(
    endpoint: &Endpoint,
    remote: EndpointAddr,
    alpn: &[u8],
    frame: IrohFrame,
    response_limit: usize,
) -> Result<IrohFrame, IrohTransportError> {
    let connection = endpoint.connect(remote, alpn).await?;
    let (mut send, mut recv) = connection.open_bi().await?;
    write_frame(&mut send, &frame).await?;
    send.finish()?;
    send.stopped().await?;
    read_frame(&mut recv, response_limit).await
}

pub async fn write_frame(
    send: &mut SendStream,
    frame: &IrohFrame,
) -> Result<(), IrohTransportError> {
    let encoded = encode_frame(frame)?;
    send.write_all(&encoded).await?;
    Ok(())
}

pub async fn read_frame(
    recv: &mut RecvStream,
    limit: usize,
) -> Result<IrohFrame, IrohTransportError> {
    let bytes = recv.read_to_end(limit + 4).await?;
    decode_frame(&bytes, limit)
}

pub fn encode_frame(frame: &IrohFrame) -> Result<Vec<u8>, IrohTransportError> {
    let token = frame.token.as_bytes();
    let token_len: u16 = token
        .len()
        .try_into()
        .map_err(|_| IrohTransportError::InvalidTokenLength)?;
    let body_len = 1 + 2 + token.len() + frame.payload.len();
    let mut encoded = Vec::with_capacity(4 + body_len);
    encoded.extend_from_slice(&(body_len as u32).to_be_bytes());
    encoded.push(frame.op);
    encoded.extend_from_slice(&token_len.to_be_bytes());
    encoded.extend_from_slice(token);
    encoded.extend_from_slice(&frame.payload);
    Ok(encoded)
}

pub fn decode_frame(bytes: &[u8], limit: usize) -> Result<IrohFrame, IrohTransportError> {
    if bytes.len() < 7 {
        return Err(IrohTransportError::Truncated);
    }
    let body_len = u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]) as usize;
    if body_len > limit {
        return Err(IrohTransportError::FrameTooLarge(body_len));
    }
    if bytes.len() != 4 + body_len {
        return Err(IrohTransportError::Truncated);
    }
    let op = bytes[4];
    let token_len = u16::from_be_bytes([bytes[5], bytes[6]]) as usize;
    let token_start = 7;
    let token_end = token_start + token_len;
    if token_end > bytes.len() {
        return Err(IrohTransportError::InvalidTokenLength);
    }
    let token = std::str::from_utf8(&bytes[token_start..token_end])
        .map_err(|_| IrohTransportError::InvalidToken)?
        .to_string();
    Ok(IrohFrame {
        op,
        token,
        payload: bytes[token_end..].to_vec(),
    })
}

pub fn iroh_uri(endpoint_id: impl std::fmt::Display, token: &str) -> String {
    if token.is_empty() {
        format!("iroh://{endpoint_id}")
    } else {
        format!("iroh://{endpoint_id}?token={token}")
    }
}

pub fn iroh_uri_for_addr(endpoint_addr: &EndpointAddr, token: &str) -> String {
    iroh_uri_for_addr_options(endpoint_addr, token, true)
}

pub fn iroh_uri_for_relay_addr(endpoint_addr: &EndpointAddr, token: &str) -> String {
    iroh_uri_for_addr_options(endpoint_addr, token, false)
}

fn iroh_uri_for_addr_options(
    endpoint_addr: &EndpointAddr,
    token: &str,
    include_direct: bool,
) -> String {
    let mut params = Vec::new();
    if !token.is_empty() {
        params.push(format!("token={token}"));
    }
    for relay in endpoint_addr.relay_urls() {
        params.push(format!("relay={relay}"));
    }
    if include_direct {
        for addr in endpoint_addr.ip_addrs() {
            if !addr.ip().is_unspecified() {
                params.push(format!("addr={addr}"));
            }
        }
    }
    if params.is_empty() {
        format!("iroh://{}", endpoint_addr.id)
    } else {
        format!("iroh://{}?{}", endpoint_addr.id, params.join("&"))
    }
}

pub fn parse_iroh_uri(value: &str) -> Option<(&str, Option<&str>)> {
    let rest = value.strip_prefix("iroh://")?;
    let (endpoint_id, query) = rest.split_once('?').unwrap_or((rest, ""));
    let token = query.split('&').find_map(|part| {
        let (key, value) = part.split_once('=')?;
        (key == "token").then_some(value)
    });
    Some((endpoint_id, token))
}

pub fn parse_iroh_target(value: &str) -> Option<IrohTarget> {
    let rest = value.strip_prefix("iroh://")?;
    let (endpoint_id, query) = rest.split_once('?').unwrap_or((rest, ""));
    let endpoint_id: EndpointId = endpoint_id.parse().ok()?;
    let mut token = None;
    let mut addrs = Vec::new();
    for part in query.split('&').filter(|part| !part.is_empty()) {
        let Some((key, value)) = part.split_once('=') else {
            continue;
        };
        match key {
            "token" => token = Some(value.to_string()),
            "relay" => {
                let relay: RelayUrl = value.parse().ok()?;
                addrs.push(TransportAddr::Relay(relay));
            }
            "addr" => {
                let addr: SocketAddr = value.parse().ok()?;
                addrs.push(TransportAddr::Ip(addr));
            }
            _ => {}
        }
    }
    Some(IrohTarget {
        endpoint_addr: EndpointAddr::from_parts(endpoint_id, addrs),
        token,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frame_round_trips() {
        let frame = IrohFrame {
            op: 7,
            token: "secret".into(),
            payload: b"payload".to_vec(),
        };
        let encoded = encode_frame(&frame).unwrap();
        let decoded = decode_frame(&encoded, DEFAULT_FRAME_LIMIT).unwrap();
        assert_eq!(decoded, frame);
    }

    #[test]
    fn parses_iroh_uri_token() {
        assert_eq!(
            parse_iroh_uri("iroh://abc?token=secret"),
            Some(("abc", Some("secret")))
        );
    }

    #[test]
    fn parses_iroh_target_addresses() {
        let value = "iroh://1992d53c02cdc04566e5c0edb1ce83305cd550297953a047a445ea3264b54b18?token=secret&relay=https://euw1-1.relay.iroh.network./&addr=127.0.0.1:1234";
        let target = parse_iroh_target(value).unwrap();
        assert_eq!(target.token.as_deref(), Some("secret"));
        assert_eq!(target.endpoint_addr.relay_urls().count(), 1);
        assert_eq!(target.endpoint_addr.ip_addrs().count(), 1);
    }
}
