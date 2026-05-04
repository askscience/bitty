use std::net::IpAddr;
use thiserror::Error;

pub const HTTP_SCHEME: &str = "http://";
pub const HTTPS_SCHEME: &str = "https://";
pub const IROH_SCHEME: &str = "iroh://";

#[derive(Debug, Error, PartialEq, Eq)]
pub enum EndpointError {
    #[error("endpoint is empty")]
    Empty,
    #[error("endpoint has unsupported scheme: {0}")]
    UnsupportedScheme(String),
    #[error("endpoint contains control characters")]
    ControlCharacters,
    #[error("endpoint resolves to a forbidden host: {0}")]
    ForbiddenHost(String),
}

pub fn normalize_endpoint(endpoint: &str) -> String {
    let endpoint = endpoint.trim();
    if endpoint.starts_with(HTTP_SCHEME) || endpoint.starts_with(HTTPS_SCHEME) {
        endpoint.into()
    } else {
        format!("{HTTP_SCHEME}{endpoint}")
    }
}

pub fn validate_grpc_endpoint(endpoint: &str) -> Result<(), EndpointError> {
    let endpoint = endpoint.trim();
    if endpoint.is_empty() {
        return Err(EndpointError::Empty);
    }
    if endpoint.chars().any(char::is_control) {
        return Err(EndpointError::ControlCharacters);
    }
    if let Some((scheme, _)) = endpoint.split_once("://") {
        match scheme {
            "http" | "https" => {}
            other => return Err(EndpointError::UnsupportedScheme(other.into())),
        }
    }
    Ok(())
}

pub fn validate_worker_endpoint_for_dial(endpoint: &str) -> Result<(), EndpointError> {
    let endpoint = endpoint.trim();
    if endpoint.starts_with(IROH_SCHEME) {
        return validate_iroh_endpoint(endpoint);
    }
    validate_grpc_endpoint(endpoint)?;
    let host = endpoint_host(endpoint);
    if is_forbidden_worker_host(host) {
        return Err(EndpointError::ForbiddenHost(host.into()));
    }
    Ok(())
}

fn validate_iroh_endpoint(endpoint: &str) -> Result<(), EndpointError> {
    if endpoint.trim().is_empty() {
        return Err(EndpointError::Empty);
    }
    if endpoint.chars().any(char::is_control) {
        return Err(EndpointError::ControlCharacters);
    }
    Ok(())
}

fn endpoint_host(endpoint: &str) -> &str {
    let without_scheme = endpoint
        .strip_prefix(HTTP_SCHEME)
        .or_else(|| endpoint.strip_prefix(HTTPS_SCHEME))
        .unwrap_or(endpoint);
    let authority = without_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(without_scheme);
    let without_userinfo = authority.rsplit('@').next().unwrap_or(authority);
    if let Some(rest) = without_userinfo.strip_prefix('[') {
        return rest.split(']').next().unwrap_or(rest);
    }
    without_userinfo
        .split(':')
        .next()
        .unwrap_or(without_userinfo)
}

fn is_forbidden_worker_host(host: &str) -> bool {
    let host = host.trim_matches('.').to_ascii_lowercase();
    if host.is_empty()
        || host == "0.0.0.0"
        || host == "::"
        || host == "169.254.169.254"
        || host == "metadata.google.internal"
    {
        return true;
    }
    if let Ok(ip) = host.parse::<IpAddr>() {
        return ip.is_unspecified()
            || (matches!(ip, IpAddr::V4(ipv4) if ipv4.octets()[0] == 169 && ipv4.octets()[1] == 254));
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalizes_bare_grpc_endpoint() {
        assert_eq!(
            normalize_endpoint("127.0.0.1:50051"),
            "http://127.0.0.1:50051"
        );
    }

    #[test]
    fn preserves_http_and_https_endpoint() {
        assert_eq!(normalize_endpoint("http://host:1"), "http://host:1");
        assert_eq!(normalize_endpoint("https://host:1"), "https://host:1");
    }

    #[test]
    fn rejects_empty_grpc_endpoint() {
        assert_eq!(validate_grpc_endpoint(""), Err(EndpointError::Empty));
    }

    #[test]
    fn rejects_unsupported_grpc_scheme() {
        assert_eq!(
            validate_grpc_endpoint("ftp://example.com"),
            Err(EndpointError::UnsupportedScheme("ftp".into()))
        );
    }

    #[test]
    fn rejects_metadata_worker_endpoint() {
        assert_eq!(
            validate_worker_endpoint_for_dial("http://169.254.169.254/latest"),
            Err(EndpointError::ForbiddenHost("169.254.169.254".into()))
        );
    }

    #[test]
    fn accepts_iroh_worker_endpoint() {
        assert!(validate_worker_endpoint_for_dial("iroh://abc?token=secret").is_ok());
    }
}
