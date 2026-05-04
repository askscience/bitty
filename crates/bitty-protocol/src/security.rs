use std::net::SocketAddr;

pub const BITTY_TOKEN_HEADER: &str = "x-bitty-token";

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuthMode {
    InsecureLocal,
    PreSharedToken(String),
}

impl AuthMode {
    pub fn accepts(&self, token: Option<&str>, remote_addr: Option<SocketAddr>) -> bool {
        match self {
            Self::InsecureLocal => remote_addr
                .map(|addr| addr.ip().is_loopback())
                .unwrap_or(true),
            Self::PreSharedToken(expected) => {
                !expected.is_empty()
                    && token
                        .map(|actual| constant_time_eq(actual.as_bytes(), expected.as_bytes()))
                        .unwrap_or(false)
            }
        }
    }
}

pub fn constant_time_eq(left: &[u8], right: &[u8]) -> bool {
    if left.len() != right.len() {
        return false;
    }
    left.iter()
        .zip(right)
        .fold(0_u8, |diff, (left, right)| diff | (left ^ right))
        == 0
}

pub fn validate_cluster_token(token: &str) -> Result<(), &'static str> {
    if token.trim().is_empty() {
        return Err("cluster token must not be empty");
    }
    if token.chars().any(char::is_control) {
        return Err("cluster token must not contain control characters");
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insecure_local_only_accepts_loopback_or_internal_requests() {
        assert!(AuthMode::InsecureLocal.accepts(None, None));
        assert!(AuthMode::InsecureLocal.accepts(None, Some("127.0.0.1:50051".parse().unwrap())));
        assert!(!AuthMode::InsecureLocal.accepts(None, Some("10.0.0.2:50051".parse().unwrap())));
    }

    #[test]
    fn pre_shared_token_requires_exact_non_empty_match() {
        let auth = AuthMode::PreSharedToken("secret".into());
        assert!(auth.accepts(Some("secret"), Some("10.0.0.2:1".parse().unwrap())));
        assert!(!auth.accepts(Some("wrong"), Some("10.0.0.2:1".parse().unwrap())));
        assert!(!AuthMode::PreSharedToken(String::new()).accepts(Some(""), None));
    }
}
