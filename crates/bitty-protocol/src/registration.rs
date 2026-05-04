use crate::pb::RegisterWorkerRequest;
use crate::BITTY_PROTOCOL_VERSION;

const SUPPORTED_BACKENDS: &[&str] = &["", "bitnet", "stub"];

pub fn validate_register_worker(request: &RegisterWorkerRequest) -> Result<(), String> {
    let v = request.protocol_version;
    if v != 0 && v != BITTY_PROTOCOL_VERSION {
        return Err(format!(
            "unsupported protocol_version {v} (server supports {BITTY_PROTOCOL_VERSION})"
        ));
    }
    let backend = request.inference_backend_id.trim();
    if !SUPPORTED_BACKENDS.contains(&backend) {
        return Err(format!(
            "unsupported inference_backend_id `{backend}` (supported: bitnet, stub)"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(version: u32, backend: &str) -> RegisterWorkerRequest {
        RegisterWorkerRequest {
            profile: None,
            protocol_version: version,
            inference_backend_id: backend.into(),
        }
    }

    #[test]
    fn accepts_default_and_current_version() {
        assert!(validate_register_worker(&req(0, "")).is_ok());
        assert!(validate_register_worker(&req(1, "")).is_ok());
    }

    #[test]
    fn rejects_unknown_version() {
        assert!(validate_register_worker(&req(2, "")).is_err());
    }

    #[test]
    fn rejects_unknown_backend() {
        assert!(validate_register_worker(&req(1, "llama.cpp")).is_err());
        assert!(validate_register_worker(&req(1, "bitnet")).is_ok());
        assert!(validate_register_worker(&req(1, "stub")).is_ok());
    }
}
