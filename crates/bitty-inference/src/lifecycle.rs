use bitty_protocol::TokenOutput;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::time::Duration;

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchPolicy {
    pub max_delay_ms: u64,
    pub max_batch_size: usize,
}

impl Default for BatchPolicy {
    fn default() -> Self {
        Self {
            max_delay_ms: 50,
            max_batch_size: 16,
        }
    }
}

impl BatchPolicy {
    pub fn delay(&self) -> Duration {
        Duration::from_millis(self.max_delay_ms)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PrefixCacheKey(pub String);

impl PrefixCacheKey {
    pub fn from_tokens(tokens: &[u32]) -> Self {
        let mut hasher = Sha256::new();
        for token in tokens {
            hasher.update(token.to_le_bytes());
        }
        Self(format!("{:x}", hasher.finalize()))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RequestLifecycle {
    pub request_id: String,
    pub prompt_tokens: Vec<u32>,
    pub generated: Vec<TokenOutput>,
    pub phase: InferencePhase,
}

impl RequestLifecycle {
    pub fn new(request_id: impl Into<String>, prompt_tokens: Vec<u32>) -> Self {
        Self {
            request_id: request_id.into(),
            prompt_tokens,
            generated: Vec::new(),
            phase: InferencePhase::Prefill,
        }
    }

    pub fn push_token(&mut self, token: TokenOutput) {
        self.phase = InferencePhase::Decode;
        self.generated.push(token);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InferencePhase {
    Prefill,
    Decode,
    Finished,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpeculativeDecision {
    pub accepted: usize,
    pub rollback_from: Option<usize>,
}

impl SpeculativeDecision {
    pub fn verify(draft_tokens: &[u32], verified_tokens: &[u32]) -> Self {
        let accepted = draft_tokens
            .iter()
            .zip(verified_tokens)
            .take_while(|(draft, verified)| draft == verified)
            .count();
        let rollback_from = (accepted < draft_tokens.len()).then_some(accepted);
        Self {
            accepted,
            rollback_from,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct DecodePipeline {
    in_flight: VecDeque<u32>,
}

impl DecodePipeline {
    pub fn enqueue(&mut self, token_position: u32) {
        self.in_flight.push_back(token_position);
    }

    pub fn complete(&mut self) -> Option<u32> {
        self.in_flight.pop_front()
    }

    pub fn len(&self) -> usize {
        self.in_flight.len()
    }

    pub fn is_empty(&self) -> bool {
        self.in_flight.is_empty()
    }
}
