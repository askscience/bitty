use bitty_inference::BatchPolicy;
use bitty_protocol::{ActivationDType, ActivationTensor, TokenOutput};
use std::collections::VecDeque;
use tokio::time::{sleep, Duration};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct InferenceRequest {
    pub request_id: String,
    pub prompt_tokens: Vec<u32>,
    pub max_new_tokens: u32,
}

#[derive(Clone, Debug)]
pub struct RequestRouter {
    policy: BatchPolicy,
    pending: VecDeque<InferenceRequest>,
}

impl RequestRouter {
    pub fn new(policy: BatchPolicy) -> Self {
        Self {
            policy,
            pending: VecDeque::new(),
        }
    }

    pub fn enqueue(&mut self, request: InferenceRequest) {
        self.pending.push_back(request);
    }

    pub async fn next_batch(&mut self) -> Vec<InferenceRequest> {
        if self.pending.is_empty() {
            return Vec::new();
        }

        sleep(Duration::from_millis(self.policy.max_delay_ms)).await;
        let mut batch = Vec::new();
        while batch.len() < self.policy.max_batch_size {
            let Some(request) = self.pending.pop_front() else {
                break;
            };
            batch.push(request);
        }
        batch
    }

    pub fn initial_activation(request: &InferenceRequest) -> ActivationTensor {
        let payload = request
            .prompt_tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect();
        ActivationTensor::new(
            request.request_id.clone(),
            0,
            0,
            0,
            vec![request.prompt_tokens.len() as u32],
            ActivationDType::Fp16,
            payload,
        )
    }

    pub fn token_from_activation(activation: &ActivationTensor, finished: bool) -> TokenOutput {
        let token_id = activation
            .payload
            .chunks_exact(4)
            .last()
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .unwrap_or(0);
        TokenOutput {
            request_id: activation.request_id.clone(),
            token_position: activation.token_position,
            token_id,
            text: format!("<tok:{token_id}>"),
            finished,
            log_prob: 0.0,
            gen_latency_us: 0,
        }
    }
}
