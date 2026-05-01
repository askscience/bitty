use governor::{DefaultKeyedRateLimiter, Quota, RateLimiter};
use std::num::NonZeroU32;
use std::sync::Arc;

pub type SharedRateLimiter = Arc<DefaultKeyedRateLimiter<String>>;

pub fn request_rate_limiter(requests_per_second: NonZeroU32) -> SharedRateLimiter {
    Arc::new(RateLimiter::keyed(Quota::per_second(requests_per_second)))
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum AuthMode {
    InsecureLocal,
    PreSharedToken(String),
    MutualTls,
}

impl AuthMode {
    pub fn accepts_token(&self, token: Option<&str>) -> bool {
        match self {
            Self::InsecureLocal => true,
            Self::PreSharedToken(expected) => token == Some(expected.as_str()),
            Self::MutualTls => token.is_some(),
        }
    }
}
