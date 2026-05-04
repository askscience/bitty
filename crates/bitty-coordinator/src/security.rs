use governor::{DefaultKeyedRateLimiter, Quota, RateLimiter};
use std::num::NonZeroU32;
use std::sync::Arc;

pub use bitty_protocol::security::AuthMode;

pub type SharedRateLimiter = Arc<DefaultKeyedRateLimiter<String>>;

pub fn request_rate_limiter(requests_per_second: NonZeroU32) -> SharedRateLimiter {
    Arc::new(RateLimiter::keyed(Quota::per_second(requests_per_second)))
}
