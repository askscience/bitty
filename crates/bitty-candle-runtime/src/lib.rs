#[cfg(not(target_env = "msvc"))]
use jemallocator::Jemalloc;

#[cfg(not(target_env = "msvc"))]
#[global_allocator]
static GLOBAL: Jemalloc = Jemalloc;

mod device;
pub mod dequant;
pub mod kv_cache;
pub mod layers;
pub mod load;
pub mod model;
pub mod sampling;
pub mod tokenizer;

pub use device::auto_device;
pub use tokenizer::{ChatMessage, GgufTokenizerOverrides, Tokenizer};
pub use model::CandleModel;
pub use layers::ModelConfig;
pub use sampling::sample_token;
pub use candle_nn::ops::rms_norm;
