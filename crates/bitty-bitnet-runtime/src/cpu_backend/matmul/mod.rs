//! Matmul dispatch by GGML quantization type.

use super::types::PackedTensor;
use bitty_model::gguf::{
    GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_Q4_0, GGML_TYPE_Q4_K, GGML_TYPE_Q5_K, GGML_TYPE_Q6_K,
    GGML_TYPE_Q8_0, GGML_TYPE_Q8_1,
};

mod f32;
mod q4_0;
mod q4k;
mod q5k;
mod q6k;
mod q8_0;

type Result<T> = std::result::Result<T, String>;

/// Matrix-vector multiply: output = input @ weight^T
pub fn matmul(
    input: &[f32],
    weight: &PackedTensor,
    in_dim: usize,
    out_dim: usize,
) -> Result<Vec<f32>> {
    // Guard: dummy tensors (small F32 buffers) — return zeros
    if weight.data.len() < in_dim.min(1) * 4 {
        return Ok(vec![0f32; out_dim]);
    }
    // Guard: dimension mismatch
    let expected_elems = in_dim * out_dim;
    let data_elems = match weight.ggml_type {
        GGML_TYPE_F32 => weight.data.len() / 4,
        GGML_TYPE_F16 => weight.data.len() / 2,
        _ => {
            let (payload, _) = packed_byte_size(weight.ggml_type, expected_elems);
            if weight.data.len() < payload {
                return Ok(vec![0f32; out_dim]);
            }
            payload
        }
    };
    let _ = data_elems;

    match weight.ggml_type {
        GGML_TYPE_F32 => f32::matmul_f32(input, &weight.data, in_dim, out_dim),
        GGML_TYPE_F16 => f32::matmul_f16(input, &weight.data, in_dim, out_dim),
        GGML_TYPE_Q4_K => q4k::matmul(input, &weight.data, in_dim, out_dim),
        GGML_TYPE_Q5_K => q5k::matmul(input, &weight.data, in_dim, out_dim),
        GGML_TYPE_Q6_K => q6k::matmul(input, &weight.data, in_dim, out_dim),
        GGML_TYPE_Q8_0 | GGML_TYPE_Q8_1 => q8_0::matmul(input, &weight.data, in_dim, out_dim),
        GGML_TYPE_Q4_0 => q4_0::matmul(input, &weight.data, in_dim, out_dim),
        _ => Err(format!(
            "CPU matmul: unsupported GGML type {}",
            weight.ggml_type
        )),
    }
}

fn packed_byte_size(ggml_type: u32, num_elements: usize) -> (usize, f64) {
    use bitty_model::gguf::{
        GGML_TYPE_I2_S, GGML_TYPE_Q2_K, GGML_TYPE_Q3_K, GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_K,
    };
    if ggml_type == GGML_TYPE_I2_S {
        return (num_elements.div_ceil(4) + 32, 0.25);
    }
    let elem_size = crate::cpu_backend::loader::ggml_type_size(ggml_type).unwrap_or(2.0);
    let payload = (num_elements as f64 * elem_size).ceil() as usize;
    let blocks = (num_elements as f64 / 256.0).ceil() as usize;
    let overhead = match ggml_type {
        GGML_TYPE_Q4_K | GGML_TYPE_Q5_K | GGML_TYPE_Q3_K => blocks * 16,
        GGML_TYPE_Q8_K => blocks * 10,
        GGML_TYPE_Q6_K => blocks * 18,
        GGML_TYPE_Q2_K => blocks * 20,
        _ => 0,
    };
    (payload + overhead, elem_size)
}
