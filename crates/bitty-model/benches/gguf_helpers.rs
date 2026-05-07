use bitty_model::gguf::{
    bytes_per_element, decode_i2_s_block, ggml_type_name, layer_id_from_tensor_name,
    quantization_from_ggml_type, GGML_TYPE_BF16, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_I2_S,
    GGML_TYPE_IQ1_S, GGML_TYPE_IQ2_S, GGML_TYPE_IQ2_XS, GGML_TYPE_IQ2_XXS, GGML_TYPE_IQ3_M,
    GGML_TYPE_IQ3_S, GGML_TYPE_IQ3_XXS, GGML_TYPE_IQ4_NL, GGML_TYPE_IQ4_XS, GGML_TYPE_Q2_K,
    GGML_TYPE_Q3_K, GGML_TYPE_Q4_0, GGML_TYPE_Q4_1, GGML_TYPE_Q4_K, GGML_TYPE_Q5_0, GGML_TYPE_Q5_1,
    GGML_TYPE_Q5_K, GGML_TYPE_Q6_K, GGML_TYPE_Q8_0, GGML_TYPE_Q8_1, GGML_TYPE_Q8_K,
    GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0,
};
use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn bench_layer_id_extraction(c: &mut Criterion) {
    c.benchmark_group("layer_id_from_tensor_name")
        .sample_size(200)
        .bench_function("blk_prefix", |b| {
            b.iter(|| {
                let _ = black_box(layer_id_from_tensor_name(black_box("blk.12.attn_q.weight")));
            });
        })
        .bench_function("model_layers_prefix", |b| {
            b.iter(|| {
                let _ = black_box(layer_id_from_tensor_name(black_box(
                    "model.layers.7.mlp.down_proj.weight",
                )));
            });
        })
        .bench_function("layers_prefix", |b| {
            b.iter(|| {
                let _ = black_box(layer_id_from_tensor_name(black_box(
                    "layers.99.attn.output.weight",
                )));
            });
        })
        .bench_function("no_layer_id_embedding", |b| {
            b.iter(|| {
                let _ = black_box(layer_id_from_tensor_name(black_box("token_embd.weight")));
            });
        })
        .bench_function("no_layer_id_output", |b| {
            b.iter(|| {
                let _ = black_box(layer_id_from_tensor_name(black_box("output.weight")));
            });
        });
}

fn bench_ggml_type_name(c: &mut Criterion) {
    let all_types: &[u32] = &[
        GGML_TYPE_F32,
        GGML_TYPE_F16,
        GGML_TYPE_Q4_0,
        GGML_TYPE_Q4_1,
        GGML_TYPE_Q5_0,
        GGML_TYPE_Q5_1,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q8_1,
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q6_K,
        GGML_TYPE_Q8_K,
        GGML_TYPE_IQ2_XXS,
        GGML_TYPE_IQ2_XS,
        GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ3_S,
        GGML_TYPE_IQ2_S,
        GGML_TYPE_IQ1_S,
        GGML_TYPE_IQ4_NL,
        GGML_TYPE_IQ3_M,
        GGML_TYPE_IQ4_XS,
        GGML_TYPE_BF16,
        GGML_TYPE_I2_S,
        GGML_TYPE_TQ1_0,
        GGML_TYPE_TQ2_0,
    ];

    c.benchmark_group("ggml_type_name")
        .sample_size(200)
        .bench_function("lookup_hot_types", |b| {
            let hot: &[u32] = &[
                GGML_TYPE_F16,
                GGML_TYPE_Q4_K,
                GGML_TYPE_Q8_0,
                GGML_TYPE_Q2_K,
            ];
            b.iter(|| {
                for &t in hot {
                    let _ = black_box(ggml_type_name(black_box(t)));
                }
            });
        })
        .bench_function("lookup_all_types", |b| {
            b.iter(|| {
                for &t in all_types {
                    let _ = black_box(ggml_type_name(black_box(t)));
                }
            });
        });
}

fn bench_bytes_per_element(c: &mut Criterion) {
    let hot: &[u32] = &[
        GGML_TYPE_F16,
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q2_K,
        GGML_TYPE_Q3_K,
        GGML_TYPE_I2_S,
        GGML_TYPE_BF16,
    ];

    c.benchmark_group("bytes_per_element")
        .sample_size(200)
        .bench_function("hot_types", |b| {
            b.iter(|| {
                for &t in hot {
                    let _ = black_box(bytes_per_element(black_box(t)));
                }
            });
        })
        .bench_function("fp16_most_common", |b| {
            b.iter(|| {
                let _ = black_box(bytes_per_element(black_box(GGML_TYPE_F16)));
            });
        })
        .bench_function("q4_k_common", |b| {
            b.iter(|| {
                let _ = black_box(bytes_per_element(black_box(GGML_TYPE_Q4_K)));
            });
        });
}

fn bench_quantization_from_ggml_type(c: &mut Criterion) {
    let all_common: &[u32] = &[
        GGML_TYPE_F32,
        GGML_TYPE_F16,
        GGML_TYPE_Q8_0,
        GGML_TYPE_Q6_K,
        GGML_TYPE_Q5_K,
        GGML_TYPE_Q4_K,
        GGML_TYPE_Q3_K,
        GGML_TYPE_Q2_K,
        GGML_TYPE_I2_S,
        GGML_TYPE_IQ2_XXS,
        GGML_TYPE_IQ3_XXS,
        GGML_TYPE_IQ4_XS,
    ];

    c.benchmark_group("quantization_from_ggml_type")
        .sample_size(200)
        .bench_function("common_types_batch", |b| {
            b.iter(|| {
                for &t in all_common {
                    let _ = black_box(quantization_from_ggml_type(black_box(t)));
                }
            });
        })
        .bench_function("q4_k_single", |b| {
            b.iter(|| {
                let _ = black_box(quantization_from_ggml_type(black_box(GGML_TYPE_Q4_K)));
            });
        })
        .bench_function("i2_s_single", |b| {
            b.iter(|| {
                let _ = black_box(quantization_from_ggml_type(black_box(GGML_TYPE_I2_S)));
            });
        });
}

fn bench_decode_i2_s_block(c: &mut Criterion) {
    let mut block = [0u8; 32];
    fn fill_block(block: &mut [u8; 32]) {
        block[0] = 0b01_10_00_01;
        for i in 1..32 {
            block[i] = (i as u8).wrapping_mul(17);
        }
    }
    fill_block(&mut block);

    c.benchmark_group("decode_i2_s_block")
        .sample_size(200)
        .bench_function("single_32byte_block", |b| {
            b.iter(|| {
                let _ = black_box(decode_i2_s_block(black_box(&block)));
            });
        })
        .bench_function("batch_64blocks_2KB", |b| {
            let blocks: Vec<[u8; 32]> = vec![block; 64];
            b.iter(|| {
                for blk in &blocks {
                    let _ = black_box(decode_i2_s_block(black_box(blk)));
                }
            });
        });
}

criterion_group!(
    benches,
    bench_layer_id_extraction,
    bench_ggml_type_name,
    bench_bytes_per_element,
    bench_quantization_from_ggml_type,
    bench_decode_i2_s_block,
);
criterion_main!(benches);
