//! Runs a real model through the stage engine. Needs the patched llama.cpp build and a small
//! Qwen3.5 GGUF, so it only runs when both are given:
//!
//!   BITTY_STAGE_BIN=/path/to/bitty-stage BITTY_TEST_MODEL=/path/to/Qwen3.5-0.8B-Q4_K_M.gguf \
//!       cargo test -p bitty-inference --test stage_engine -- --nocapture

use bitty_inference::{LayerExecutor, StageExecutor, StageOptions};
use bitty_protocol::{ActivationDType, ActivationTensor, AssignedLayerRange, Quantization};

fn range(start: u32, end: u32) -> AssignedLayerRange {
    AssignedLayerRange {
        start_layer: start,
        end_layer_exclusive: end,
        quantization: Quantization::Q4,
    }
}

fn token_activation(request: &str, step: u32, tokens: &[u32]) -> ActivationTensor {
    ActivationTensor::new(
        request,
        step,
        0,
        0,
        vec![tokens.len() as u32],
        ActivationDType::Tokens,
        tokens.iter().flat_map(|token| token.to_le_bytes()).collect(),
    )
}

/// Greedy generation through a chain of stages, the way the coordinator drives workers.
async fn generate(
    chain: &[(StageExecutor, AssignedLayerRange)],
    request: &str,
    prompt: &[u32],
    steps: u32,
) -> (Vec<u32>, Vec<Vec<f32>>) {
    let mut input = prompt.to_vec();
    let mut tokens = Vec::new();
    let mut all_logits = Vec::new();
    for step in 0..steps {
        let mut activation = token_activation(request, step, &input);
        for (executor, range) in chain {
            activation = executor.execute_range(range, activation).await.expect("stage runs");
        }
        let (last, _) = chain.last().unwrap();
        let logits = last.final_logits(activation).await.expect("final logits");
        assert!(logits.verify_checksum());
        let token = logits
            .logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(index, _)| index as u32)
            .unwrap();
        tokens.push(token);
        all_logits.push(logits.logits);
        input = vec![token];
    }
    for (executor, _) in chain {
        executor.end_request(request).await;
    }
    (tokens, all_logits)
}

#[tokio::test(flavor = "multi_thread")]
async fn chained_stages_match_the_whole_model() {
    let (Some(_), Some(model)) = (
        std::env::var_os("BITTY_STAGE_BIN"),
        std::env::var_os("BITTY_TEST_MODEL"),
    ) else {
        eprintln!("skipped: set BITTY_STAGE_BIN and BITTY_TEST_MODEL to run the stage engine test");
        return;
    };
    let options = StageOptions {
        context: 512,
        sessions: 2,
        ..StageOptions::default()
    };
    let load = || async {
        let executor = StageExecutor::load(&model).await.expect("model header").with_options(options);
        assert!(executor.engine_available());
        executor
    };

    let whole = load().await;
    let layers = whole.metadata().layer_count;
    assert!(layers >= 3);
    let whole_chain = vec![(whole.clone(), range(0, layers))];

    // the vocabulary lives in the model file: tokenize through a running stage
    whole
        .execute_range(&range(0, layers), token_activation("warmup", 0, &[1]))
        .await
        .expect("whole model runs");
    whole.end_request("warmup").await;
    let prompt = whole
        .tokenize("The capital of Italy is Rome. The capital of France is", true)
        .await
        .expect("tokenize");
    assert!(prompt.len() > 4);

    let (reference, reference_logits) = generate(&whole_chain, "reference", &prompt, 8).await;
    let mut text = String::new();
    for token in &reference {
        text.push_str(&whole.decode_token_text(*token).await);
    }
    eprintln!("whole model: {text:?}");
    assert!(text.contains("Paris"), "unexpected continuation {text:?}");

    let cut_a = layers / 3;
    let cut_b = 2 * layers / 3;
    let chain = vec![
        (load().await, range(0, cut_a)),
        (load().await, range(cut_a, cut_b)),
        (load().await, range(cut_b, layers)),
    ];
    let (chained, chained_logits) = generate(&chain, "chained", &prompt, 8).await;
    assert_eq!(chained, reference, "the chain must sample the same tokens");
    let worst = reference_logits
        .iter()
        .flatten()
        .zip(chained_logits.iter().flatten())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("max |logit diff| between whole model and chain: {worst}");
    assert!(worst < 1e-3, "logits drifted by {worst}");

    // a second request reuses the running stages and starts from a clean state
    let (again, _) = generate(&chain, "chained-again", &prompt, 8).await;
    assert_eq!(again, reference);

    // a request the stage never opened is refused instead of running on someone else's state
    let stray = chain[0]
        .0
        .execute_range(&chain[0].1, token_activation("stray", 3, &[1]))
        .await;
    assert!(stray.is_err());
}

#[test]
fn vocabulary_process_handles_text_without_weights() {
    let (Some(engine), Some(model)) = (
        std::env::var_os("BITTY_STAGE_BIN"),
        std::env::var_os("BITTY_TEST_MODEL"),
    ) else {
        eprintln!("skipped: set BITTY_STAGE_BIN and BITTY_TEST_MODEL");
        return;
    };
    let codec = bitty_inference::TextCodec::open_with_engine(engine.as_ref(), model.as_ref()).unwrap();
    assert!(codec.vocab_size() > 1000);

    let text = |tokens: &[u32]| {
        let mut stream = bitty_inference::TextStream::default();
        let mut text = String::new();
        for token in tokens {
            text.push_str(&stream.push(&codec.piece(*token).unwrap()));
        }
        text + &stream.finish()
    };
    let raw = codec.tokenize("Perché no? 🙂").unwrap();
    assert_eq!(text(&raw), "Perché no? 🙂");

    let quiet = text(&codec.chat_tokens(&[("user", "Hi")], false).unwrap());
    assert!(quiet.contains("<|im_start|>user\nHi<|im_end|>"), "{quiet}");
    assert!(quiet.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"), "{quiet}");
    let thinking = text(&codec.chat_tokens(&[("user", "Hi")], true).unwrap());
    assert!(thinking.ends_with("<|im_start|>assistant\n<think>\n"), "{thinking}");

    // the end of a turn stops generation, ordinary text does not
    let end = codec.tokenize("<|im_end|>").unwrap();
    assert!(codec.is_end(*end.last().unwrap()));
    assert!(!codec.is_end(raw[0]));
}
