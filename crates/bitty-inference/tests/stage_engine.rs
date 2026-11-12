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
        tokens
            .iter()
            .flat_map(|token| token.to_le_bytes())
            .collect(),
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
            activation = executor
                .execute_range(range, activation)
                .await
                .expect("stage runs");
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
        let executor = StageExecutor::load(&model)
            .await
            .expect("model header")
            .with_options(options);
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
        .tokenize(
            "The capital of Italy is Rome. The capital of France is",
            true,
        )
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

    // hidden states sent as f16 between the stages (BITTY_WIRE_F16=1): the same tokens here,
    // logits off by the precision of f16
    let half = |executor: StageExecutor| {
        executor.with_options(StageOptions {
            wire_f16: true,
            ..options
        })
    };
    let half_chain = vec![
        (half(load().await), range(0, cut_a)),
        (half(load().await), range(cut_a, cut_b)),
        (half(load().await), range(cut_b, layers)),
    ];
    let (halved, halved_logits) = generate(&half_chain, "chained-f16", &prompt, 8).await;
    assert_eq!(
        halved, reference,
        "f16 between the stages changed the tokens"
    );
    // only the best logits come back: at the edge of that set one run can have a token the
    // other left out, so the logits both runs have are compared
    let worst_half = reference_logits
        .iter()
        .flatten()
        .zip(halved_logits.iter().flatten())
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);
    eprintln!("max |logit diff| with f16 between the stages: {worst_half}");
    // the logits that decide: the best ten of every position
    let best_half = reference_logits
        .iter()
        .zip(&halved_logits)
        .map(|(a, b)| {
            let mut order: Vec<usize> = (0..a.len()).filter(|&i| a[i].is_finite()).collect();
            order.sort_by(|&x, &y| a[y].total_cmp(&a[x]));
            order
                .iter()
                .take(10)
                .map(|&i| (a[i] - b[i]).abs())
                .fold(0.0f32, f32::max)
        })
        .fold(0.0f32, f32::max);
    eprintln!("max |logit diff| among the best ten with f16: {best_half}");
    assert!(worst_half < 1.0, "f16 moved the logits by {worst_half}");

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
    let codec =
        bitty_inference::TextCodec::open_with_engine(engine.as_ref(), model.as_ref()).unwrap();
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
    assert!(
        quiet.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        "{quiet}"
    );
    let thinking = text(&codec.chat_tokens(&[("user", "Hi")], true).unwrap());
    assert!(
        thinking.ends_with("<|im_start|>assistant\n<think>\n"),
        "{thinking}"
    );

    // the end of a turn stops generation, ordinary text does not
    let end = codec.tokenize("<|im_end|>").unwrap();
    assert!(codec.is_end(*end.last().unwrap()));
    assert!(!codec.is_end(raw[0]));
}

/// A guess that is stopped half way, or never started, leaves the session where it was: the
/// token that follows gets the logits it would have got without the guess.
#[test]
fn a_stopped_guess_leaves_no_trace() {
    use bitty_inference::stage_process::{StageOutput, StageProcess, StageProcessConfig};
    let (Some(engine), Some(model)) = (
        std::env::var_os("BITTY_STAGE_BIN"),
        std::env::var_os("BITTY_TEST_MODEL"),
    ) else {
        eprintln!("skipped: set BITTY_STAGE_BIN and BITTY_TEST_MODEL");
        return;
    };
    let layers = bitty_model::ModelMetadata::from_gguf_path(&model)
        .unwrap()
        .layer_count;
    let stage = |start: u32, end: u32| {
        StageProcess::spawn(&StageProcessConfig {
            engine: engine.clone().into(),
            model: model.clone().into(),
            start_layer: start,
            end_layer_exclusive: end,
            threads: 2,
            gpu_layers: 0,
            context: 1024,
            sessions: 2,
            top_logits: 0,
            rollback: 4,
        })
        .unwrap()
    };
    let hidden = |output: StageOutput| match output {
        StageOutput::Hidden(values) => values
            .iter()
            .flat_map(|value| value.to_le_bytes())
            .collect::<Vec<u8>>(),
        other => panic!("expected hidden states, got {other:?}"),
    };
    let logits = |output: StageOutput| match output {
        StageOutput::Logits(values) => values,
        other => panic!("expected logits, got {other:?}"),
    };
    let same = |left: &[f32], right: &[f32]| {
        left.len() == right.len()
            && left
                .iter()
                .zip(right)
                .all(|(left, right)| (left - right).abs() < 1e-3)
    };

    let prompt: Vec<u32> = (1000..1024).collect();
    let (next, wrong) = (2000u32, 3000u32);
    let at = prompt.len() as u32;
    let (mut first, mut last) = (stage(0, layers / 2), stage(layers / 2, layers));

    // what the model answers without any guess
    let through = hidden(first.decode_tokens(1, 0, &prompt, false).unwrap());
    last.decode_hidden(1, 0, at, &through, false).unwrap();
    let through = hidden(first.decode_tokens(1, at, &[next], false).unwrap());
    let expected = logits(last.decode_hidden(1, at, 1, &through, false).unwrap());

    let prompt_hidden = hidden(first.decode_tokens(2, 0, &prompt, false).unwrap());
    last.decode_hidden(2, 0, at, &prompt_hidden, false).unwrap();

    // a guess stopped before it reached the engine
    first.abort_handle().stop(7);
    assert!(first.guess_tokens(2, at, &[wrong], 7).unwrap().is_none());

    // a long guess stopped while the engine works on it, on both kinds of stage
    let long = vec![wrong; 400];
    let stop = first.abort_handle();
    let stopper = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(30));
        stop.stop(8);
    });
    let started = std::time::Instant::now();
    let stopped = first.guess_tokens(2, at, &long, 8).unwrap();
    stopper.join().unwrap();
    eprintln!(
        "first stage: {} after {:?}",
        if stopped.is_none() {
            "stopped"
        } else {
            "finished"
        },
        started.elapsed()
    );
    assert!(stopped.is_none(), "400 tokens were decoded in 30 ms");

    let long_hidden = prompt_hidden.repeat(400 / prompt.len() + 1);
    let width = prompt_hidden.len() / prompt.len();
    let stop = last.abort_handle();
    let stopper = std::thread::spawn(move || {
        std::thread::sleep(std::time::Duration::from_millis(30));
        stop.stop(9);
    });
    let stopped = last
        .guess_hidden(2, at, 400, &long_hidden[..400 * width], 9)
        .unwrap();
    stopper.join().unwrap();
    assert!(stopped.is_none(), "400 tokens were decoded in 30 ms");

    // a guess that ran to its end and was wrong: the next request starts where it started
    let through = hidden(first.guess_tokens(2, at, &[wrong], 10).unwrap().unwrap());
    let wrong_logits = logits(last.guess_hidden(2, at, 1, &through, 10).unwrap().unwrap());
    assert!(!same(&wrong_logits, &expected));
    // a stop that comes too late changes nothing
    first.abort_handle().stop(10);
    last.abort_handle().stop(10);

    let through = hidden(first.decode_tokens(2, at, &[next], false).unwrap());
    let answer = logits(last.decode_hidden(2, at, 1, &through, false).unwrap());
    assert!(same(&answer, &expected), "the session kept part of a guess");

    // and a guess that was right goes on as if it had been certain
    let through = hidden(first.guess_tokens(2, at + 1, &[next], 11).unwrap().unwrap());
    last.guess_hidden(2, at + 1, 1, &through, 11)
        .unwrap()
        .unwrap();
    first.decode_tokens(2, at + 2, &[next], false).unwrap();
}
