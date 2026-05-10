/// CPU-side token sampling. GPU acceleration of the full softmax-argmax pipeline
/// is deferred (the logits are small enough that a CPU roundtrip is fine for now).
pub fn sample(logits: &[f32], temperature: f32, top_k: usize) -> u32 {
    if logits.is_empty() {
        return 0;
    }
    if temperature <= 0.0 || !temperature.is_finite() {
        return argmax(logits);
    }

    let inv_t = 1.0 / temperature.max(1e-6);
    let mut scored: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, v * inv_t))
        .collect();

    if top_k > 0 && top_k < scored.len() {
        scored.select_nth_unstable_by(top_k, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(top_k);
    }

    let max = scored.iter().fold(f32::NEG_INFINITY, |acc, &(_, v)| if v > acc { v } else { acc });
    let mut sum = 0f32;
    for e in scored.iter_mut() {
        let p = (e.1 - max).exp();
        e.1 = p;
        sum += p;
    }
    if sum <= 0.0 || !sum.is_finite() {
        return argmax(logits);
    }

    let u: f32 = rand::random();
    let target = u * sum;
    let mut acc = 0f32;
    for &(i, p) in &scored {
        acc += p;
        if acc >= target {
            return i as u32;
        }
    }
    scored.last().map(|&(i, _)| i as u32).unwrap_or(0)
}

fn argmax(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in logits.iter().enumerate() {
        if v.is_finite() && v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    best_idx as u32
}
