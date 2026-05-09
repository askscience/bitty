pub fn argmax(logits: &[f32]) -> u32 {
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

pub fn sample_with_temperature(
    logits: &[f32],
    temperature: f32,
    top_k: usize,
    rng_state: &mut u64,
) -> u32 {
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
        .filter(|(_, &v)| v.is_finite())
        .map(|(i, &v)| (i, v * inv_t))
        .collect();

    if scored.is_empty() {
        return 0;
    }

    let k = if top_k > 0 && top_k < scored.len() {
        top_k
    } else {
        scored.len()
    };

    if k < scored.len() {
        scored.select_nth_unstable_by(k, |a, b| {
            b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
    }

    let max_val = scored
        .iter()
        .map(|(_, v)| *v)
        .fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = scored.iter().map(|(_, v)| (v - max_val).exp()).sum();

    if sum <= 0.0 {
        return scored.first().map(|(i, _)| *i as u32).unwrap_or(0);
    }

    let dice = xorshift_f32(rng_state) * sum;
    let mut cumulative = 0.0f32;
    for (idx, val) in &scored {
        cumulative += (val - max_val).exp();
        if cumulative >= dice {
            return *idx as u32;
        }
    }

    scored.last().map(|(i, _)| *i as u32).unwrap_or(0)
}

fn xorshift_f32(state: &mut u64) -> f32 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    let bits = ((*state >> 12) & 0x00FF_FFFF) as u32 | 0x3F80_0000;
    f32::from_bits(bits) - 1.0
}
