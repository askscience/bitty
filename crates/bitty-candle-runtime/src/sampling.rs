use rand::Rng;

pub fn sample_token(
    logits: &mut [f32],
    temperature: f32,
    top_k: usize,
    top_p: f32,
    repeat_penalty: f32,
    recent_tokens: &[u32],
) -> u32 {
    let vocab_size = logits.len();

    if repeat_penalty != 1.0 && !recent_tokens.is_empty() {
        for &token_id in recent_tokens {
            let idx = token_id as usize;
            if idx < vocab_size {
                if logits[idx] > 0.0 {
                    logits[idx] /= repeat_penalty;
                } else {
                    logits[idx] *= repeat_penalty;
                }
            }
        }
    }

    if temperature <= 0.0 {
        let mut best_idx = 0;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i;
            }
        }
        return best_idx as u32;
    }

    if temperature != 1.0 {
        let inv_temp = 1.0 / temperature;
        for logit in logits.iter_mut() {
            *logit *= inv_temp;
        }
    }

    if top_k > 0 && top_k < vocab_size {
        let mut heap: Vec<usize> = (0..top_k).collect();

        for i in (0..(top_k / 2)).rev() {
            sift_down(&mut heap, i, top_k, logits);
        }

        for i in top_k..vocab_size {
            if logits[i] > logits[heap[0]] {
                heap[0] = i;
                sift_down(&mut heap, 0, top_k, logits);
            }
        }

        let threshold = logits[heap[0]];
        for logit in logits.iter_mut().take(vocab_size) {
            if *logit < threshold {
                *logit = f32::NEG_INFINITY;
            }
        }
    }

    let max_val = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<(usize, f32)> = logits
        .iter()
        .enumerate()
        .map(|(i, &v)| (i, (v - max_val).exp()))
        .collect();

    let total_prob: f32 = probs.iter().map(|(_, p)| p).sum();

    if top_p > 0.0 && top_p < 1.0 {
        probs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        let mut cumulative_prob = 0.0;
        let mut cutoff_idx = probs.len();
        for (i, (_, p)) in probs.iter().enumerate() {
            cumulative_prob += p / total_prob;
            if cumulative_prob > top_p {
                cutoff_idx = i + 1;
                break;
            }
        }
        probs.truncate(cutoff_idx);
    }

    let new_total_prob: f32 = probs.iter().map(|(_, p)| p).sum();
    let mut rng = rand::rng();
    let r = rng.random::<f32>() * new_total_prob;
    let mut cumsum = 0.0f32;
    for (i, p) in probs {
        cumsum += p;
        if cumsum >= r {
            return i as u32;
        }
    }

    (vocab_size - 1) as u32
}

fn sift_down(heap: &mut [usize], mut i: usize, n: usize, values: &[f32]) {
    loop {
        let mut min = i;
        let l = 2 * i + 1;
        let r = 2 * i + 2;
        if l < n && values[heap[l]] < values[heap[min]] {
            min = l;
        }
        if r < n && values[heap[r]] < values[heap[min]] {
            min = r;
        }
        if min == i {
            break;
        }
        heap.swap(i, min);
        i = min;
    }
}
