//! Picks the next token from the logits of the final stage.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Defaults follow the settings Qwen recommends for its models.
pub const DEFAULT_TOP_K: u32 = 20;
pub const DEFAULT_TOP_P: f32 = 0.95;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SamplingOptions {
    /// 0 always takes the most likely token.
    pub temperature: f32,
    /// Keep only the `top_k` most likely tokens; 0 keeps all.
    pub top_k: u32,
    /// Keep the most likely tokens whose probabilities add up to `top_p`; 1 keeps all.
    pub top_p: f32,
    /// 0 picks a different seed every time.
    pub seed: u64,
}

impl Default for SamplingOptions {
    fn default() -> Self {
        Self {
            temperature: 0.7,
            top_k: DEFAULT_TOP_K,
            top_p: DEFAULT_TOP_P,
            seed: 0,
        }
    }
}

pub struct Sampler {
    options: SamplingOptions,
    rng: StdRng,
}

/// A chosen token and the natural logarithm of the probability it was given.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Sampled {
    pub token: u32,
    pub log_prob: f32,
}

impl Sampler {
    pub fn new(options: SamplingOptions) -> Self {
        let rng = if options.seed == 0 {
            StdRng::from_entropy()
        } else {
            StdRng::seed_from_u64(options.seed)
        };
        Self { options, rng }
    }

    /// Returns `None` when there are no usable logits.
    pub fn sample(&mut self, logits: &[f32]) -> Option<Sampled> {
        self.sample_candidates(
            logits
                .iter()
                .enumerate()
                .map(|(token, logit)| (token as u32, *logit))
                .collect(),
        )
    }

    /// Samples from pairs of token and logit: the whole vocabulary or only its best part, as
    /// [`top_logits`] cuts it. The result is the same as long as the part holds at least
    /// `top_k` tokens; only the probability reported at temperature 0 is then computed on the
    /// part and not on everything.
    pub fn sample_candidates(&mut self, mut candidates: Vec<(u32, f32)>) -> Option<Sampled> {
        candidates.retain(|(_, logit)| !logit.is_nan());
        if candidates.is_empty() {
            return None;
        }
        let greedy = self.options.temperature <= 0.0;
        let keep = if greedy {
            1
        } else if self.options.top_k == 0 {
            candidates.len()
        } else {
            (self.options.top_k as usize).min(candidates.len())
        };
        let log_sum = greedy.then(|| {
            let best = candidates.iter().map(|(_, logit)| *logit).fold(f32::MIN, f32::max);
            log_sum_exp(candidates.iter().map(|(_, logit)| *logit), best)
        });
        if keep < candidates.len() {
            candidates.select_nth_unstable_by(keep - 1, by_logit);
            candidates.truncate(keep);
        }
        candidates.sort_unstable_by(by_logit);
        if let Some(log_sum) = log_sum {
            // the probability reported is the one at temperature 1
            return Some(Sampled {
                token: candidates[0].0,
                log_prob: -log_sum,
            });
        }

        let top = candidates[0].1;
        let mut weights: Vec<f32> = candidates
            .iter()
            .map(|(_, logit)| ((logit - top) / self.options.temperature).exp())
            .collect();
        let total: f32 = weights.iter().sum();
        if self.options.top_p > 0.0 && self.options.top_p < 1.0 {
            let mut cumulative = 0.0;
            let mut kept = weights.len();
            for (index, weight) in weights.iter().enumerate() {
                cumulative += weight / total;
                if cumulative >= self.options.top_p {
                    kept = index + 1;
                    break;
                }
            }
            weights.truncate(kept);
        }
        let total: f32 = weights.iter().sum();
        let mut draw = self.rng.gen::<f32>() * total;
        let mut chosen = weights.len() - 1;
        for (index, weight) in weights.iter().enumerate() {
            if draw < *weight {
                chosen = index;
                break;
            }
            draw -= weight;
        }
        Some(Sampled {
            token: candidates[chosen].0,
            log_prob: (weights[chosen] / total).ln(),
        })
    }
}

/// Best logit first; equal logits go by token id, so the order never depends on how the
/// candidates arrived.
fn by_logit(left: &(u32, f32), right: &(u32, f32)) -> std::cmp::Ordering {
    right.1.total_cmp(&left.1).then(left.0.cmp(&right.0))
}

/// How many logits the final stage sends to the coordinator: `BITTY_TOP_LOGITS`, 1024 by
/// default. 0 sends the whole vocabulary.
pub fn top_logits_limit() -> usize {
    std::env::var("BITTY_TOP_LOGITS")
        .ok()
        .and_then(|value| value.trim().parse::<usize>().ok())
        .unwrap_or(1024)
}

/// The `limit` best logits as pairs of token and logit, best first.
pub fn top_logits(logits: &[f32], limit: usize) -> Vec<(u32, f32)> {
    let mut candidates: Vec<(u32, f32)> = logits
        .iter()
        .enumerate()
        .filter(|(_, logit)| !logit.is_nan())
        .map(|(token, logit)| (token as u32, *logit))
        .collect();
    if limit > 0 && limit < candidates.len() {
        candidates.select_nth_unstable_by(limit - 1, by_logit);
        candidates.truncate(limit);
    }
    candidates.sort_unstable_by(by_logit);
    candidates
}

/// Wire form of [`top_logits`]: little-endian u32 token id and f32 logit for each pair.
pub fn encode_top_logits(pairs: &[(u32, f32)]) -> Vec<u8> {
    pairs
        .iter()
        .flat_map(|(token, logit)| token.to_le_bytes().into_iter().chain(logit.to_le_bytes()))
        .collect()
}

pub fn decode_top_logits(payload: &[u8]) -> Option<Vec<(u32, f32)>> {
    if payload.len() % 8 != 0 {
        return None;
    }
    Some(
        payload
            .chunks_exact(8)
            .map(|pair| {
                (
                    u32::from_le_bytes([pair[0], pair[1], pair[2], pair[3]]),
                    f32::from_le_bytes([pair[4], pair[5], pair[6], pair[7]]),
                )
            })
            .collect(),
    )
}

/// ln(sum(exp(x - max))) for a stable log-probability of the best logit.
fn log_sum_exp(logits: impl Iterator<Item = f32>, max: f32) -> f32 {
    logits.map(|logit| (logit - max).exp()).sum::<f32>().ln()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn options(temperature: f32, top_k: u32, top_p: f32) -> SamplingOptions {
        SamplingOptions {
            temperature,
            top_k,
            top_p,
            seed: 7,
        }
    }

    #[test]
    fn zero_temperature_takes_the_best_token() {
        let mut sampler = Sampler::new(options(0.0, 20, 0.95));
        let sampled = sampler.sample(&[0.1, 3.0, f32::NAN, 2.0]).unwrap();
        assert_eq!(sampled.token, 1);
        assert!(sampled.log_prob < 0.0 && sampled.log_prob > -1.0);
        assert!(sampler.sample(&[]).is_none());
    }

    #[test]
    fn top_k_and_top_p_limit_the_choice() {
        let logits = [5.0, 4.9, 4.8, -2.0, -3.0];
        let mut sampler = Sampler::new(options(1.0, 2, 1.0));
        for _ in 0..200 {
            assert!(sampler.sample(&logits).unwrap().token < 2);
        }
        // one token holds nearly all the probability: top_p leaves only that one
        let mut sampler = Sampler::new(options(1.0, 0, 0.5));
        for _ in 0..200 {
            assert_eq!(sampler.sample(&[10.0, 1.0, 0.5, 0.0]).unwrap().token, 0);
        }
    }

    #[test]
    fn sampling_follows_the_distribution_and_the_seed() {
        let logits = [1.0, 1.0, -20.0];
        let mut sampler = Sampler::new(options(1.0, 0, 1.0));
        let draws: Vec<u32> = (0..400).map(|_| sampler.sample(&logits).unwrap().token).collect();
        let zeros = draws.iter().filter(|token| **token == 0).count();
        assert!((120..280).contains(&zeros), "{zeros}");
        assert!(draws.iter().all(|token| *token < 2));
        let mut again = Sampler::new(options(1.0, 0, 1.0));
        let repeat: Vec<u32> = (0..400).map(|_| again.sample(&logits).unwrap().token).collect();
        assert_eq!(draws, repeat);
    }

    #[test]
    fn the_best_part_of_the_logits_samples_like_the_whole() {
        let logits: Vec<f32> = (0..5000).map(|i| ((i * 37 % 1013) as f32) / 50.0).collect();
        let part = top_logits(&logits, 64);
        assert_eq!(part.len(), 64);
        assert!(part.windows(2).all(|pair| pair[0].1 >= pair[1].1));
        assert_eq!(decode_top_logits(&encode_top_logits(&part)).unwrap(), part);

        let mut whole = Sampler::new(options(0.8, 20, 0.95));
        let mut cut = Sampler::new(options(0.8, 20, 0.95));
        for _ in 0..50 {
            let expected = whole.sample(&logits).unwrap();
            let got = cut.sample_candidates(part.clone()).unwrap();
            assert_eq!(expected.token, got.token);
        }
    }
}
