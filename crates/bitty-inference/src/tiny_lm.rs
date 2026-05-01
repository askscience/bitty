const VOCAB: usize = 96;
const NEWLINE_INDEX: usize = 95;

const DEFAULT_CORPUS: &str = r#"
Distributed inference turns many small machines into one cooperative model.
The coordinator profiles every worker and assigns layers according to measured speed.
Fast nodes handle prefill work while slower nodes help with steady decode traffic.
Each token moves around the ring as an activation tensor with a checksum.
When a worker disappears the coordinator rebuilds the topology and keeps serving.
Batching nearby requests improves throughput without changing the model output.
Speculative decoding asks a small draft model to propose likely next tokens.
The main model verifies those tokens and rolls back only the rejected suffix.
Rust workers keep weight shards resident in memory and report metrics continuously.
Tiny local tests should be deterministic, cheap, and useful before large models arrive.
"#;

#[derive(Clone, Debug)]
pub struct TinyLanguageModel {
    transitions: [[u32; VOCAB]; VOCAB],
}

impl Default for TinyLanguageModel {
    fn default() -> Self {
        Self::train(DEFAULT_CORPUS)
    }
}

impl TinyLanguageModel {
    pub fn train(corpus: &str) -> Self {
        let mut transitions = [[0_u32; VOCAB]; VOCAB];
        let normalized = normalize(corpus);

        for pair in normalized.windows(2) {
            let prev = byte_to_index(pair[0]);
            let next = byte_to_index(pair[1]);
            transitions[prev][next] = transitions[prev][next].saturating_add(8);
        }

        Self { transitions }
    }

    pub fn generate(&self, prompt: &str, max_chars: usize, seed: u64) -> String {
        let mut rng = XorShift64::new(seed ^ stable_hash(prompt.as_bytes()));
        let mut output = String::with_capacity(prompt.len() + max_chars);
        output.push_str(prompt);

        let mut previous = prompt
            .bytes()
            .rev()
            .find(|byte| byte_to_index(*byte) < VOCAB)
            .unwrap_or(b'\n');

        for _ in 0..max_chars {
            let next = self.sample_next(previous, &mut rng);
            output.push(index_to_byte(next) as char);
            previous = index_to_byte(next);
        }

        output
    }

    pub fn next_token_distribution(&self, previous: u8) -> &[u32; VOCAB] {
        &self.transitions[byte_to_index(previous)]
    }

    fn sample_next(&self, previous: u8, rng: &mut XorShift64) -> usize {
        let row = self.next_token_distribution(previous);
        let total = row.iter().map(|count| *count as u64).sum::<u64>();
        if total == 0 {
            return byte_to_index(b' ');
        }

        let mut ticket = rng.next() % total.max(1);

        for (index, count) in row.iter().enumerate() {
            let count = *count as u64;
            if ticket < count {
                return index;
            }
            ticket -= count;
        }

        byte_to_index(b' ')
    }
}

fn normalize(input: &str) -> Vec<u8> {
    input
        .bytes()
        .map(|byte| match byte {
            b'\n' | b'\r' => b'\n',
            32..=126 => byte,
            _ => b' ',
        })
        .collect()
}

fn byte_to_index(byte: u8) -> usize {
    match byte {
        b'\n' | b'\r' => NEWLINE_INDEX,
        32..=126 => (byte - 32) as usize,
        _ => 0,
    }
}

fn index_to_byte(index: usize) -> u8 {
    match index {
        NEWLINE_INDEX => b'\n',
        0..=94 => index as u8 + 32,
        _ => b' ',
    }
}

fn stable_hash(bytes: &[u8]) -> u64 {
    let mut hash = 0xcbf29ce484222325_u64;
    for byte in bytes {
        hash ^= *byte as u64;
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[derive(Clone, Debug)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    fn next(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tiny_model_generates_deterministically() {
        let model = TinyLanguageModel::default();
        let first = model.generate("The coordinator", 64, 42);
        let second = model.generate("The coordinator", 64, 42);

        assert_eq!(first, second);
        assert!(first.starts_with("The coordinator"));
        assert!(first.len() > "The coordinator".len());
    }

    #[test]
    fn tiny_model_learns_prompt_adjacent_tokens() {
        let model = TinyLanguageModel::train("aaab aaac aaad");
        let distribution = model.next_token_distribution(b'a');

        assert!(distribution[byte_to_index(b'a')] > distribution[byte_to_index(b'z')]);
    }
}
