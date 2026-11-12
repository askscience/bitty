//! Guesses for the next tokens taken from text already seen, with no model.
//!
//! The ring pays the network once per pass, whatever the number of tokens in it, so every
//! guess that turns out right saves a whole trip. The guesses here cost nothing to make:
//!
//! - [`lookup_in_text`] finds the last few tokens earlier in the request itself (its prompt
//!   and what the model has written so far) and proposes what followed them there. Models
//!   copy a lot from what they were given: quotes, summaries, code, corrections.
//! - [`GuessHistory`] does the same in the answers the ring gave before, so a phrase or a
//!   format that keeps coming back is guessed from the second time on.
//!
//! A guess never changes the text: the workers check it, and it is kept only while it is the
//! token the model itself chooses.

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

/// The longest run of tokens looked for. Three tokens are rarely found by chance.
const LONGEST_MATCH: usize = 3;
/// The shortest run of tokens that is worth a guess inside the request. One token alone
/// finds anything.
const SHORTEST_MATCH: usize = 2;
/// Tokens kept in the history unless `BITTY_GUESS_HISTORY_TOKENS` says otherwise: about 4 MB.
const DEFAULT_HISTORY_TOKENS: usize = 1 << 20;
/// Places remembered for each run of three tokens: the most recent ones.
const PLACES_PER_KEY: usize = 8;
/// Marks the boundary between two answers in the history: no guess runs across it.
const BOUNDARY: u32 = u32::MAX;
/// The start of the history file, followed by a version and the token count.
const FILE_MAGIC: &[u8; 4] = b"BGHC";
const FILE_VERSION: u32 = 1;

/// What `BITTY_GUESS_CACHE` asks for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GuessCacheMode {
    /// No guesses from text.
    Off,
    /// Only guesses from the request itself: nothing is kept after it.
    Request,
    /// Guesses from the request and from the answers given before. The default.
    All,
}

pub fn guess_cache_mode() -> GuessCacheMode {
    parse_mode(std::env::var("BITTY_GUESS_CACHE").ok().as_deref())
}

fn parse_mode(value: Option<&str>) -> GuessCacheMode {
    match value
        .map(|value| value.trim().to_ascii_lowercase())
        .as_deref()
    {
        Some("off" | "0" | "no" | "false" | "none") => GuessCacheMode::Off,
        Some("request") => GuessCacheMode::Request,
        _ => GuessCacheMode::All,
    }
}

/// Up to `count` tokens that followed the last tokens of `text` the last time they appeared
/// earlier in it. Tries the last three tokens first, then the last two; `None` when neither
/// appears before.
pub fn lookup_in_text(text: &[u32], count: usize) -> Option<Vec<u32>> {
    find_in_text(text, count).map(|(_, guesses)| guesses)
}

/// [`lookup_in_text`], with the place in `text` the guesses were taken from.
fn find_in_text(text: &[u32], count: usize) -> Option<(usize, Vec<u32>)> {
    if count == 0 {
        return None;
    }
    for length in (SHORTEST_MATCH..=LONGEST_MATCH).rev() {
        if text.len() <= length {
            continue;
        }
        let key = &text[text.len() - length..];
        // the most recent place first: the nearer text is the more likely to repeat
        let found = (0..text.len() - length)
            .rev()
            .find(|&start| &text[start..start + length] == key);
        if let Some(start) = found {
            let from = start + length;
            let end = (from + count).min(text.len());
            if from < end {
                return Some((from, text[from..end].to_vec()));
            }
        }
    }
    None
}

/// How far around the place a copy had reached the last token is looked for, once the last
/// two tokens are found nowhere: a corrected word is seldom more tokens than this away from
/// the one it replaces.
const NEAR_BEHIND: usize = 2;
const NEAR_AHEAD: usize = 8;

/// The guesses of one request from its own text, which keeps track of where in that text
/// the model was copying from.
///
/// [`lookup_in_text`] loses a copy as soon as the model changes a word: the last three
/// tokens, and often the last two, are then found nowhere. A model that corrects a text goes
/// on copying right after the word it changed, so when nothing else is found the last token
/// alone is looked for near the place the copy had reached, and what follows it there is
/// sent. Alone and anywhere it would find anything; there it is seldom wrong.
#[derive(Clone, Debug, Default)]
pub struct TextLookup {
    /// Where in the text the next token would be if the copy went on.
    next_source: Option<usize>,
}

impl TextLookup {
    pub fn new() -> Self {
        Self::default()
    }

    /// Up to `count` guesses for what follows `text`.
    pub fn propose(&mut self, text: &[u32], count: usize) -> Option<Vec<u32>> {
        if let Some((from, guesses)) = find_in_text(text, count) {
            self.next_source = Some(from);
            return Some(guesses);
        }
        let (Some(hint), Some(&last)) = (self.next_source, text.last()) else {
            return None;
        };
        if count == 0 {
            return None;
        }
        // positions before the last token only: it cannot be found in itself
        let window = hint.saturating_sub(NEAR_BEHIND)..(hint + NEAR_AHEAD).min(text.len() - 1);
        let at = window
            .clone()
            .find(|&place| place >= hint && text[place] == last)
            .or_else(|| window.rev().find(|&place| text[place] == last))?;
        let from = at + 1;
        let end = (from + count).min(text.len());
        if from >= end {
            return None;
        }
        self.next_source = Some(from);
        Some(text[from..end].to_vec())
    }

    /// Tells the lookup how many tokens the last pass added to the text: the copy has moved
    /// on by as many.
    pub fn advance(&mut self, added: usize) {
        if let Some(next) = self.next_source.as_mut() {
            *next += added;
        }
    }
}

/// The answers the ring gave before, as token ids, with an index of where each run of three
/// tokens appears. Bounded: past the limit the oldest half is dropped.
#[derive(Debug)]
pub struct GuessHistory {
    tokens: Vec<u32>,
    index: HashMap<[u32; 3], Vec<u32>>,
    limit: usize,
    file: Option<PathBuf>,
    /// Answers added since the history was last written to its file.
    unsaved: usize,
}

impl GuessHistory {
    /// An empty history of at most `limit` tokens, kept only in memory.
    pub fn new(limit: usize) -> Self {
        Self {
            tokens: Vec::new(),
            index: HashMap::new(),
            limit: limit.max(LONGEST_MATCH * 4),
            file: None,
            unsaved: 0,
        }
    }

    /// The history saved in `file`, or an empty one when there is none or it cannot be read.
    /// [`GuessHistory::save`] writes it back there. The limit is `BITTY_GUESS_HISTORY_TOKENS`.
    pub fn open(file: impl Into<PathBuf>) -> Self {
        let file = file.into();
        let limit = std::env::var("BITTY_GUESS_HISTORY_TOKENS")
            .ok()
            .and_then(|value| value.trim().parse::<usize>().ok())
            .unwrap_or(DEFAULT_HISTORY_TOKENS);
        let mut history = Self::new(limit);
        if let Some(tokens) = read_tokens(&file) {
            history.tokens = tokens;
            history.trim();
            history.reindex();
        }
        history.file = Some(file);
        history
    }

    /// Tokens held, boundaries included.
    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }

    /// Adds an answer: `lead` is the end of the prompt it answered (the turn markers of the
    /// chat template), so that its first tokens can be guessed too; `answer` is what the
    /// model wrote.
    pub fn add(&mut self, lead: &[u32], answer: &[u32]) {
        if answer.is_empty() {
            return;
        }
        let lead = &lead[lead.len().saturating_sub(LONGEST_MATCH)..];
        let start = self.tokens.len();
        self.tokens.extend(
            lead.iter()
                .chain(answer)
                .copied()
                .filter(|token| *token != BOUNDARY),
        );
        self.tokens.push(BOUNDARY);
        if self.tokens.len() > self.limit {
            self.trim();
            self.reindex();
        } else {
            self.index_from(start);
        }
        self.unsaved += 1;
    }

    /// Up to `count` tokens that followed the last three tokens of `text` in earlier answers.
    /// Of the places where they appear, the continuation that most of them share wins; among
    /// equals, the most recent.
    pub fn propose(&self, text: &[u32], count: usize) -> Option<Vec<u32>> {
        if count == 0 || text.len() < LONGEST_MATCH {
            return None;
        }
        let key: [u32; 3] = text[text.len() - LONGEST_MATCH..].try_into().ok()?;
        let places = self.index.get(&key)?;
        // how many places go on with each first token, and the most recent place for each
        let mut votes: HashMap<u32, (usize, usize)> = HashMap::new();
        for (order, place) in places.iter().enumerate() {
            let next = *place as usize + LONGEST_MATCH;
            let Some(&token) = self.tokens.get(next).filter(|token| **token != BOUNDARY) else {
                continue;
            };
            let entry = votes.entry(token).or_insert((0, order));
            entry.0 += 1;
            entry.1 = order;
        }
        let (_, (_, order)) = votes
            .into_iter()
            .max_by_key(|(_, (count, order))| (*count, *order))?;
        let from = places[order] as usize + LONGEST_MATCH;
        let guesses: Vec<u32> = self.tokens[from..]
            .iter()
            .take(count)
            .take_while(|token| **token != BOUNDARY)
            .copied()
            .collect();
        (!guesses.is_empty()).then_some(guesses)
    }

    /// Writes the history to its file when answers were added since the last time. The file
    /// is written beside and then renamed, so it is never seen half written.
    pub fn save(&mut self) -> std::io::Result<()> {
        let Some(file) = self.file.clone() else {
            return Ok(());
        };
        if self.unsaved == 0 {
            return Ok(());
        }
        write_tokens(&file, &self.tokens)?;
        self.unsaved = 0;
        Ok(())
    }

    /// Drops the oldest answers until at most half the limit is left, at a boundary.
    fn trim(&mut self) {
        if self.tokens.len() <= self.limit {
            return;
        }
        let keep_from = self.tokens.len() - self.limit / 2;
        let cut = self.tokens[keep_from..]
            .iter()
            .position(|token| *token == BOUNDARY)
            .map_or(self.tokens.len(), |offset| keep_from + offset + 1);
        self.tokens.drain(..cut);
    }

    fn reindex(&mut self) {
        self.index.clear();
        self.index_from(0);
    }

    fn index_from(&mut self, start: usize) {
        let first = start.saturating_sub(LONGEST_MATCH - 1);
        for place in first..self.tokens.len().saturating_sub(LONGEST_MATCH - 1) {
            let window = &self.tokens[place..place + LONGEST_MATCH];
            if window.contains(&BOUNDARY) {
                continue;
            }
            let key: [u32; 3] = window.try_into().expect("three tokens");
            let places = self.index.entry(key).or_default();
            if places.last() == Some(&(place as u32)) {
                continue;
            }
            places.push(place as u32);
            if places.len() > PLACES_PER_KEY {
                places.remove(0);
            }
        }
    }
}

/// Weight, in passes, of the chance carried over from earlier requests against the passes
/// of the request itself.
const PRIOR_PASSES: f64 = 4.0;
/// The chance that a guess is right before anything was seen: a match of three tokens.
const FIRST_CHANCE: f64 = 0.6;
/// Weight of a new pass in the average time of the passes with as many guesses.
const TIME_WEIGHT: f64 = 0.25;
/// Before two sizes of pass were timed: how much longer each guess makes a pass, as a share
/// of a pass. Measured on a ring of four with three hops through a relay: about 0.2.
const ASSUMED_COST_PER_GUESS: f64 = 0.2;
/// The memory of the chance across requests fades by this much at every pass, so the leader
/// follows what it is asked now.
const CHANCE_FADE: f64 = 0.98;

/// How many guesses a pass should carry, from what passes cost on this ring and how often
/// guesses turn out right.
///
/// A pass with `k` guesses that are each right with chance `p` (as long as the ones before
/// them were) gives `(1 - p^(k+1)) / (1 - p)` tokens on average, and costs the time of a
/// pass with `k` guesses. That time is measured: the average of the passes with as many
/// guesses, and a straight line through those averages for the sizes not tried yet. The
/// count that gives the most tokens a second wins, 0 included: on a ring where long passes
/// cost a lot and guesses are seldom right, no guess at all is the fastest.
///
/// The times belong to one plan of the ring: [`GuessTuning::follow_plan`] forgets them when
/// it changes.
#[derive(Clone, Debug, Default)]
pub struct GuessTuning {
    plan: String,
    /// The average time of a pass in ms, by the number of guesses it carried.
    times: Vec<Option<f64>>,
    /// Guesses that were right, and passes where one was wrong, over earlier requests.
    right: f64,
    wrong: f64,
}

/// How the guesses of one request went: guesses right, and passes where one was wrong.
#[derive(Clone, Copy, Debug, Default)]
pub struct RequestGuesses {
    pub right: usize,
    pub wrong: usize,
}

impl RequestGuesses {
    /// Counts a pass that sent `sent` guesses and kept `kept`.
    pub fn record(&mut self, sent: usize, kept: usize) {
        self.right += kept;
        if kept < sent {
            self.wrong += 1;
        }
    }
}

impl GuessTuning {
    /// Forgets the times when the ring has another plan: the passes cost something else.
    pub fn follow_plan(&mut self, plan: &str) {
        if self.plan != plan {
            self.plan = plan.to_string();
            self.times.clear();
        }
    }

    /// The time of a pass that carried `guesses` guesses. The prompt is not a pass of this
    /// kind and is not to be recorded.
    pub fn record_pass(&mut self, guesses: usize, ms: f64) {
        if !ms.is_finite() || ms <= 0.0 {
            return;
        }
        if self.times.len() <= guesses {
            self.times.resize(guesses + 1, None);
        }
        let slot = &mut self.times[guesses];
        *slot = Some(match *slot {
            Some(average) => average + (ms - average) * TIME_WEIGHT,
            None => ms,
        });
    }

    /// Counts the outcome of a pass with guesses for later requests.
    pub fn record_outcome(&mut self, sent: usize, kept: usize) {
        if sent == 0 {
            return;
        }
        self.right = self.right * CHANCE_FADE + kept as f64;
        self.wrong = self.wrong * CHANCE_FADE + f64::from(u8::from(kept < sent));
    }

    /// The chance that a guess is right for this request: what the request has shown, with
    /// what the earlier ones showed as a start.
    pub fn chance(&self, request: RequestGuesses) -> f64 {
        let earlier =
            (self.right + FIRST_CHANCE * PRIOR_PASSES) / (self.right + self.wrong + PRIOR_PASSES);
        let (right, wrong) = (request.right as f64, request.wrong as f64);
        ((right + earlier * PRIOR_PASSES) / (right + wrong + PRIOR_PASSES)).clamp(0.0, 0.999)
    }

    /// The time of a pass with `guesses` guesses: measured, or from the line through the
    /// measured sizes. `None` before any pass was timed.
    pub fn pass_ms(&self, guesses: usize) -> Option<f64> {
        if let Some(Some(measured)) = self.times.get(guesses) {
            return Some(*measured);
        }
        let points: Vec<(f64, f64)> = self
            .times
            .iter()
            .enumerate()
            .filter_map(|(size, time)| time.map(|time| (size as f64, time)))
            .collect();
        let (base, slope) = match points.as_slice() {
            [] => return None,
            [(size, time)] => {
                let base = time / (1.0 + ASSUMED_COST_PER_GUESS * size);
                (base, base * ASSUMED_COST_PER_GUESS)
            }
            points => {
                let n = points.len() as f64;
                let mean_x = points.iter().map(|(x, _)| x).sum::<f64>() / n;
                let mean_y = points.iter().map(|(_, y)| y).sum::<f64>() / n;
                let spread: f64 = points.iter().map(|(x, _)| (x - mean_x).powi(2)).sum();
                let joint: f64 = points
                    .iter()
                    .map(|(x, y)| (x - mean_x) * (y - mean_y))
                    .sum();
                // a pass never gets cheaper with more tokens in it
                let slope = (joint / spread).max(0.0);
                ((mean_y - slope * mean_x).max(1.0), slope)
            }
        };
        Some(base + slope * guesses as f64)
    }

    /// How many guesses a pass should carry, at most `limit`, when each is right with
    /// `chance`. 0 means a pass without guesses is the fastest.
    pub fn best_count(&self, chance: f64, limit: usize) -> usize {
        let tokens = |guesses: usize| {
            if chance >= 0.999 {
                guesses as f64 + 1.0
            } else {
                (1.0 - chance.powi(guesses as i32 + 1)) / (1.0 - chance)
            }
        };
        let cost = |guesses: usize| {
            self.pass_ms(guesses)
                .unwrap_or(1.0 + ASSUMED_COST_PER_GUESS * guesses as f64)
        };
        (0..=limit)
            .map(|guesses| (guesses, tokens(guesses) / cost(guesses)))
            .fold(
                (0, f64::MIN),
                |best, next| if next.1 > best.1 { next } else { best },
            )
            .0
    }

    /// One line for the log: the time of a pass by its guesses, and the chance across
    /// requests.
    pub fn summary(&self) -> String {
        let times: Vec<String> = self
            .times
            .iter()
            .enumerate()
            .filter_map(|(size, time)| time.map(|time| format!("{size}: {time:.0}")))
            .collect();
        format!(
            "ms per pass by guesses [{}], a guess is right {:.0}% of the time",
            times.join(", "),
            self.chance(RequestGuesses::default()) * 100.0
        )
    }
}

/// The name of the history file of a model: its file name and vocabulary size, since the
/// token ids of one model mean nothing in another.
pub fn history_file_name(model: &Path, vocab_size: u32) -> String {
    let stem = model
        .file_stem()
        .map(|stem| stem.to_string_lossy().into_owned())
        .unwrap_or_else(|| "model".into());
    let stem: String = stem
        .chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' || c == '.' {
                c
            } else {
                '_'
            }
        })
        .collect();
    format!("{stem}.v{vocab_size}.guesses")
}

fn read_tokens(file: &Path) -> Option<Vec<u32>> {
    let mut bytes = Vec::new();
    std::fs::File::open(file)
        .ok()?
        .read_to_end(&mut bytes)
        .ok()?;
    if bytes.len() < 12 || &bytes[..4] != FILE_MAGIC {
        return None;
    }
    let word = |at: usize| u32::from_le_bytes(bytes[at..at + 4].try_into().expect("four bytes"));
    if word(4) != FILE_VERSION {
        return None;
    }
    let count = word(8) as usize;
    let body = &bytes[12..];
    if body.len() != count * 4 {
        return None;
    }
    Some(
        body.chunks_exact(4)
            .map(|chunk| u32::from_le_bytes(chunk.try_into().expect("four bytes")))
            .collect(),
    )
}

fn write_tokens(file: &Path, tokens: &[u32]) -> std::io::Result<()> {
    if let Some(dir) = file.parent() {
        std::fs::create_dir_all(dir)?;
    }
    let mut bytes = Vec::with_capacity(12 + tokens.len() * 4);
    bytes.extend_from_slice(FILE_MAGIC);
    bytes.extend_from_slice(&FILE_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(tokens.len() as u32).to_le_bytes());
    for token in tokens {
        bytes.extend_from_slice(&token.to_le_bytes());
    }
    let partial = file.with_extension("guesses.partial");
    {
        let mut out = std::fs::File::create(&partial)?;
        out.write_all(&bytes)?;
        out.sync_all()?;
    }
    std::fs::rename(&partial, file)
}

#[cfg(test)]
mod tests {
    use super::*;
    use proptest::prelude::*;

    #[test]
    fn the_request_text_gives_what_followed_the_last_tokens() {
        // "... 7 8 9 10 11 ... 7 8 9" -> 10 11
        let text = [1, 7, 8, 9, 10, 11, 12, 2, 3, 7, 8, 9];
        assert_eq!(lookup_in_text(&text, 2), Some(vec![10, 11]));
        assert_eq!(
            lookup_in_text(&text, 10),
            Some(vec![10, 11, 12, 2, 3, 7, 8, 9])
        );
    }

    #[test]
    fn two_tokens_are_tried_when_three_are_not_found() {
        let text = [5, 8, 9, 4, 1, 8, 9];
        assert_eq!(lookup_in_text(&text, 3), Some(vec![4, 1, 8]));
    }

    #[test]
    fn one_token_alone_is_no_guess() {
        assert_eq!(lookup_in_text(&[4, 9, 5, 6, 9], 3), None);
        assert_eq!(lookup_in_text(&[1, 2], 3), None);
        assert_eq!(lookup_in_text(&[1, 2, 3, 1, 2, 3], 0), None);
    }

    #[test]
    fn the_most_recent_place_wins_in_the_request() {
        let text = [7, 8, 9, 1, 0, 7, 8, 9, 2, 0, 7, 8, 9];
        assert_eq!(lookup_in_text(&text, 1), Some(vec![2]));
    }

    /// Writes `answer` after `prompt` the way the ring does: a pass sends the guesses, keeps
    /// those that match and adds one token of the answer after them. Returns the passes.
    fn passes_to_write(prompt: &[u32], answer: &[u32], near: bool) -> usize {
        let mut text = prompt.to_vec();
        let mut lookup = TextLookup::new();
        let (mut written, mut passes) = (0, 0);
        while written < answer.len() {
            let guesses = if near {
                lookup.propose(&text, 4)
            } else {
                lookup_in_text(&text, 4)
            }
            .unwrap_or_default();
            // the guesses the model agrees with, and the token it chooses after them
            let right = guesses
                .iter()
                .zip(&answer[written..])
                .take_while(|(guess, token)| guess == token)
                .count();
            let added = (right + 1).min(answer.len() - written);
            text.extend_from_slice(&answer[written..written + added]);
            written += added;
            lookup.advance(added);
            passes += 1;
        }
        passes
    }

    #[test]
    fn a_copy_goes_on_after_a_corrected_word() {
        // the prompt asks to correct a text; the answer is the text with two words changed
        let mut prompt: Vec<u32> = vec![900, 901, 902];
        let original: Vec<u32> = (1..=30).collect();
        prompt.extend(&original);
        prompt.extend([903, 904]);
        let mut answer = original.clone();
        answer[8] = 500;
        answer[19] = 501;
        let plain = passes_to_write(&prompt, &answer, false);
        let near = passes_to_write(&prompt, &answer, true);
        assert!(near < plain, "near {near}, plain {plain}");
        eprintln!("passes: near {near}, plain {plain}");
    }

    #[test]
    fn the_near_lookup_needs_a_copy_to_follow() {
        let mut lookup = TextLookup::new();
        // nothing was copied yet: the last token alone is no guess
        assert_eq!(lookup.propose(&[4, 9, 5, 6, 9], 3), None);
    }

    #[test]
    fn the_history_guesses_an_answer_seen_before() {
        let mut history = GuessHistory::new(1000);
        // the chat template ends every prompt the same way
        let lead = [100, 101, 102];
        history.add(&[50, 60, 100, 101, 102], &[10, 11, 12, 13]);
        assert_eq!(
            history.propose(&[9, 100, 101, 102], 3),
            Some(vec![10, 11, 12])
        );
        assert_eq!(
            history.propose(&[100, 101, 102, 10, 11], 9),
            Some(vec![12, 13])
        );
        // no guess runs past the end of an answer
        assert_eq!(history.propose(&[11, 12, 13], 4), None);
        history.add(&lead, &[20, 21]);
        assert_eq!(history.len(), 4 + 3 + 1 + 2 + 3 + 1);
    }

    #[test]
    fn the_continuation_most_answers_share_wins() {
        let mut history = GuessHistory::new(1000);
        let lead = [100, 101, 102];
        history.add(&lead, &[1, 2, 3]);
        history.add(&lead, &[1, 2, 3]);
        history.add(&lead, &[7, 8, 9]);
        assert_eq!(history.propose(&lead, 2), Some(vec![1, 2]));
        history.add(&lead, &[7, 8, 9]);
        // two against two: the most recent
        assert_eq!(history.propose(&lead, 2), Some(vec![7, 8]));
    }

    #[test]
    fn the_history_drops_its_oldest_answers_past_the_limit() {
        let mut history = GuessHistory::new(40);
        for answer in 0..20u32 {
            history.add(&[], &[1000 + answer, 1, 2, 3, 4]);
        }
        assert!(history.len() <= 40);
        // the first answers are gone, the last one is still there
        assert_eq!(history.propose(&[1000, 1, 2], 1), None);
        assert_eq!(history.propose(&[1019, 1, 2], 2), Some(vec![3, 4]));
    }

    #[test]
    fn the_history_comes_back_from_its_file() {
        let dir = std::env::temp_dir().join(format!("bitty-guesses-{}", std::process::id()));
        let file = dir.join("model.v8.guesses");
        let mut history = GuessHistory::open(&file);
        assert!(history.is_empty());
        history.add(&[100, 101, 102], &[10, 11, 12]);
        history.save().unwrap();
        let again = GuessHistory::open(&file);
        assert_eq!(again.propose(&[100, 101, 102], 2), Some(vec![10, 11]));
        // a broken file is an empty history, not an error
        std::fs::write(&file, b"BGHC broken").unwrap();
        assert!(GuessHistory::open(&file).is_empty());
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn many_guesses_go_out_when_they_are_right_and_passes_are_cheap() {
        let mut tuning = GuessTuning::default();
        tuning.follow_plan("a");
        tuning.record_pass(0, 800.0);
        tuning.record_pass(4, 900.0);
        // right nine times in ten: the longest pass gives the most
        assert_eq!(tuning.best_count(0.9, 8), 8);
        // right one time in fifty: a guess is not worth its cost
        assert_eq!(tuning.best_count(0.02, 8), 0);
    }

    #[test]
    fn expensive_long_passes_get_fewer_guesses() {
        // the ring measured today: 0.8 s with none, 1.5 s with 4, 3.3 s with 8
        let mut tuning = GuessTuning::default();
        tuning.record_pass(0, 790.0);
        tuning.record_pass(4, 1500.0);
        tuning.record_pass(8, 3300.0);
        let best = tuning.best_count(0.95, 8);
        assert!((3..8).contains(&best), "{best}");
        // the measured times are used as they are, the line only between them
        assert_eq!(tuning.pass_ms(4), Some(1500.0));
        let between = tuning.pass_ms(6).unwrap();
        assert!(between > 1500.0 && between < 3300.0, "{between}");
    }

    #[test]
    fn one_timed_size_is_enough_to_start() {
        let mut tuning = GuessTuning::default();
        assert_eq!(tuning.pass_ms(3), None);
        // with nothing timed, the assumed cost decides: long passes, guesses being right
        assert!(tuning.best_count(0.9, 8) >= 5);
        tuning.record_pass(0, 1000.0);
        assert!((tuning.pass_ms(5).unwrap() - 2000.0).abs() < 1e-6);
    }

    #[test]
    fn a_new_plan_forgets_the_times() {
        let mut tuning = GuessTuning::default();
        tuning.follow_plan("a");
        tuning.record_pass(2, 500.0);
        tuning.follow_plan("a");
        assert_eq!(tuning.pass_ms(2), Some(500.0));
        tuning.follow_plan("b");
        assert_eq!(tuning.pass_ms(2), None);
    }

    #[test]
    fn the_chance_follows_the_request_and_starts_from_the_earlier_ones() {
        let mut tuning = GuessTuning::default();
        let fresh = RequestGuesses::default();
        assert!((tuning.chance(fresh) - FIRST_CHANCE).abs() < 1e-9);
        for _ in 0..50 {
            tuning.record_outcome(4, 0);
        }
        // earlier requests guessed badly: a new one starts low
        assert!(tuning.chance(fresh) < 0.1);
        // and climbs as its own guesses are right
        let mut request = RequestGuesses::default();
        for _ in 0..10 {
            request.record(8, 8);
        }
        assert!(tuning.chance(request) > 0.9);
    }

    #[test]
    fn the_mode_is_read_from_the_setting() {
        assert_eq!(parse_mode(None), GuessCacheMode::All);
        assert_eq!(parse_mode(Some("off")), GuessCacheMode::Off);
        assert_eq!(parse_mode(Some(" Request ")), GuessCacheMode::Request);
    }

    #[test]
    fn the_file_name_is_safe() {
        let name = history_file_name(Path::new("/m/Qwen3.5 35B?.gguf"), 248320);
        assert_eq!(name, "Qwen3.5_35B_.v248320.guesses");
    }

    /// The slow way: every place where the key appears, and what followed it.
    fn slow_lookup(text: &[u32], count: usize) -> Option<Vec<u32>> {
        for length in (SHORTEST_MATCH..=LONGEST_MATCH).rev() {
            if text.len() <= length {
                continue;
            }
            let key = &text[text.len() - length..];
            let mut best = None;
            for start in 0..text.len() - length {
                if &text[start..start + length] == key {
                    best = Some(start);
                }
            }
            if let Some(start) = best {
                let from = start + length;
                let end = (from + count).min(text.len());
                if from < end {
                    return Some(text[from..end].to_vec());
                }
            }
        }
        None
    }

    proptest! {
        #[test]
        fn lookup_matches_a_slow_search(text in proptest::collection::vec(0u32..6, 0..60), count in 0usize..6) {
            prop_assert_eq!(lookup_in_text(&text, count), slow_lookup(&text, count));
        }

        #[test]
        fn a_history_guess_follows_its_key_somewhere(
            answers in proptest::collection::vec(proptest::collection::vec(0u32..5, 1..20), 1..12),
            text in proptest::collection::vec(0u32..5, 3..10),
        ) {
            let mut history = GuessHistory::new(10_000);
            for answer in &answers {
                history.add(&[], answer);
            }
            if let Some(guess) = history.propose(&text, 4) {
                // key followed by the guess appears whole inside one answer
                let key = &text[text.len() - 3..];
                let wanted: Vec<u32> = key.iter().chain(&guess).copied().collect();
                prop_assert!(answers.iter().any(|answer| answer.windows(wanted.len()).any(|w| w == wanted.as_slice())));
            }
        }
    }
}
