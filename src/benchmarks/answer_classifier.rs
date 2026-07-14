pub use crate::error_classes::WrongAnswerClass;
use std::collections::{HashMap, HashSet};

/// Result of trying to extract an answer, including the class if wrong.
pub struct ExtractionResult {
    pub predicted_answer: Option<char>,
    pub wrong_class: Option<WrongAnswerClass>,
}

/// Classifies a wrong answer into the most specific WrongAnswerClass.
///
/// Classification hierarchy:
/// 1. If `predicted_answer` is Some and is a valid letter but wrong → WrongAnswerKey
/// 2. If `predicted_answer` is Some but is an invalid letter → InvalidAnswerKey
/// 3. If `predicted_answer` is None → apply heuristics: NoAnswer, Looping, Truncated,
///    Refused, Uncertainty, OffTopic
pub fn classify_wrong_answer(
    response: &str,
    question: &str,
    expected_answer: char,
    predicted_answer: Option<char>,
) -> WrongAnswerClass {
    let trimmed = response.trim();

    // Path 1 & 2: we extracted something
    if let Some(pred) = predicted_answer {
        // Is it a valid letter within the expected range (A–J)?
        if pred.is_ascii_alphabetic() && pred == pred.to_ascii_uppercase() && pred <= 'J' {
            if pred != expected_answer {
                return WrongAnswerClass::WrongAnswerKey;
            }
            // Shouldn't reach here, but if somehow pred == expected, fall through
        } else {
            // Invalid letter (e.g., "L" on a 4-choice question, or a non-letter character)
            return WrongAnswerClass::InvalidAnswerKey;
        }
    }

    // Path 3: no valid answer extracted — apply heuristics on the raw response

    // Truncation check on the original response (before trimming)
    if response.ends_with(' ') || response.ends_with('\n') || response.ends_with('\t') {
        return WrongAnswerClass::Truncated;
    }

    // NoAnswer: very short output (< 20 chars) with no extracted answer
    if trimmed.len() < 20 {
        return WrongAnswerClass::NoAnswer;
    }

    // Check looping (repeated phrases)
    if detect_looping(trimmed) {
        return WrongAnswerClass::Looping;
    }

    // Check truncation (ends mid-sentence with conjunctions, etc.)
    if detect_truncated(trimmed) {
        return WrongAnswerClass::Truncated;
    }

    // Check refusal (ethical/safety/policy)
    if detect_refusal(trimmed) {
        return WrongAnswerClass::Refused;
    }

    // Check uncertainty (honest "I don't know")
    if detect_uncertainty(trimmed) {
        return WrongAnswerClass::Uncertainty;
    }

    // Off-topic: long ramble with minimal overlap to the question
    if trimmed.len() > 50 && check_off_topic(trimmed, question) {
        return WrongAnswerClass::OffTopic;
    }

    // Fallback: couldn't classify — default to NoAnswer
    WrongAnswerClass::NoAnswer
}

/// Detects looping by finding a repeated phrase of at least 3 words occurring 3+ times.
fn detect_looping(text: &str) -> bool {
    let words: Vec<&str> = text.split_whitespace().collect();
    let min_phrase_len = 3;
    let min_repeats = 3;
    let text_len = words.len();

    if text_len < min_phrase_len * min_repeats {
        return false;
    }

    // Simple approach: check for exact repeated substrings of 3 words
    for phrase_len in min_phrase_len..=text_len / min_repeats {
        let mut counts: HashMap<Vec<&str>, usize> = HashMap::new();
        for i in 0..=(text_len - phrase_len) {
            let phrase = words[i..i + phrase_len].to_vec();
            let counter = counts.entry(phrase).or_insert(0);
            *counter += 1;
        }
        if counts.values().any(|c| *c >= min_repeats) {
            return true;
        }
    }

    false
}

/// Detects if the response looks truncated (ends mid-sentence).
/// Only checks for conjunction endings, open delimiters — whitespace truncation is handled upstream.
fn detect_truncated(text: &str) -> bool {
    let trimmed = text.trim_end();
    // Check if it ends with a conjunction or subordinator
    let truncated_endings = [
        " and ",
        " but ",
        " because ",
        " so ",
        " or ",
        " yet ",
        " for ",
        " although ",
        " if ",
        " when ",
        " while ",
        " that ",
        " which ",
        " who ",
        " where ",
        " why ",
        " whom ",
        " whether ",
        " how ",
    ];
    for ending in truncated_endings {
        // Check if text ends with this word (case insensitive)
        if trimmed.to_lowercase().ends_with(ending.trim()) {
            return true;
        }
    }
    // Check if it ends with an opening parenthesis, quote, or bracket
    if trimmed.ends_with('(') || trimmed.ends_with('[') || trimmed.ends_with('"') {
        return true;
    }
    false
}

/// Detects uncertainty expressions.
fn detect_uncertainty(text: &str) -> bool {
    let lower = text.to_lowercase();
    let uncertainty_patterns = [
        "i don't know",
        "i'm not sure",
        "i don't think i can",
        "i can't say for sure",
        "i'm uncertain",
        "i do not know",
        "i am not sure",
        "i'm not certain",
        "i cannot determine",
        "not sure",
        "i don't have enough information",
        "i don't think",
    ];
    for pattern in &uncertainty_patterns {
        if lower.contains(pattern) {
            return true;
        }
    }
    false
}

/// Detects ethical/policy refusals.
/// Note: does not include uncertainty phrases; those belong to Uncertainty class.
fn detect_refusal(text: &str) -> bool {
    let lower = text.to_lowercase();
    let refusal_patterns = [
        "i can't help with",
        "i won't",
        "i'm not allowed to",
        "i should not",
        "i refuse",
        "i can't assist with",
        "i will not",
        "i'm not able to help",
        "i cannot help",
        "i won't help",
        "i'm prohibited from",
        "i am not permitted",
        "i can't provide",
        "i'm unable to",
        "i can't do that",
    ];
    for pattern in &refusal_patterns {
        if lower.contains(pattern) {
            return true;
        }
    }
    false
}

/// Checks if the response is off-topic (shares very few words with the question).
/// Returns true if the response is likely hallucinated/off-topic.
///
/// Heuristic: compare overlap of content words (non-function words). If less than 15%
/// of the question's content words appear in the response, flag as off-topic.
/// This is a rough heuristic and may produce false positives/negatives.
fn check_off_topic(response: &str, question: &str) -> bool {
    let common_words = [
        "the",
        "a",
        "an",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "could",
        "should",
        "may",
        "might",
        "must",
        "shall",
        "can",
        "this",
        "that",
        "which",
        "who",
        "whom",
        "whose",
        "what",
        "where",
        "when",
        "how",
        "why",
        "in",
        "on",
        "at",
        "to",
        "for",
        "of",
        "and",
        "or",
        "but",
        "with",
        "without",
        "as",
        "if",
        "then",
        "than",
        "so",
        "very",
        "just",
        "not",
        "no",
        "nor",
        "from",
        "by",
        "into",
        "over",
        "after",
        "under",
        "before",
        "between",
        "about",
        "against",
        "through",
        "during",
        "here",
        "there",
        "where",
        "therefore",
        "however",
        "thus",
        "hence",
        "also",
        "yet",
        "still",
        "while",
        "whether",
        "either",
        "neither",
        "nor",
        "now",
        "then",
        "again",
        "further",
        "more",
        "most",
        "less",
        "least",
        "much",
        "many",
        "few",
        "little",
        "some",
        "any",
        "every",
        "each",
        "other",
        "another",
        "own",
        "same",
        "such",
        "only",
        "own",
        "back",
        "even",
        "new",
        "old",
        "long",
        "great",
        "little",
        "own",
        "other",
        "right",
        "big",
        "high",
        "such",
        "important",
        "few",
        "good",
        "better",
        "best",
        "well",
        "well",
        "best",
    ];
    let common_set: HashSet<&str> = common_words.into_iter().collect();

    let response_words: Vec<&str> = response
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 2)
        .filter(|w| !common_set.contains(&w.to_lowercase().as_str()))
        .collect();
    let response_set: HashSet<String> = response_words.iter().map(|w| w.to_lowercase()).collect();

    let question_words: Vec<&str> = question
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()))
        .filter(|w| w.len() > 2)
        .filter(|w| !common_set.contains(&w.to_lowercase().as_str()))
        .collect();
    let question_set: HashSet<String> = question_words.iter().map(|w| w.to_lowercase()).collect();

    let overlap = response_set.intersection(&question_set).count();
    let question_content_words = question_set.len();

    if question_content_words == 0 {
        false // no meaningful question words, can't judge
    } else {
        let overlap_ratio = overlap as f64 / question_content_words as f64;
        overlap_ratio < 0.15
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrong_answer_key() {
        // Valid wrong letter → WrongAnswerKey
        assert_eq!(
            classify_wrong_answer(
                "The square root of 4 is 5, so the answer is A.",
                "What is the square root of 4?",
                'B',
                Some('A')
            ),
            WrongAnswerClass::WrongAnswerKey
        );
        // Short wrong answer → WrongAnswerKey
        assert_eq!(
            classify_wrong_answer(
                "The answer is A.",
                "What is the square root of 4?",
                'B',
                Some('A')
            ),
            WrongAnswerClass::WrongAnswerKey
        );
    }

    #[test]
    fn test_invalid_answer_key() {
        // "L" is beyond the valid A-J range → InvalidAnswerKey
        assert_eq!(
            classify_wrong_answer(
                "The answer is L.",
                "Which of the following? A) ... B) ... C) ...",
                'A',
                Some('L')
            ),
            WrongAnswerClass::InvalidAnswerKey
        );
        // Non-letter character → InvalidAnswerKey
        assert_eq!(
            classify_wrong_answer("The answer is 3.", "Choose A, B, C, or D.", 'B', Some('3')),
            WrongAnswerClass::InvalidAnswerKey
        );
    }

    #[test]
    fn test_no_answer() {
        assert_eq!(
            classify_wrong_answer("Hmm", "What is 2+2?", 'C', None),
            WrongAnswerClass::NoAnswer
        );
    }

    #[test]
    fn test_looping() {
        let text = "The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog. The quick brown fox jumps over the lazy dog.";
        assert_eq!(
            classify_wrong_answer(text, "Question?", 'A', None),
            WrongAnswerClass::Looping
        );
    }

    #[test]
    fn test_truncated() {
        assert_eq!(
            classify_wrong_answer(
                "The answer is definitely A because the function ",
                "What function is used to sort?",
                'C',
                None
            ),
            WrongAnswerClass::Truncated
        );
        assert_eq!(
            classify_wrong_answer(
                "I believe it is A because it is a great answer and ",
                "Which option?",
                'B',
                None
            ),
            WrongAnswerClass::Truncated
        );
    }

    #[test]
    fn test_uncertainty() {
        assert_eq!(
            classify_wrong_answer(
                "I don't know what the answer is, but maybe it's A.",
                "What is the capital of France?",
                'C',
                None
            ),
            WrongAnswerClass::Uncertainty
        );
    }

    #[test]
    fn test_refusal() {
        assert_eq!(
            classify_wrong_answer(
                "I cannot help you with that. I won't provide assistance on this.",
                "How do I hack this?",
                'A',
                None
            ),
            WrongAnswerClass::Refused
        );
    }

    #[test]
    fn test_off_topic() {
        // Long ramble about weather with no valid answer extracted → OffTopic
        assert_eq!(
            classify_wrong_answer(
                "The weather today is really nice, I think it might rain later in the afternoon. The forecast says 70 degrees, which is perfect for a walk in the park.",
                "What is the derivative of x^2?",
                'A',
                None
            ),
            WrongAnswerClass::OffTopic
        );
        // Overlapping content → NoAnswer (not enough words to be off-topic)
        assert_eq!(
            classify_wrong_answer(
                "Hmm, calculus says derivative of x^2 is 2x, but I can't extract a letter.",
                "What is the derivative of x^2?",
                'A',
                None
            ),
            WrongAnswerClass::NoAnswer
        );
    }

    #[test]
    fn test_refusal_doesnt_catch_uncertainty() {
        // "i don't think i can" should go to Uncertainty, not Refused
        assert_eq!(
            classify_wrong_answer(
                "I don't think I can answer that, I'm not sure.",
                "Who won the 1998 World Cup?",
                'A',
                None
            ),
            WrongAnswerClass::Uncertainty
        );
    }
}
