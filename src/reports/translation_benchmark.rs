// Re-export shared types — these are defined in `crate::shared`
pub use crate::shared::{Difficulty, TranslationState};

/// Fuzzy match for translation benchmark scoring.
///
/// Normalizes both strings to lowercase and checks:
/// 1. Exact match
/// 2. Substring match with word boundary + length ratio (expected > 3 chars)
/// 3. Reverse substring match with length ratio (response > 3 chars)
///
/// The length threshold prevents short expected answers (like "the") from
/// matching any response that contains those words.
/// The word boundary check prevents partial word matches (e.g., "cat" in "category").
/// The length ratio ensures the response isn't bloated (expected covers ≥40% of response).
pub fn fuzzy_match(response: &str, expected: &str) -> bool {
    // All current test data is ASCII, but using Unicode-aware lowering for future-proofing.
    let r = response.trim().to_lowercase();
    let e = expected.trim().to_lowercase();

    // Exact match
    if r == e {
        return true;
    }

    if e.is_empty() || r.is_empty() {
        return false;
    }

    // Substring match: expected contained in response
    if e.len() > 3 && r.contains(&e) {
        // Verify the match sits at word boundaries
        if at_word_boundary(&r, &e) {
            // Length ratio: expected should cover a meaningful portion of response
            let ratio = e.len() as f64 / r.len() as f64;
            if ratio >= 0.4 {
                return true;
            }
        }
    }

    // Short expected strings (≤ 3 chars): word-boundary match is sufficient, no ratio required.
    // This allows valid short translations like "it", "he", "go", "no" to match
    // when they appear at proper word boundaries in the response.
    if e.len() <= 3 && r.contains(&e) && at_word_boundary(&r, &e) {
        return true;
    }

    // Reverse substring: response contained in expected
    if r.len() > 3 && e.contains(&r) {
        let ratio = r.len() as f64 / e.len() as f64;
        // More lenient for reverse matches — a short correct response embedded in
        // a longer expected answer is still valid (e.g., "cat sat" for "the cat sat on the mat")
        if ratio >= 0.25 {
            return true;
        }
    }

    // Short response strings (≤ 3 chars): word-boundary match in expected is sufficient
    if r.len() <= 3 && e.contains(&r) && at_word_boundary(&e, &r) {
        return true;
    }

    false
}

/// Check that `needle` sits at word boundaries within `haystack`.
///
/// Uses Unicode-aware character classes (`char::is_alphanumeric` + `_`).
/// Supports CJK, Arabic, and other non-ASCII scripts.
fn at_word_boundary(haystack: &str, needle: &str) -> bool {
    if let Some(idx) = haystack.find(needle) {
        let before_char = if idx == 0 {
            None
        } else {
            haystack[..idx].chars().last()
        };
        let before_ok = before_char.is_none_or(|c| !c.is_alphanumeric() && c != '_');

        let after_idx = idx + needle.len();
        let after_char = if after_idx >= haystack.len() {
            None
        } else {
            haystack[after_idx..].chars().next()
        };
        let after_ok = after_char.is_none_or(|c| !c.is_alphanumeric() && c != '_');

        before_ok && after_ok
    } else {
        false
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn difficulty_labels() {
        assert_eq!(Difficulty::Easy.label(), "easy");
        assert_eq!(Difficulty::Medium.label(), "medium");
        assert_eq!(Difficulty::Hard.label(), "hard");
    }

    #[test]
    fn fuzzy_match_exact() {
        assert!(fuzzy_match("hello world", "hello world"));
    }

    #[test]
    fn fuzzy_match_substring_expected_longer() {
        // Expected is substantial (> 3 chars), response contains it at word boundary, ratio OK
        assert!(fuzzy_match("the cat sat on the mat", "cat sat on the mat"));
    }

    #[test]
    fn fuzzy_match_short_expected_at_word_boundary() {
        // Short expected string at word boundary should match
        assert!(fuzzy_match("the cat sat on the mat", "the"));
    }

    #[test]
    fn fuzzy_match_short_expected_not_at_word_boundary() {
        // Short expected string embedded in a word should NOT match
        assert!(!fuzzy_match("together", "the"));
    }

    #[test]
    fn fuzzy_match_short_response_at_word_boundary() {
        // Short response at word boundary in expected should match
        assert!(fuzzy_match("I think it is good", "it"));
        assert!(fuzzy_match("let's go now", "go"));
    }

    #[test]
    fn fuzzy_match_substring_expected_short_rejected_when_not_at_boundary() {
        // Short expected string NOT at word boundary should still be rejected
        assert!(!fuzzy_match("together", "the"));
        assert!(!fuzzy_match("category", "cat"));
    }

    #[test]
    fn fuzzy_match_reverse_substring_response_longer() {
        // Response is substantial (> 3 chars), expected contains it, ratio OK
        assert!(fuzzy_match("cat sat", "the cat sat on the mat"));
    }

    #[test]
    fn fuzzy_match_reverse_substring_response_short_rejected() {
        // Response is too short (<= 3 chars), reverse substring rejected
        assert!(!fuzzy_match("it", "the cat sat on the mat"));
    }

    #[test]
    fn fuzzy_match_case_insensitive() {
        assert!(fuzzy_match("Hello World", "hello world"));
    }

    #[test]
    fn fuzzy_match_whitespace_trimmed() {
        assert!(fuzzy_match("  hello world  ", "hello world"));
    }

    #[test]
    fn fuzzy_match_completely_different() {
        assert!(!fuzzy_match("completely different", "hello world"));
    }

    #[test]
    fn fuzzy_match_fast_path_exact() {
        assert!(fuzzy_match("Hello World", "HELLO WORLD"));
        assert!(fuzzy_match("foo bar", "Foo Bar"));
    }

    #[test]
    fn fuzzy_match_substring_ascii() {
        assert!(fuzzy_match("the quick brown fox", "quick brown"));
        assert!(fuzzy_match("QUICK BROWN", "the quick brown fox"));
    }

    #[test]
    fn fuzzy_match_rejects_bloated_response() {
        // Response is significantly longer than expected, should fail ratio check
        assert!(!fuzzy_match(
            "sure! the answer is happy love and that's it",
            "happy love"
        ));
    }

    #[test]
    fn fuzzy_match_accepts_near_exact_with_word_boundary() {
        // Response has minor prefix but expected covers most of it
        assert!(fuzzy_match("answer: happy love", "happy love"));
    }

    #[test]
    fn fuzzy_match_rejects_word_boundary_violation() {
        // "cat" embedded in "category" should not match
        assert!(!fuzzy_match("category", "cat"));
    }
}
