use serde::{Deserialize, Serialize};

/// Unified enum for classifying wrong answers across all benchmarks.
///
/// QA-style classification (multiple-choice):
/// - `WrongAnswerKey`: valid letter (A–J etc.) but simply incorrect
/// - `InvalidAnswerKey`: non-letter or letter outside valid options (e.g., "L" on a 4-choice question)
/// - `NoAnswer`: empty/unrecognizable output
/// - `Uncertainty`: "I don't know", "I'm not sure"
/// - `Refused`: ethical/safety/policy refusal
/// - `Looping`: repeated text patterns
/// - `Truncated`: generation cut off mid-thought
/// - `OffTopic`: long answer on a completely unrelated topic (hallucination/rambling)
///
/// Coding/structural benchmark classes:
/// - `MalformedJson`: output was expected to be valid JSON but parsing failed
/// - `Timeout`: model or execution timed out
/// - `MissingTest`: no test or expected output available (dataset issue)
/// - `FailedTest`: model's code did not pass the available tests
#[derive(Clone, Debug, PartialEq, Eq, Ord, PartialOrd, Hash, Serialize, Deserialize)]
pub enum WrongAnswerClass {
    WrongAnswerKey,
    InvalidAnswerKey,
    NoAnswer,
    Uncertainty,
    Refused,
    Looping,
    Truncated,
    OffTopic,
    MalformedJson,
    Timeout,
    MissingTest,
    FailedTest,
}

impl WrongAnswerClass {
    /// Human-readable label for display (e.g., in reports).
    pub fn display(&self) -> &'static str {
        match self {
            Self::WrongAnswerKey => "Wrong Answer Key",
            Self::InvalidAnswerKey => "Invalid Answer Key",
            Self::NoAnswer => "No Answer",
            Self::Uncertainty => "Uncertainty",
            Self::Refused => "Refused",
            Self::Looping => "Looping",
            Self::Truncated => "Truncated",
            Self::OffTopic => "Off-Topic / Hallucination",
            Self::MalformedJson => "Malformed JSON",
            Self::Timeout => "Timeout",
            Self::MissingTest => "Missing Test",
            Self::FailedTest => "Failed Test",
        }
    }
}
