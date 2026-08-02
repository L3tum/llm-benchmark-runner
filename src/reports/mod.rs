mod categories;
pub mod console;
pub mod generator;
pub mod html;
pub mod markdown;
pub mod model;
pub mod report_helpers;
pub mod translation_benchmark;

// Difficulty and TranslationState are defined in `crate::shared` and
// re-exported from both `crate::benchmarks` and `crate::shared`.
// New code should import from `crate::shared`.
