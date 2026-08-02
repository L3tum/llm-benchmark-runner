//! LLM Benchmark Runner library.
//!
//! This crate provides benchmark implementations for evaluating LLM models
//! across various tasks including MMLU-Pro, KLD, SuperGPQA, MultiPL-E,
//! and translation benchmarks.

pub mod benchmarks;
pub mod client;
pub mod config;
mod docker_runner;
mod download;
mod error_classes;
pub mod report;
pub mod reports;
pub mod runner;
pub mod shared;
mod token_tracker;
pub mod utils;
