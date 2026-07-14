use super::categories::{get_category_order, slugify_name};
use super::generator::{HtmlReport, ReportContext, ReportGenerator};
use super::model::*;
use anyhow::Result;
use askama::Template;
use serde::Serialize;
use std::collections::HashMap;

/// Custom filter for slugifying names
fn slugify(name: &str) -> String {
    slugify_name(name.to_string())
}

pub struct HtmlReportGenerator;

impl ReportGenerator for HtmlReportGenerator {
    type Output = String;

    fn generate(&self, ctx: &ReportContext<'_>) -> Result<Self::Output> {
        let input = ctx.input;

        // Build category data for template
        let categories = get_category_order();
        let category_data: Vec<CategoryData> = categories
            .iter()
            .map(|cat| build_category_data_for_cat(cat, input))
            .collect();

        let generated_at = &input.generated_at;
        let models = &input.models.join(", ");
        let models_list = &input.models;
        let minebench_data = input.tests.get(&TestName::new("minebench"));
        let minebench_map: HashMap<String, MinebenchResult> = minebench_data
            .map(|td| {
                td.model_results
                    .iter()
                    .map(|(m, br)| (m.clone(), to_minebench_result(m.clone(), br)))
                    .collect()
            })
            .unwrap_or_default();
        // Build 2D token usage matrix (model x model)
        let token_usage_results: HashMap<String, HashMap<String, MinebenchResult>> = models_list
            .iter()
            .map(|model| {
                let inner = minebench_map
                    .iter()
                    .map(|(m, d)| (m.clone(), d.clone()))
                    .collect();
                (model.clone(), inner)
            })
            .collect();

        let template = ReportTemplate {
            generated_at,
            models,
            models_list,
            token_usage_results: &token_usage_results,
            summary: &input.summary,
            category_data,
        };
        Ok(template.render()?)
    }
}

#[allow(unused_assignments)]
fn build_category_data_for_cat(cat: &BenchmarkCategory, input: &ReportInput) -> CategoryData {
    let mut model_count = 0;
    let mut has_results = false;

    // Extract per-benchmark test data
    let mmlu_pro_data = input.tests.get(&TestName::new("mmlu_pro"));
    let gpqa_data = input.tests.get(&TestName::new("gpqa"));
    let aime_data = input.tests.get(&TestName::new("aime"));
    let math500_data = input.tests.get(&TestName::new("math500"));
    let minebench_data = input.tests.get(&TestName::new("minebench"));
    let supergpqa_data = input.tests.get(&TestName::new("supergpqa"));
    let coding_eval_data = input.tests.get(&TestName::new("coding_eval"));
    let swe_bench_data = input.tests.get(&TestName::new("swe_bench"));
    let kld_data = input.tests.get(&TestName::new("kld"));

    // Helper: convert generic model results into MMLU-Pro results
    let mmlu_results: Option<Vec<(String, MmluProResult)>> = mmlu_pro_data.map(|td| {
        let mut all_results: Vec<(String, MmluProResult)> = td
            .model_results
            .iter()
            .map(|(m, br)| (m.clone(), to_mmlu_result(br)))
            .collect();
        if let Some((_, best)) = all_results.iter().max_by(|a, b| {
            a.1.accuracy
                .partial_cmp(&b.1.accuracy)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let best_accuracy = best.accuracy;
            for (_, r) in &mut all_results {
                r.best = r.accuracy == best_accuracy;
            }
        }
        all_results
    });
    // GPQA — build with best=false, then mark the best
    let gpqa_results: Option<Vec<(String, GpqaResult)>> = gpqa_data.map(|td| {
        let mut all_results: Vec<(String, GpqaResult)> = td
            .model_results
            .iter()
            .map(|(m, br)| {
                let mut r = to_gpqa_result(br);
                r.best = false;
                (m.clone(), r)
            })
            .collect();
        if let Some((_, best)) = all_results.iter().max_by(|a, b| {
            a.1.accuracy
                .partial_cmp(&b.1.accuracy)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let best_accuracy = best.accuracy;
            for (_, r) in &mut all_results {
                r.best = r.accuracy == best_accuracy;
            }
        }
        all_results
    });
    // SuperGPQA — same logic
    let supergpqa_results: Option<Vec<(String, SuperGpqaResult)>> = supergpqa_data.map(|td| {
        let mut all_results: Vec<(String, SuperGpqaResult)> = td
            .model_results
            .iter()
            .map(|(m, br)| {
                let mut r = to_supergpqa_result(br);
                r.best = false;
                (m.clone(), r)
            })
            .collect();
        if let Some((_, best)) = all_results.iter().max_by(|a, b| {
            a.1.accuracy
                .partial_cmp(&b.1.accuracy)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let best_accuracy = best.accuracy;
            for (_, r) in &mut all_results {
                r.best = r.accuracy == best_accuracy;
            }
        }
        all_results
    });
    // AIME — mark the best
    let aime_results: Option<Vec<(String, AimeResult)>> = aime_data.map(|td| {
        let mut all_results: Vec<(String, AimeResult)> = td
            .model_results
            .iter()
            .map(|(m, br)| (m.clone(), to_aime_result(br)))
            .collect();
        if let Some((_, best)) = all_results.iter().max_by(|a, b| {
            a.1.accuracy
                .partial_cmp(&b.1.accuracy)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let best_accuracy = best.accuracy;
            for (_, r) in &mut all_results {
                r.best = r.accuracy == best_accuracy;
            }
        }
        all_results
    });
    // MATH-500 — mark the best
    let math500_results: Option<Vec<(String, Math500Result)>> = math500_data.map(|td| {
        let mut all_results: Vec<(String, Math500Result)> = td
            .model_results
            .iter()
            .map(|(m, br)| (m.clone(), to_math500_result(br)))
            .collect();
        if let Some((_, best)) = all_results.iter().max_by(|a, b| {
            a.1.accuracy
                .partial_cmp(&b.1.accuracy)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let best_accuracy = best.accuracy;
            for (_, r) in &mut all_results {
                r.best = r.accuracy == best_accuracy;
            }
        }
        all_results
    });
    // Minebench
    let minebench_results: Option<Vec<(String, MinebenchResult)>> = minebench_data.map(|td| {
        td.model_results
            .iter()
            .map(|(m, br)| (m.clone(), to_minebench_result(m.clone(), br)))
            .collect()
    });
    // Coding Eval
    let coding_eval_results: Option<Vec<(String, CodingEvalResult)>> = coding_eval_data.map(|td| {
        let mut all_results: Vec<(String, CodingEvalResult)> = td
            .model_results
            .iter()
            .map(|(m, br)| (m.clone(), to_coding_eval_result(m.clone(), br)))
            .collect();
        if let Some((_, best)) = all_results.iter().max_by(|a, b| {
            a.1.pass_score
                .partial_cmp(&b.1.pass_score)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let best_score = best.pass_score;
            for (_, r) in &mut all_results {
                r.best = r.pass_score == best_score;
            }
        }
        all_results
    });
    // SWE-Bench
    let swe_bench_results: Option<Vec<(String, SweBenchResult)>> = swe_bench_data.map(|td| {
        let mut all_results: Vec<(String, SweBenchResult)> = td
            .model_results
            .iter()
            .map(|(m, br)| (m.clone(), to_swe_bench_result(m.clone(), br)))
            .collect();
        if let Some((_, best)) = all_results.iter().max_by(|a, b| {
            a.1.resolution_rate
                .partial_cmp(&b.1.resolution_rate)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let best_rate = best.resolution_rate;
            for (_, r) in &mut all_results {
                r.best = r.resolution_rate == best_rate;
            }
        }
        all_results
    });
    // KLD: pairwise + avg
    let kld_pairwise: HashMap<String, KldPairResult> = if let Some(td) = kld_data {
        td.aggregate
            .as_ref()
            .and_then(|agg| agg.breakdowns.get("pairwise_kld"))
            .map(|table| {
                let mut pairs = HashMap::new();
                for (row, metrics) in &table.rows {
                    if let Some(kld_score) = metrics.get("avg_kld") {
                        pairs.insert(
                            row.clone(),
                            KldPairResult {
                                models: vec!["Model A".into(), "Model B".into()],
                                avg_kld: match &kld_score.value {
                                    ScoreValue::Float(f) => *f,
                                    _ => 0.0,
                                },
                                avg_kld_str: kld_score.display_value(),
                                num_prompts_evaluated: 0,
                            },
                        );
                    }
                }
                pairs
            })
            .unwrap_or_default()
    } else {
        HashMap::new()
    };
    let kld_avg_results: Vec<KldAvgResult> = if let Some(td) = kld_data {
        let mut all_results: Vec<KldAvgResult> = td
            .model_results
            .iter()
            .filter_map(|(m, br)| {
                br.scores.get("avg_kld_to_others").map(|score| {
                    let output_tokens = br
                        .scores
                        .get("output_tokens")
                        .map(|s| s.display_value())
                        .unwrap_or_else(|| "–".into());
                    let thinking_tokens = br
                        .scores
                        .get("thinking_tokens")
                        .map(|s| s.display_value())
                        .unwrap_or_else(|| "–".into());
                    let avg_kld = match &score.value {
                        ScoreValue::Float(f) => *f,
                        _ => 0.0,
                    };
                    KldAvgResult {
                        model: m.clone(),
                        klds: vec![],
                        avg_kld_to_others: avg_kld,
                        avg_kld_to_others_str: score.display_value(),
                        output_tokens,
                        thinking_tokens,
                        best: false, // will be marked below
                    }
                })
            })
            .collect();
        // For KLD, lower is better — mark the minimum
        if let Some(best) = all_results.iter().min_by(|a, b| {
            a.avg_kld_to_others
                .partial_cmp(&b.avg_kld_to_others)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            let best_kld = best.avg_kld_to_others;
            for r in &mut all_results {
                r.best = r.avg_kld_to_others == best_kld;
            }
        }
        all_results
    } else {
        vec![]
    };

    match cat {
        BenchmarkCategory::Knowledge => {
            has_results = mmlu_results.is_some() || gpqa_results.is_some();
            model_count = mmlu_results
                .as_ref()
                .map(|v: &Vec<(String, MmluProResult)>| v.len())
                .unwrap_or(0)
                .max(
                    gpqa_results
                        .as_ref()
                        .map(|v: &Vec<(String, GpqaResult)>| v.len())
                        .unwrap_or(0),
                );
            CategoryData {
                name: cat.display().to_string(),
                name_slug: slugify_name(cat.display()),
                has_results,
                mmlu_pro_results: mmlu_results.unwrap_or_default(),
                gpqa_results: gpqa_results.unwrap_or_default(),
                supergpqa_results: supergpqa_results.unwrap_or_default(),
                aime_results: vec![],
                math500_results: vec![],
                minebench_results: vec![],
                coding_eval_results: vec![],
                swe_bench_results: vec![],
                kld_results: HashMap::new(),
                kld_avg_results: vec![],
                model_count,
            }
        }
        BenchmarkCategory::Math => {
            has_results = aime_results.is_some() || math500_results.is_some();
            model_count = aime_results
                .as_ref()
                .map(|v: &Vec<(String, AimeResult)>| v.len())
                .unwrap_or(0)
                .max(
                    math500_results
                        .as_ref()
                        .map(|v: &Vec<(String, Math500Result)>| v.len())
                        .unwrap_or(0),
                );
            CategoryData {
                name: cat.display().to_string(),
                name_slug: slugify_name(cat.display()),
                has_results,
                mmlu_pro_results: vec![],
                gpqa_results: vec![],
                supergpqa_results: vec![],
                aime_results: aime_results.unwrap_or_default(),
                math500_results: math500_results.unwrap_or_default(),
                minebench_results: vec![],
                coding_eval_results: vec![],
                swe_bench_results: vec![],
                kld_results: HashMap::new(),
                kld_avg_results: vec![],
                model_count,
            }
        }
        BenchmarkCategory::ShortContextCoding => {
            has_results = coding_eval_results.is_some();
            model_count = coding_eval_results.as_ref().map(|v| v.len()).unwrap_or(0);
            CategoryData {
                name: cat.display().to_string(),
                name_slug: slugify_name(cat.display()),
                has_results,
                mmlu_pro_results: vec![],
                gpqa_results: vec![],
                supergpqa_results: vec![],
                aime_results: vec![],
                math500_results: vec![],
                minebench_results: vec![],
                coding_eval_results: coding_eval_results.unwrap_or_default(),
                swe_bench_results: vec![],
                kld_results: HashMap::new(),
                kld_avg_results: vec![],
                model_count,
            }
        }
        BenchmarkCategory::LongContextCoding => {
            has_results = swe_bench_results.is_some();
            model_count = swe_bench_results.as_ref().map(|v| v.len()).unwrap_or(0);
            CategoryData {
                name: cat.display().to_string(),
                name_slug: slugify_name(cat.display()),
                has_results,
                mmlu_pro_results: vec![],
                gpqa_results: vec![],
                supergpqa_results: vec![],
                aime_results: vec![],
                math500_results: vec![],
                minebench_results: vec![],
                coding_eval_results: vec![],
                swe_bench_results: swe_bench_results
                    .map(|v| v.into_iter().map(|(_, r)| r).collect())
                    .unwrap_or_default(),
                kld_results: HashMap::new(),
                kld_avg_results: vec![],
                model_count,
            }
        }
        BenchmarkCategory::Creative => {
            has_results = minebench_results.is_some();
            model_count = minebench_results.as_ref().map(|v| v.len()).unwrap_or(0);
            CategoryData {
                name: cat.display().to_string(),
                name_slug: slugify_name(cat.display()),
                has_results,
                mmlu_pro_results: vec![],
                gpqa_results: vec![],
                supergpqa_results: vec![],
                aime_results: vec![],
                math500_results: vec![],
                minebench_results: minebench_results.unwrap_or_default(),
                coding_eval_results: vec![],
                swe_bench_results: vec![],
                kld_results: HashMap::new(),
                kld_avg_results: vec![],
                model_count,
            }
        }
        BenchmarkCategory::Similarity => {
            has_results = !kld_pairwise.is_empty() || !kld_avg_results.is_empty();
            model_count = kld_avg_results.len();
            CategoryData {
                name: cat.display().to_string(),
                name_slug: slugify_name(cat.display()),
                has_results,
                mmlu_pro_results: vec![],
                gpqa_results: vec![],
                supergpqa_results: vec![],
                aime_results: vec![],
                math500_results: vec![],
                minebench_results: vec![],
                coding_eval_results: vec![],
                swe_bench_results: vec![],
                kld_results: kld_pairwise,
                kld_avg_results,
                model_count,
            }
        }
        BenchmarkCategory::Reasoning
        | BenchmarkCategory::Research
        | BenchmarkCategory::InstructionFollowing
        | BenchmarkCategory::Safety => CategoryData {
            name: cat.display().to_string(),
            name_slug: slugify_name(cat.display()),
            has_results: false,
            mmlu_pro_results: vec![],
            gpqa_results: vec![],
            supergpqa_results: vec![],
            aime_results: vec![],
            math500_results: vec![],
            minebench_results: vec![],
            coding_eval_results: vec![],
            swe_bench_results: vec![],
            kld_results: HashMap::new(),
            kld_avg_results: vec![],
            model_count: 0,
        },
        BenchmarkCategory::Other(_) => CategoryData {
            name: cat.display().to_string(),
            name_slug: slugify_name(cat.display()),
            has_results: false,
            mmlu_pro_results: vec![],
            gpqa_results: vec![],
            supergpqa_results: vec![],
            aime_results: vec![],
            math500_results: vec![],
            minebench_results: vec![],
            coding_eval_results: vec![],
            swe_bench_results: vec![],
            kld_results: HashMap::new(),
            kld_avg_results: vec![],
            model_count: 0,
        },
    }
}

// Conversion helpers

fn to_mmlu_result(br: &BenchmarkResult) -> MmluProResult {
    let accuracy = br
        .scores
        .get("accuracy")
        .map(|s| match &s.value {
            ScoreValue::Float(f) => *f,
            _ => 0.0,
        })
        .unwrap_or(0.0);
    let total_questions = br
        .scores
        .get("total_questions")
        .map(|s| match &s.value {
            ScoreValue::Integer(i) => *i,
            _ => 0,
        })
        .unwrap_or(0);
    let output_tokens = br
        .scores
        .get("output_tokens")
        .map(|s| s.display_value())
        .unwrap_or_else(|| "–".into());
    let thinking_tokens = br
        .scores
        .get("thinking_tokens")
        .map(|s| s.display_value())
        .unwrap_or_else(|| "–".into());
    let results_by_subject = br
        .breakdowns
        .get("subjects")
        .map(|table| {
            table
                .rows
                .iter()
                .map(|(name, metrics)| {
                    let acc = metrics
                        .get("accuracy")
                        .map(|s| match &s.value {
                            ScoreValue::Float(f) => *f,
                            _ => 0.0,
                        })
                        .unwrap_or(0.0);
                    let correct = metrics
                        .get("correct")
                        .map(|s| match &s.value {
                            ScoreValue::Integer(i) => *i,
                            _ => 0,
                        })
                        .unwrap_or(0);
                    let wrong = metrics
                        .get("wrong")
                        .map(|s| match &s.value {
                            ScoreValue::Integer(i) => *i,
                            _ => 0,
                        })
                        .unwrap_or(0);
                    (
                        name.clone(),
                        SubjectResult {
                            acc,
                            acc_pct: format!("{:.2}", acc),
                            correct,
                            wrong,
                        },
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    MmluProResult {
        accuracy,
        accuracy_pct: format!("{:.2}", accuracy),
        total_questions,
        output_tokens,
        thinking_tokens,
        results_by_subject,
        best: false,
    }
}

fn to_gpqa_result(br: &BenchmarkResult) -> GpqaResult {
    let accuracy = br
        .scores
        .get("accuracy")
        .map(|s| match &s.value {
            ScoreValue::Float(f) => *f,
            _ => 0.0,
        })
        .unwrap_or(0.0);
    let total_questions = br
        .scores
        .get("total_questions")
        .map(|s| match &s.value {
            ScoreValue::Integer(i) => *i,
            _ => 0,
        })
        .unwrap_or(0);
    let output_tokens = br
        .scores
        .get("output_tokens")
        .map(|s| s.display_value())
        .unwrap_or_else(|| "–".into());
    let thinking_tokens = br
        .scores
        .get("thinking_tokens")
        .map(|s| s.display_value())
        .unwrap_or_else(|| "–".into());
    let results_by_subject = br
        .breakdowns
        .get("subjects")
        .map(|table| {
            table
                .rows
                .iter()
                .map(|(name, metrics)| {
                    let acc = metrics
                        .get("accuracy")
                        .map(|s| match &s.value {
                            ScoreValue::Float(f) => *f,
                            _ => 0.0,
                        })
                        .unwrap_or(0.0);
                    let correct = metrics
                        .get("correct")
                        .map(|s| match &s.value {
                            ScoreValue::Integer(i) => *i,
                            _ => 0,
                        })
                        .unwrap_or(0);
                    let wrong = metrics
                        .get("wrong")
                        .map(|s| match &s.value {
                            ScoreValue::Integer(i) => *i,
                            _ => 0,
                        })
                        .unwrap_or(0);
                    (
                        name.clone(),
                        SubjectResult {
                            acc,
                            acc_pct: format!("{:.2}", acc),
                            correct,
                            wrong,
                        },
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    GpqaResult {
        accuracy,
        accuracy_pct: format!("{:.2}", accuracy),
        total_questions,
        output_tokens,
        thinking_tokens,
        results_by_subject,
        best: false,
    }
}

fn to_supergpqa_result(br: &BenchmarkResult) -> SuperGpqaResult {
    let accuracy = br
        .scores
        .get("accuracy")
        .map(|s| match &s.value {
            ScoreValue::Float(f) => *f,
            _ => 0.0,
        })
        .unwrap_or(0.0);
    let total_questions = br
        .scores
        .get("total_questions")
        .map(|s| match &s.value {
            ScoreValue::Integer(i) => *i,
            _ => 0,
        })
        .unwrap_or(0);
    let output_tokens = br
        .scores
        .get("output_tokens")
        .map(|s| s.display_value())
        .unwrap_or_else(|| "–".into());
    let thinking_tokens = br
        .scores
        .get("thinking_tokens")
        .map(|s| s.display_value())
        .unwrap_or_else(|| "–".into());

    // Extract all four breakdown tables
    let build_table = |table_opt: Option<&BreakdownTable>| {
        table_opt
            .map(|table| {
                table
                    .rows
                    .iter()
                    .map(|(name, metrics)| {
                        let acc = metrics
                            .get("accuracy")
                            .map(|s| match &s.value {
                                ScoreValue::Float(f) => *f,
                                _ => 0.0,
                            })
                            .unwrap_or(0.0);
                        let correct = metrics
                            .get("correct")
                            .map(|s| match &s.value {
                                ScoreValue::Integer(i) => *i,
                                _ => 0,
                            })
                            .unwrap_or(0);
                        let wrong = metrics
                            .get("wrong")
                            .map(|s| match &s.value {
                                ScoreValue::Integer(i) => *i,
                                _ => 0,
                            })
                            .unwrap_or(0);
                        (
                            name.clone(),
                            SubjectResult {
                                acc,
                                acc_pct: format!("{:.2}", acc),
                                correct,
                                wrong,
                            },
                        )
                    })
                    .collect()
            })
            .unwrap_or_default()
    };

    let results_by_discipline = build_table(br.breakdowns.get("discipline"));
    let results_by_field = build_table(br.breakdowns.get("field"));
    let results_by_subfield = build_table(br.breakdowns.get("subfield"));
    let results_by_difficulty = build_table(br.breakdowns.get("difficulty"));

    SuperGpqaResult {
        accuracy,
        accuracy_pct: format!("{:.2}", accuracy),
        total_questions,
        output_tokens,
        thinking_tokens,
        results_by_discipline,
        results_by_field,
        results_by_subfield,
        results_by_difficulty,
        best: false,
    }
}

fn to_aime_result(br: &BenchmarkResult) -> AimeResult {
    let accuracy = br
        .scores
        .get("accuracy")
        .map(|s| match &s.value {
            ScoreValue::Float(f) => *f,
            _ => 0.0,
        })
        .unwrap_or(0.0);
    let total_questions = br
        .scores
        .get("total_questions")
        .map(|s| match &s.value {
            ScoreValue::Integer(i) => *i,
            _ => 0,
        })
        .unwrap_or(0);
    let correct = br
        .scores
        .get("correct")
        .map(|s| match &s.value {
            ScoreValue::Integer(i) => *i,
            _ => 0,
        })
        .unwrap_or(0);
    let output_tokens = br
        .scores
        .get("output_tokens")
        .map(|s| s.display_value())
        .unwrap_or_else(|| "–".into());
    let thinking_tokens = br
        .scores
        .get("thinking_tokens")
        .map(|s| s.display_value())
        .unwrap_or_else(|| "–".into());
    AimeResult {
        accuracy,
        accuracy_pct: format!("{:.2}", accuracy),
        total_questions,
        output_tokens,
        thinking_tokens,
        correct,
        best: false,
    }
}

fn to_math500_result(br: &BenchmarkResult) -> Math500Result {
    let accuracy = br
        .scores
        .get("accuracy")
        .map(|s| match &s.value {
            ScoreValue::Float(f) => *f,
            _ => 0.0,
        })
        .unwrap_or(0.0);
    let total_questions = br
        .scores
        .get("total_questions")
        .map(|s| match &s.value {
            ScoreValue::Integer(i) => *i,
            _ => 0,
        })
        .unwrap_or(0);
    let output_tokens = br
        .scores
        .get("output_tokens")
        .map(|s| s.display_value())
        .unwrap_or_else(|| "–".into());
    let thinking_tokens = br
        .scores
        .get("thinking_tokens")
        .map(|s| s.display_value())
        .unwrap_or_else(|| "–".into());
    let results_by_subject = br
        .breakdowns
        .get("subjects")
        .map(|table| {
            table
                .rows
                .iter()
                .map(|(name, metrics)| {
                    let acc = metrics
                        .get("accuracy")
                        .map(|s| match &s.value {
                            ScoreValue::Float(f) => *f,
                            _ => 0.0,
                        })
                        .unwrap_or(0.0);
                    let correct = metrics
                        .get("correct")
                        .map(|s| match &s.value {
                            ScoreValue::Integer(i) => *i,
                            _ => 0,
                        })
                        .unwrap_or(0);
                    let wrong = metrics
                        .get("wrong")
                        .map(|s| match &s.value {
                            ScoreValue::Integer(i) => *i,
                            _ => 0,
                        })
                        .unwrap_or(0);
                    (
                        name.clone(),
                        SubjectResult {
                            acc,
                            acc_pct: format!("{:.2}", acc),
                            correct,
                            wrong,
                        },
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    Math500Result {
        accuracy,
        accuracy_pct: format!("{:.2}", accuracy),
        total_questions,
        output_tokens,
        thinking_tokens,
        results_by_subject,
        best: false,
    }
}

fn to_minebench_result(model: String, br: &BenchmarkResult) -> MinebenchResult {
    let json_valid = br
        .scores
        .get("valid_json")
        .map(|s| match &s.value {
            ScoreValue::Bool(b) => *b,
            _ => false,
        })
        .unwrap_or(false);
    let valid_buildings = br
        .scores
        .get("valid_buildings")
        .map(|s| match &s.value {
            ScoreValue::Integer(i) => *i,
            _ => 0,
        })
        .unwrap_or(0);
    let total_buildings = br
        .scores
        .get("total_buildings")
        .map(|s| match &s.value {
            ScoreValue::Integer(i) => *i,
            _ => 0,
        })
        .unwrap_or(0);
    let output_file = br
        .artifacts
        .iter()
        .find(|a| a.label == "Output")
        .map(|a| a.path.clone())
        .unwrap_or_default();
    let output_tokens = br
        .scores
        .get("output_tokens")
        .map(|s| s.display_value())
        .unwrap_or_else(|| "–".into());
    let thinking_tokens = br
        .scores
        .get("thinking_tokens")
        .map(|s| s.display_value())
        .unwrap_or_else(|| "–".into());
    MinebenchResult {
        model,
        json_valid,
        json_valid_str: if json_valid {
            "✓".into()
        } else {
            "✗".into()
        },
        valid_buildings,
        total_buildings,
        output_file,
        output_tokens,
        thinking_tokens,
    }
}

fn to_coding_eval_result(model: String, br: &BenchmarkResult) -> CodingEvalResult {
    let pass_score = br
        .scores
        .get("pass_at_1")
        .map(|s| match &s.value {
            ScoreValue::Float(f) => *f,
            _ => 0.0,
        })
        .unwrap_or(0.0);
    let passed = br
        .scores
        .get("passed")
        .map(|s| match &s.value {
            ScoreValue::Integer(i) => *i,
            _ => 0,
        })
        .unwrap_or(0);
    let total_questions = br
        .scores
        .get("total_questions")
        .map(|s| match &s.value {
            ScoreValue::Integer(i) => *i,
            _ => 0,
        })
        .unwrap_or(0);
    let timeout_count = br
        .scores
        .get("timeout_count")
        .map(|s| match &s.value {
            ScoreValue::Integer(i) => *i,
            _ => 0,
        })
        .unwrap_or(0);
    let output_tokens = br
        .scores
        .get("output_tokens")
        .map(|s| s.display_value())
        .unwrap_or_else(|| "–".into());
    let thinking_tokens = br
        .scores
        .get("thinking_tokens")
        .map(|s| s.display_value())
        .unwrap_or_else(|| "–".into());
    let results_by_taskset = br
        .breakdowns
        .get("tasksets")
        .map(|table| {
            table
                .rows
                .iter()
                .map(|(taskset, metrics)| {
                    let pass_at_1 = metrics
                        .get("pass@1")
                        .map(|s| s.display_value())
                        .unwrap_or_else(|| "–".into());
                    let pass_at_2 = metrics
                        .get("pass@2")
                        .map(|s| s.display_value())
                        .unwrap_or_else(|| "–".into());
                    let pass_at_3 = metrics
                        .get("pass@3")
                        .map(|s| s.display_value())
                        .unwrap_or_else(|| "–".into());
                    let passed = metrics
                        .get("passed")
                        .map(|s| match &s.value {
                            ScoreValue::Integer(i) => *i,
                            _ => 0,
                        })
                        .unwrap_or(0);
                    let total = metrics
                        .get("total")
                        .map(|s| match &s.value {
                            ScoreValue::Integer(i) => *i,
                            _ => 0,
                        })
                        .unwrap_or(0);
                    (
                        taskset.clone(),
                        CodingEvalTasksetResult {
                            pass_at_1_pct: pass_at_1,
                            pass_at_2_pct: pass_at_2,
                            pass_at_3_pct: pass_at_3,
                            passed,
                            total,
                            timeout_count: 0,
                            skipped_later_attempts: 0,
                        },
                    )
                })
                .collect()
        })
        .unwrap_or_default();
    let failures: Vec<FailureResult> = br
        .diagnostics
        .iter()
        .map(|d| {
            let mut parts = d.message.split(": ");
            let (taskset, rest) = (
                parts.next().unwrap_or("").to_string(),
                parts.next().unwrap_or("").to_string(),
            );
            let (task_id, error) = if let Some(idx) = rest.find(" (") {
                (rest[..idx].to_string(), rest[idx..].to_string())
            } else {
                (rest.clone(), rest)
            };
            FailureResult {
                taskset,
                task_id: task_id.trim().to_string(),
                entry_point: String::new(),
                error_summary: error.trim().to_string(),
            }
        })
        .collect();
    CodingEvalResult {
        model,
        pass_score,
        pass_at_1_pct: format!("{:.2}", pass_score * 100.0),
        pass_at_2_pct: String::new(),
        pass_at_3_pct: String::new(),
        passed,
        total_questions,
        timeout_count,
        skipped_later_attempts: 0,
        output_tokens,
        thinking_tokens,
        results_by_taskset,
        failures,
        best: false,
    }
}

fn to_swe_bench_result(model: String, br: &BenchmarkResult) -> SweBenchResult {
    let resolution_rate = br
        .scores
        .get("resolution_rate")
        .map(|s| match &s.value {
            ScoreValue::Float(f) => *f,
            _ => 0.0,
        })
        .unwrap_or(0.0);
    let resolved = br
        .scores
        .get("resolved")
        .map(|s| match &s.value {
            ScoreValue::Integer(i) => *i,
            _ => 0,
        })
        .unwrap_or(0);
    let total_questions = br
        .scores
        .get("total_questions")
        .map(|s| match &s.value {
            ScoreValue::Integer(i) => *i,
            _ => 0,
        })
        .unwrap_or(0);
    let output_tokens = br
        .scores
        .get("output_tokens")
        .map(|s| s.display_value())
        .unwrap_or_else(|| "–".into());
    let thinking_tokens = br
        .scores
        .get("thinking_tokens")
        .map(|s| s.display_value())
        .unwrap_or_else(|| "–".into());
    let error_summary = br
        .diagnostics
        .iter()
        .map(|d| d.message.as_str())
        .collect::<Vec<_>>()
        .join(" | ");
    SweBenchResult {
        model,
        dataset: String::new(), // not available from generic model
        resolved,
        total_questions,
        resolution_rate,
        resolution_rate_pct: format!("{:.2}", resolution_rate),
        output_tokens,
        thinking_tokens,
        harness_passed: true,
        error_summary,
        best: false,
    }
}

// Struct definitions below

/// Per-subject result breakdown
#[derive(Serialize, Clone)]
struct SubjectResult {
    acc: f64,
    acc_pct: String,
    correct: i64,
    wrong: i64,
}

/// Strongly-typed MMLU-Pro result for template
#[derive(Serialize, Clone)]
struct MmluProResult {
    accuracy: f64,
    accuracy_pct: String,
    total_questions: i64,
    output_tokens: String,
    thinking_tokens: String,
    results_by_subject: HashMap<String, SubjectResult>,
    best: bool,
}

/// Strongly-typed GPQA result
#[derive(Serialize, Clone)]
struct GpqaResult {
    accuracy: f64,
    accuracy_pct: String,
    total_questions: i64,
    output_tokens: String,
    thinking_tokens: String,
    results_by_subject: HashMap<String, SubjectResult>,
    best: bool,
}

/// Strongly-typed SuperGPQA result with multiple breakdown tables
#[derive(Serialize, Clone)]
struct SuperGpqaResult {
    accuracy: f64,
    accuracy_pct: String,
    total_questions: i64,
    output_tokens: String,
    thinking_tokens: String,
    results_by_discipline: HashMap<String, SubjectResult>,
    results_by_field: HashMap<String, SubjectResult>,
    results_by_subfield: HashMap<String, SubjectResult>,
    results_by_difficulty: HashMap<String, SubjectResult>,
    best: bool,
}

/// Strongly-typed AIME result
#[derive(Serialize, Clone)]
struct AimeResult {
    accuracy: f64,
    accuracy_pct: String,
    total_questions: i64,
    output_tokens: String,
    thinking_tokens: String,
    correct: i64,
    best: bool,
}

/// Strongly-typed MATH-500 result
#[derive(Serialize, Clone)]
struct Math500Result {
    accuracy: f64,
    accuracy_pct: String,
    total_questions: i64,
    output_tokens: String,
    thinking_tokens: String,
    results_by_subject: HashMap<String, SubjectResult>,
    best: bool,
}

/// KLD pairwise result
#[derive(Serialize, Clone)]
struct KldPairResult {
    models: Vec<String>,
    avg_kld: f64,
    avg_kld_str: String,
    num_prompts_evaluated: i64,
}

/// KLD average-to-others result
#[derive(Serialize, Clone)]
struct KldAvgResult {
    model: String,
    klds: Vec<f64>,
    avg_kld_to_others: f64,
    avg_kld_to_others_str: String,
    output_tokens: String,
    thinking_tokens: String,
    best: bool,
}

/// Strongly-typed Minebench result
#[derive(Serialize, Clone)]
struct MinebenchResult {
    model: String,
    json_valid: bool,
    json_valid_str: String,
    valid_buildings: i64,
    total_buildings: i64,
    output_file: String,
    output_tokens: String,
    thinking_tokens: String,
}

/// Coding eval taskset-level result
#[derive(Serialize, Clone)]
struct CodingEvalTasksetResult {
    pass_at_1_pct: String,
    pass_at_2_pct: String,
    pass_at_3_pct: String,
    passed: i64,
    total: i64,
    timeout_count: i64,
    skipped_later_attempts: i64,
}

/// Strongly-typed Coding Eval result
#[derive(Serialize, Clone)]
struct CodingEvalResult {
    model: String,
    pass_score: f64,
    pass_at_1_pct: String,
    pass_at_2_pct: String,
    pass_at_3_pct: String,
    passed: i64,
    total_questions: i64,
    timeout_count: i64,
    skipped_later_attempts: i64,
    output_tokens: String,
    thinking_tokens: String,
    results_by_taskset: HashMap<String, CodingEvalTasksetResult>,
    failures: Vec<FailureResult>,
    best: bool,
}

/// Failure summary
#[derive(Serialize, Clone)]
struct FailureResult {
    taskset: String,
    task_id: String,
    entry_point: String,
    error_summary: String,
}

/// SWE-Bench result
#[derive(Serialize, Clone)]
struct SweBenchResult {
    model: String,
    dataset: String,
    resolved: i64,
    total_questions: i64,
    resolution_rate: f64,
    resolution_rate_pct: String,
    output_tokens: String,
    thinking_tokens: String,
    harness_passed: bool,
    error_summary: String,
    best: bool,
}

/// Per-category data for the template.
#[derive(Serialize)]
struct CategoryData {
    name: String,
    name_slug: String,
    has_results: bool,
    model_count: usize,
    mmlu_pro_results: Vec<(String, MmluProResult)>,
    gpqa_results: Vec<(String, GpqaResult)>,
    supergpqa_results: Vec<(String, SuperGpqaResult)>,
    aime_results: Vec<(String, AimeResult)>,
    math500_results: Vec<(String, Math500Result)>,
    minebench_results: Vec<(String, MinebenchResult)>,
    coding_eval_results: Vec<(String, CodingEvalResult)>,
    swe_bench_results: Vec<SweBenchResult>,
    kld_results: HashMap<String, KldPairResult>,
    kld_avg_results: Vec<KldAvgResult>,
}

/// Token usage result per benchmark per model.
#[derive(Serialize)]
struct TokenUsageResult {
    output_tokens: String,
    thinking_tokens: String,
}

/// Report template with category data.
#[derive(Template)]
#[template(path = "report.html")]
struct ReportTemplate<'a> {
    generated_at: &'a str,
    models: &'a str,
    models_list: &'a [String],
    token_usage_results: &'a HashMap<String, HashMap<String, MinebenchResult>>,
    summary: &'a [String],
    category_data: Vec<CategoryData>,
}

impl HtmlReport for HtmlReportGenerator {}
