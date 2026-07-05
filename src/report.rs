use anyhow::Result;
use askama::Template;
use serde::Serialize;
use std::collections::HashMap;
use std::fs;
use std::path::Path;

/// Strongly-typed MMLU-Pro result for template
#[derive(Serialize)]
struct MmluProResult {
    accuracy: f64,
    accuracy_pct: String, // pre-rounded
    total_questions: i64,
    results_by_subject: HashMap<String, SubjectResult>,
    best: bool,
}

/// Strongly-typed GPQA Diamond result (per-category accuracy)
#[derive(Serialize)]
struct GpqaResult {
    accuracy: f64,
    accuracy_pct: String,
    total_questions: i64,
    results_by_subject: HashMap<String, SubjectResult>,
    best: bool,
}

/// Strongly-typed AIME result (overall count correct)
#[derive(Serialize)]
struct AimeResult {
    accuracy: f64,
    accuracy_pct: String,
    total_questions: i64,
    correct: i64,
    best: bool,
}

/// Strongly-typed MATH-500 result (per-subject accuracy)
#[derive(Serialize)]
struct Math500Result {
    accuracy: f64,
    accuracy_pct: String,
    total_questions: i64,
    results_by_subject: HashMap<String, SubjectResult>,
    best: bool,
}

#[derive(Serialize)]
struct SubjectResult {
    acc: f64,
    acc_pct: String, // pre-rounded
    corr: i64,
    wrong: i64,
}

/// Strongly-typed KLD pairwise result
#[derive(Serialize)]
struct KldPairResult {
    models: Vec<String>,
    avg_kld: f64,
    avg_kld_str: String, // pre-rounded
    num_prompts_evaluated: i64,
}

/// Strongly-typed average KLD to others result
#[derive(Serialize)]
struct KldAvgResult {
    avg_kld_to_others: f64,
    avg_kld_to_others_str: String, // pre-rounded
    klds: Vec<f64>,
    best: bool,
}

/// Container for all KLD results
#[derive(Serialize)]
struct KldResults {
    pairwise: HashMap<String, KldPairResult>,
    avg_kld_to_others: HashMap<String, KldAvgResult>,
}

pub fn generate_reports(results: &serde_json::Value, output_dir: &Path) -> Result<()> {
    let models_evaluated: Vec<String> = results
        .get("models")
        .and_then(|v| v.as_object())
        .map(|obj| obj.keys().cloned().collect())
        .unwrap_or_default();

    let mmlu_pro_results = extract_mmlu_results(results);
    let gpqa_results = extract_gpqa_results(results);
    let aime_results = extract_aime_results(results);
    let math500_results = extract_math500_results(results);
    let kld_results = convert_kld_results(results);

    let summary = generate_summary(
        &mmlu_pro_results,
        &gpqa_results,
        &aime_results,
        &math500_results,
        &kld_results,
    );
    let timestamp = chrono::Utc::now()
        .format("%Y-%m-%d %H:%M:%S UTC")
        .to_string();

    let models_evaluated_str = models_evaluated.join(", ");
    // HTML report using askama template
    let html = ReportTemplate {
        timestamp: &timestamp,
        models_evaluated: &models_evaluated_str,
        mmlu_pro_results: &mmlu_pro_results,
        gpqa_results: &gpqa_results,
        aime_results: &aime_results,
        math500_results: &math500_results,
        kld_results: &kld_results.pairwise,
        avg_kld_to_others: &kld_results.avg_kld_to_others,
        summary: &summary,
    }
    .render()
    .map_err(|e| anyhow::anyhow!("Template rendering error: {}", e))?;

    fs::write(output_dir.join("benchmark_report.html"), html)?;
    println!("HTML report: benchmark_report.html");

    // Markdown report
    let md = generate_markdown_report(
        &timestamp,
        &models_evaluated,
        &mmlu_pro_results,
        &gpqa_results,
        &aime_results,
        &math500_results,
        &kld_results,
        &summary,
    );
    fs::write(output_dir.join("benchmark_report.md"), md)?;
    println!("Markdown report: benchmark_report.md");

    // Save raw results as JSON
    let json = serde_json::to_string_pretty(results)?;
    fs::write(output_dir.join("results.json"), json)?;
    println!("Raw results: results.json");

    Ok(())
}

#[derive(Template)]
#[template(path = "report.html", escape = "html")]
pub struct ReportTemplate<'a> {
    timestamp: &'a str,
    models_evaluated: &'a str, // pre-joined comma-separated
    mmlu_pro_results: &'a HashMap<String, MmluProResult>,
    gpqa_results: &'a HashMap<String, GpqaResult>,
    aime_results: &'a HashMap<String, AimeResult>,
    math500_results: &'a HashMap<String, Math500Result>,
    kld_results: &'a HashMap<String, KldPairResult>,
    avg_kld_to_others: &'a HashMap<String, KldAvgResult>,
    summary: &'a Vec<String>,
}

fn extract_mmlu_results(results: &serde_json::Value) -> HashMap<String, MmluProResult> {
    let mut map = HashMap::new();
    let models = results.get("models").and_then(|v| v.as_object());
    if let Some(models) = models {
        // Find best accuracy first
        let best_acc = models
            .values()
            .filter_map(|data| data.get("mmlu_pro"))
            .filter_map(|mmlu| mmlu.get("accuracy").and_then(|v| v.as_f64()))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for (name, data) in models {
            if let Some(mmlu) = data.get("mmlu_pro").and_then(|v| v.as_object()) {
                let accuracy = mmlu.get("accuracy").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let total_questions = mmlu
                    .get("total_questions")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let best = best_acc == Some(accuracy);

                // Parse per-subject results
                let results_by_subject: HashMap<String, SubjectResult> = mmlu
                    .get("results_by_subject")
                    .and_then(|v| v.as_object())
                    .map(|obj| {
                        obj.iter()
                            .filter_map(|(cat, val)| {
                                val.as_object().map(|val_obj| {
                                    (
                                        cat.clone(),
                                        SubjectResult {
                                            acc: val_obj
                                                .get("acc")
                                                .and_then(|v| v.as_f64())
                                                .unwrap_or(0.0),
                                            acc_pct: format!(
                                                "{:.2}",
                                                val_obj
                                                    .get("acc")
                                                    .and_then(|v| v.as_f64())
                                                    .unwrap_or(0.0)
                                            ),
                                            corr: val_obj
                                                .get("corr")
                                                .and_then(|v| v.as_i64())
                                                .unwrap_or(0),
                                            wrong: val_obj
                                                .get("wrong")
                                                .and_then(|v| v.as_i64())
                                                .unwrap_or(0),
                                        },
                                    )
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                let accuracy_pct = format!("{:.2}", accuracy);
                map.insert(
                    name.clone(),
                    MmluProResult {
                        accuracy,
                        accuracy_pct,
                        total_questions,
                        results_by_subject,
                        best,
                    },
                );
            }
        }
    }
    map
}

fn extract_gpqa_results(results: &serde_json::Value) -> HashMap<String, GpqaResult> {
    let mut map = HashMap::new();
    let models = results.get("models").and_then(|v| v.as_object());
    if let Some(models) = models {
        let best_acc = models
            .values()
            .filter_map(|data| data.get("gpqa"))
            .filter_map(|gpqa| gpqa.get("accuracy").and_then(|v| v.as_f64()))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for (name, data) in models {
            if let Some(gpqa) = data.get("gpqa").and_then(|v| v.as_object()) {
                let accuracy = gpqa.get("accuracy").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let total_questions = gpqa
                    .get("total_questions")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let best = best_acc == Some(accuracy);

                let results_by_subject: HashMap<String, SubjectResult> = gpqa
                    .get("results_by_subject")
                    .and_then(|v| v.as_object())
                    .map(|obj| {
                        obj.iter()
                            .filter_map(|(cat, val)| {
                                val.as_object().map(|val_obj| {
                                    (
                                        cat.clone(),
                                        SubjectResult {
                                            acc: val_obj
                                                .get("acc")
                                                .and_then(|v| v.as_f64())
                                                .unwrap_or(0.0),
                                            acc_pct: format!(
                                                "{:.2}",
                                                val_obj
                                                    .get("acc")
                                                    .and_then(|v| v.as_f64())
                                                    .unwrap_or(0.0)
                                            ),
                                            corr: val_obj
                                                .get("corr")
                                                .and_then(|v| v.as_i64())
                                                .unwrap_or(0),
                                            wrong: val_obj
                                                .get("wrong")
                                                .and_then(|v| v.as_i64())
                                                .unwrap_or(0),
                                        },
                                    )
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                let accuracy_pct = format!("{:.2}", accuracy);
                map.insert(
                    name.clone(),
                    GpqaResult {
                        accuracy,
                        accuracy_pct,
                        total_questions,
                        results_by_subject,
                        best,
                    },
                );
            }
        }
    }
    map
}

fn extract_aime_results(results: &serde_json::Value) -> HashMap<String, AimeResult> {
    let mut map = HashMap::new();
    let models = results.get("models").and_then(|v| v.as_object());
    if let Some(models) = models {
        let best_acc = models
            .values()
            .filter_map(|data| data.get("aime"))
            .filter_map(|aime| aime.get("accuracy").and_then(|v| v.as_f64()))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for (name, data) in models {
            if let Some(aime) = data.get("aime").and_then(|v| v.as_object()) {
                let accuracy = aime.get("accuracy").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let total_questions = aime
                    .get("total_questions")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let correct = aime.get("correct").and_then(|v| v.as_i64()).unwrap_or(0);
                let best = best_acc == Some(accuracy);
                let accuracy_pct = format!("{:.2}", accuracy);

                map.insert(
                    name.clone(),
                    AimeResult {
                        accuracy,
                        accuracy_pct,
                        total_questions,
                        correct,
                        best,
                    },
                );
            }
        }
    }
    map
}

fn extract_math500_results(results: &serde_json::Value) -> HashMap<String, Math500Result> {
    let mut map = HashMap::new();
    let models = results.get("models").and_then(|v| v.as_object());
    if let Some(models) = models {
        let best_acc = models
            .values()
            .filter_map(|data| data.get("math500"))
            .filter_map(|math| math.get("accuracy").and_then(|v| v.as_f64()))
            .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

        for (name, data) in models {
            if let Some(math) = data.get("math500").and_then(|v| v.as_object()) {
                let accuracy = math.get("accuracy").and_then(|v| v.as_f64()).unwrap_or(0.0);
                let total_questions = math
                    .get("total_questions")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0);
                let best = best_acc == Some(accuracy);

                let results_by_subject: HashMap<String, SubjectResult> = math
                    .get("results_by_subject")
                    .and_then(|v| v.as_object())
                    .map(|obj| {
                        obj.iter()
                            .filter_map(|(cat, val)| {
                                val.as_object().map(|val_obj| {
                                    (
                                        cat.clone(),
                                        SubjectResult {
                                            acc: val_obj
                                                .get("acc")
                                                .and_then(|v| v.as_f64())
                                                .unwrap_or(0.0),
                                            acc_pct: format!(
                                                "{:.2}",
                                                val_obj
                                                    .get("acc")
                                                    .and_then(|v| v.as_f64())
                                                    .unwrap_or(0.0)
                                            ),
                                            corr: val_obj
                                                .get("corr")
                                                .and_then(|v| v.as_i64())
                                                .unwrap_or(0),
                                            wrong: val_obj
                                                .get("wrong")
                                                .and_then(|v| v.as_i64())
                                                .unwrap_or(0),
                                        },
                                    )
                                })
                            })
                            .collect()
                    })
                    .unwrap_or_default();

                let accuracy_pct = format!("{:.2}", accuracy);
                map.insert(
                    name.clone(),
                    Math500Result {
                        accuracy,
                        accuracy_pct,
                        total_questions,
                        results_by_subject,
                        best,
                    },
                );
            }
        }
    }
    map
}

fn convert_kld_results(results: &serde_json::Value) -> KldResults {
    let mut pairwise = HashMap::new();
    let mut avg_kld_to_others = HashMap::new();

    if let Some(kld_pairwise) = results.get("kld_pairwise").and_then(|v| v.as_object()) {
        // Handle avg_kld_to_others
        if let Some(avg_map) = kld_pairwise
            .get("avg_kld_to_others")
            .and_then(|v| v.as_object())
        {
            // Find the model with lowest avg KLD
            let best_avg = avg_map
                .values()
                .filter_map(|v| v.get("avg_kld_to_others").and_then(|x| x.as_f64()))
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

            for (name, val) in avg_map {
                if let (Some(avg), Some(klds)) = (
                    val.get("avg_kld_to_others").and_then(|v| v.as_f64()),
                    val.get("klds").and_then(|v| v.as_array()),
                ) {
                    let best = best_avg == Some(avg);
                    let klds_vec: Vec<f64> = klds.iter().filter_map(|v| v.as_f64()).collect();
                    avg_kld_to_others.insert(
                        name.clone(),
                        KldAvgResult {
                            avg_kld_to_others: avg,
                            avg_kld_to_others_str: format!("{:.3}", avg),
                            klds: klds_vec,
                            best,
                        },
                    );
                }
            }
        }

        // Handle pairwise results
        for (key, val) in kld_pairwise {
            if key == "avg_kld_to_others" {
                continue;
            }
            if let (Some(models), Some(avg_kld), Some(num_prompts)) = (
                val.get("models").and_then(|v| v.as_array()),
                val.get("avg_kld").and_then(|v| v.as_f64()),
                val.get("num_prompts_evaluated").and_then(|v| v.as_i64()),
            ) {
                let models_vec: Vec<String> = models
                    .iter()
                    .filter_map(|v| v.as_str().map(|s| s.to_string()))
                    .collect();
                pairwise.insert(
                    key.clone(),
                    KldPairResult {
                        models: models_vec,
                        avg_kld,
                        avg_kld_str: format!("{:.3}", avg_kld),
                        num_prompts_evaluated: num_prompts,
                    },
                );
            }
        }
    }

    KldResults {
        pairwise,
        avg_kld_to_others,
    }
}

fn generate_summary(
    mmlu_pro_results: &HashMap<String, MmluProResult>,
    gpqa_results: &HashMap<String, GpqaResult>,
    aime_results: &HashMap<String, AimeResult>,
    math500_results: &HashMap<String, Math500Result>,
    kld_results: &KldResults,
) -> Vec<String> {
    let mut summary = Vec::new();

    if let Some((model, data)) = mmlu_pro_results
        .iter()
        .max_by(|(_a, a_data), (_b, b_data)| {
            a_data
                .accuracy
                .partial_cmp(&b_data.accuracy)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
    {
        summary.push(format!(
            "Highest MMLU-Pro accuracy: {} ({:.1}%)",
            model,
            data.accuracy * 100.0
        ));
    }

    if let Some((model, data)) = gpqa_results.iter().max_by(|(_a, a_data), (_b, b_data)| {
        a_data
            .accuracy
            .partial_cmp(&b_data.accuracy)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        summary.push(format!(
            "Highest GPQA Diamond accuracy: {} ({:.1}%)",
            model,
            data.accuracy * 100.0
        ));
    }

    if let Some((model, data)) = aime_results.iter().max_by(|(_a, a_data), (_b, b_data)| {
        a_data
            .accuracy
            .partial_cmp(&b_data.accuracy)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        summary.push(format!(
            "Highest AIME 2025 accuracy: {} ({:.1}%)",
            model,
            data.accuracy * 100.0
        ));
    }

    if let Some((model, data)) = math500_results.iter().max_by(|(_a, a_data), (_b, b_data)| {
        a_data
            .accuracy
            .partial_cmp(&b_data.accuracy)
            .unwrap_or(std::cmp::Ordering::Equal)
    }) {
        summary.push(format!(
            "Highest MATH-500 accuracy: {} ({:.1}%)",
            model,
            data.accuracy * 100.0
        ));
    }

    // KLD summary
    if let Some((model, data)) =
        kld_results
            .avg_kld_to_others
            .iter()
            .min_by(|(_a, a_data), (_b, b_data)| {
                a_data
                    .avg_kld_to_others
                    .partial_cmp(&b_data.avg_kld_to_others)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    {
        summary.push(format!(
            "Lowest avg KLD to others: {} ({:.3})",
            model, data.avg_kld_to_others
        ));
    }
    for data in kld_results.pairwise.values() {
        summary.push(format!(
            "KLD {} vs {}: {:.3} ({} prompts)",
            data.models[0], data.models[1], data.avg_kld, data.num_prompts_evaluated
        ));
    }
    summary
}

#[allow(clippy::too_many_arguments)]
fn generate_markdown_report(
    timestamp: &str,
    models_evaluated: &[String],
    mmlu_pro_results: &HashMap<String, MmluProResult>,
    gpqa_results: &HashMap<String, GpqaResult>,
    aime_results: &HashMap<String, AimeResult>,
    math500_results: &HashMap<String, Math500Result>,
    kld_results: &KldResults,
    summary: &[String],
) -> String {
    let mut md = format!(
        "# Model Benchmark Report\n\n**Generated on:** {}\n\n## Models Evaluated\n\n{}\n",
        timestamp,
        models_evaluated.join(", ")
    );

    // MMLU-Pro
    md.push_str("\n## MMLU-Pro Accuracy (higher is better)\n\n| Model | Overall Accuracy | Total Questions |\n|-------|-----------------|-----------------|\n");
    for (model, data) in mmlu_pro_results {
        md.push_str(&format!(
            "| {} | {:.1}% | {} |\n",
            model,
            data.accuracy * 100.0,
            data.total_questions
        ));
    }

    md.push_str("\n### Per-Subject Breakdown\n\n| Model | Subject | Accuracy | Correct | Wrong |\n|-------|---------|----------|---------|------|\n");
    for (model, data) in mmlu_pro_results {
        for (subject, sdata) in &data.results_by_subject {
            md.push_str(&format!(
                "| {} | {} | {:.1}% | {} | {} |\n",
                model,
                subject,
                sdata.acc * 100.0,
                sdata.corr,
                sdata.wrong
            ));
        }
    }

    // GPQA Diamond
    if !gpqa_results.is_empty() {
        md.push_str("\n## GPQA Diamond Accuracy (higher is better)\n\n| Model | Overall Accuracy | Total Questions |\n|-------|-----------------|-----------------|\n");
        for (model, data) in gpqa_results {
            md.push_str(&format!(
                "| {} | {:.1}% | {} |\n",
                model,
                data.accuracy * 100.0,
                data.total_questions
            ));
        }

        md.push_str("\n### Per-Category Breakdown\n\n| Model | Category | Accuracy | Correct | Wrong |\n|-------|----------|----------|---------|------|\n");
        for (model, data) in gpqa_results {
            for (category, sdata) in &data.results_by_subject {
                md.push_str(&format!(
                    "| {} | {} | {:.1}% | {} | {} |\n",
                    model,
                    category,
                    sdata.acc * 100.0,
                    sdata.corr,
                    sdata.wrong
                ));
            }
        }
    }

    // AIME
    if !aime_results.is_empty() {
        md.push_str("\n## AIME 2025 Accuracy (higher is better)\n\n| Model | Accuracy | Total Questions | Correct |\n|-------|----------|-----------------|---------|\n");
        for (model, data) in aime_results {
            md.push_str(&format!(
                "| {} | {:.1}% | {} | {} |\n",
                model,
                data.accuracy * 100.0,
                data.total_questions,
                data.correct
            ));
        }
    }

    // MATH-500
    if !math500_results.is_empty() {
        md.push_str("\n## MATH-500 Accuracy (higher is better)\n\n| Model | Overall Accuracy | Total Questions |\n|-------|-----------------|-----------------|\n");
        for (model, data) in math500_results {
            md.push_str(&format!(
                "| {} | {:.1}% | {} |\n",
                model,
                data.accuracy * 100.0,
                data.total_questions
            ));
        }

        md.push_str("\n### Per-Subject Breakdown\n\n| Model | Subject | Accuracy | Correct | Wrong |\n|-------|---------|----------|---------|------|\n");
        for (model, data) in math500_results {
            for (subject, sdata) in &data.results_by_subject {
                md.push_str(&format!(
                    "| {} | {} | {:.1}% | {} | {} |\n",
                    model,
                    subject,
                    sdata.acc * 100.0,
                    sdata.corr,
                    sdata.wrong
                ));
            }
        }
    }

    // KLD
    if !kld_results.avg_kld_to_others.is_empty() || !kld_results.pairwise.is_empty() {
        md.push_str("\n## KLD (Kullback-Leibler Divergence)\n\nAverage KL divergence (lower = more similar output distributions).\n\n### Average KLD to All Other Models\n\n| Model | Avg KLD |\n|-------|---------|\n");
        for (model, data) in &kld_results.avg_kld_to_others {
            md.push_str(&format!("| {} | {:.3} |\n", model, data.avg_kld_to_others));
        }

        if !kld_results.pairwise.is_empty() {
            md.push_str("\n### Pairwise KLD\n\n| Model A | Model B | Average KLD | Samples |\n|---------|---------|-------------|---------|\n");
            for data in kld_results.pairwise.values() {
                md.push_str(&format!(
                    "| {} | {} | {:.3} | {} |\n",
                    data.models[0], data.models[1], data.avg_kld, data.num_prompts_evaluated
                ));
            }
        }
    }

    md.push_str("\n## Summary\n\n");
    for item in summary {
        md.push_str(&format!("- {}\n", item));
    }
    md
}
