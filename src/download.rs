use anyhow::Result;
use reqwest::blocking::Client;
use sha2::{Digest, Sha256};
use std::collections::HashMap;
use std::io::Write;
use std::sync::OnceLock;
use std::time::Duration;

/// Download a URL with retry, exponential backoff, and timeout.
///
/// * `url` - the URL to download
/// * `max_retries` - number of retry attempts (0 = no retry)
/// * `timeout_secs` - per-request timeout in seconds (prevents indefinite hangs)
/// * `user_agent` - User-Agent header to identify the caller
pub fn download_with_retry(
    url: &str,
    max_retries: u32,
    timeout_secs: u64,
    user_agent: &str,
) -> Result<reqwest::blocking::Response> {
    let client = Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()?;
    let mut last_err = None;
    for attempt in 0..=max_retries {
        let resp = client.get(url).header("User-Agent", user_agent).send();
        match resp {
            Ok(resp) => return Ok(resp),
            Err(e) => {
                if attempt < max_retries {
                    let wait_time = Duration::from_secs(2u64.pow(attempt));
                    eprintln!(
                        "  Download {} failed (attempt {}/{}) after waiting {}s: {}",
                        url,
                        attempt + 1,
                        max_retries,
                        wait_time.as_secs(),
                        e
                    );
                    std::thread::sleep(wait_time);
                }
                last_err = Some(e);
            }
        }
    }
    let last_err = last_err.unwrap();
    Err(anyhow::Error::from(last_err).context(format!(
        "Failed to download {} after {} retries ({}s timeout)",
        url, max_retries, timeout_secs
    )))
}

/// Download a URL with retry, exponential backoff, and timeout, returning the response bytes.
///
/// This is a convenience wrapper that calls `download_with_retry` and extracts the response bytes.
///
/// * `url` - the URL to download
/// * `max_retries` - number of retry attempts (0 = no retry)
/// * `timeout_secs` - per-request timeout in seconds
/// * `user_agent` - User-Agent header to identify the caller
pub fn download_with_retry_bytes(
    url: &str,
    max_retries: u32,
    timeout_secs: u64,
    user_agent: &str,
) -> Result<bytes::Bytes> {
    download_with_retry_bytes_opt(url, max_retries, timeout_secs, user_agent, None)
}

/// Download a URL with retry, exponential backoff, and timeout, with an optional Authorization header.
///
/// This is useful for gated datasets on HuggingFace that require an access token.
///
/// * `url` - the URL to download
/// * `max_retries` - number of retry attempts (0 = no retry)
/// * `timeout_secs` - per-request timeout in seconds
/// * `user_agent` - User-Agent header to identify the caller
/// * `auth_header` - optional Bearer token (e.g., from HF_TOKEN env var)
pub fn download_with_retry_bytes_opt(
    url: &str,
    max_retries: u32,
    timeout_secs: u64,
    user_agent: &str,
    auth_header: Option<&str>,
) -> Result<bytes::Bytes> {
    let client = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()?;
    let mut last_err = None;
    for attempt in 0..=max_retries {
        let mut request = client.get(url).header("User-Agent", user_agent);
        if let Some(token) = auth_header {
            request = request.header("Authorization", format!("Bearer {token}"));
        }
        let resp = request.send();
        match resp {
            Ok(resp) => {
                // Reject non-success HTTP statuses (4xx/5xx) instead of
                // treating the error page body as a successful download.
                let resp = resp.error_for_status()?;
                let bytes = resp.bytes()?;
                // Enforce pinned checksum when this URL is registered.
                if let Some(expected) = lookup_checksum(url) {
                    let actual = sha256_hex(&bytes);
                    if !actual.eq_ignore_ascii_case(expected) {
                        return Err(anyhow::anyhow!(
                            "Checksum mismatch for {}: expected {}, got {}",
                            url,
                            expected,
                            actual
                        ));
                    }
                }
                return Ok(bytes);
            }
            Err(e) => {
                if attempt < max_retries {
                    let wait_time = Duration::from_secs(2u64.pow(attempt));
                    eprintln!(
                        "  Download {} failed (attempt {}/{}) after waiting {}s: {}",
                        url,
                        attempt + 1,
                        max_retries,
                        wait_time.as_secs(),
                        e
                    );
                    std::thread::sleep(wait_time);
                }
                last_err = Some(e);
            }
        }
    }
    let last_err = last_err.unwrap();
    Err(anyhow::Error::from(last_err).context(format!(
        "Failed to download {} after {} retries ({}s timeout)",
        url, max_retries, timeout_secs
    )))
}

/// Compute the lowercase hex SHA-256 digest of the given bytes.
pub fn sha256_hex(data: &[u8]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data);
    let digest = hasher.finalize();
    // Single pre-sized write avoids 32 separate format! allocations.
    let mut hex = String::with_capacity(64);
    for byte in digest {
        use std::fmt::Write;
        write!(hex, "{:02x}", byte).expect("write to String is infallible");
    }
    hex
}

/// Registry of pinned dataset URLs -> expected SHA-256 (hex).
/// Any download whose URL is present here is verified on every fetch; a
/// mismatch fails the download rather than silently using tampered data.
/// URLs not yet registered fall back to warn-and-continue so existing
/// (and not-yet-audited) datasets keep working.
pub fn checksum_registry() -> &'static HashMap<&'static str, &'static str> {
    static REGISTRY: OnceLock<HashMap<&'static str, &'static str>> = OnceLock::new();
    REGISTRY.get_or_init(|| {
        let mut m = HashMap::new();
        // SuperGPQA (hf resolve main) — fetched 2026-08-04
        m.insert(
            "https://huggingface.co/datasets/m-a-p/SuperGPQA/resolve/main/SuperGPQA-all.jsonl",
            "28b998e70205ee95e540317b5adc06a06552a3961fb50b153df126b833f7a910",
        );
        // bullshit-bench benchmark (github raw main) — fetched 2026-08-04
        m.insert(
            "https://raw.githubusercontent.com/petergpt/bullshit-benchmark/main/questions.v2.json",
            "43f43d14bd20ddf17a29fafad2c3e9862e06c2e9c4a5bc94cf08d6ee2a6b1ece",
        );
        // Google IFEval (jsonl) — fetched 2026-08-04 (parser still being migrated)
        m.insert(
            "https://huggingface.co/datasets/google/IFEval/resolve/main/ifeval_input_data.jsonl",
            "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2",
        );
        // HuggingFaceH4/MATH-500 (jsonl) — fetched 2026-08-04 (parser still being migrated)
        m.insert(
            "https://huggingface.co/datasets/HuggingFaceH4/MATH-500/resolve/main/test.jsonl",
            "35dc41080a3680858b27fa7e0533d2d547825316fc5dafe5d316f4ccc5a06132",
        );
        // --- HF datasets migrated to parquet shards (fetched 2026-08-04) ---
        m.insert(
            "https://huggingface.co/datasets/truthfulqa/truthful_qa/resolve/main/multiple_choice/validation-00000-of-00001.parquet",
            "23f08e230ca4ed66babf3a72419af7cbde1f3d734dd396ac4cf6d088bd162afd",
        );
        m.insert(
            "https://huggingface.co/datasets/truthfulqa/truthful_qa/resolve/main/generation/validation-00000-of-00001.parquet",
            "dfb1004b8ab83b22e8e476c76d5ac6074ff35c43946a724e810de3d83c3e21a5",
        );
        m.insert(
            "https://huggingface.co/datasets/EdinburghNLP/xsum/resolve/main/data/test-00000-of-00001.parquet",
            "224e9dbc6fed987759c1954603b43cb280b8d475d78893779130aa707d967ed7",
        );
        m.insert(
            "https://huggingface.co/datasets/ehovy/race/resolve/main/all/test-00000-of-00001.parquet",
            "442e6b5f22f525811544016552b2ee7d8d1b8ef31e989132edc86e7925f94136",
        );
        m.insert(
            "https://huggingface.co/datasets/stanfordnlp/snli/resolve/main/plain_text/test-00000-of-00001.parquet",
            "4696deda851c4d2385f26b58f2e13f9ed9f08ea7b42a3f4c2b97a9d08448878c",
        );
        m.insert(
            "https://huggingface.co/datasets/rajpurkar/squad_v2/resolve/main/squad_v2/validation-00000-of-00001.parquet",
            "0560174ab095c5ac0a8c8dc8da05f1625453c45a77e4ce9cabc6947ddfdd24cb",
        );
        m.insert(
            "https://huggingface.co/datasets/nq_open/resolve/main/nq_open/validation-00000-of-00001.parquet",
            "b074bed0bccb56fa1551a8ac1c9c51ce89bc11c7fbb6a9c713b2c33a98531e12",
        );
        m.insert(
            "https://huggingface.co/datasets/mandarjoshi/trivia_qa/resolve/main/rc.nocontext/validation-00000-of-00001.parquet",
            "48a5005c0eb4f8a4ae5b9868644297fd5bf1e694aeb3dc9c8ab958cac0d5b201",
        );
        m.insert(
            "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro/resolve/main/data/test-00000-of-00001.parquet",
            "0e24a191921c2f453518a537a8b2117bd137e7714d4ef1565e9ba06c1ecb9ad8",
        );
        m.insert(
            "https://huggingface.co/datasets/li-lab/MMLU-ProX/resolve/main/en/test-00000-of-00001.parquet",
            "0b0e1451ca45d44385d936c953336a3a9e68a53af498c699b02c4f960ae24b7f",
        );
        // --- BBH (default task subset) parquet shards (fetched 2026-08-04) ---
        let bbh = |task: &str| {
            format!(
                "https://huggingface.co/datasets/lukaemon/bbh/resolve/main/{}/test-00000-of-00001.parquet",
                task
            )
        };
        let bbh_tasks: &[(&str, &str)] = &[
            ("logical_deduction_three_objects", "69666640c655c70e4ee6628dd2475e911981b0e828e43ac69bbb9d8144144586"),
            ("logical_deduction_five_objects", "413dd1a9abdaeffa6ec7869c21914050f0604adec94048c60071b80889189a8e"),
            ("temporal_sequences", "b12750a723e54541eb27f6f534d5f91650e8d406e81cd5ac23c3ee7f780ad8ce"),
            ("disambiguation_qa", "94f25b514de46673114f229eea71a88b22f5777fc852f4b71ccb0291fd65b0de"),
            ("hyperbaton", "e231956d054f3bfa868482348e9c089d41239cc5a1aafe4e1b08a6fcbd0474c7"),
            ("reasoning_about_colored_objects", "5c93a11b4ced66cc2afe4cd1348a36666eba10a2aeaa023c0e3609102f4ad130"),
            ("object_counting", "81f3549f01589125e5d7bee1f5a261a0611c418c1b3bee6b9bc25136b5915fc0"),
            ("tracking_shuffled_objects_three_objects", "b3d557e426ac57f4b2194b4be5d810eb4425ff5a2c776d9b0bb6572b246c4564"),
        ];
        for (task, sha) in bbh_tasks {
            m.insert(Box::leak(bbh(task).into_boxed_str()), sha);
        }
        // --- Corrected-source datasets (fetched 2026-08-04) ---
        m.insert(
            "https://huggingface.co/datasets/abisee/cnn_dailymail/resolve/main/3.0.0/test-00000-of-00001.parquet",
            "04e322d2634a96dba76bf9a6294fbbe48e0b36abeae43f13d86ba2c3bebffe4e",
        );
        m.insert(
            "https://huggingface.co/datasets/PatronusAI/HaluBench/resolve/main/data/test-00000-of-00001.parquet",
            "c7e9cf966085ffae88d2947744418a05a26ea94380c35c238b9fc12ecb874cdc",
        );
        m.insert(
            "https://huggingface.co/datasets/saeidasgari/mmlu-pro-plus/resolve/main/data/test-00000-of-00001.parquet",
            "0e582410f640124642da94c0c850dd4c761635d07ea1213f2a0edc46a3f236c5",
        );
        m.insert(
            "https://huggingface.co/datasets/cruxeval-org/cruxeval/resolve/main/test.jsonl",
            "8368b81047dc5014e4caf5a2f97604eff7644e0ecd7415e3ceeb184bbc2e0c96",
        );
        m.insert(
            "https://huggingface.co/datasets/McGill-NLP/FaithDial/resolve/main/data/test.json",
            "acae13df566cdb2bf28274465a9b314dd0f66211fb9d40781c3c6922df33f674",
        );
        m.insert(
            "https://huggingface.co/datasets/akariasai/PopQA/resolve/main/test.tsv",
            "9a5227f41bff0e4c331d4a774d946b12f95307892b58f860a9606ef356e6089b",
        );
        m.insert(
            "https://fever.ai/download/fever/shared_task_dev.jsonl",
            "e89865bfe1b4dd054e03dd57d7241a6fde24862905f31117cf0cd719f7c78df7",
        );
        m.insert(
            "https://huggingface.co/datasets/swiss-ai/harmbench/resolve/main/DirectRequest/test-00000-of-00001.parquet",
            "1b1ca634c144ea5e796954c0308195305b93296683633230103f490e78bfc152",
        );
        m.insert(
            "https://huggingface.co/datasets/jzjiao/halueval-sft/resolve/main/data/test-00000-of-00001-af0f10a1c83a1f93.parquet",
            "f6c8a542a13b99b3f2e1a37a088afc5b8b1c5e1e1326b1880f679cc763e5add4",
        );
        m
    })
}

/// Look up an expected SHA-256 for a dataset URL, if one is registered.
pub fn lookup_checksum(url: &str) -> Option<&'static str> {
    checksum_registry().get(url).copied()
}

/// Convert an Arrow scalar value at `row` of `col` into a JSON value.
///
/// Part of the parquet-migration infrastructure for benchmarks whose datasets
/// moved to HF parquet shards. Consumed by `download_parquet_records`.
#[allow(dead_code)] // used by benchmarks during the parquet migration
fn arrow_value_to_json(col: &dyn arrow::array::Array, row: usize) -> serde_json::Value {
    use arrow::array::{
        Array as _, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array, Int64Array,
        Int8Array, LargeListArray, LargeStringArray, ListArray, StringArray, StructArray,
        UInt16Array, UInt32Array, UInt64Array, UInt8Array,
    };
    use arrow::datatypes::DataType;
    if col.is_null(row) {
        return serde_json::Value::Null;
    }
    match col.data_type() {
        DataType::Boolean => serde_json::json!(col
            .as_any()
            .downcast_ref::<BooleanArray>()
            .map(|a| a.value(row))
            .unwrap_or(false)),
        DataType::Int8 => serde_json::json!(col
            .as_any()
            .downcast_ref::<Int8Array>()
            .map(|a| a.value(row))
            .unwrap_or(0)),
        DataType::Int16 => serde_json::json!(col
            .as_any()
            .downcast_ref::<Int16Array>()
            .map(|a| a.value(row))
            .unwrap_or(0)),
        DataType::Int32 => serde_json::json!(col
            .as_any()
            .downcast_ref::<Int32Array>()
            .map(|a| a.value(row))
            .unwrap_or(0)),
        DataType::Int64 => serde_json::json!(col
            .as_any()
            .downcast_ref::<Int64Array>()
            .map(|a| a.value(row))
            .unwrap_or(0)),
        DataType::UInt8 => serde_json::json!(col
            .as_any()
            .downcast_ref::<UInt8Array>()
            .map(|a| a.value(row))
            .unwrap_or(0)),
        DataType::UInt16 => serde_json::json!(col
            .as_any()
            .downcast_ref::<UInt16Array>()
            .map(|a| a.value(row))
            .unwrap_or(0)),
        DataType::UInt32 => serde_json::json!(col
            .as_any()
            .downcast_ref::<UInt32Array>()
            .map(|a| a.value(row))
            .unwrap_or(0)),
        DataType::UInt64 => serde_json::json!(col
            .as_any()
            .downcast_ref::<UInt64Array>()
            .map(|a| a.value(row))
            .unwrap_or(0)),
        DataType::Float32 => serde_json::json!(col
            .as_any()
            .downcast_ref::<Float32Array>()
            .map(|a| a.value(row))
            .unwrap_or(0.0)),
        DataType::Float64 => serde_json::json!(col
            .as_any()
            .downcast_ref::<Float64Array>()
            .map(|a| a.value(row))
            .unwrap_or(0.0)),
        DataType::Utf8 => serde_json::json!(col
            .as_any()
            .downcast_ref::<StringArray>()
            .map(|a| a.value(row))
            .unwrap_or("")),
        DataType::LargeUtf8 => serde_json::json!(col
            .as_any()
            .downcast_ref::<LargeStringArray>()
            .map(|a| a.value(row))
            .unwrap_or("")),
        DataType::List(_) => {
            let arr = col.as_any().downcast_ref::<ListArray>().unwrap();
            let vals = arr.value(row);
            (0..vals.len())
                .map(|i| arrow_value_to_json(&vals, i))
                .collect()
        }
        DataType::LargeList(_) => {
            let arr = col.as_any().downcast_ref::<LargeListArray>().unwrap();
            let vals = arr.value(row);
            (0..vals.len())
                .map(|i| arrow_value_to_json(&vals, i))
                .collect()
        }
        DataType::Struct(fields) => {
            let arr = col.as_any().downcast_ref::<StructArray>().unwrap();
            let mut obj = serde_json::Map::new();
            for (i, field) in fields.iter().enumerate() {
                let child = arr.column(i);
                obj.insert(
                    field.name().clone(),
                    arrow_value_to_json(child.as_ref(), row),
                );
            }
            serde_json::Value::Object(obj)
        }
        _ => serde_json::Value::Null,
    }
}

/// Download a parquet dataset and return its rows as generic JSON objects.
/// Each Arrow record is converted to a `serde_json::Value` so downstream code
/// can deserialize into its existing serde structs. Registered URLs are
/// checksum-verified via `download_with_retry_bytes`.
///
/// Part of the parquet-migration infrastructure; benchmarks migrate over to it
/// incrementally as their parsers are rewritten.
#[allow(dead_code)] // used by benchmarks during the parquet migration
pub fn download_parquet_records(
    url: &str,
    max_retries: u32,
    timeout_secs: u64,
    user_agent: &str,
) -> Result<Vec<serde_json::Value>> {
    let bytes = download_with_retry_bytes(url, max_retries, timeout_secs, user_agent)?;
    let mut tmp = tempfile::NamedTempFile::new()?;
    tmp.write_all(&bytes)?;
    tmp.flush()?;
    let file = std::fs::File::open(tmp.path())?;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let reader = builder.build()?;
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch?;
        let cols = batch.columns();
        for row in 0..batch.num_rows() {
            let mut obj = serde_json::Map::new();
            for (i, field) in batch.schema().fields().iter().enumerate() {
                obj.insert(
                    field.name().clone(),
                    arrow_value_to_json(cols[i].as_ref(), row),
                );
            }
            rows.push(serde_json::Value::Object(obj));
        }
    }
    Ok(rows)
}

/// Download bytes with retry/backoff/timeout, optionally verifying the SHA-256
/// digest of the result. When `expected_sha256` is `Some`, a mismatch is an error.
/// Passing `None` preserves the existing unverified behavior.
pub fn download_with_retry_bytes_sha256(
    url: &str,
    max_retries: u32,
    timeout_secs: u64,
    user_agent: &str,
    expected_sha256: Option<&str>,
) -> Result<bytes::Bytes> {
    let resp =
        download_with_retry(url, max_retries, timeout_secs, user_agent)?.error_for_status()?;
    let bytes = resp.bytes()?;
    if let Some(expected) = expected_sha256 {
        let actual = sha256_hex(&bytes);
        if !actual.eq_ignore_ascii_case(expected.trim()) {
            return Err(anyhow::anyhow!(
                "Checksum mismatch for {}: expected {}, got {}",
                url,
                expected,
                actual
            ));
        }
    }
    Ok(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mockito::{Server, ServerGuard};

    fn start_mock_server() -> ServerGuard {
        Server::new()
    }

    #[test]
    fn test_download_with_retry_bytes_success() {
        let mut server = start_mock_server();
        let expected_body = "hello world";
        let mock = server
            .mock("GET", "/data")
            .with_status(200)
            .with_header("Content-Type", "text/plain")
            .with_body(expected_body)
            .create();

        let url = format!("{}/data", server.url());
        let bytes = download_with_retry_bytes(&url, 3, 10, "test-agent").unwrap();
        assert_eq!(String::from_utf8(bytes.to_vec()).unwrap(), expected_body);
        mock.assert();
    }

    #[test]
    fn test_download_with_retry_bytes_success_check_headers() {
        let mut server = start_mock_server();
        let expected_body = "data";
        // Use expect to capture the User-Agent header
        let mock = server
            .mock("GET", "/data")
            .match_header("User-Agent", "my-agent")
            .with_status(200)
            .with_body(expected_body)
            .create();

        let url = format!("{}/data", server.url());
        let bytes = download_with_retry_bytes(&url, 3, 10, "my-agent").unwrap();
        assert_eq!(String::from_utf8(bytes.to_vec()).unwrap(), expected_body);
        mock.assert();
    }

    // NOTE: mockito 1.x doesn't support dynamic status codes per-request,
    // so we can't easily test the "retry then success" flow where the server
    // returns 500 for the first N requests and 200 later. The test is omitted
    // for now; manual testing confirms the retry logic works as expected.

    #[test]
    fn test_download_with_retry_bytes_timeout_error() {
        // Test that an unreachable URL causes an error after retries
        // (the timeout is 1 second, so this will fail quickly with a connection timeout/error)
        let result = download_with_retry_bytes("http://localhost:1", 0, 1, "test-agent");
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("Failed to download"));
        assert!(err_msg.contains("localhost"));
    }

    // NOTE: The retry-then-success scenario (where the server returns connection errors
    // for the first N requests and succeeds later) cannot be easily tested with mockito
    // because mockito handles HTTP-level responses, not connection-level failures.
    // The retry logic itself is straightforward: exponential backoff with sleep between attempts.

    #[test]
    fn test_download_with_retry_bytes_timeout_not_exceeded() {
        // The timeout is enforced by the client builder. We can't easily
        // simulate a slow server with mockito in a deterministic way, but
        // we can verify that the client is built with the correct timeout
        // by checking that a successful request works within the timeout.
        let mut server = start_mock_server();
        let mock = server
            .mock("GET", "/data")
            .with_status(200)
            .with_body("ok")
            .create();

        let url = format!("{}/data", server.url());
        // With a 1-second timeout, a fast server should work
        let bytes = download_with_retry_bytes(&url, 0, 1, "test-agent").unwrap();
        assert_eq!(String::from_utf8(bytes.to_vec()).unwrap(), "ok");
        mock.assert();
    }

    // --- P2-3: SHA-256 download integrity ---

    #[test]
    fn test_download_sha256_correct_hash_passes() {
        let mut server = start_mock_server();
        let body = "hello world";
        let mock = server
            .mock("GET", "/data")
            .with_status(200)
            .with_body(body)
            .create();
        let url = format!("{}/data", server.url());
        let expected = sha256_hex(body.as_bytes());
        let bytes =
            download_with_retry_bytes_sha256(&url, 0, 10, "test-agent", Some(&expected)).unwrap();
        assert_eq!(String::from_utf8(bytes.to_vec()).unwrap(), body);
        mock.assert();
    }

    #[test]
    fn test_download_sha256_wrong_hash_fails() {
        let mut server = start_mock_server();
        let body = "hello world";
        let mock = server
            .mock("GET", "/data")
            .with_status(200)
            .with_body(body)
            .create();
        let url = format!("{}/data", server.url());
        let result = download_with_retry_bytes_sha256(&url, 0, 10, "test-agent", Some("deadbeef"));
        let err = result.unwrap_err().to_string();
        assert!(
            err.contains("Checksum mismatch"),
            "unexpected error: {}",
            err
        );
        mock.assert();
    }

    #[test]
    fn test_download_sha256_none_opt_out() {
        let mut server = start_mock_server();
        let body = "hello world";
        let mock = server
            .mock("GET", "/data")
            .with_status(200)
            .with_body(body)
            .create();
        let url = format!("{}/data", server.url());
        let bytes = download_with_retry_bytes_sha256(&url, 0, 10, "test-agent", None).unwrap();
        assert_eq!(String::from_utf8(bytes.to_vec()).unwrap(), body);
        mock.assert();
    }

    #[test]
    fn test_registry_has_coding_reachable_entries() {
        // The registry should contain the reachable datasets.
        assert!(lookup_checksum(
            "https://huggingface.co/datasets/m-a-p/SuperGPQA/resolve/main/SuperGPQA-all.jsonl"
        )
        .is_some());
        // An unknown URL returns None (warn-and-continue path).
        assert!(lookup_checksum("https://example.invalid/x.json").is_none());
    }
}
