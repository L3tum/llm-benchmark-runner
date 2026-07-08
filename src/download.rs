use anyhow::Result;
use std::time::Duration;

/// Download a URL with retry and exponential backoff.
/// `max_retries`: number of retry attempts (0 = no retry).
pub fn download_with_retry(url: &str, max_retries: u32) -> Result<reqwest::blocking::Response> {
    let mut last_err = None;
    for attempt in 0..=max_retries {
        match reqwest::blocking::get(url) {
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
        "Failed to download {} after {} retries",
        url, max_retries
    )))
}
