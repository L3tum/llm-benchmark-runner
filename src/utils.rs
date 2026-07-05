use std::time::Duration;

/// Format a Duration as "H:MM:SS" or "MM:SS"
pub fn format_duration(d: Duration) -> String {
    let secs = d.as_secs();
    if secs >= 3600 {
        let h = secs / 3600;
        let m = (secs % 3600) / 60;
        let s = secs % 60;
        format!("{}:{:02}:{:02}", h, m, s)
    } else {
        let m = secs / 60;
        let s = secs % 60;
        format!("{:01}:{:02}", m, s)
    }
}
