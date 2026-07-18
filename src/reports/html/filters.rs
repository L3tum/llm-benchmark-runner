pub fn slugify(name: &str) -> Result<String, askama::Error> {
    Ok(name
        .to_lowercase()
        .replace(" ", "-")
        .replace("/", "-")
        .replace("_", "-")
        .replace("  ", "-")
        .trim()
        .to_string())
}
