pub fn slugify(name: &str) -> Result<String, askama::Error> {
    Ok(crate::utils::slugify(name))
}
