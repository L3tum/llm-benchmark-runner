use anyhow::{bail, Context, Result};
use std::collections::HashMap;
use yaml_serde::Value;

/// Expand macros and variables in a YAML config file.
///
/// 1. Extract `variables` and `macros` sections (remove from config).
/// 2. Recursively expand `!macro` tags and interpolate `{{variable}}` strings.
/// 3. Return the clean Value tree for typed deserialization.
pub fn expand_config(content: &str) -> Result<Value> {
    let parsed = yaml_serde::from_str(content).context("Failed to parse YAML")?;
    let (variables, macros, config) = extract_metadata(parsed)?;
    let mut stack = Vec::new();
    expand_value(config, variables, &macros, &mut stack, "<config root>")
}

type MacroExpansionMetadata = (HashMap<String, String>, HashMap<String, Value>, Value);

fn extract_metadata(value: Value) -> Result<MacroExpansionMetadata> {
    match value {
        Value::Mapping(mut mapping) => {
            let variables_val = mapping.remove(Value::String("variables".to_string()));
            let macros_val = mapping.remove(Value::String("macros".to_string()));

            let variables = parse_variables(variables_val)?;
            let macros = parse_macros(macros_val)?;
            let config = if mapping.is_empty() {
                Value::Null
            } else {
                Value::Mapping(mapping)
            };
            Ok((variables, macros, config))
        }
        other => Ok((HashMap::new(), HashMap::new(), other)),
    }
}

fn parse_variables(value: Option<Value>) -> Result<HashMap<String, String>> {
    let mut vars = HashMap::new();
    if let Some(Value::Mapping(mapping)) = value {
        for (k, v) in mapping {
            let key = match k {
                Value::String(s) => s,
                other => bail!("Variable key must be a string, got: {:?}", other),
            };
            let val = match v {
                Value::String(s) => s,
                Value::Number(n) => n.to_string(),
                Value::Bool(b) => b.to_string(),
                Value::Null => String::from(""),
                other => bail!(
                    "Variable '{}' value must be a scalar, got: {:?}",
                    key,
                    other
                ),
            };
            vars.insert(key, val);
        }
    }
    Ok(vars)
}

fn parse_macros(value: Option<Value>) -> Result<HashMap<String, Value>> {
    let mut macros = HashMap::new();
    if let Some(Value::Mapping(mapping)) = value {
        for (k, v) in mapping {
            let key = match k {
                Value::String(s) => s,
                other => bail!("Macro name must be a string, got: {:?}", other),
            };
            macros.insert(key, v);
        }
    }
    Ok(macros)
}

/// Expand a single YAML value, interpolating strings and expanding macros.
fn expand_value(
    value: Value,
    vars: HashMap<String, String>,
    macros: &HashMap<String, Value>,
    stack: &mut Vec<String>,
    path: &str,
) -> Result<Value> {
    match value {
        Value::Tagged(tagged) => {
            // Handle !macro tag
            let tag_str = format!("{}", tagged.tag);
            if tag_str.contains("macro") {
                let args = extract_macro_args(&tagged.value, path)?;
                expand_macro_with_vars(Value::Tagged(tagged), args, vars, macros, stack, path)
            } else {
                // Unknown custom tag: unwrap and expand the inner value
                expand_value(tagged.value, vars, macros, stack, path)
            }
        }
        Value::String(s) => {
            let expanded = interpolate_string(&s, &vars, path)?;
            Ok(Value::String(expanded))
        }
        Value::Number(n) => Ok(Value::Number(n)),
        Value::Bool(b) => Ok(Value::Bool(b)),
        Value::Null => Ok(Value::Null),
        Value::Sequence(seq) => {
            let mut result = Vec::new();
            for (i, item) in seq.into_iter().enumerate() {
                let item_path = format!("{}[{}]", path, i);
                let expanded = expand_value(item, vars.clone(), macros, stack, &item_path)?;
                result.push(expanded);
            }
            Ok(Value::Sequence(result))
        }
        Value::Mapping(mapping) => {
            let mut result = mapping;
            // Collect expansions to avoid simultaneous borrow
            let expansions: Vec<_> = result
                .iter()
                .map(|(k, v)| {
                    let val_path = format!("{}.<val>", path);
                    (
                        k.clone(),
                        expand_value(v.clone(), vars.clone(), macros, stack, &val_path),
                    )
                })
                .collect();
            for (k, v) in expansions {
                let expanded_val = v?;
                result.insert(k, expanded_val);
            }
            Ok(Value::Mapping(result))
        }
    }
}

fn extract_macro_args(tagged: &Value, _path: &str) -> Result<HashMap<String, Value>> {
    // Expected format: [name, {args}]
    match tagged {
        Value::Sequence(seq) if !seq.is_empty() => {
            // First element is the macro name (as a string), rest can be args
            let _macro_name = match &seq[0] {
                Value::String(s) => s.clone(),
                other => {
                    bail!("Macro name must be a string, got: {:?}", other);
                }
            };
            // Look up the macro template
            // We'll store the name back as a string for expansion
            Ok(HashMap::new())
        }
        _ => bail!(
            "Macro tag must be a sequence [name, args], got: {:?}",
            tagged
        ),
    }
}

fn expand_macro_with_vars(
    tagged: Value,
    _args: HashMap<String, Value>, // unused for now, we extract name from the sequence
    vars: HashMap<String, String>,
    macros: &HashMap<String, Value>,
    stack: &mut Vec<String>,
    _path: &str,
) -> Result<Value> {
    // The tagged value should be a sequence: [name, {args}]
    let (macro_name, args_map) = match &tagged {
        Value::Tagged(tagged_value) => {
            // Unwrap the tag, we need the inner sequence
            if let Value::Sequence(seq) = &tagged_value.value {
                let name = match seq.first() {
                    Some(Value::String(s)) => s.clone(),
                    _ => return Err(anyhow::anyhow!("Macro name must be a string")),
                };
                let args = if seq.len() >= 2 {
                    match &seq[1] {
                        Value::Mapping(m) => Some(m.clone()),
                        other => {
                            return Err(anyhow::anyhow!(
                                "Macro arguments must be a mapping, got: {:?}",
                                other
                            ))
                        }
                    }
                } else {
                    None
                };
                (name, args)
            } else {
                return Err(anyhow::anyhow!("Macro tag content must be a sequence"));
            }
        }
        other => {
            return Err(anyhow::anyhow!("Expected Tagged value, got: {:?}", other));
        }
    };

    // Cycle detection
    if stack.contains(&macro_name) {
        let cycle = stack.join(" -> ") + " -> " + &macro_name;
        return Err(anyhow::anyhow!(
            "Circular macro reference detected: {}",
            cycle
        ));
    }
    if stack.len() > 50 {
        return Err(anyhow::anyhow!("Macro expansion depth exceeded (max 50)"));
    }

    stack.push(macro_name.clone());
    let template = macros
        .get(&macro_name)
        .ok_or_else(|| anyhow::anyhow!("Macro '{}' not found in macros section", macro_name))?;

    // Merge args over global vars to create the local variable scope
    let mut merged_vars = vars;
    if let Some(args) = args_map {
        for (k, v) in args {
            let key = match k {
                Value::String(s) => s.clone(),
                other => {
                    stack.pop();
                    return Err(anyhow::anyhow!(
                        "Macro argument key must be a string, got: {:?}",
                        other
                    ));
                }
            };
            let val = match v {
                Value::String(s) => s.clone(),
                Value::Number(n) => n.to_string(),
                Value::Bool(b) => b.to_string(),
                Value::Null => String::from(""),
                other => {
                    stack.pop();
                    return Err(anyhow::anyhow!(
                        "Macro argument '{}' value must be a scalar, got: {:?}",
                        key,
                        other
                    ));
                }
            };
            merged_vars.insert(key, val);
        }
    }

    // Recursively expand the template with the merged vars
    let result = expand_value(
        template.clone(),
        merged_vars,
        macros,
        stack,
        &format!("!macro[{}]", macro_name),
    );
    stack.pop();
    result
}

fn interpolate_string(s: &str, vars: &HashMap<String, String>, path: &str) -> Result<String> {
    let mut result = String::new();
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '{' && chars.peek() == Some(&'{') {
            chars.next(); // consume the second {
            let mut key = String::new();
            while let Some(&c) = chars.peek() {
                if c == '}' {
                    chars.next();
                    if let Some(&'}') = chars.peek() {
                        chars.next(); // consume closing }
                        break;
                    }
                }
                key.push(c);
                chars.next();
            }
            if key.is_empty() {
                bail!("Empty variable in interpolation at {}", path);
            }
            let val = vars
                .get(&key)
                .ok_or_else(|| anyhow::anyhow!("Undefined variable '{}' at {}", key, path))?;
            result.push_str(val);
        } else {
            result.push(c);
        }
    }

    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_simple_interpolation() {
        let vars = HashMap::from([("name".to_string(), "World".to_string())]);
        let result = interpolate_string("Hello {{name}}!", &vars, "test").unwrap();
        assert_eq!(result, "Hello World!");
    }

    #[test]
    fn test_undefined_variable() {
        let vars = HashMap::from([("name".to_string(), "World".to_string())]);
        let result = interpolate_string("Hello {{unknown}}!", &vars, "test");
        assert!(result.is_err());
    }

    #[test]
    fn test_multiple_interpolations() {
        let vars = HashMap::from([
            ("greeting".to_string(), "Hello".to_string()),
            ("name".to_string(), "World".to_string()),
        ]);
        let result = interpolate_string("{{greeting}}, {{name}}!", &vars, "test").unwrap();
        assert_eq!(result, "Hello, World!");
    }

    #[test]
    fn test_empty_variable() {
        let vars = HashMap::new();
        let result = interpolate_string("Hello!", &vars, "test").unwrap();
        assert_eq!(result, "Hello!");
    }

    #[test]
    fn test_macro_expansion() {
        let content = r#"
macros:
  greet:
    msg: "Hello {{name}}!"

hello:
  !macro [greet, {name: World}]
"#;
        let result = expand_config(content).unwrap();
        if let Value::Mapping(map) = result {
            if let Some(Value::Mapping(inner)) = map.get(Value::String("hello".to_string())) {
                if let Some(Value::String(msg)) = inner.get(Value::String("msg".to_string())) {
                    assert_eq!(msg, "Hello World!");
                } else {
                    panic!("msg should be a string");
                }
            } else {
                panic!("hello should be a mapping");
            }
        } else {
            panic!("result should be a mapping");
        }
    }

    #[test]
    fn test_variables_and_macro() {
        let content = r#"
variables:
  greeting: "Hello"

macros:
  greet:
    msg: "{{greeting}} {{name}}!"

hello:
  !macro [greet, {name: World}]
"#;
        let result = expand_config(content).unwrap();
        if let Value::Mapping(map) = result {
            if let Some(Value::Mapping(inner)) = map.get(Value::String("hello".to_string())) {
                if let Some(Value::String(msg)) = inner.get(Value::String("msg".to_string())) {
                    assert_eq!(msg, "Hello World!");
                } else {
                    panic!("msg should be a string");
                }
            } else {
                panic!("hello should be a mapping");
            }
        } else {
            panic!("result should be a mapping");
        }
    }

    #[test]
    fn test_nested_macro() {
        let content = r#"
macros:
  outer:
    msg: !macro [inner, {name: World}]
    extra: "field"
  inner:
    msg: "Hello {{name}}!"

result:
  !macro [outer, {}]
"#;
        let result = expand_config(content).unwrap();
        if let Value::Mapping(map) = result {
            if let Some(Value::Mapping(inner)) = map.get(Value::String("result".to_string())) {
                if let Some(Value::String(msg)) = inner.get(Value::String("msg".to_string())) {
                    assert_eq!(msg, "Hello World!");
                }
                if let Some(Value::String(extra)) = inner.get(Value::String("extra".to_string())) {
                    assert_eq!(extra, "field");
                }
            }
        }
    }

    #[test]
    fn test_circular_macro() {
        let content = r#"
macros:
  a:
    !macro [b, {}]
  b:
    !macro [a, {}]

result:
  !macro [a, {}]
"#;
        let result = expand_config(content);
        assert!(result.is_err());
    }
}
