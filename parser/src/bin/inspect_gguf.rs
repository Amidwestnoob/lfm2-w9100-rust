//! inspect_gguf — CLI tool that parses a GGUF file and outputs JSON.
//!
//! Usage: inspect_gguf PATH
//!
//! Output JSON schema:
//! {
//!   "version": u32,
//!   "alignment": u32,
//!   "data_offset": u64,
//!   "metadata": { "key": {"type": "<variant>", "value": <value>}, ... },
//!   "tensors": [
//!     {"name": str, "dimensions": [u64], "ggml_type": u32, "offset": u64, "nbytes": u64},
//!     ...
//!   ]
//! }

use std::fs::File;
use std::io::BufReader;
use std::process::exit;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() != 2 {
        eprintln!("Usage: inspect_gguf PATH");
        exit(1);
    }

    let path = &args[1];
    let file = match File::open(path) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error opening file: {}", e);
            exit(1);
        }
    };

    let file_len = match file.metadata() {
        Ok(m) => m.len(),
        Err(e) => {
            eprintln!("Error reading file metadata: {}", e);
            exit(1);
        }
    };

    let mut reader = BufReader::new(file);
    let gguf = match parser::gguf::parse(&mut reader, file_len) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("Parse error: {}", e);
            exit(1);
        }
    };

    // Build JSON output
    let mut metadata_json = serde_json::Map::new();
    for (key, value) in &gguf.metadata {
        metadata_json.insert(key.clone(), value_to_json(value));
    }

    let tensors_json: Vec<serde_json::Value> = gguf
        .tensors
        .iter()
        .map(|t| {
            serde_json::json!({
                "name": t.name,
                "dimensions": t.dimensions,
                "ggml_type": t.ggml_type,
                "offset": t.offset,
                "nbytes": t.nbytes,
            })
        })
        .collect();

    let output = serde_json::json!({
        "version": gguf.version,
        "alignment": gguf.alignment,
        "data_offset": gguf.data_offset,
        "metadata": metadata_json,
        "tensors": tensors_json,
    });

    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}

fn value_to_json(value: &parser::gguf::Value) -> serde_json::Value {
    let (type_str, val) = match value {
        parser::gguf::Value::U8(v) => ("U8", serde_json::json!(v)),
        parser::gguf::Value::I8(v) => ("I8", serde_json::json!(v)),
        parser::gguf::Value::U16(v) => ("U16", serde_json::json!(v)),
        parser::gguf::Value::I16(v) => ("I16", serde_json::json!(v)),
        parser::gguf::Value::U32(v) => ("U32", serde_json::json!(v)),
        parser::gguf::Value::I32(v) => ("I32", serde_json::json!(v)),
        parser::gguf::Value::F32(v) => ("F32", serde_json::json!(v)),
        parser::gguf::Value::Bool(v) => ("Bool", serde_json::json!(v)),
        parser::gguf::Value::String(v) => ("String", serde_json::json!(v)),
        parser::gguf::Value::Array(vals) => {
            let arr: Vec<serde_json::Value> = vals.iter().map(value_to_json).collect();
            return serde_json::json!({"type": "Array", "value": arr});
        }
        parser::gguf::Value::U64(v) => ("U64", serde_json::json!(v)),
        parser::gguf::Value::I64(v) => ("I64", serde_json::json!(v)),
        parser::gguf::Value::F64(v) => ("F64", serde_json::json!(v)),
    };
    serde_json::json!({"type": type_str, "value": val})
}
