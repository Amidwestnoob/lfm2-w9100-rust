//! Shared test helpers: locate the handoff packet and build the tokenizer
//! from a metadata-only GGUF generated in memory from `metadata.json`.
#![allow(dead_code)]

use std::path::{Path, PathBuf};
use std::sync::OnceLock;
use w9100_tokenizer::gguf::{metadata_from_typed_json, write_metadata_only, MetaValue};
use w9100_tokenizer::Tokenizer;

/// Packet location: `W9100_TOKENIZER_PACKET`, else the workshop layout
/// (`contributions/w9100-rust-tokenizer` -> `progress/w9100-20261005/tokenizer-handoff`),
/// else the local handoff clone.
pub fn packet_dir() -> PathBuf {
    if let Ok(p) = std::env::var("W9100_TOKENIZER_PACKET") {
        return PathBuf::from(p);
    }
    let manifest = Path::new(env!("CARGO_MANIFEST_DIR"));
    let candidates = [
        manifest.join("../../progress/w9100-20261005/tokenizer-handoff"),
        manifest.join("../../../progress/w9100-20261005/tokenizer-handoff"),
        manifest.join("tests/fixtures/lfm2"),
    ];
    for c in candidates {
        if c.join("metadata.json").is_file() {
            return c;
        }
    }
    panic!(
        "handoff packet not found: set W9100_TOKENIZER_PACKET to progress/w9100-20261005/tokenizer-handoff \
         (contribution-workshop commit 16dd1f28fe2af407e0945ffc5ff8a50f1859eea6)"
    );
}

pub fn packet_file(rel: &str) -> PathBuf {
    packet_dir().join(rel)
}

pub fn data_file(rel: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data").join(rel)
}

pub fn read_json(path: &Path) -> serde_json::Value {
    let text = std::fs::read_to_string(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    serde_json::from_str(&text).unwrap_or_else(|e| panic!("parse {}: {e}", path.display()))
}

pub fn metadata_entries() -> &'static Vec<(String, MetaValue)> {
    static M: OnceLock<Vec<(String, MetaValue)>> = OnceLock::new();
    M.get_or_init(|| {
        let text = std::fs::read_to_string(packet_file("metadata.json")).expect("metadata.json");
        metadata_from_typed_json(&text).expect("typed metadata")
    })
}

/// Metadata-only GGUF bytes (Rust port of write_metadata_gguf.py).
pub fn gguf_bytes() -> &'static Vec<u8> {
    static G: OnceLock<Vec<u8>> = OnceLock::new();
    G.get_or_init(|| write_metadata_only(metadata_entries()))
}

pub fn tokenizer() -> &'static Tokenizer {
    static T: OnceLock<Tokenizer> = OnceLock::new();
    T.get_or_init(|| Tokenizer::from_gguf_bytes(gguf_bytes()).expect("load tokenizer from metadata-only GGUF"))
}

pub fn unhex(s: &str) -> Vec<u8> {
    assert!(s.len() % 2 == 0, "odd hex");
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).unwrap()).collect()
}

pub fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

pub fn ids_of(v: &serde_json::Value) -> Vec<u32> {
    v.as_array().expect("id array").iter().map(|x| x.as_u64().expect("id") as u32).collect()
}

/// Input text bytes of an encode request.
pub fn request_text(r: &serde_json::Value) -> Vec<u8> {
    if let Some(h) = r.get("text_hex").and_then(|v| v.as_str()) {
        unhex(h)
    } else {
        r["text"].as_str().expect("text").as_bytes().to_vec()
    }
}
