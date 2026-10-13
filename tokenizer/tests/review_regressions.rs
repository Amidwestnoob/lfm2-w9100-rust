//! Astra r001 review regressions; existing oracle fixtures stay unchanged.
mod common;
use common::*;
use w9100_tokenizer::{gguf, ChatTemplate, InvalidUtf8Policy, StreamDecoder, StreamOptions, TokenizerError, Utf8Stream};

#[test]
fn lfm_chat_review_cases_match_jinja2() {
    let cases = read_json(&data_file("review-chat-cases.json"));
    let tmpl = ChatTemplate::from_tokenizer(tokenizer()).unwrap();
    let mut failures = Vec::new();
    for c in cases.as_array().unwrap() {
        let got = tmpl.render_json(&c["settings"]);
        let matches = if c["expected_error"] == true {
            got.is_err()
        } else {
            got.as_ref().ok().map(String::as_str) == c["expected_text"].as_str()
        };
        if !matches { failures.push(c["id"].clone()); }
    }
    assert!(failures.is_empty(), "Jinja2 mismatches: {failures:?}");
}

#[test]
fn template_review_cases_match_jinja2() {
    let cases = read_json(&data_file("review-template-cases.json"));
    let mut failures = Vec::new();
    for c in cases.as_array().unwrap() {
        let got = ChatTemplate::new(c["template"].as_str().unwrap())
            .and_then(|t| t.render_json(&c["settings"]));
        if got.as_ref().ok().map(String::as_str) != c["expected_text"].as_str() {
            failures.push(format!("{}: {got:?}", c["id"]));
        }
    }
    assert!(failures.is_empty(), "Jinja2 mismatches: {failures:?}");
}

#[test]
fn batch_utf8_error_keeps_unreturned_prefix() {
    let tok = tokenizer();
    let byte = |b: u8| (0..tok.vocab_size() as u32).find(|&id| tok.token_to_piece(id, true).unwrap() == [b]).unwrap();
    let mut d = StreamDecoder::new(tok, StreamOptions::default());
    assert_eq!(d.push_token(byte(0xc3)).unwrap(), "");
    let ids = [byte(0xa9), byte(b'a'), byte(0xff)];
    assert_eq!(d.push_tokens(&ids), Err(TokenizerError::InvalidUtf8 {
        valid_up_to: 3, bytes: vec![0xc3, 0xa9, b'a', 0xff],
    }));
    assert_eq!(d.pending(), &[0xc3, 0xa9, b'a', 0xff]);
    assert_eq!(d.tokens_seen(), 1, "failed batch accepts no tokens");
}

#[test]
fn finish_preserves_permanent_utf8_error_kind_and_clears_state() {
    let mut s = Utf8Stream::new(InvalidUtf8Policy::Error);
    let expected = TokenizerError::InvalidUtf8 { valid_up_to: 1, bytes: b"a\xff".to_vec() };
    assert_eq!(s.push(b"a\xff"), Err(expected.clone()));
    assert_eq!(s.finish(), Err(expected));
    assert_eq!(s.finish().unwrap(), "");
    assert_eq!(s.push(b"ok").unwrap(), "ok");
}

#[test]
fn gguf_untrusted_lengths_return_errors_not_capacity_panics() {
    // Public streaming reader: advertised bound can exceed the actual stream.
    // Neither a string nor a string-array count may allocate before reading it.
    let mut header = b"GGUF".to_vec();
    header.extend_from_slice(&3u32.to_le_bytes());
    header.extend_from_slice(&0u64.to_le_bytes());
    header.extend_from_slice(&1u64.to_le_bytes());
    let mut string = header.clone();
    string.extend_from_slice(&((isize::MAX as u64) + 1).to_le_bytes());
    let mut array = header;
    array.extend_from_slice(&1u64.to_le_bytes());
    array.push(b'k');
    array.extend_from_slice(&gguf::ty::ARRAY.to_le_bytes());
    array.extend_from_slice(&gguf::ty::STRING.to_le_bytes());
    array.extend_from_slice(&((isize::MAX as u64) / 24 + 1).to_le_bytes());
    let mut failures = Vec::new();
    for (label, input) in [("string", string), ("array", array)] {
        let result = std::panic::catch_unwind(|| gguf::read_metadata(input.as_slice(), u64::MAX));
        if !matches!(result, Ok(Err(_))) { failures.push(label); }
    }
    assert!(failures.is_empty(), "malformed counts did not return errors: {failures:?}");
}

#[test]
fn cli_bad_hex_is_an_input_error_not_a_panic() {
    let owned = std::env::temp_dir().join(format!("w9100-review-hex-{}.gguf", std::process::id()));
    let model = std::env::var("W9100_TOKENIZER_GGUF").unwrap_or_else(|_| {
        std::fs::write(&owned, gguf_bytes()).unwrap();
        owned.to_str().unwrap().to_string()
    });
    for text in ["a€", "éé", "🙂", "gg", "+1", "0"] {
        let result = std::process::Command::new(env!("CARGO_BIN_EXE_w9100-tok"))
            .args(["encode", "--model", &model, "--text-hex", text]).output().unwrap();
        assert_eq!(result.status.code(), Some(2), "{text:?}: {}", String::from_utf8_lossy(&result.stderr));
        assert!(!String::from_utf8_lossy(&result.stderr).contains("panicked"));
    }
    if model == owned.to_string_lossy() { std::fs::remove_file(owned).unwrap(); }
}
