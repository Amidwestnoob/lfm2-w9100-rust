//! Handoff packet acceptance: 531 oracle cases, 128000 pieces (both render
//! modes), 28 chat renders, 3 native prompt re-encodes. Expected values are
//! read here only to compare; the library never sees them.

mod common;
use common::*;
use w9100_tokenizer::{ChatTemplate, DecodeOptions, EncodeOptions, Tokenizer};

#[test]
fn packet_oracle_cases_all_match() {
    let tok = tokenizer();
    let cases = read_json(&packet_file("fixtures/oracle-cases.json"));
    let cases = cases.as_array().unwrap();
    assert_eq!(cases.len(), 531);
    let (mut enc, mut dec, mut redecode) = (0, 0, 0);
    let mut failures = Vec::new();
    for c in cases {
        let id = c["id"].as_str().unwrap();
        match c["op"].as_str().unwrap() {
            "encode" => {
                let opts = EncodeOptions {
                    add_special: c["add_special"].as_bool().unwrap(),
                    parse_special: c["parse_special"].as_bool().unwrap(),
                };
                let got = tok.encode(&request_text(c), opts).unwrap();
                if got != ids_of(&c["expected_token_ids"]) {
                    failures.push(format!("encode {id}"));
                }
                // the oracle also records decode(ids, remove_special=false, render_special=true)
                let back = tok.decode(&got, DecodeOptions { remove_special: false, render_special: true }).unwrap();
                if hex(&back) != c["expected_decode_bytes_hex"].as_str().unwrap() {
                    failures.push(format!("re-decode {id}"));
                }
                enc += 1;
                redecode += 1;
            }
            "decode" => {
                let opts = DecodeOptions {
                    remove_special: c["remove_special"].as_bool().unwrap(),
                    render_special: c["render_special"].as_bool().unwrap(),
                };
                let got = tok.decode(&ids_of(&c["token_ids"]), opts).unwrap();
                if hex(&got) != c["expected_bytes_hex"].as_str().unwrap() {
                    failures.push(format!("decode {id}"));
                }
                dec += 1;
            }
            other => panic!("unknown op {other}"),
        }
    }
    println!("encode={enc} decode={dec} re-decode={redecode} failures={}", failures.len());
    assert!(failures.is_empty(), "mismatches: {failures:?}");
    assert_eq!(enc + dec, 531);
}

#[test]
fn native_prompts_reencode_exactly() {
    let tok = tokenizer();
    let cases = read_json(&packet_file("fixtures/oracle-cases.json"));
    let native: Vec<_> = cases.as_array().unwrap().iter().filter(|c| c["id"].as_str().unwrap().starts_with("native-prompt-reencode-")).collect();
    assert_eq!(native.len(), 3);
    for c in native {
        assert!(!c["add_special"].as_bool().unwrap() && c["parse_special"].as_bool().unwrap());
        let got = tok.encode(&request_text(c), EncodeOptions { add_special: false, parse_special: true }).unwrap();
        assert_eq!(got, ids_of(&c["native_prompt_token_ids"]), "{}", c["id"]);
        assert_eq!(got[0], 124894);
        assert_ne!(got[1], 124894, "double BOS");
        // the reconstructed text is decode(remove_special=false, render_special=true) of the IDs
        let text = tok.decode(&got, DecodeOptions { remove_special: false, render_special: true }).unwrap();
        assert_eq!(text, request_text(c));
    }
}

#[test]
fn golden_native_prompts_roundtrip() {
    let tok = tokenizer();
    let golden = read_json(&packet_file("fixtures/golden-native.json"));
    let mut n = 0;
    fn walk(v: &serde_json::Value, out: &mut Vec<Vec<u32>>) {
        match v {
            serde_json::Value::Object(m) => {
                for (k, x) in m {
                    if (k == "prompt_tokens" || k == "input_tokens" || k == "prompt_token_ids") && x.is_array() {
                        out.push(x.as_array().unwrap().iter().map(|i| i.as_u64().unwrap() as u32).collect());
                    } else {
                        walk(x, out);
                    }
                }
            }
            serde_json::Value::Array(a) => a.iter().for_each(|x| walk(x, out)),
            _ => {}
        }
    }
    let mut prompts = Vec::new();
    walk(&golden, &mut prompts);
    for ids in prompts {
        let text = tok.decode(&ids, DecodeOptions { remove_special: false, render_special: true }).unwrap();
        let again = tok.encode(&text, EncodeOptions { add_special: false, parse_special: true }).unwrap();
        assert_eq!(again, ids);
        n += 1;
    }
    assert_eq!(n, 3, "expected three golden prompts");
}

#[test]
fn all_token_pieces_both_render_modes() {
    let tok = tokenizer();
    let text = std::fs::read_to_string(packet_file("fixtures/all-token-pieces.jsonl")).unwrap();
    let mut n = 0u32;
    let mut failures = 0;
    for line in text.lines() {
        let e: serde_json::Value = serde_json::from_str(line).unwrap();
        let id = e["token_id"].as_u64().unwrap() as u32;
        assert_eq!(id, n);
        if hex(tok.token_to_piece(id, true).unwrap()) != e["render_special_true_hex"].as_str().unwrap() {
            failures += 1;
        }
        if hex(tok.token_to_piece(id, false).unwrap()) != e["render_special_false_hex"].as_str().unwrap() {
            failures += 1;
        }
        n += 1;
    }
    assert_eq!(n, 128000);
    assert_eq!(failures, 0);
}

#[test]
fn chat_cases_all_match_and_no_double_bos() {
    let tok = tokenizer();
    let src = std::fs::read_to_string(packet_file("chat_template.jinja")).unwrap();
    assert_eq!(Some(src.as_str()), tok.chat_template(), "metadata template equals chat_template.jinja");
    let tmpl = ChatTemplate::from_tokenizer(tok).unwrap();
    let cases = read_json(&packet_file("fixtures/chat-cases.json"));
    let cases = cases.as_array().unwrap();
    assert_eq!(cases.len(), 28);
    let oracle = read_json(&packet_file("fixtures/oracle-cases.json"));
    let mut tokenized_checked = 0;
    for c in cases {
        let got = tmpl.render_json(&c["settings"]).unwrap();
        assert_eq!(got, c["expected_text"].as_str().unwrap(), "{}", c["id"]);
        let ids = tok.encode(got.as_bytes(), EncodeOptions { add_special: false, parse_special: true }).unwrap();
        assert_eq!(ids[0], 124894);
        assert_ne!(ids.get(1), Some(&124894), "duplicated BOS in {}", c["id"]);
        // same text tokenized by the CPU oracle, when the packet has it
        for o in oracle.as_array().unwrap() {
            if o["op"] == "encode" && o.get("text").and_then(|t| t.as_str()) == Some(got.as_str()) {
                let opts = EncodeOptions { add_special: o["add_special"].as_bool().unwrap(), parse_special: o["parse_special"].as_bool().unwrap() };
                assert_eq!(tok.encode(got.as_bytes(), opts).unwrap(), ids_of(&o["expected_token_ids"]));
                tokenized_checked += 1;
            }
        }
    }
    println!("chat texts also checked against oracle tokenization: {tokenized_checked}");
}

#[test]
fn structured_chat_api_matches_fixture() {
    let tok = tokenizer();
    let tmpl = ChatTemplate::from_tokenizer(tok).unwrap();
    let cases = read_json(&packet_file("fixtures/chat-cases.json"));
    let c = cases.as_array().unwrap().iter().find(|c| c["id"] == "chat-system-user-gen1-preserve0").unwrap();
    let mut req = w9100_tokenizer::ChatRequest::for_tokenizer(tok, c["settings"]["messages"].clone()).unwrap();
    req.add_generation_prompt = true;
    req.preserve_thinking = Some(false);
    req.tools = Some(serde_json::json!([]));
    assert_eq!(tmpl.render(&req).unwrap(), c["expected_text"].as_str().unwrap());
    let ids = tok.encode_chat(&tmpl, &req).unwrap();
    assert_eq!(ids.iter().filter(|&&i| i == 124894).count(), 1);
}

#[test]
fn gguf_file_and_bytes_load_identically() {
    let dir = std::env::temp_dir().join(format!("w9100-tok-test-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("tokenizer-only.gguf");
    std::fs::write(&path, gguf_bytes()).unwrap();
    let a = Tokenizer::from_gguf_file(&path).unwrap();
    let b = tokenizer();
    assert_eq!(a.vocab_size(), 128000);
    assert_eq!(a.special_tokens(), b.special_tokens());
    for id in (0..128000u32).step_by(997) {
        assert_eq!(a.token_to_piece(id, true).unwrap(), b.token_to_piece(id, true).unwrap());
    }
    std::fs::remove_dir_all(&dir).ok();
    // A python-generated file (write_metadata_gguf.py) can be compared byte for byte.
    if let Ok(p) = std::env::var("W9100_TOKENIZER_GGUF") {
        let py = std::fs::read(&p).unwrap();
        assert_eq!(&py, gguf_bytes(), "Rust writer output differs from {p}");
        println!("python-generated GGUF is byte-identical ({} bytes)", py.len());
    }
    assert_eq!(gguf_bytes().len(), 8_222_048, "same size as the packet's generated file");
}

#[test]
fn lfm2_configuration() {
    let tok = tokenizer();
    assert_eq!(tok.model(), "gpt2");
    assert_eq!(tok.pre(), "lfm2");
    assert_eq!(tok.bos(), Some(124894));
    assert_eq!(tok.eos(), Some(124900));
    assert_eq!(tok.pad(), Some(124893));
    assert_eq!(tok.token_text(124895).unwrap(), b"<|endoftext|>");
    assert_ne!(tok.eos(), Some(124895), "<|endoftext|> is not the configured EOS");
    assert!(tok.add_bos(), "family default add_bos = true (no add_bos_token key)");
    assert!(!tok.add_eos());
    assert!(tok.ignore_merges());
    assert!(tok.clean_spaces());
    assert!(tok.load_notes().ambiguous_special_order.is_empty());
}
