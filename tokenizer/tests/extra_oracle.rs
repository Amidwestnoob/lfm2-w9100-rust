//! Deterministic extra strings beyond the packet corpus, compared with the
//! pinned CPU llama.cpp oracle (tests/data/extra-oracle-cases.json, produced
//! by cpu_oracle.cpp from tests/data/extra-requests.json), extra chat cases
//! rendered by the Jinja2 oracle, and reference facts from a probe binary.

mod common;
use common::*;
use w9100_tokenizer::{ChatTemplate, DecodeOptions, EncodeOptions, TokenizerError};

#[test]
fn extra_requests_match_cpu_oracle() {
    let tok = tokenizer();
    let reqs = read_json(&data_file("extra-requests.json"));
    let exp = read_json(&data_file("extra-oracle-cases.json"));
    let (reqs, exp) = (reqs.as_array().unwrap(), exp.as_array().unwrap());
    assert_eq!(reqs.len(), exp.len());
    let mut failures = Vec::new();
    for (r, e) in reqs.iter().zip(exp) {
        assert_eq!(r["id"], e["id"]);
        let id = r["id"].as_str().unwrap();
        if r["op"] == "encode" {
            let opts = EncodeOptions { add_special: r["add_special"].as_bool().unwrap(), parse_special: r["parse_special"].as_bool().unwrap() };
            let got = tok.encode(&request_text(r), opts).unwrap();
            if got != ids_of(&e["expected_token_ids"]) {
                failures.push(format!("encode {id}"));
            }
            let back = tok.decode(&got, DecodeOptions { remove_special: false, render_special: true }).unwrap();
            if hex(&back) != e["expected_decode_bytes_hex"].as_str().unwrap() {
                failures.push(format!("re-decode {id}"));
            }
        } else {
            let opts = DecodeOptions { remove_special: r["remove_special"].as_bool().unwrap(), render_special: r["render_special"].as_bool().unwrap() };
            let got = tok.decode(&ids_of(&r["token_ids"]), opts).unwrap();
            if hex(&got) != e["expected_bytes_hex"].as_str().unwrap() {
                failures.push(format!("decode {id}"));
            }
        }
    }
    println!("extra oracle cases: {} compared, {} failures", reqs.len(), failures.len());
    assert!(failures.is_empty(), "{failures:?}");
    assert_eq!(reqs.len(), 708);
}

#[test]
fn extra_chat_cases_match_jinja2_oracle() {
    let tok = tokenizer();
    let tmpl = ChatTemplate::from_tokenizer(tok).unwrap();
    let cases = read_json(&data_file("extra-chat-cases.json"));
    let mut n = 0;
    for c in cases.as_array().unwrap() {
        let got = tmpl.render_json(&c["settings"]);
        if c.get("expected_error").and_then(|v| v.as_bool()) == Some(true) {
            assert!(matches!(got, Err(TokenizerError::Template(_))), "{} must fail like the oracle, got {got:?}", c["id"]);
        } else {
            assert_eq!(got.unwrap(), c["expected_text"].as_str().unwrap(), "{}", c["id"]);
        }
        n += 1;
    }
    assert_eq!(n, 31);
}

#[test]
fn reference_probe_facts() {
    // tests/data/reference-probe.json is the output of a test-only C++ probe
    // linked against the pinned CPU llama.cpp build.
    let tok = tokenizer();
    let p = read_json(&data_file("reference-probe.json"));
    let st = tok.special_tokens();
    assert_eq!(st.bos.map(i64::from), p["bos"].as_i64());
    assert_eq!(st.eos.map(i64::from), p["eos"].as_i64());
    assert_eq!(st.eot.map(i64::from), p["eot"].as_i64());
    assert_eq!(st.pad.map(i64::from), p["pad"].as_i64());
    assert_eq!(st.linefeed.map(i64::from), p["nl"].as_i64());
    assert_eq!(tok.add_bos() as i64, p["add_bos"].as_i64().unwrap());
    assert_eq!(tok.add_eos() as i64, p["add_eos"].as_i64().unwrap());
    let eog: Vec<u32> = tok.eog_tokens().iter().copied().collect();
    assert_eq!(eog, ids_of(&p["eog"]));
    // where the reference throws, this crate returns explicit errors
    assert!(p["above_10ffff"].as_str().unwrap().starts_with("exception"));
    assert_eq!(
        tok.encode(b"a\xf4\x90\x80\x80b", EncodeOptions::default()),
        Err(TokenizerError::InvalidCodepoint { codepoint: 0x110000 })
    );
    assert!(p["decode_128000"].as_str().unwrap().starts_with("exception"));
    assert!(matches!(tok.decode(&[128000], DecodeOptions::default()), Err(TokenizerError::InvalidTokenId { id: 128000, .. })));
}
