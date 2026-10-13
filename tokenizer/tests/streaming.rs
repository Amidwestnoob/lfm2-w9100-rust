//! Stateful streaming UTF-8 decoder: split sequences, byte-by-byte input,
//! invalid bytes under both policies, finish with pending bytes,
//! reset, error atomicity and parallel independent decoders.

mod common;
use common::*;
use std::collections::HashMap;
use w9100_tokenizer::{InvalidUtf8Policy, StreamDecoder, StreamOptions, TokenizerError, Utf8Stream};

/// Token IDs whose piece is exactly one byte (byte-level BPE alphabet).
fn byte_tokens() -> HashMap<u8, u32> {
    let tok = tokenizer();
    let mut m = HashMap::new();
    for id in 0..tok.vocab_size() as u32 {
        let p = tok.token_to_piece(id, true).unwrap();
        if p.len() == 1 {
            m.entry(p[0]).or_insert(id);
        }
    }
    m
}

fn opts(policy: InvalidUtf8Policy) -> StreamOptions {
    StreamOptions { render_special: true, invalid: policy }
}

#[test]
fn every_byte_value_has_a_token() {
    assert_eq!(byte_tokens().len(), 256);
}

#[test]
fn multibyte_split_across_tokens_waits_without_replacement() {
    let tok = tokenizer();
    let bt = byte_tokens();
    for policy in [InvalidUtf8Policy::Error, InvalidUtf8Policy::Replace] {
        let mut d = StreamDecoder::new(tok, opts(policy));
        for ch in ["é", "€", "🙂", "日"] {
            let b = ch.as_bytes();
            for (k, byte) in b.iter().enumerate() {
                let out = d.push_token(bt[byte]).unwrap();
                if k + 1 < b.len() {
                    assert_eq!(out, "", "no premature output for {ch}");
                    assert_eq!(d.pending(), &b[..=k]);
                } else {
                    assert_eq!(out, ch);
                    assert!(d.pending().is_empty());
                }
            }
        }
        assert_eq!(d.finish().unwrap(), "");
    }
}

#[test]
fn byte_by_byte_and_chunked_utf8_stream() {
    let text = "a é € 🙂 日本 \u{10FFFF} z\n";
    let b = text.as_bytes();
    for policy in [InvalidUtf8Policy::Error, InvalidUtf8Policy::Replace] {
        for chunk in 1..=b.len() {
            let mut s = Utf8Stream::new(policy);
            let mut out = String::new();
            for c in b.chunks(chunk) {
                out.push_str(&s.push(c).unwrap());
            }
            out.push_str(&s.finish().unwrap());
            assert_eq!(out, text, "chunk {chunk}");
            assert!(!out.contains('\u{FFFD}') || text.contains('\u{FFFD}'));
        }
    }
}

#[test]
fn streaming_packet_cases_equals_concatenated_pieces() {
    let tok = tokenizer();
    let cases = read_json(&packet_file("fixtures/oracle-cases.json"));
    let mut n = 0;
    for c in cases.as_array().unwrap().iter().filter(|c| c["op"] == "encode") {
        let ids = ids_of(&c["expected_token_ids"]);
        let pieces: Vec<u8> = ids.iter().flat_map(|&i| tok.token_to_piece(i, true).unwrap().to_vec()).collect();
        let mut d = StreamDecoder::new(tok, opts(InvalidUtf8Policy::Replace));
        let mut out = String::new();
        for &id in &ids {
            out.push_str(&d.push_token(id).unwrap());
        }
        out.push_str(&d.finish().unwrap());
        assert_eq!(out, String::from_utf8_lossy(&pieces), "{}", c["id"]);
        if let Ok(valid) = std::str::from_utf8(&pieces) {
            let mut d = StreamDecoder::new(tok, opts(InvalidUtf8Policy::Error));
            let s = d.push_tokens(&ids).unwrap() + &d.finish().unwrap();
            assert_eq!(s, valid);
        }
        n += 1;
    }
    assert_eq!(n, 487);
}

#[test]
fn invalid_bytes_under_both_policies() {
    let mut s = Utf8Stream::new(InvalidUtf8Policy::Replace);
    assert_eq!(s.push(b"a\xffb").unwrap(), "a\u{FFFD}b");
    assert_eq!(s.push(b"\xf0\x9f").unwrap(), "");
    assert_eq!(s.push(b"x").unwrap(), "\u{FFFD}x");
    assert_eq!(s.push(b"\xed\xa0\x80").unwrap(), "\u{FFFD}\u{FFFD}\u{FFFD}", "surrogate encoding is invalid");
    assert_eq!(s.push(b"\xc0\xaf").unwrap(), "\u{FFFD}\u{FFFD}", "overlong");

    let mut s = Utf8Stream::new(InvalidUtf8Policy::Error);
    assert_eq!(s.push(b"ok").unwrap(), "ok");
    match s.push(b"a\xffb") {
        Err(TokenizerError::InvalidUtf8 { valid_up_to, bytes }) => {
            assert_eq!(valid_up_to, 1);
            assert_eq!(bytes, b"a\xffb");
        }
        other => panic!("{other:?}"),
    }
    s.reset();
    assert_eq!(s.push(b"\xf0\x9f").unwrap(), "");
    assert!(s.push(b"x").is_err());
}

#[test]
fn finish_with_incomplete_sequence() {
    let tok = tokenizer();
    let bt = byte_tokens();
    let mut d = StreamDecoder::new(tok, opts(InvalidUtf8Policy::Error));
    assert_eq!(d.push_token(bt[&b'h']).unwrap(), "h");
    assert_eq!(d.push_token(bt[&0xf0]).unwrap(), "");
    assert_eq!(d.push_token(bt[&0x9f]).unwrap(), "");
    assert_eq!(d.tokens_seen(), 3);
    assert_eq!(d.finish(), Err(TokenizerError::IncompleteUtf8 { pending: vec![0xf0, 0x9f] }));
    // the decoder is empty and reusable after finish
    assert!(d.pending().is_empty());
    assert_eq!(d.tokens_seen(), 0);
    assert_eq!(d.finish().unwrap(), "");
    assert_eq!(d.push_token(bt[&b'k']).unwrap(), "k");

    let mut d = StreamDecoder::new(tok, opts(InvalidUtf8Policy::Replace));
    d.push_token(bt[&0xe2]).unwrap();
    d.push_token(bt[&0x82]).unwrap();
    assert_eq!(d.finish().unwrap(), "\u{FFFD}");
    assert_eq!(d.finish().unwrap(), "");
}

#[test]
fn reset_drops_state() {
    let tok = tokenizer();
    let bt = byte_tokens();
    let mut d = StreamDecoder::new(tok, opts(InvalidUtf8Policy::Error));
    d.push_token(bt[&0xf0]).unwrap();
    d.reset();
    assert!(d.pending().is_empty());
    assert_eq!(d.tokens_seen(), 0);
    assert_eq!(d.push_token(bt[&b'a']).unwrap(), "a");
    assert_eq!(d.tokens_seen(), 1);
}

#[test]
fn bad_token_leaves_stream_unchanged() {
    let tok = tokenizer();
    let bt = byte_tokens();
    let mut d = StreamDecoder::new(tok, opts(InvalidUtf8Policy::Error));
    d.push_token(bt[&0xe2]).unwrap();
    assert!(matches!(d.push_token(128000), Err(TokenizerError::InvalidTokenId { .. })));
    assert_eq!(d.pending(), &[0xe2]);
    assert_eq!(d.tokens_seen(), 1);
    // push_tokens validates every ID before it consumes any
    assert!(d.push_tokens(&[bt[&0x82], u32::MAX]).is_err());
    assert_eq!(d.pending(), &[0xe2]);
    assert_eq!(d.push_tokens(&[bt[&0x82], bt[&0xac]]).unwrap(), "€");
}

#[test]
fn special_tokens_in_stream() {
    let tok = tokenizer();
    let ids = [124894u32, 124899, 124901, 124900];
    let mut on = StreamDecoder::new(tok, StreamOptions { render_special: true, invalid: InvalidUtf8Policy::Error });
    assert_eq!(on.push_tokens(&ids).unwrap(), "<|startoftext|><|im_start|><think><|im_end|>");
    let mut off = StreamDecoder::new(tok, StreamOptions::default());
    assert_eq!(off.push_tokens(&ids).unwrap(), "<think>");
}

#[test]
fn parallel_independent_decoders() {
    let tok = tokenizer();
    let bt = byte_tokens();
    let text = "Grüße 🙂 日本語 ok";
    let ids: Vec<u32> = text.bytes().map(|b| bt[&b]).collect();
    std::thread::scope(|s| {
        for t in 0..8 {
            let ids = &ids;
            s.spawn(move || {
                // each thread interleaves two decoders on the same tokenizer
                let mut a = StreamDecoder::new(tok, StreamOptions::default());
                let mut b = StreamDecoder::new(tok, StreamOptions::default());
                let (mut oa, mut ob) = (String::new(), String::new());
                for (i, &id) in ids.iter().enumerate() {
                    oa.push_str(&a.push_token(id).unwrap());
                    if (i + t) % 3 != 0 {
                        ob.push_str(&b.push_token(id).unwrap());
                    } else {
                        ob.push_str(&b.push_tokens(&[id]).unwrap());
                    }
                }
                assert_eq!(oa + &a.finish().unwrap(), text);
                assert_eq!(ob + &b.finish().unwrap(), text);
            });
        }
    });
}
