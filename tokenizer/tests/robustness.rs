//! Error handling and edge cases: malformed metadata, out-of-range IDs,
//! empty input, flag independence, Unicode/whitespace/code/number text,
//! deterministic random input, and parallel use.

mod common;
use common::*;
use std::collections::BTreeMap;
use w9100_tokenizer::gguf::{write_metadata_only, MetaArray, MetaValue, Metadata};
use w9100_tokenizer::{attr, DecodeOptions, EncodeOptions, Tokenizer, TokenizerError};

const BOS: u32 = 124894;
const IM_START: u32 = 124899;
const IM_END: u32 = 124900;
const THINK: u32 = 124901;

fn flags() -> [EncodeOptions; 4] {
    [(false, false), (true, false), (false, true), (true, true)].map(|(a, p)| EncodeOptions { add_special: a, parse_special: p })
}

fn base_md() -> Metadata {
    metadata_entries().iter().cloned().collect::<BTreeMap<_, _>>()
}

fn load_err(md: &Metadata) -> TokenizerError {
    match Tokenizer::from_metadata(md) {
        Ok(_) => panic!("malformed metadata was accepted"),
        Err(e) => e,
    }
}

// ---------------------------------------------------------------- metadata

#[test]
fn malformed_gguf_container() {
    let good = gguf_bytes();
    // truncation anywhere in the header or metadata is an error, never a panic
    // (the trailing alignment padding is not needed for metadata, as in gguf.cpp,
    // so cuts stay inside the key/value data)
    let mut end = good.len();
    while good[end - 1] == 0 {
        end -= 1;
    }
    assert!(good.len() - end < 32 + 8);
    let mut cuts: Vec<usize> = (0..64).collect();
    cuts.extend((64..end).step_by(end / 300));
    cuts.push(end - 1);
    for cut in cuts {
        let r = Tokenizer::from_gguf_bytes(&good[..cut]);
        assert!(matches!(r, Err(TokenizerError::Gguf(_))), "cut at {cut}: {r:?}");
    }
    let mut bad = good.clone();
    bad[0] = b'X';
    assert!(matches!(Tokenizer::from_gguf_bytes(&bad), Err(TokenizerError::Gguf(m)) if m.contains("magic")));
    let mut bad = good.clone();
    bad[4..8].copy_from_slice(&2u32.to_le_bytes());
    assert!(matches!(Tokenizer::from_gguf_bytes(&bad), Err(TokenizerError::Gguf(m)) if m.contains("version")));
    // huge key/value count
    let mut bad = good.clone();
    bad[16..24].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(matches!(Tokenizer::from_gguf_bytes(&bad), Err(TokenizerError::Gguf(_))));
    // first key length set huge
    let mut bad = good.clone();
    bad[24..32].copy_from_slice(&(1u64 << 60).to_le_bytes());
    assert!(matches!(Tokenizer::from_gguf_bytes(&bad), Err(TokenizerError::Gguf(_))));
    // first value type set to an unknown type
    let klen = u64::from_le_bytes(good[24..32].try_into().unwrap()) as usize;
    let tpos = 32 + klen;
    let mut bad = good.clone();
    bad[tpos..tpos + 4].copy_from_slice(&99u32.to_le_bytes());
    assert!(matches!(Tokenizer::from_gguf_bytes(&bad), Err(TokenizerError::Gguf(m)) if m.contains("type")));
    // missing file
    assert!(matches!(Tokenizer::from_gguf_file("/nonexistent/w9100.gguf"), Err(TokenizerError::Gguf(_))));
    // empty input
    assert!(Tokenizer::from_gguf_bytes(&[]).is_err());
}

#[test]
fn gguf_structural_errors() {
    // nested arrays, invalid bool, huge array count, duplicate keys
    let entries = |v: MetaValue| vec![("k".to_string(), v)];
    let mut nested = write_metadata_only(&entries(MetaValue::Array(MetaArray::U8(vec![1]))));
    // patch the element type (offset: 24 header + 8+1 key + 4 type) from U8 to ARRAY
    nested[37..41].copy_from_slice(&9u32.to_le_bytes());
    assert!(matches!(Tokenizer::from_gguf_bytes(&nested), Err(TokenizerError::Gguf(m)) if m.contains("nested")), "nested array");

    // gguf.cpp reads BOOL as int8 and maps every non-zero value to true
    let mut odd_bool = write_metadata_only(&entries(MetaValue::Bool(false)));
    odd_bool[37] = 7;
    let md = w9100_tokenizer::gguf::read_metadata(&odd_bool[..], odd_bool.len() as u64).unwrap();
    assert_eq!(md["k"], MetaValue::Bool(true));
    let mut md = base_md();
    let mut bytes = write_metadata_only(&[("tokenizer.ggml.add_bos_token".to_string(), MetaValue::Bool(false))]);
    let off = 24 + 8 + "tokenizer.ggml.add_bos_token".len() + 4;
    bytes[off] = 0xff;
    let parsed = w9100_tokenizer::gguf::read_metadata(&bytes[..], bytes.len() as u64).unwrap();
    md.insert("tokenizer.ggml.add_bos_token".into(), parsed["tokenizer.ggml.add_bos_token"].clone());
    assert!(Tokenizer::from_metadata(&md).unwrap().add_bos());

    // a GGUF without tokenizer keys
    assert_eq!(Tokenizer::from_gguf_bytes(&odd_bool).unwrap_err(), TokenizerError::MissingKey("tokenizer.ggml.model"));

    let mut huge = write_metadata_only(&entries(MetaValue::Array(MetaArray::U32(vec![1]))));
    huge[41..49].copy_from_slice(&(u64::MAX / 2).to_le_bytes());
    assert!(matches!(Tokenizer::from_gguf_bytes(&huge), Err(TokenizerError::Gguf(_))));

    let dup = write_metadata_only(&[("k".into(), MetaValue::U32(1)), ("k".into(), MetaValue::U32(2))]);
    assert!(matches!(Tokenizer::from_gguf_bytes(&dup), Err(TokenizerError::Gguf(m)) if m.contains("duplicate")));
}

#[test]
fn malformed_tokenizer_metadata() {
    let mut md = base_md();
    md.remove("tokenizer.ggml.model");
    assert_eq!(load_err(&md), TokenizerError::MissingKey("tokenizer.ggml.model"));

    let mut md = base_md();
    md.remove("tokenizer.ggml.tokens");
    assert!(matches!(load_err(&md), TokenizerError::MissingKey(_)));

    let mut md = base_md();
    md.remove("tokenizer.ggml.merges");
    assert!(matches!(load_err(&md), TokenizerError::MissingKey(_)));

    let mut md = base_md();
    md.insert("tokenizer.ggml.model".into(), MetaValue::U32(1));
    assert!(matches!(load_err(&md), TokenizerError::WrongType { .. }));

    let mut md = base_md();
    md.insert("tokenizer.ggml.tokens".into(), MetaValue::Array(MetaArray::U32(vec![1, 2])));
    assert!(matches!(load_err(&md), TokenizerError::WrongType { .. }));

    let mut md = base_md();
    md.insert("tokenizer.ggml.bos_token_id".into(), MetaValue::String(b"1".to_vec()));
    assert!(matches!(load_err(&md), TokenizerError::WrongType { .. }));

    let mut md = base_md();
    md.insert("tokenizer.ggml.model".into(), MetaValue::String(b"llama".to_vec()));
    assert!(matches!(load_err(&md), TokenizerError::Unsupported(_)));

    let mut md = base_md();
    md.insert("tokenizer.ggml.pre".into(), MetaValue::String(b"qwen2".to_vec()));
    assert!(matches!(load_err(&md), TokenizerError::Unsupported(_)));

    let mut md = base_md();
    md.insert("tokenizer.ggml.pre".into(), MetaValue::String(b"no-such-pre".to_vec()));
    assert!(matches!(load_err(&md), TokenizerError::Unsupported(_)));

    // short token_type array
    let mut md = base_md();
    if let Some(MetaValue::Array(MetaArray::I32(v))) = md.get_mut("tokenizer.ggml.token_type") {
        v.truncate(1000);
    } else {
        panic!("token_type is not I32");
    }
    assert!(matches!(load_err(&md), TokenizerError::InvalidMetadata(_)));

    // duplicate token text
    let mut md = base_md();
    if let Some(MetaValue::Array(MetaArray::String(v))) = md.get_mut("tokenizer.ggml.tokens") {
        v[1] = v[0].clone();
    }
    assert!(matches!(load_err(&md), TokenizerError::InvalidMetadata(m) if m.contains("duplicate")));

    // empty vocabulary
    let mut md = base_md();
    md.insert("tokenizer.ggml.tokens".into(), MetaValue::Array(MetaArray::String(vec![])));
    assert!(Tokenizer::from_metadata(&md).is_err());

    // merge with no space separator: llama-vocab.cpp stores it as the pair
    // ("", "") and continues; this crate does the same. With
    // ignore_merges=true and whole-word vocab hits the output can change only
    // for words that need that one merge, so a plain sentence is unchanged.
    let mut md = base_md();
    if let Some(MetaValue::Array(MetaArray::String(v))) = md.get_mut("tokenizer.ggml.merges") {
        v[0] = b"nospace".to_vec();
    }
    let t = Tokenizer::from_metadata(&md).unwrap();
    assert_eq!(t.encode(b"hello world", EncodeOptions::default()).unwrap(), tokenizer().encode(b"hello world", EncodeOptions::default()).unwrap());

    // merges of the wrong element type
    let mut md = base_md();
    md.insert("tokenizer.ggml.merges".into(), MetaValue::Array(MetaArray::I32(vec![1])));
    assert!(matches!(load_err(&md), TokenizerError::WrongType { .. }));

    // special token ID outside the vocabulary: the reference warns and keeps
    // the default ID (11 for BPE BOS); this crate does the same and records it
    let mut md = base_md();
    md.insert("tokenizer.ggml.bos_token_id".into(), MetaValue::U32(500_000));
    let t = Tokenizer::from_metadata(&md).unwrap();
    assert_eq!(t.bos(), Some(11));
    assert_eq!(t.load_notes().bad_special_ids.len(), 1);
    assert!(tokenizer().load_notes().bad_special_ids.is_empty());
}

#[test]
fn optional_flags_follow_metadata() {
    let mut md = base_md();
    md.insert("tokenizer.ggml.add_bos_token".into(), MetaValue::Bool(false));
    let t = Tokenizer::from_metadata(&md).unwrap();
    assert!(!t.add_bos());
    let plain = t.encode(b"hi", EncodeOptions::default()).unwrap();
    assert_eq!(t.encode(b"hi", EncodeOptions { add_special: true, parse_special: false }).unwrap(), plain);
    md.insert("tokenizer.ggml.add_eos_token".into(), MetaValue::Bool(true));
    let t = Tokenizer::from_metadata(&md).unwrap();
    let ids = t.encode(b"hi", EncodeOptions { add_special: true, parse_special: false }).unwrap();
    assert_eq!(ids.last(), Some(&IM_END));
    assert_ne!(ids[0], BOS);
}

// ---------------------------------------------------------------- IDs

#[test]
fn out_of_range_ids_are_errors() {
    let tok = tokenizer();
    for id in [128000u32, 128001, 200000, u32::MAX] {
        assert!(matches!(tok.token_to_piece(id, true), Err(TokenizerError::InvalidTokenId { .. })));
        assert!(matches!(tok.token_text(id), Err(TokenizerError::InvalidTokenId { .. })));
        assert!(tok.token_attr(id).is_err());
        for (r, s) in [(false, false), (false, true), (true, false), (true, true)] {
            let e = tok.decode(&[1, 2, id, 3], DecodeOptions { remove_special: r, render_special: s });
            assert!(matches!(e, Err(TokenizerError::InvalidTokenId { .. })), "{id}");
        }
    }
    assert_eq!(tok.checked_ids(&[0, 127999]).unwrap(), vec![0, 127999]);
    assert!(matches!(tok.checked_ids(&[-1]), Err(TokenizerError::InvalidTokenId { id: -1, .. })));
    assert!(matches!(tok.checked_ids(&[128000]), Err(TokenizerError::InvalidTokenId { id: 128000, .. })));
    assert!(tok.checked_ids(&[i64::MIN, i64::MAX]).is_err());
    assert!(!tok.is_eog(128000));
    assert!(tok.is_eog(124895) && tok.is_eog(IM_END));
}

// ---------------------------------------------------------------- empty input

#[test]
fn empty_input_all_flag_combinations() {
    let tok = tokenizer();
    for o in flags() {
        let ids = tok.encode(b"", o).unwrap();
        assert_eq!(ids, if o.add_special { vec![BOS] } else { vec![] }, "{o:?}");
    }
    for (r, s) in [(false, false), (false, true), (true, false), (true, true)] {
        assert!(tok.decode(&[], DecodeOptions { remove_special: r, render_special: s }).unwrap().is_empty());
    }
    // BOS only, with removal
    assert!(tok.decode(&[BOS], DecodeOptions { remove_special: true, render_special: true }).unwrap().is_empty());
    assert_eq!(tok.decode(&[BOS], DecodeOptions { remove_special: false, render_special: true }).unwrap(), b"<|startoftext|>");
    assert!(tok.decode(&[BOS], DecodeOptions { remove_special: false, render_special: false }).unwrap().is_empty());
}

// ---------------------------------------------------------------- flags

#[test]
fn encode_flags_are_independent() {
    let tok = tokenizer();
    let texts: [&[u8]; 5] = [
        b"<|im_start|>user\nhi<|im_end|>\n",
        b"plain text, no specials",
        b"<think>reasoning</think> answer",
        b"<|startoftext|>already has BOS",
        b"x<|im_end|><|im_end|>y<|im_start|>",
    ];
    for t in texts {
        let nn = tok.encode(t, EncodeOptions { add_special: false, parse_special: false }).unwrap();
        let an = tok.encode(t, EncodeOptions { add_special: true, parse_special: false }).unwrap();
        let np = tok.encode(t, EncodeOptions { add_special: false, parse_special: true }).unwrap();
        let ap = tok.encode(t, EncodeOptions { add_special: true, parse_special: true }).unwrap();
        // add_special only prepends BOS (add_bos=true, add_eos=false)
        assert_eq!(an[0], BOS);
        assert_eq!(&an[1..], &nn[..]);
        assert_eq!(ap[0], BOS);
        assert_eq!(&ap[1..], &np[..]);
        // parse_special=false never emits control tokens
        for &id in &nn {
            assert_eq!(tok.token_attr(id).unwrap() & attr::CONTROL, 0, "control token {id} without parse_special");
        }
        // user-defined tokens (<think>, </think>) are matched in both modes
        if t.starts_with(b"<think>") {
            assert_eq!(nn[0], THINK);
            assert_eq!(np[0], THINK);
        }
        if t.starts_with(b"<|im_start|>") {
            assert_eq!(np[0], IM_START);
            assert_ne!(nn[0], IM_START);
        }
    }
    // reference behaviour: an explicit BOS in the text plus add_special gives two BOS tokens
    let ids = tok.encode(b"<|startoftext|>x", EncodeOptions { add_special: true, parse_special: true }).unwrap();
    assert_eq!(&ids[..2], &[BOS, BOS]);
}

#[test]
fn decode_flags_are_independent() {
    let tok = tokenizer();
    let ids = [BOS, IM_START, 1, 2, 3, THINK, IM_END];
    let ff = tok.decode(&ids, DecodeOptions { remove_special: false, render_special: false }).unwrap();
    let ft = tok.decode(&ids, DecodeOptions { remove_special: false, render_special: true }).unwrap();
    let tf = tok.decode(&ids, DecodeOptions { remove_special: true, render_special: false }).unwrap();
    let tt = tok.decode(&ids, DecodeOptions { remove_special: true, render_special: true }).unwrap();
    let body: Vec<u8> = [1u32, 2, 3].iter().flat_map(|&i| tok.token_to_piece(i, false).unwrap().to_vec()).collect();
    // render_special=false hides control tokens but keeps user-defined text
    assert_eq!(ff, [body.clone(), b"<think>".to_vec()].concat());
    assert_eq!(tf, ff);
    // render_special=true shows control tokens; remove_special drops only the leading BOS
    assert_eq!(ft, [b"<|startoftext|><|im_start|>".to_vec(), body.clone(), b"<think><|im_end|>".to_vec()].concat());
    assert_eq!(tt, [b"<|im_start|>".to_vec(), body, b"<think><|im_end|>".to_vec()].concat());
    // pieces are independent of decode state
    assert_eq!(tok.token_to_piece(IM_END, true).unwrap(), b"<|im_end|>");
    assert!(tok.token_to_piece(IM_END, false).unwrap().is_empty());
    assert_eq!(tok.token_to_piece(THINK, false).unwrap(), b"<think>");
}

// ---------------------------------------------------------------- text classes

#[test]
fn unicode_whitespace_code_numbers_roundtrip() {
    let tok = tokenizer();
    let samples = [
        "héllo wörld — ÆØÅ ß ﬁ", "日本語のテキストと中文混合", "한국어 문장입니다", "Ελληνικά και русский текст",
        "emoji 🙂👍🏽 👨‍👩‍👧 🇺🇸", "combining e\u{301} a\u{308} n\u{303}", "zero\u{200b}width\u{feff}joiner",
        "tabs\tand\nnewlines\r\n\n\n   indent", "    leading and trailing    ", "\u{3000}ideographic\u{00a0}nbsp",
        "fn main() { let x: Vec<u8> = vec![1, 2, 3]; println!(\"{x:?}\"); }",
        "def f(a, b):\n    return {'k': [a ** 2, b // 3]}\n", "if (a && b ||!c) { x->y = *p; }",
        "<div class=\"a\">&amp; &lt;tag&gt;</div>", "SELECT * FROM t WHERE id = 42;",
        "1234567890 3.14159 -0.0001 1e-9 0xFF 1,000,000", "12345678901234567890", "٣٤٥ ১২৩ ๑๒๓",
        "DON'T you're I'M we've", "a", " ", "\n", "\u{10FFFF}\u{E000}\u{FFFD}",
    ];
    for s in samples {
        for o in flags() {
            let ids = tok.encode(s.as_bytes(), o).unwrap();
            let back = tok.decode(&ids, DecodeOptions { remove_special: true, render_special: false }).unwrap();
            assert_eq!(String::from_utf8(back).unwrap(), s, "round trip {s:?} {o:?}");
            // concatenated pieces equal the text too (no clean-up needed for these)
            let pieces: Vec<u8> = ids.iter().filter(|&&i| i != BOS).flat_map(|&i| tok.token_to_piece(i, true).unwrap().to_vec()).collect();
            assert_eq!(pieces, s.as_bytes());
        }
    }
    // clean_spaces (reference llama_vocab::impl::detokenize) removes the space
    // before ! ? . , and in some contractions; the CPU oracle shows the same
    // (extra case x-enc-rand-33-add0-parse0)
    let ids = tok.encode(b"a || !c . d ?", EncodeOptions::default()).unwrap();
    assert_eq!(tok.decode(&ids, DecodeOptions::default()).unwrap(), b"a ||!c. d?");
    let pieces: Vec<u8> = ids.iter().flat_map(|&i| tok.token_to_piece(i, true).unwrap().to_vec()).collect();
    assert_eq!(pieces, b"a || !c . d ?");

    // LLAMA3 split: digit runs are pre-tokenized in groups of at most three
    let ids = tok.encode(b"12345678901234567890", EncodeOptions::default()).unwrap();
    for id in ids {
        let p = tok.token_to_piece(id, true).unwrap();
        assert!(p.len() <= 3 && p.iter().all(u8::is_ascii_digit), "{p:?}");
    }
}

#[test]
fn invalid_utf8_input_is_handled_like_reference() {
    let tok = tokenizer();
    // invalid bytes become U+FFFD per byte (reference unicode_cpts_from_utf8)
    let ids = tok.encode(b"a\xffb\xc3", EncodeOptions::default()).unwrap();
    let back = tok.decode(&ids, DecodeOptions::default()).unwrap();
    assert_eq!(back, "a\u{FFFD}b\u{FFFD}".as_bytes());
    // above U+10FFFF: the reference throws, this crate returns an error in every flag mode
    for o in flags() {
        assert!(matches!(tok.encode(b"\xf7\xbf\xbf\xbf", o), Err(TokenizerError::InvalidCodepoint { .. })));
    }
    // encode_str rejects nothing valid
    assert_eq!(tok.encode_str("hi", true, false).unwrap()[0], BOS);
}

#[test]
fn deterministic_random_input_never_panics() {
    let tok = tokenizer();
    let mut state: u64 = 0x5eed_2026_1005;
    let mut next = || {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        state
    };
    let alphabet: Vec<&[u8]> = vec![
        b"a", b"Z", b"7", b" ", b"  ", b"\n", b"\t", b"'s", b"'", b"<", b">", b"|", b"<|im_start|>", b"<|im_end|>",
        b"<think>", b"</think>", b"<|pad|>", "é".as_bytes(), "🙂".as_bytes(), "日".as_bytes(), b"\xff", b"\xc3", b"\xe2\x82", b"\xf4\x90\x80\x80",
    ];
    let mut checked = 0;
    for _ in 0..3000 {
        let n = (next() % 24) as usize;
        let mut s = Vec::new();
        for _ in 0..n {
            s.extend_from_slice(alphabet[(next() % alphabet.len() as u64) as usize]);
        }
        let has_big = s.windows(4).any(|w| w == b"\xf4\x90\x80\x80");
        for o in flags() {
            match tok.encode(&s, o) {
                Ok(ids) => {
                    assert!(!has_big);
                    for id in &ids {
                        assert!((*id as usize) < tok.vocab_size());
                    }
                    let d = tok.decode(&ids, DecodeOptions { remove_special: o.add_special, render_special: true }).unwrap();
                    let _ = d;
                    checked += 1;
                }
                Err(TokenizerError::InvalidCodepoint { .. }) => assert!(has_big),
                Err(e) => panic!("unexpected error {e:?} for {s:?}"),
            }
        }
    }
    assert!(checked > 5_000, "{checked}");
}

// ---------------------------------------------------------------- threads

#[test]
fn tokenizer_is_send_and_sync() {
    fn check<T: Send + Sync>() {}
    check::<Tokenizer>();
    check::<w9100_tokenizer::ChatTemplate>();
}

#[test]
fn concurrent_shared_and_separate_instances() {
    let shared = tokenizer();
    let cases = read_json(&packet_file("fixtures/oracle-cases.json"));
    let enc: Vec<(Vec<u8>, EncodeOptions, Vec<u32>)> = cases
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["op"] == "encode")
        .map(|c| {
            (
                request_text(c),
                EncodeOptions { add_special: c["add_special"].as_bool().unwrap(), parse_special: c["parse_special"].as_bool().unwrap() },
                ids_of(&c["expected_token_ids"]),
            )
        })
        .collect();
    // second, independent instance loaded on another thread at the same time
    let separate = std::thread::scope(|s| {
        let h = s.spawn(|| Tokenizer::from_gguf_bytes(gguf_bytes()).unwrap());
        let h2 = s.spawn(|| Tokenizer::from_gguf_bytes(gguf_bytes()).unwrap());
        (h.join().unwrap(), h2.join().unwrap())
    });
    let instances = [shared, &separate.0, &separate.1];
    std::thread::scope(|s| {
        for t in 0..8 {
            let enc = &enc;
            let tok = instances[t % 3];
            s.spawn(move || {
                for round in 0..3 {
                    for (i, (text, o, exp)) in enc.iter().enumerate() {
                        if (i + t + round) % 2 == 0 {
                            assert_eq!(&tok.encode(text, *o).unwrap(), exp);
                            let d = tok.decode(exp, DecodeOptions { remove_special: false, render_special: true }).unwrap();
                            assert!(!d.is_empty() || exp.is_empty());
                        }
                    }
                }
            });
        }
    });
}
