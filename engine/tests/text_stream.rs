//! Offline integration regression: the accepted native token fixtures are test
//! inputs only. Production code neither reads them nor forwards to an oracle.
use lfm2_engine::text_output::DecodedText;
use std::path::Path;
use w9100_tokenizer::{
    DecodeOptions, EncodeOptions, InvalidUtf8Policy, StreamDecoder, StreamOptions, Tokenizer,
};

#[test]
fn cleanup_matches_accepted_decoder_across_adversarial_chunk_boundaries() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let tok = Tokenizer::from_gguf_file(root.join("model/LFM2.5-8B-A1B-Q6_K.gguf")).unwrap();
    let mut samples = vec![
        "Hello , world ! I 'm fine . We 're ready ?".to_owned(),
        "long prefix to force incremental emission: ' ' ' , . 's 'm 're 've 't 'd 'll".to_owned(),
        "Unicode café 🦀 ' 日本語 , hello .".repeat(8),
    ];
    let alphabet = [
        ' ', ' ', '\'', 's', 'm', 'r', 'e', 'v', 'l', 't', 'd', '!', ',', '.', '?', 'a', 'é', '🦀',
    ];
    let mut state = 0x12345678u64;
    for _ in 0..1000 {
        let mut sample = String::new();
        for _ in 0..120 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            sample.push(alphabet[state as usize % alphabet.len()]);
        }
        samples.push(sample);
    }
    for sample in samples {
        let ids = tok
            .encode(
                sample.as_bytes(),
                EncodeOptions {
                    add_special: false,
                    parse_special: false,
                },
            )
            .unwrap();
        let expected = tok
            .decode(
                &ids,
                DecodeOptions {
                    remove_special: true,
                    render_special: false,
                },
            )
            .unwrap();
        // One character at a time exercises cleanup boundaries independently of BPE.
        let mut cleanup = DecodedText::new(tok.clean_spaces());
        let mut text = String::new();
        for c in sample.chars() {
            text.push_str(&cleanup.push(&c.to_string()));
        }
        text.push_str(&cleanup.finish(""));
        assert_eq!(text.as_bytes(), expected, "{sample:?}");
        // Then exercise real BPE token pieces and StreamDecoder UTF-8 buffering.
        let mut decoder = StreamDecoder::new(&tok, StreamOptions::default());
        let mut cleanup = DecodedText::new(tok.clean_spaces());
        let mut text = String::new();
        for id in ids {
            text.push_str(&cleanup.push(&decoder.push_token(id).unwrap()));
        }
        text.push_str(&cleanup.finish(&decoder.finish().unwrap()));
        assert_eq!(text.as_bytes(), expected, "{sample:?}");
    }
}

#[test]
fn all_native_sequences_decode_identically_incrementally() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).parent().unwrap();
    let tok = Tokenizer::from_gguf_file(root.join("model/LFM2.5-8B-A1B-Q6_K.gguf")).unwrap();
    let fixture: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(root.join("model/golden.json")).unwrap())
            .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let ids: Vec<u32> = case["output_tokens"]
            .as_array()
            .unwrap()
            .iter()
            .map(|x| u32::try_from(x.as_u64().unwrap()).unwrap())
            .collect();
        let mut decoder = StreamDecoder::new(
            &tok,
            StreamOptions {
                render_special: false,
                invalid: InvalidUtf8Policy::Replace,
            },
        );
        let mut streamed = String::new();
        let mut cleanup = DecodedText::new(tok.clean_spaces());
        for &id in &ids {
            streamed.push_str(&cleanup.push(&decoder.push_token(id).unwrap()));
        }
        streamed.push_str(&cleanup.finish(&decoder.finish().unwrap()));
        let decoded = tok
            .decode(
                &ids,
                DecodeOptions {
                    remove_special: true,
                    render_special: false,
                },
            )
            .unwrap();
        assert_eq!(
            streamed,
            String::from_utf8_lossy(&decoded),
            "{}",
            case["name"]
        );
    }
}
