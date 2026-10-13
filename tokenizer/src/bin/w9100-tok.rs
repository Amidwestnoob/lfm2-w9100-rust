//! `w9100-tok`: CPU command line for the Rust tokenizer, plus the test
//! exporter for the handoff `verify_candidate.py` contract.
//!
//! All tokenization, decoding and template work is done by the library. The
//! export commands only read request inputs (`op`, `text`/`text_hex`, flags,
//! `token_ids`; chat `id` and `settings`). They do not read expected answers.

use std::io::{BufWriter, Read, Write};
use std::process::ExitCode;
use w9100_tokenizer::{ChatTemplate, DecodeOptions, EncodeOptions, StreamDecoder, StreamOptions, Tokenizer, TokenizerError};

const USAGE: &str = "usage:
  w9100-tok info          --model F.gguf
  w9100-tok encode        --model F.gguf [--add-special] [--parse-special] (--text S | --text-hex H | --stdin)
  w9100-tok decode        --model F.gguf --ids 1,2,3 [--remove-special] [--render-special]
  w9100-tok stream        --model F.gguf --ids 1,2,3 [--render-special]
  w9100-tok chat          --model F.gguf --settings ctx.json [--template T.jinja] [--tokens]
  w9100-tok export-cases  --model F.gguf --requests requests.json --out cases.jsonl
  w9100-tok export-pieces --model F.gguf --out pieces.jsonl
  w9100-tok export-chats  (--model F.gguf | --template T.jinja) --cases chat-cases.json --out chats.jsonl";

struct Args {
    cmd: String,
    kv: std::collections::HashMap<String, String>,
    flags: std::collections::HashSet<String>,
}

fn parse_args() -> Result<Args, String> {
    let mut it = std::env::args().skip(1);
    let cmd = it.next().ok_or_else(|| USAGE.to_string())?;
    let mut kv = std::collections::HashMap::new();
    let mut flags = std::collections::HashSet::new();
    let takes_value = ["--model", "--text", "--text-hex", "--ids", "--settings", "--template", "--requests", "--out", "--cases"];
    let mut rest: Vec<String> = it.collect();
    rest.reverse();
    while let Some(a) = rest.pop() {
        if takes_value.contains(&a.as_str()) {
            let v = rest.pop().ok_or_else(|| format!("{a} needs a value"))?;
            kv.insert(a, v);
        } else if a.starts_with("--") {
            flags.insert(a);
        } else {
            return Err(format!("unexpected argument {a}\n{USAGE}"));
        }
    }
    Ok(Args { cmd, kv, flags })
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

fn unhex(s: &str) -> Result<Vec<u8>, String> {
    if !s.bytes().all(|b| b.is_ascii_hexdigit()) {
        return Err("hex input must contain only ASCII hexadecimal digits".into());
    }
    if s.len() % 2 != 0 {
        return Err("odd hex length".into());
    }
    (0..s.len()).step_by(2).map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| e.to_string())).collect()
}

fn ids_arg(tok: &Tokenizer, s: &str) -> Result<Vec<u32>, String> {
    let raw: Result<Vec<i64>, String> = s
        .split(',')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .map(|p| p.parse::<i64>().map_err(|_| format!("bad token id {p:?}")))
        .collect();
    tok.checked_ids(&raw?).map_err(|e| e.to_string())
}

fn need<'a>(a: &'a Args, k: &str) -> Result<&'a str, String> {
    a.kv.get(k).map(String::as_str).ok_or_else(|| format!("missing {k}\n{USAGE}"))
}

fn load(a: &Args) -> Result<Tokenizer, String> {
    Tokenizer::from_gguf_file(need(a, "--model")?).map_err(|e| e.to_string())
}

fn read_json(path: &str) -> Result<serde_json::Value, String> {
    let text = std::fs::read_to_string(path).map_err(|e| format!("read {path}: {e}"))?;
    serde_json::from_str(&text).map_err(|e| format!("parse {path}: {e}"))
}

fn writer(path: &str) -> Result<BufWriter<std::fs::File>, String> {
    Ok(BufWriter::new(std::fs::File::create(path).map_err(|e| format!("create {path}: {e}"))?))
}

fn e2s(e: TokenizerError) -> String {
    e.to_string()
}

fn run() -> Result<(), String> {
    let a = parse_args()?;
    match a.cmd.as_str() {
        "info" => {
            let tok = load(&a)?;
            let st = tok.special_tokens();
            let out = serde_json::json!({
                "model": tok.model(), "pre": tok.pre(), "vocab_size": tok.vocab_size(),
                "bos": st.bos, "eos": st.eos, "eot": st.eot, "pad": st.pad, "linefeed": st.linefeed,
                "add_bos": tok.add_bos(), "add_eos": tok.add_eos(), "ignore_merges": tok.ignore_merges(),
                "clean_spaces": tok.clean_spaces(), "eog": tok.eog_tokens(),
                "has_chat_template": tok.chat_template().is_some(),
                "notes": {"ambiguous_autodetect": tok.load_notes().ambiguous_autodetect,
                          "ambiguous_special_order": tok.load_notes().ambiguous_special_order,
                          "bad_special_ids": tok.load_notes().bad_special_ids},
            });
            println!("{out}");
        }
        "encode" => {
            let tok = load(&a)?;
            let text: Vec<u8> = if let Some(t) = a.kv.get("--text") {
                t.as_bytes().to_vec()
            } else if let Some(h) = a.kv.get("--text-hex") {
                unhex(h)?
            } else if a.flags.contains("--stdin") {
                let mut b = Vec::new();
                std::io::stdin().read_to_end(&mut b).map_err(|e| e.to_string())?;
                b
            } else {
                return Err(format!("encode needs --text, --text-hex or --stdin\n{USAGE}"));
            };
            let opts = EncodeOptions { add_special: a.flags.contains("--add-special"), parse_special: a.flags.contains("--parse-special") };
            let ids = tok.encode(&text, opts).map_err(e2s)?;
            println!("{}", serde_json::json!({ "token_ids": ids }));
        }
        "decode" => {
            let tok = load(&a)?;
            let ids = ids_arg(&tok, need(&a, "--ids")?)?;
            let opts = DecodeOptions { remove_special: a.flags.contains("--remove-special"), render_special: a.flags.contains("--render-special") };
            let bytes = tok.decode(&ids, opts).map_err(e2s)?;
            let text = String::from_utf8(bytes.clone()).ok();
            println!("{}", serde_json::json!({ "bytes_hex": hex(&bytes), "text": text }));
        }
        "stream" => {
            let tok = load(&a)?;
            let ids = ids_arg(&tok, need(&a, "--ids")?)?;
            let mut dec = StreamDecoder::new(&tok, StreamOptions { render_special: a.flags.contains("--render-special"), ..Default::default() });
            for id in ids {
                let chunk = dec.push_token(id).map_err(e2s)?;
                println!("{}", serde_json::json!({ "token_id": id, "text": chunk, "pending_hex": hex(dec.pending()) }));
            }
            let tail = dec.finish().map_err(e2s)?;
            println!("{}", serde_json::json!({ "finish": tail }));
        }
        "chat" => {
            let tok = load(&a)?;
            let tmpl = match a.kv.get("--template") {
                Some(p) => ChatTemplate::new(&std::fs::read_to_string(p).map_err(|e| e.to_string())?).map_err(e2s)?,
                None => ChatTemplate::from_tokenizer(&tok).map_err(e2s)?,
            };
            let ctx = read_json(need(&a, "--settings")?)?;
            let text = tmpl.render_json(&ctx).map_err(e2s)?;
            if a.flags.contains("--tokens") {
                let ids = tok.encode(text.as_bytes(), EncodeOptions { add_special: false, parse_special: true }).map_err(e2s)?;
                println!("{}", serde_json::json!({ "text": text, "token_ids": ids }));
            } else {
                println!("{}", serde_json::json!({ "text": text }));
            }
        }
        "export-cases" => {
            let tok = load(&a)?;
            let reqs = read_json(need(&a, "--requests")?)?;
            let reqs = reqs.as_array().ok_or("requests must be a JSON array")?;
            let mut w = writer(need(&a, "--out")?)?;
            for r in reqs {
                let id = r.get("id").and_then(|v| v.as_str()).ok_or("request without id")?;
                let flag = |k: &str| r.get(k).and_then(|v| v.as_bool()).ok_or(format!("{id}: missing bool {k}"));
                let line = match r.get("op").and_then(|v| v.as_str()) {
                    Some("encode") => {
                        let text = if let Some(h) = r.get("text_hex").and_then(|v| v.as_str()) {
                            unhex(h)?
                        } else {
                            r.get("text").and_then(|v| v.as_str()).ok_or(format!("{id}: missing text"))?.as_bytes().to_vec()
                        };
                        let opts = EncodeOptions { add_special: flag("add_special")?, parse_special: flag("parse_special")? };
                        match tok.encode(&text, opts) {
                            Ok(ids) => serde_json::json!({ "id": id, "token_ids": ids }),
                            Err(e) => serde_json::json!({ "id": id, "error": e.to_string() }),
                        }
                    }
                    Some("decode") => {
                        let raw: Vec<i64> = r.get("token_ids").and_then(|v| v.as_array()).ok_or(format!("{id}: missing token_ids"))?
                            .iter().map(|x| x.as_i64().ok_or(format!("{id}: non-integer token id"))).collect::<Result<_, _>>()?;
                        let opts = DecodeOptions { remove_special: flag("remove_special")?, render_special: flag("render_special")? };
                        match tok.checked_ids(&raw).and_then(|ids| tok.decode(&ids, opts)) {
                            Ok(b) => serde_json::json!({ "id": id, "bytes_hex": hex(&b) }),
                            Err(e) => serde_json::json!({ "id": id, "error": e.to_string() }),
                        }
                    }
                    other => return Err(format!("{id}: unsupported op {other:?}")),
                };
                writeln!(w, "{line}").map_err(|e| e.to_string())?;
            }
            w.flush().map_err(|e| e.to_string())?;
            eprintln!("exported {} cases", reqs.len());
        }
        "export-pieces" => {
            let tok = load(&a)?;
            let mut w = writer(need(&a, "--out")?)?;
            for id in 0..tok.vocab_size() as u32 {
                let on = tok.token_to_piece(id, true).map_err(e2s)?;
                let off = tok.token_to_piece(id, false).map_err(e2s)?;
                writeln!(w, "{}", serde_json::json!({ "token_id": id, "render_special_true_hex": hex(on), "render_special_false_hex": hex(off) }))
                    .map_err(|e| e.to_string())?;
            }
            w.flush().map_err(|e| e.to_string())?;
            eprintln!("exported {} pieces", tok.vocab_size());
        }
        "export-chats" => {
            let tmpl = if let Some(p) = a.kv.get("--template") {
                ChatTemplate::new(&std::fs::read_to_string(p).map_err(|e| e.to_string())?).map_err(e2s)?
            } else {
                ChatTemplate::from_tokenizer(&load(&a)?).map_err(e2s)?
            };
            // Only `id` and `settings` are read from each case.
            #[derive(serde::Deserialize)]
            struct CaseInput {
                id: String,
                settings: serde_json::Value,
            }
            let text = std::fs::read_to_string(need(&a, "--cases")?).map_err(|e| e.to_string())?;
            let cases: Vec<CaseInput> = serde_json::from_str(&text).map_err(|e| e.to_string())?;
            let mut w = writer(need(&a, "--out")?)?;
            for c in &cases {
                let line = match tmpl.render_json(&c.settings) {
                    Ok(t) => serde_json::json!({ "id": c.id, "text": t }),
                    Err(e) => serde_json::json!({ "id": c.id, "error": e.to_string() }),
                };
                writeln!(w, "{line}").map_err(|e| e.to_string())?;
            }
            w.flush().map_err(|e| e.to_string())?;
            eprintln!("exported {} chats", cases.len());
        }
        "-h" | "--help" | "help" => println!("{USAGE}"),
        other => return Err(format!("unknown command {other}\n{USAGE}")),
    }
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::from(2)
        }
    }
}
