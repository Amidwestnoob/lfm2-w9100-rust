//! Rust tokenizer/chat -> Rust ash/Vulkan engine -> Rust incremental decoder.
//! `--tokens` remains supported. Text and chat always use the real GPU path;
//! `--cpu --tokens` is an explicit retained diagnostic reference, never fallback.
use anyhow::{anyhow, bail, ensure, Result};
use lfm2_engine::text_output::DecodedText;
use lfm2_engine::{gpu_lfm::GpuEngine, lfm::Engine, model_file::ModelFile};
use std::io::Write;
use std::time::Instant;
use w9100_tokenizer::{
    ChatTemplate, DecodeOptions, EncodeOptions, InvalidUtf8Policy, StreamDecoder, StreamOptions,
    Tokenizer,
};

#[derive(Debug)]
enum Input {
    Tokens(String),
    Text(String),
    Chat(String),
}
#[derive(Debug)]
struct Args {
    model: String,
    input: Input,
    n: usize,
    eos: u32,
    stream: bool,
    cpu: bool,
}
fn parse_args(args: impl IntoIterator<Item = String>) -> Result<Args> {
    let mut args = args.into_iter();
    let mut model = None;
    let mut input = None;
    let (mut n, mut eos, mut stream, mut cpu) = (64, 124900, false, false);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--stream" => stream = true,
            "--cpu" => cpu = true,
            "--model" | "--tokens" | "--text" | "--chat-json" | "--n" | "--eos" => {
                let value = args.next().ok_or_else(|| anyhow!("{arg} needs a value"))?;
                match arg.as_str() {
                    "--model" => model = Some(value),
                    "--n" => n = value.parse()?,
                    "--eos" => eos = value.parse()?,
                    _ => {
                        ensure!(
                            input.is_none(),
                            "choose exactly one of --tokens, --text, --chat-json"
                        );
                        input = Some(match arg.as_str() {
                            "--tokens" => Input::Tokens(value),
                            "--text" => Input::Text(value),
                            _ => Input::Chat(value),
                        });
                    }
                }
            }
            _ => bail!("unknown arg {arg}"),
        }
    }
    let input = input.ok_or_else(|| anyhow!("need --tokens, --text or --chat-json"))?;
    ensure!(
        !cpu || (matches!(input, Input::Tokens(_)) && !stream),
        "--cpu is only an explicit --tokens diagnostic; text/chat/stream require the GPU"
    );
    Ok(Args {
        model: model.ok_or_else(|| anyhow!("need --model"))?,
        input,
        n,
        eos,
        stream,
        cpu,
    })
}
fn parse_tokens(s: &str) -> Result<Vec<u32>> {
    s.split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(|s| s.parse().map_err(|_| anyhow!("bad token id {s:?}")))
        .collect()
}
fn prompt_tokens(tok: &Tokenizer, input: &Input) -> Result<Vec<u32>> {
    let ids = match input {
        Input::Tokens(s) => parse_tokens(s)?,
        Input::Text(s) => tok.encode(
            s.as_bytes(),
            EncodeOptions {
                add_special: true,
                parse_special: false,
            },
        )?,
        Input::Chat(path) => {
            let settings: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(path)?)?;
            let template = ChatTemplate::from_tokenizer(tok)?;
            let rendered = template.render_json(&settings)?;
            tok.encode(
                rendered.as_bytes(),
                EncodeOptions {
                    add_special: false,
                    parse_special: true,
                },
            )?
        }
    };
    ensure!(!ids.is_empty(), "prompt token list must not be empty");
    ensure!(
        ids.iter().all(|&id| (id as usize) < tok.vocab_size()),
        "prompt contains an out-of-range token ID"
    );
    Ok(ids)
}
fn argmax(logits: &[f32]) -> u32 {
    // Independent lane reductions allow SIMD compare/select without changing
    // greedy ties, NaN handling, or any floating-point arithmetic.
    let mut values = [f32::NEG_INFINITY; 8];
    let mut indices = [0usize; 8];
    let mut chunks = logits.chunks_exact(8);
    for (block, chunk) in chunks.by_ref().enumerate() {
        for lane in 0..8 {
            if chunk[lane] > values[lane] {
                values[lane] = chunk[lane];
                indices[lane] = block * 8 + lane;
            }
        }
    }
    let (mut best, mut value) = (0, f32::NEG_INFINITY);
    for lane in 0..8 {
        if values[lane] > value || (values[lane] == value && indices[lane] < best) {
            best = indices[lane];
            value = values[lane];
        }
    }
    let offset = logits.len() - chunks.remainder().len();
    for (i, &v) in chunks.remainder().iter().enumerate() {
        if v > value {
            best = offset + i;
            value = v;
        }
    }
    best as u32
}
fn emit_delta(delta: &str) -> Result<()> {
    // Flush on every generated token, not once after inference has completed.
    let mut stdout = std::io::stdout().lock();
    serde_json::to_writer(&mut stdout, &serde_json::json!({"text_delta": delta}))?;
    writeln!(stdout)?;
    stdout.flush()?;
    Ok(())
}
fn retain_logits(step: usize, logits: &[f32]) -> Result<()> {
    if let Ok(path) = std::env::var("LFM_FORWARD_OUT_DIR") {
        let bytes: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(
            std::path::Path::new(&path).join(format!("forward-{step}.logits.bin")),
            bytes,
        )?;
    }
    Ok(())
}
fn main() -> Result<()> {
    let load_start = Instant::now();
    let tokenize_start = Instant::now();
    let args = parse_args(std::env::args().skip(1))?;
    let tokenizer = Tokenizer::from_gguf_file(&args.model)?;
    let prompt = prompt_tokens(&tokenizer, &args.input)?;
    let tokenize_ms = tokenize_start.elapsed().as_secs_f64() * 1000.0;
    let ctx = prompt
        .len()
        .checked_add(args.n)
        .and_then(|n| n.checked_add(16))
        .ok_or_else(|| anyhow!("context length overflow"))?;
    let mf = ModelFile::open(&args.model)?;
    let mut eos = args.eos;
    if args.cpu {
        let mut engine = Engine::load(&mf)?;
        if eos == 124900 {
            eos = engine.hp.eos;
        }
        engine.set_ctx(ctx);
        let mut logits = engine.forward(&prompt, 0)?;
        let mut out = Vec::new();
        for step in 0..args.n {
            let token = argmax(&logits);
            if token == eos {
                break;
            }
            out.push(token);
            logits = engine.forward(&[token], (prompt.len() + step) as i64)?;
        }
        println!(
            "{}",
            serde_json::json!({"greedy": true, "device": "cpu-reference", "eos": eos, "n_vocab": engine.hp.vocab, "tokens": out})
        );
        return Ok(());
    }
    let mut engine = GpuEngine::load(&mf)?;
    if std::env::var("LFM_TRACE").is_ok() {
        eprintln!("[trace] load complete, device={}", engine.gpu.device_name);
    }
    if eos == 124900 {
        eos = engine.hp.eos;
    }
    engine.set_ctx(ctx);
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let prompt_start = Instant::now();
    engine
        .gpu
        .ctx
        .set_submission_context("prefill", prompt.len());
    let mut logits = engine.forward(&prompt, 0)?;
    let prompt_ms = prompt_start.elapsed().as_secs_f64() * 1000.0;
    retain_logits(0, &logits)?;
    if let Ok(path) = std::env::var("LFM_LOGITS_OUT") {
        let bytes: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
        std::fs::write(path, bytes)?;
    }
    let mut decoder = StreamDecoder::new(
        &tokenizer,
        StreamOptions {
            render_special: false,
            invalid: InvalidUtf8Policy::Replace,
        },
    );
    let mut out = Vec::new();
    let mut streamed = String::new();
    let mut cleanup = DecodedText::new(tokenizer.clean_spaces());
    let gen_start = Instant::now();
    for step in 0..args.n {
        let token = argmax(&logits);
        if token == eos {
            break;
        }
        out.push(token);
        let delta = cleanup.push(&decoder.push_token(token)?);
        streamed.push_str(&delta);
        if args.stream {
            emit_delta(&delta)?;
        }
        engine.gpu.ctx.set_submission_context(
            if step + 1 == args.n {
                "final_forward"
            } else {
                "decode"
            },
            prompt.len(),
        );
        logits = engine.forward(&[token], (prompt.len() + step) as i64)?;
        retain_logits(step + 1, &logits)?;
    }
    let gen_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
    let final_delta = cleanup.finish(&decoder.finish()?);
    streamed.push_str(&final_delta);
    if args.stream && !final_delta.is_empty() {
        emit_delta(&final_delta)?;
    }
    let decoded = tokenizer.decode(
        &out,
        DecodeOptions {
            remove_special: true,
            render_special: false,
        },
    )?;
    let decoded_text = String::from_utf8_lossy(&decoded);
    // The incremental cleanup must match the accepted whole-sequence decoder.
    ensure!(
        decoded_text == streamed,
        "incremental and whole-sequence decoding differ"
    );
    eprintln!(
        "BENCH_TIMING {}",
        serde_json::json!({"load_ms":load_ms,"tokenize_ms":tokenize_ms,"warmup_ms":0.0,"prompt_ms":prompt_ms,"gen_ms":gen_ms,"n_prompt_tokens":prompt.len(),"n_gen_tokens":out.len()})
    );
    println!(
        "{}",
        serde_json::json!({
            "prompt_tokens": prompt, "tokens": out, "decoded_text": decoded_text,
            "greedy": true, "device": engine.gpu.device_name,
            "gpu_dispatch_count": engine.gpu.dispatch_count(), "eos": eos, "n_vocab": engine.hp.vocab
        })
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn lane_argmax_matches_scalar_including_ties_nan_and_tails() {
        fn scalar(xs: &[f32]) -> u32 {
            let (mut best, mut value) = (0, f32::NEG_INFINITY);
            for (i, &v) in xs.iter().enumerate() {
                if v > value {
                    best = i as u32;
                    value = v;
                }
            }
            best
        }
        let mut seed = 17u32;
        for n in 0..257 {
            let mut xs = Vec::new();
            for _ in 0..n {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                xs.push(f32::from_bits(seed));
            }
            assert_eq!(argmax(&xs), scalar(&xs));
            for v in [f32::NAN, f32::NEG_INFINITY, f32::INFINITY, 0.0, -0.0, 1.0] {
                xs.fill(v);
                assert_eq!(argmax(&xs), scalar(&xs));
            }
            if n > 8 {
                xs[7] = 2.0;
                xs[8] = 2.0;
                assert_eq!(argmax(&xs), 7);
            }
        }
    }
    fn parse(a: &[&str]) -> Result<Args> {
        parse_args(a.iter().map(|s| s.to_string()))
    }
    #[test]
    fn cli_preserves_tokens_and_adds_text_chat_stream() {
        let a = parse(&["--model", "m", "--tokens", "1,2", "--n", "0"]).unwrap();
        assert_eq!(a.n, 0);
        assert!(matches!(a.input, Input::Tokens(_)));
        let a = parse(&["--model", "m", "--text", "", "--stream"]).unwrap();
        assert!(a.stream && matches!(a.input, Input::Text(_)));
        assert!(matches!(
            parse(&["--model", "m", "--chat-json", "c"]).unwrap().input,
            Input::Chat(_)
        ));
    }
    #[test]
    fn cli_rejects_ambiguous_missing_and_cpu_text() {
        for args in [
            vec!["--model"],
            vec!["--model", "m", "--tokens", "1", "--text", "hi"],
            vec!["--model", "m", "--text", "hi", "--cpu"],
            vec!["--model", "m", "--tokens", "1", "--n", "-1"],
        ] {
            assert!(parse(&args).is_err());
        }
        assert!(parse_tokens("1,-1").is_err());
        assert_eq!(parse_tokens("1, 2").unwrap(), [1, 2]);
    }
}
