//! Sequential JSON-lines persistent engine. No network listener or deployment.
use anyhow::{ensure, Result};
use lfm2_engine::{gpu_lfm::GpuEngine, model_file::ModelFile};
use serde::Deserialize;
use serde_json::json;
use std::{
    collections::HashMap,
    io::{BufRead, Write},
    time::Instant,
};
use w9100_tokenizer::{ChatTemplate, DecodeOptions, EncodeOptions, Tokenizer};
const CONTEXT: usize = 4096;
const GENERATION: usize = 1024;
#[derive(Deserialize)]
struct Request {
    session_id: String,
    action: String,
    #[serde(default)]
    tokens: Option<Vec<u32>>,
    #[serde(default)]
    text: Option<String>,
    #[serde(default)]
    chat: Option<serde_json::Value>,
    n: usize,
    #[serde(default)]
    eos: Option<u32>,
}
struct Session {
    prefix: Vec<u32>,
    cache: Vec<u8>,
    next: u32,
}
fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}
fn main() -> Result<()> {
    let cold = Instant::now();
    let mut args = std::env::args().skip(1);
    let mut model = None;
    let mut file = None;
    let mut soak_seconds = 0.0f64;
    while let Some(a) = args.next() {
        match a.as_str() {
            "--model" => model = args.next(),
            "--requests" => file = args.next(),
            "--soak-seconds" => {
                soak_seconds = args
                    .next()
                    .ok_or_else(|| anyhow::anyhow!("missing duration"))?
                    .parse()?
            }
            _ => anyhow::bail!("unknown argument {a}"),
        }
    }
    let model = model.ok_or_else(|| anyhow::anyhow!("--model required"))?;
    let tok = Tokenizer::from_gguf_file(&model)?;
    let mf = ModelFile::open(&model)?;
    // Model weights + context/working buffers must fit alongside production.
    let vram = std::path::Path::new("/sys/class/drm/card0/device");
    let total: u64 = std::fs::read_to_string(vram.join("mem_info_vram_total"))?
        .trim()
        .parse()?;
    let used: u64 = std::fs::read_to_string(vram.join("mem_info_vram_used"))?
        .trim()
        .parse()?;
    let weight_bytes: u64 = mf.tensors.iter().map(|t| if std::env::var("LFM_Q6_ALIGNED").as_deref()==Ok("1") && t.ggml_type==14 {t.nbytes/210*212} else {t.nbytes}).sum();
    ensure!(
        total.saturating_sub(used) > weight_bytes + 768 * 1024 * 1024,
        "insufficient co-resident VRAM headroom"
    );
    let mut engine = GpuEngine::load(&mf)?;
    engine.set_ctx(CONTEXT);
    let load_ms = ms(cold);
    let input: Box<dyn BufRead> = if let Some(f) = file {
        Box::new(std::io::BufReader::new(std::fs::File::open(f)?))
    } else {
        Box::new(std::io::BufReader::new(std::io::stdin()))
    };
    ensure!(
        soak_seconds.is_finite() && (0.0..=720.0).contains(&soak_seconds),
        "soak duration out of range"
    );
    let input: Box<dyn Iterator<Item = std::io::Result<String>>> = if soak_seconds > 0.0 {
        let lines: Vec<String> = input.lines().collect::<std::io::Result<_>>()?;
        ensure!(!lines.is_empty(), "empty soak requests");
        Box::new(lines.into_iter().cycle().map(Ok))
    } else {
        Box::new(input.lines())
    };
    let soak_start = Instant::now();
    let mut sessions: HashMap<String, Session> = HashMap::new();
    let mut active: Option<String> = None;
    let mut requests = 0usize;
    for line in input {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let request_start = Instant::now();
        let r: Request = serde_json::from_str(&line)?;
        ensure!(
            !r.session_id.is_empty() && r.session_id.len() <= 128,
            "invalid session ID"
        );
        ensure!(
            ["fresh", "continue", "reset", "interleave"].contains(&r.action.as_str()),
            "invalid action"
        );
        ensure!(r.n <= GENERATION, "generation exceeds 1024");
        ensure!(
            usize::from(r.tokens.is_some())
                + usize::from(r.text.is_some())
                + usize::from(r.chat.is_some())
                <= 1,
            "ambiguous input"
        );
        let suffix = if let Some(ids) = r.tokens {
            ids
        } else if let Some(text) = r.text {
            tok.encode(
                text.as_bytes(),
                EncodeOptions {
                    add_special: r.action == "fresh" || r.action == "reset",
                    parse_special: false,
                },
            )?
        } else if let Some(chat) = r.chat {
            let text = ChatTemplate::from_tokenizer(&tok)?.render_json(&chat)?;
            tok.encode(
                text.as_bytes(),
                EncodeOptions {
                    add_special: false,
                    parse_special: true,
                },
            )?
        } else {
            vec![]
        };
        ensure!(
            suffix.iter().all(|&id| (id as usize) < engine.hp.vocab),
            "out-of-range token"
        );
        let new = r.action == "fresh"
            || r.action == "reset"
            || (r.action == "interleave" && !sessions.contains_key(&r.session_id));
        ensure!(
            r.action != "fresh" || !sessions.contains_key(&r.session_id),
            "fresh session already exists"
        );
        ensure!(
            new || sessions.contains_key(&r.session_id),
            "unknown continuation session"
        );
        let old_len = if new {
            0
        } else {
            sessions[&r.session_id].prefix.len()
        };
        ensure!(
            old_len + suffix.len() > 0 && old_len + suffix.len() + r.n <= CONTEXT,
            "context exceeds 4096 or empty prompt"
        );
        ensure!(
            sessions.contains_key(&r.session_id) || sessions.len() < 8,
            "at most eight sessions"
        );
        if active.as_ref() != Some(&r.session_id) {
            if let Some(id) = &active {
                let s = sessions.get_mut(id).unwrap();
                s.cache = engine.snapshot_resident_cache(s.prefix.len())?;
            }
            if !new {
                let s = &sessions[&r.session_id];
                engine.restore_session(s.prefix.len(), &s.cache)?;
            }
        }
        if new {
            engine.reset_session()?;
            sessions.insert(
                r.session_id.clone(),
                Session {
                    prefix: vec![],
                    cache: vec![],
                    next: 0,
                },
            );
        }
        active = Some(r.session_id.clone());
        let s = sessions.get_mut(&r.session_id).unwrap();
        let prefill_start = Instant::now();
        engine
            .gpu
            .ctx
            .set_submission_context("prefill", old_len + suffix.len());
        for chunk in suffix.chunks(lfm2_engine::work_bound::PREFILL_CHUNK) {
            let profile_start = lfm2_engine::vulkan::profile_clock_ns();
            s.next = engine.forward_token(chunk, s.prefix.len())?;
            engine.gpu.ctx.profile_forward(profile_start, lfm2_engine::vulkan::profile_clock_ns(), s.prefix.len(), chunk.len())?;
            s.prefix.extend_from_slice(chunk);
        }
        let prefill_ms = ms(prefill_start);
        let prompt = s.prefix.clone();
        let first_token_ms = ms(request_start);
        let generation_start = Instant::now();
        let mut out = Vec::new();
        for _ in 0..r.n {
            let token = s.next;
            if token == r.eos.unwrap_or(engine.hp.eos) {
                break;
            }
            out.push(token);
            engine
                .gpu
                .ctx
                .set_submission_context("decode", prompt.len());
            let profile_start = lfm2_engine::vulkan::profile_clock_ns();
            s.next = engine.forward_token(&[token], s.prefix.len())?;
            engine.gpu.ctx.profile_forward(profile_start, lfm2_engine::vulkan::profile_clock_ns(), s.prefix.len(), 1)?;
            s.prefix.push(token);
        }
        let decode_ms = ms(generation_start);
        let decoded = tok.decode(
            &out,
            DecodeOptions {
                remove_special: true,
                render_special: false,
            },
        )?;
        requests += 1;
        println!(
            "{}",
            json!({"type":"request_result","session_id":r.session_id,"action":r.action,
            "process_id":std::process::id(),"model_load_count":1,"request_count":requests,"session_count":sessions.len(),
            "prompt_tokens":prompt,"tokens":out,"greedy":true,"decoded_text":String::from_utf8_lossy(&decoded),
            "device":engine.device_name(),"load_ms":load_ms,"prefill_ms":prefill_ms,"decode_ms":decode_ms,
            "time_to_first_token_ms":first_token_ms,"cold_first_token_ms":load_ms+first_token_ms,
            "request_ms":ms(request_start),"cold_request_ms":load_ms+ms(request_start),
            "gpu_argmax_dispatches":engine.gpu.gpu_argmax_dispatches.get(),"logit_readback_bytes_per_token":8,
            "decode_command_buffer_reuses":engine.gpu.ctx.decode_command_buffer_reuses.get(),
            "prefill_weight_reuse_tokens":engine.gpu.prefill_weight_reuse_tokens.get(),
            "expert_grouped_dispatches":engine.gpu.expert_grouped_dispatches.get(),"max_context":CONTEXT,"max_generation":GENERATION})
        );
        std::io::stdout().flush()?;
        if soak_seconds > 0.0 && soak_start.elapsed().as_secs_f64() >= soak_seconds && requests >= 8
        {
            let status = std::env::var("LFM_MONITOR_STATUS")?;
            let monitor: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(status)?)?;
            ensure!(
                monitor["new_gpu_faults"] == 0 && monitor["protected_health_failures"] == 0,
                "supervisor unhealthy"
            );
            println!(
                "{}",
                json!({"type":"soak_result","elapsed_seconds":soak_start.elapsed().as_secs_f64(),"completed_requests":requests,"model_load_count":1,"new_gpu_faults":monitor["new_gpu_faults"],"protected_health_failures":monitor["protected_health_failures"],"supervisor_status":monitor})
            );
            break;
        }
    }
    Ok(())
}
