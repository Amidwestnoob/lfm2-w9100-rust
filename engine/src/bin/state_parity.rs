//! Real-GPU unchanged-vs-bounded resident state comparison. Never long/CPU inference.
use anyhow::{ensure, Result};
use lfm2_engine::{gpu_lfm::GpuEngine, model_file::ModelFile};
use serde_json::json;
use std::path::Path;

fn argmax(xs: &[f32]) -> u32 {
    let (mut best, mut value) = (0, f32::NEG_INFINITY);
    for (i, &v) in xs.iter().enumerate() {
        if v > value {
            best = i as u32;
            value = v;
        }
    }
    best
}
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    ensure!(a.len() == 4, "state_parity MODEL golden.json OUTDIR");
    let model = ModelFile::open(&a[1])?;
    let fixtures: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&a[2])?)?;
    let ids: Vec<u32> = serde_json::from_value(fixtures["cases"][0]["prompt_tokens"].clone())?;
    let out = Path::new(&a[3]);
    std::fs::create_dir_all(out)?;
    let mut cases = Vec::new();
    for n in [2usize, 13, 33] {
        let prompt = &ids[..n];
        let (
            mut logits_mismatches,
            mut cache_mismatches,
            mut logits_elements,
            mut cache_bytes_compared,
        ) = (0usize, 0usize, 0usize, 0usize);
        let mut reference_tokens = Vec::new();
        let mut step_results = Vec::new();
        for bounded in [false, true] {
            // Sequential fresh engines, not two resident models competing for VRAM.
            let mut engine = GpuEngine::load(&model)?;
            engine.set_ctx(n + 32 + 16);
            engine.set_bounded_submissions_for_test(bounded);
            engine.gpu.ctx.set_submission_context("test", n);
            let mut logits = engine.forward(prompt, 0)?;
            for step in 0..=32 {
                engine.gpu.ctx.set_submission_context("test", n);
                let raw: Vec<u8> = logits.iter().flat_map(|x| x.to_le_bytes()).collect();
                let cache = engine.snapshot_resident_cache(n + step)?;
                let prefix = out.join(format!(
                    "n{n}-step{step}-{}",
                    if bounded { "bounded" } else { "unchanged" }
                ));
                std::fs::write(prefix.with_extension("logits.bin"), &raw)?;
                std::fs::write(prefix.with_extension("cache.bin"), &cache)?;
                if bounded {
                    let reference = out.join(format!("n{n}-step{step}-unchanged"));
                    let rl = std::fs::read(reference.with_extension("logits.bin"))?;
                    let rc = std::fs::read(reference.with_extension("cache.bin"))?;
                    ensure!(
                        rl.len() == raw.len() && rc.len() == cache.len(),
                        "snapshot size mismatch"
                    );
                    let lm = raw
                        .chunks_exact(4)
                        .zip(rl.chunks_exact(4))
                        .filter(|(x, y)| x != y)
                        .count();
                    let cm = cache.iter().zip(&rc).filter(|(x, y)| x != y).count();
                    logits_elements += logits.len();
                    cache_bytes_compared += cache.len();
                    logits_mismatches += lm;
                    cache_mismatches += cm;
                    step_results.push(json!({"step":step,"logits_elements":logits.len(),"cache_bytes_compared":cache.len(),"logits_mismatches":lm,"cache_mismatches":cm}));
                    ensure!(
                        lm == 0 && cm == 0,
                        "GPU state/logit parity failure n={n} step={step}"
                    );
                }
                if step < 32 {
                    let token = argmax(&logits);
                    if bounded {
                        ensure!(token == reference_tokens[step], "greedy mismatch");
                    } else {
                        reference_tokens.push(token);
                    }
                    engine.gpu.ctx.set_submission_context(
                        if step == 31 {
                            "final_forward"
                        } else {
                            "decode"
                        },
                        n,
                    );
                    logits = engine.forward(&[token], (n + step) as i64)?;
                }
            }
        }
        cases.push(json!({"prompt_tokens":n,"prompt_token_ids":prompt,"tokens":reference_tokens,"logits_elements":logits_elements,"logits_mismatches":logits_mismatches,"cache_bytes_compared":cache_bytes_compared,"cache_mismatches":cache_mismatches,"final_forward_compared":true,"steps":step_results}));
    }
    println!(
        "{}",
        json!({"pass":true,"cases":cases,"cache_scope":"all initialized attention K/V bytes and every convolution slot, after prefill and every one of 32 generated-token forwards"})
    );
    Ok(())
}
