//! `q6_check` — the first real Q6_K multiply on the W9100, checked against the
//! pure-Rust CPU reference (`q6k::q6k_matvec`).
//!
//! Reads real Q6_K bytes straight out of the model file, multiplies them by a
//! deterministic f32 activation on the GPU via the fused dequantize+matmul
//! typed Rust IR kernel, and compares the result against the CPU dequantize-then-dot
//! reference. Reporting a tight relative/L2 match is REQUIRED to declare the
//! hardware Q6_K path correct; this is not a benchmark.
//!
//! Usage:
//!   q6_check --model PATH [--tensor blk.0.attn_q.weight] [--rows N] [--out out.json]

use anyhow::{anyhow, Result};
use lfm2_engine::model_file::ModelFile;
use lfm2_engine::ops::GpuOps;
use lfm2_engine::q6k::{self, QK_K_BYTES};
use std::io::Write;

fn parse_arg(args: &[String], i: &mut usize, name: &str) -> Option<String> {
    if args[*i] == name {
        *i += 1;
        Some(args[*i].clone())
    } else {
        None
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model: Option<String> = None;
    let mut tensor = "blk.0.attn_q.weight".to_string();
    let mut rows = 256usize;
    let mut n = 1usize;
    let mut out_json: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        if let Some(v) = parse_arg(&args, &mut i, "--model") {
            model = Some(v);
        } else if let Some(v) = parse_arg(&args, &mut i, "--tensor") {
            tensor = v;
        } else if let Some(v) = parse_arg(&args, &mut i, "--rows") {
            rows = v.parse()?;
        } else if let Some(v) = parse_arg(&args, &mut i, "--n") {
            n = v.parse()?;
        } else if let Some(v) = parse_arg(&args, &mut i, "--out") {
            out_json = Some(v);
        } else {
            return Err(anyhow!("unknown arg {}", args[i]));
        }
        i += 1;
    }

    let model = model.ok_or_else(|| anyhow!("need --model"))?;
    let mf = ModelFile::open(&model)?;
    let t = mf.tensor(&tensor)?;
    if t.ggml_type != 14 {
        return Err(anyhow!(
            "tensor {tensor} is ggml type {}, not Q6_K (14)",
            t.ggml_type
        ));
    }
    let in_dim = t.ne[0] as usize;
    if in_dim % q6k::QK_K != 0 {
        return Err(anyhow!("in_dim {in_dim} not a multiple of {}", q6k::QK_K));
    }
    let out_dim = rows.min(t.ne[1] as usize);
    let blocks_per_row = in_dim / q6k::QK_K;
    let row_bytes = blocks_per_row * QK_K_BYTES;
    let wbytes = mf.read_tensor_range(&tensor, 0, (row_bytes * out_dim) as u64)?;

    // Deterministic activation vector (in_dim * n).
    let x: Vec<f32> = (0..in_dim * n)
        .map(|k| {
            let kk = k % in_dim;
            let v = (kk as f32) * 0.03125f32;
            (v.sin() * 0.5 + (kk % 3) as f32 * 0.01 - 0.25) as f32
        })
        .collect();

    // CPU reference (dequantize + f64 dot per row, per column).
    let mut refv = vec![0f32; out_dim * n];
    for j in 0..n {
        let xcol = &x[j * in_dim..(j + 1) * in_dim];
        let mut col = vec![0f32; out_dim];
        q6k::q6k_matvec(&wbytes, in_dim, out_dim, xcol, &mut col);
        refv[j * out_dim..(j + 1) * out_dim].copy_from_slice(&col);
    }

    // GPU.
    let mut gpu = GpuOps::open()?;
    let wbuf = gpu.upload(&wbytes)?;
    let xbuf = gpu.upload(bytemuck::cast_slice(&x))?;
    let obuf = gpu.alloc_host(out_dim * n * 4)?;
    gpu.q6k_matmul(&wbuf, 0, in_dim, out_dim, &xbuf, n, &obuf)?;

    let mut gout = vec![0u8; out_dim * n * 4];
    obuf.read(0, &mut gout);
    let gvalues: Vec<f32> = bytemuck::cast_slice(&gout).to_vec();

    // Compare.
    let mut max_abs = 0f64;
    let mut max_rel = 0f64;
    let mut sum_sq = 0f64;
    let mut sum_ref_sq = 0f64;
    let mut exact = 0usize;
    for o in 0..out_dim * n {
        let a = gvalues[o] as f64;
        let b = refv[o] as f64;
        let d = (a - b).abs();
        max_abs = max_abs.max(d);
        let rel = if b.abs() > 1e-6 { d / b.abs() } else { d };
        max_rel = max_rel.max(rel);
        sum_sq += d * d;
        sum_ref_sq += b * b;
        if gvalues[o].to_bits() == refv[o].to_bits() {
            exact += 1;
        }
    }
    let rel_l2 = (sum_sq / sum_ref_sq.max(1e-30)).sqrt();

    let report = serde_json::json!({
        "device": gpu.device_name,
        "device_id": format!("{:04x}", gpu.device_id),
        "tensor": tensor,
        "in_dim": in_dim,
        "out_dim": out_dim,
        "blocks_per_row": blocks_per_row,
        "gpu_dispatch_count": gpu.dispatch_count(),
        "max_abs_diff": max_abs,
        "max_rel_diff": max_rel,
        "rel_l2": rel_l2,
        "bit_exact_count": exact,
        "total": out_dim,
        "tol_rel": 1e-3,
        "pass": rel_l2 < 1e-3 && gvalues.iter().all(|v| v.is_finite()),
    });

    let pass = report["pass"].as_bool().unwrap_or(false);
    println!("{report}");
    if let Some(p) = out_json {
        let mut f = std::fs::File::create(&p)?;
        f.write_all(format!("{report}\n").as_bytes())?;
    }
    if !pass {
        return Err(anyhow!(
            "Q6_K GPU multiply FAILED: rel_l2={rel_l2:.3e} max_abs={max_abs:.3e}"
        ));
    }
    Ok(())
}
