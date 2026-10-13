//! `kernel_check` — exercise all seven Rust-authored GPU kernels against CPU
//! references using real model tensor data.
//!
//! Last stdout line is JSON matching the contract schema.

use anyhow::Result;
use lfm2_engine::kernels;
use lfm2_engine::model_file::ModelFile;
use lfm2_engine::ops::GpuOps;
use lfm2_engine::q6k;
use sha2::{Digest, Sha256};

fn parse_arg(args: &[String], i: &mut usize, name: &str) -> Option<String> {
    if args[*i] == name {
        *i += 1;
        Some(args[*i].clone())
    } else {
        None
    }
}

fn cpu_q6k_matmul(wbytes: &[u8], in_dim: usize, out_dim: usize, x: &[f32], n: usize) -> Vec<f32> {
    let blocks = in_dim / q6k::QK_K;
    let row_bytes = blocks * q6k::QK_K_BYTES;
    let mut out = vec![0f32; out_dim * n];
    let mut wrow = vec![0f32; in_dim];
    for j in 0..n {
        for o in 0..out_dim {
            q6k::dequantize_q6k_row(&wbytes[o * row_bytes..(o + 1) * row_bytes], &mut wrow);
            let mut acc = 0f64;
            for k in 0..in_dim {
                acc += wrow[k] as f64 * x[k + j * in_dim] as f64;
            }
            out[o + j * out_dim] = acc as f32;
        }
    }
    out
}

fn cpu_f32_matmul(w: &[f32], in_dim: usize, out_dim: usize, x: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; out_dim * n];
    for j in 0..n {
        for o in 0..out_dim {
            let mut acc = 0f64;
            for k in 0..in_dim {
                acc += w[k + o * in_dim] as f64 * x[k + j * in_dim] as f64;
            }
            out[o + j * out_dim] = acc as f32;
        }
    }
    out
}

fn cpu_rms_norm(x: &[f32], scale: &[f32], dim: usize, n: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; dim * n];
    for j in 0..n {
        let mut ss = 0f32;
        for k in 0..dim {
            let v = x[k + j * dim];
            ss += v * v;
        }
        let inv = 1.0 / (ss / dim as f32 + eps).sqrt();
        for k in 0..dim {
            out[k + j * dim] = x[k + j * dim] * inv * scale[k];
        }
    }
    out
}

fn cpu_rope(q: &mut [f32], hd: usize, heads: usize, n: usize, start_pos: i64, theta: f32) {
    let tscale = theta.powf(-2.0 / hd as f32);
    let dim_all = hd * heads;
    for j in 0..n {
        let pos = (start_pos + j as i64) as f32;
        for p in 0..(hd / 2) {
            let th = pos * tscale.powi(p as i32);
            let (s, c) = th.sin_cos();
            for h in 0..heads {
                let i0 = p + h * hd + j * dim_all;
                let i1 = p + hd / 2 + h * hd + j * dim_all;
                let x0 = q[i0];
                let x1 = q[i1];
                q[i0] = x0 * c - x1 * s;
                q[i1] = x0 * s + x1 * c;
            }
        }
    }
}

fn cpu_swiglu(gate: &[f32], up: &[f32]) -> Vec<f32> {
    gate.iter()
        .zip(up.iter())
        .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
        .collect()
}

fn l2_error(gpu: &[f32], cpu: &[f32]) -> f64 {
    let mut sum_sq = 0f64;
    let mut sum_ref = 0f64;
    for (a, b) in gpu.iter().zip(cpu.iter()) {
        let d = *a as f64 - *b as f64;
        sum_sq += d * d;
        sum_ref += (*b as f64) * (*b as f64);
    }
    (sum_sq / sum_ref.max(1e-30)).sqrt()
}

fn test_x(n: usize, seed: f32) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let v = (i as f32) * seed;
            v.sin() * 0.5 + (i % 3) as f32 * 0.01 - 0.25
        })
        .collect()
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path: Option<String> = None;
    let mut all = false;
    let mut i = 1;
    while i < args.len() {
        if let Some(v) = parse_arg(&args, &mut i, "--model") {
            model_path = Some(v);
        } else if args[i] == "--all" {
            all = true;
        } else {
            anyhow::bail!("unknown arg {}", args[i]);
        }
        i += 1;
    }
    if !all {
        anyhow::bail!("need --all");
    }
    let model_path = model_path.ok_or_else(|| anyhow::anyhow!("need --model"))?;

    let mf = ModelFile::open(&model_path)?;
    let gpu = GpuOps::open()?;

    let mut cases = Vec::new();

    // --- swiglu ---
    {
        let n = 256usize;
        let gate = test_x(n, 0.03125);
        let up = test_x(n, 0.015625);
        let cpu = cpu_swiglu(&gate, &up);
        let gb = gpu.upload(bytemuck::cast_slice(&gate))?;
        let ub = gpu.upload(bytemuck::cast_slice(&up))?;
        let ob = gpu.alloc_host(n * 4)?;
        gpu.swiglu(&gb, &ub, n, &ob)?;
        let gpu_out = gpu.read_f32(&ob, n);
        let rel = l2_error(&gpu_out, &cpu);
        cases.push(serde_json::json!({
            "operation": "swiglu",
            "pass": rel < 1e-3 && gpu_out.iter().all(|v| v.is_finite()),
            "dispatch_count": 1,
            "tested_elements": n,
            "relative_l2": rel,
            "spirv_sha256": hex::encode(Sha256::digest(bytemuck::cast_slice(&kernels::build_swiglu()))),
        }));
    }

    // --- f32_matmul ---
    {
        let in_dim = 256usize;
        let out_dim = 128usize;
        let n = 2usize;
        let w: Vec<f32> = (0..in_dim * out_dim)
            .map(|i| ((i % 13) as f32 - 6.0) * 0.01)
            .collect();
        let x = test_x(in_dim * n, 0.041);
        let cpu = cpu_f32_matmul(&w, in_dim, out_dim, &x, n);
        let wb = gpu.upload_to_device(bytemuck::cast_slice(&w))?;
        let xb = gpu.upload(bytemuck::cast_slice(&x))?;
        let ob = gpu.alloc_host(out_dim * n * 4)?;
        gpu.f32_matmul(&wb, in_dim, out_dim, &xb, n, &ob)?;
        let gpu_out = gpu.read_f32(&ob, out_dim * n);
        let rel = l2_error(&gpu_out, &cpu);
        cases.push(serde_json::json!({
            "operation": "f32_matmul",
            "pass": rel < 1e-3 && gpu_out.iter().all(|v| v.is_finite()),
            "dispatch_count": 1,
            "tested_elements": out_dim * n,
            "relative_l2": rel,
            "spirv_sha256": hex::encode(Sha256::digest(bytemuck::cast_slice(&kernels::build_f32_matmul()))),
        }));
    }

    // --- rms_norm ---
    {
        let dim = 256usize;
        let n = 4usize;
        let x = test_x(dim * n, 0.027);
        let scale: Vec<f32> = (0..dim).map(|i| ((i % 5) as f32 + 1.0) * 0.2).collect();
        let eps = 1e-6f32;
        let cpu = cpu_rms_norm(&x, &scale, dim, n, eps);
        let xb = gpu.upload(bytemuck::cast_slice(&x))?;
        let sb = gpu.upload(bytemuck::cast_slice(&scale))?;
        let ob = gpu.alloc_host(dim * n * 4)?;
        gpu.rms_norm(&xb, &sb, dim, n, eps, &ob)?;
        let gpu_out = gpu.read_f32(&ob, dim * n);
        let rel = l2_error(&gpu_out, &cpu);
        cases.push(serde_json::json!({
            "operation": "rms_norm",
            "pass": rel < 1e-3 && gpu_out.iter().all(|v| v.is_finite()),
            "dispatch_count": 1,
            "tested_elements": dim * n,
            "relative_l2": rel,
            "spirv_sha256": hex::encode(Sha256::digest(bytemuck::cast_slice(&kernels::build_rms_norm()))),
        }));
    }

    // --- rope ---
    {
        let hd = 64usize;
        let heads = 4usize;
        let n = 3usize;
        let theta = 10000.0f32;
        let mut q: Vec<f32> = test_x(hd * heads * n, 0.023);
        let mut cpu_q = q.clone();
        cpu_rope(&mut cpu_q, hd, heads, n, 0, theta);
        let qb = gpu.upload(bytemuck::cast_slice(&q))?;
        gpu.rope(&qb, hd, heads, n, 0, theta.powf(-2.0 / hd as f32))?;
        let gpu_out = gpu.read_f32(&qb, hd * heads * n);
        let rel = l2_error(&gpu_out, &cpu_q);
        cases.push(serde_json::json!({
            "operation": "rope",
            "pass": rel < 1e-3 && gpu_out.iter().all(|v| v.is_finite()),
            "dispatch_count": 1,
            "tested_elements": hd * heads * n,
            "relative_l2": rel,
            "spirv_sha256": hex::encode(Sha256::digest(bytemuck::cast_slice(&kernels::build_rope()))),
        }));
    }

    // --- attention ---
    {
        let hd = 8usize;
        let heads = 4usize;
        let kv = 2usize;
        let n = 2usize;
        let ctx = 8usize;
        let start_pos: i64 = 3;
        let scale = 1.0 / (hd as f32).sqrt();
        let q: Vec<f32> = test_x(hd * heads * n, 0.019);
        let k: Vec<f32> = test_x(hd * kv * ctx, 0.017);
        let v: Vec<f32> = test_x(ctx * hd * kv, 0.013);

        // CPU reference
        let mut cpu_a = vec![0f32; hd * heads * n];
        for j in 0..n {
            let p = (start_pos + j as i64) as usize;
            for hh in 0..heads {
                let kh = hh * kv / heads;
                let qrow = hh * hd + j * hd * heads;
                let mut m = f32::NEG_INFINITY;
                let mut scores = vec![0f32; p + 1];
                for t in 0..=p {
                    let mut dot = 0f64;
                    for d in 0..hd {
                        dot += q[d + qrow] as f64 * k[d + kh * hd + t * hd * kv] as f64;
                    }
                    scores[t] = (dot as f32) * scale;
                    if scores[t] > m {
                        m = scores[t];
                    }
                }
                let mut ssum = 0f64;
                for t in 0..=p {
                    let e = ((scores[t] - m) as f64).exp();
                    scores[t] = e as f32;
                    ssum += e;
                }
                for t in 0..=p {
                    scores[t] /= ssum as f32;
                }
                for d in 0..hd {
                    let mut acc = 0f64;
                    for t in 0..=p {
                        acc += scores[t] as f64
                            // Production host and native reference use position-major V.
                            * v[t * hd * kv + kh * hd + d] as f64;
                    }
                    cpu_a[d + qrow] = acc as f32;
                }
            }
        }

        let qb = gpu.upload(bytemuck::cast_slice(&q))?;
        let kb = gpu.upload(bytemuck::cast_slice(&k))?;
        let vb = gpu.upload(bytemuck::cast_slice(&v))?;
        let ab = gpu.alloc_host(hd * heads * n * 4)?;
        gpu.attention(&qb, &kb, &vb, hd, heads, kv, n, start_pos, ctx, scale, &ab)?;
        let gpu_out = gpu.read_f32(&ab, hd * heads * n);
        let rel = l2_error(&gpu_out, &cpu_a);
        cases.push(serde_json::json!({
            "operation": "attention",
            "pass": rel < 1e-3 && gpu_out.iter().all(|v| v.is_finite()),
            "dispatch_count": 1,
            "tested_elements": hd * heads * n,
            "relative_l2": rel,
            "spirv_sha256": hex::encode(Sha256::digest(bytemuck::cast_slice(&kernels::build_attention()))),
        }));
    }

    // --- q6k_matmul ---
    {
        let tensor = mf
            .tensor("blk.0.attn_q.weight")
            .ok()
            .or_else(|| mf.tensor("blk.2.attn_q.weight").ok())
            .ok_or_else(|| anyhow::anyhow!("no Q6_K tensor found"))?;
        if tensor.ggml_type != 14 {
            anyhow::bail!("not Q6_K");
        }
        let in_dim = tensor.ne[0] as usize;
        let out_dim = 32usize.min(tensor.ne[1] as usize);
        let blocks = in_dim / q6k::QK_K;
        let row_bytes = blocks * q6k::QK_K_BYTES;
        let wbytes = mf.read_tensor_range(&tensor.name, 0, (row_bytes * out_dim) as u64)?;
        let n = 1usize;
        let x = test_x(in_dim * n, 0.03125);
        let cpu = cpu_q6k_matmul(&wbytes, in_dim, out_dim, &x, n);
        let wb = gpu.upload_q6(&wbytes)?;
        let xb = gpu.upload(bytemuck::cast_slice(&x))?;
        let ob = gpu.alloc_host(out_dim * n * 4)?;
        gpu.q6k_matmul(&wb, 0, in_dim, out_dim, &xb, n, &ob)?;
        let gpu_out = gpu.read_f32(&ob, out_dim * n);
        let rel = l2_error(&gpu_out, &cpu);
        cases.push(serde_json::json!({
            "operation": "q6k_matmul",
            "pass": rel < 1e-3 && gpu_out.iter().all(|v| v.is_finite()),
            "dispatch_count": 1,
            "tested_elements": out_dim * n,
            "relative_l2": rel,
            "spirv_sha256": hex::encode(Sha256::digest(bytemuck::cast_slice(&if gpu.q6_aligned {kernels::build_q6k_aligned(false,false,1)} else {kernels::build_q6k_matmul()}))),
            "q6_layout_aligned": gpu.q6_aligned,
        }));
    }

    // --- q6k_matmul_4expert ---
    {
        let tensor = mf
            .tensor("blk.2.ffn_gate_exps.weight")
            .ok()
            .or_else(|| mf.tensor("blk.3.ffn_gate_exps.weight").ok())
            .ok_or_else(|| anyhow::anyhow!("no 3D Q6_K expert tensor"))?;
        if tensor.ggml_type != 14 || tensor.ne.len() != 3 {
            anyhow::bail!("not a 3D Q6_K tensor");
        }
        let in_dim = tensor.ne[0] as usize;
        let out_dim = 16usize.min(tensor.ne[1] as usize);
        let n_expert = tensor.ne[2] as usize;
        let blocks = in_dim / q6k::QK_K;
        let row_bytes = blocks * q6k::QK_K_BYTES;
        let wbytes =
            mf.read_tensor_range(&tensor.name, 0, (row_bytes * out_dim * n_expert) as u64)?;
        // Use 4 different experts
        let experts = [0u32, 1u32, 2u32, 3u32];
        let x4 = test_x(in_dim * 4, 0.037);

        // CPU reference per expert
        let mut cpu = vec![0f32; 4 * out_dim];
        for e_idx in 0..4u32 {
            let base = e_idx as usize * out_dim * row_bytes;
            let x_col = &x4[e_idx as usize * in_dim..(e_idx as usize + 1) * in_dim];
            let result = cpu_q6k_matmul(&wbytes[base..], in_dim, out_dim, x_col, 1);
            cpu[e_idx as usize * out_dim..(e_idx as usize + 1) * out_dim].copy_from_slice(&result);
        }

        let wb = gpu.upload_q6(&wbytes)?;
        let xb = gpu.upload(bytemuck::cast_slice(&x4))?;
        let ob = gpu.alloc_host(4 * out_dim * 4)?;
        gpu.q6k_matmul_4expert(&wb, in_dim, out_dim, experts, &xb, &ob)?;
        let gpu_out = gpu.read_f32(&ob, 4 * out_dim);
        let rel = l2_error(&gpu_out, &cpu);
        cases.push(serde_json::json!({
            "operation": "q6k_matmul_4expert",
            "pass": rel < 1e-3 && gpu_out.iter().all(|v| v.is_finite()),
            "dispatch_count": 1,
            "tested_elements": 4 * out_dim,
            "relative_l2": rel,
            "spirv_sha256": hex::encode(Sha256::digest(bytemuck::cast_slice(&if gpu.q6_aligned {kernels::build_q6k_aligned(true,false,1)} else {kernels::build_q6k_matmul_4expert()}))),
            "q6_layout_aligned": gpu.q6_aligned,
        }));
    }

    let report = serde_json::json!({
        "device": gpu.device_name,
        "device_id": format!("{:04x}", gpu.device_id),
        "cases": cases,
    });

    let s = serde_json::to_string(&report)?;
    println!("{}", s);

    let all_pass = cases.iter().all(|c| c["pass"].as_bool().unwrap_or(false));
    if !all_pass {
        anyhow::bail!("one or more kernel checks failed");
    }
    Ok(())
}
