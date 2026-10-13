//! LFM2 GPU forward graph: the same LFM2.5-8B-A1B arithmetic as `lfm.rs`, but
//! with every matrix multiplication and the elementwise tensor ops (RMS-norm,
//! NEOX RoPE, causal attention/softmax, SwiGLU) dispatched as Rust-authored compute on
//! the W9100 via `ash`. The default resident graph also performs expert top-k
//! routing, short convolution and cache updates in Rust-authored GPU kernels.
//! The original host-staged forward is retained as an explicit diagnostic oracle.
//!
//! Numerical contract is identical to `lfm.rs` (and the retained native
//! backend): Q6_K weights dequantized with *signed* int8 scales, K/V cache
//! rounded through f16, NEOX RoPE, GQA `kh = h*kv/heads`, MoE top-4 with the
//! 6.103515625e-5 clamp.

use crate::model_file::{ModelFile, GGML_TYPE_F32, GGML_TYPE_Q6_K};
use crate::ops::GpuOps;
use crate::q6k;
use crate::{vulkan::GpuBuffer, Hyperparams};
use anyhow::Result;
use std::collections::HashMap;

use crate::lfm;
mod resident_graph;

pub struct GpuEngine {
    pub hp: Hyperparams,
    /// Q6_K weight buffers (raw bytes) and f32 tensor buffers, by name.
    weights: HashMap<String, GpuBuffer>,
    f32_bufs: HashMap<String, GpuBuffer>,
    /// f32 tensor host copies (for routing/conv/scalar steps).
    f32_host: HashMap<String, Vec<f32>>,
    /// token_embd.weight raw Q6_K bytes (host copy for embedding row lookup).
    embd_raw: Vec<u8>,
    /// per-attention-layer KV cache (GPU buffers), f16-rounded like native.
    keys: Vec<Option<GpuBuffer>>,
    values: Vec<Option<GpuBuffer>>,
    /// shortconv cache (host), mirroring lfm.rs.
    conv_cache: Vec<Option<Vec<f32>>>,
    ctx: usize,
    resident: Option<resident_graph::ResidentState>,
    bounded_submissions: bool,
    token_only: bool,
    /// Declared LAST so it is dropped after every `GpuBuffer` above: the buffers
    /// must `destroy_buffer` before `VulkanContext::drop` destroys the device.
    pub gpu: GpuOps,
}

fn f16_round(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

impl GpuEngine {
    pub fn load(model: &ModelFile) -> Result<Self> {
        let hp = lfm::load_hyperparams(model)?;
        anyhow::ensure!(
            (4..=64).contains(&hp.n_expert),
            "resident routing supports 4..=64 experts"
        );
        let gpu = GpuOps::open()?;

        let mut weights = HashMap::new();
        let mut f32_bufs = HashMap::new();
        let mut f32_host = HashMap::new();
        let mut embd_raw = Vec::new();

        let staging_size = model
            .tensors
            .iter()
            .map(|t| if gpu.q6_aligned && t.ggml_type==GGML_TYPE_Q6_K {t.nbytes as usize/210*212} else {t.nbytes as usize})
            .max()
            .unwrap_or(4);
        let staging = gpu.ctx.alloc_buffer(staging_size, true)?;
        for t in &model.tensors {
            if t.ggml_type != GGML_TYPE_F32 && t.ggml_type != GGML_TYPE_Q6_K {
                continue;
            }
            let size = t.nbytes as usize;
            // The previous transfer is complete before this mapped range is reused.
            {
                let bytes = unsafe {std::slice::from_raw_parts_mut(staging.mapped.unwrap().as_ptr(), size)};
                model.read_tensor_into(&t.name, bytes)?;
                if t.ggml_type==GGML_TYPE_Q6_K && t.name=="token_embd.weight" {embd_raw=bytes.to_vec();}
                if t.ggml_type==GGML_TYPE_F32 {f32_host.insert(t.name.clone(), bytemuck::cast_slice(bytes).to_vec());}
            }
            let upload_size = if gpu.q6_aligned && t.ggml_type==GGML_TYPE_Q6_K {
                let padded = size/210*212;
                let all = unsafe {std::slice::from_raw_parts_mut(staging.mapped.unwrap().as_ptr(),padded)};
                q6k::repack_aligned_in_place(all,size); padded
            } else {size};
            let buf = gpu.ctx.alloc_buffer(upload_size, false)?;
            gpu.ctx.begin_batch()?;
            gpu.ctx.record_copy(&staging, &buf, 0, 0, upload_size);
            gpu.ctx.finish_batch()?;
            if t.ggml_type == GGML_TYPE_F32 {
                f32_bufs.insert(t.name.clone(), buf);
            } else {
                weights.insert(t.name.clone(), buf);
            }
        }

        let n_layers = hp.n_layers;
        Ok(GpuEngine {
            hp,
            gpu,
            weights,
            f32_bufs,
            f32_host,
            embd_raw,
            keys: (0..n_layers).map(|_| None).collect(),
            values: (0..n_layers).map(|_| None).collect(),
            conv_cache: vec![None; n_layers],
            ctx: 4096,
            resident: None,
            bounded_submissions: true,
            token_only: false,
        })
    }

    pub fn set_ctx(&mut self, ctx: usize) {
        self.ctx = ctx;
    }

    /// Short-only unchanged submission schedule, solely for state parity testing.
    pub fn set_bounded_submissions_for_test(&mut self, bounded: bool) {
        self.bounded_submissions = bounded;
    }

    pub fn device_name(&self) -> String {
        self.gpu.device_name.clone()
    }

    fn f32_buf(&self, name: &str) -> &GpuBuffer {
        self.f32_bufs
            .get(name)
            .unwrap_or_else(|| panic!("missing f32 {name}"))
    }

    fn weight_buf(&self, name: &str) -> &GpuBuffer {
        self.weights
            .get(name)
            .unwrap_or_else(|| panic!("missing weight {name}"))
    }

    /// GPU Q6_K matmul `w[in,out] * x[in,n] -> out[out,n]`, x/out as host Vec.
    fn mm(
        &mut self,
        wname: &str,
        in_dim: usize,
        out_dim: usize,
        x: &[f32],
        n: usize,
    ) -> Result<Vec<f32>> {
        let wbuf = self.weight_buf(wname);
        if std::env::var("LFM_TRACE").is_ok() {
            eprintln!("[trace] mm {wname} in={in_dim} out={out_dim} n={n}");
        }
        let xb = self.gpu.upload(bytemuck::cast_slice(x))?;
        let ob = self.gpu.alloc_host(out_dim * n * 4)?;
        self.gpu.q6k_matmul(wbuf, 0, in_dim, out_dim, &xb, n, &ob)?;
        Ok(self.gpu.read_f32(&ob, out_dim * n))
    }

    /// GPU Q6_K matmul of one expert slice (base byte offset precomputed).
    fn mm_expert(
        &mut self,
        wname: &str,
        e: usize,
        in_dim: usize,
        out_dim: usize,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        let wbuf = self.weight_buf(wname);
        let row_bytes = (in_dim / q6k::QK_K) * if self.gpu.q6_aligned {q6k::ALIGNED_BYTES} else {q6k::QK_K_BYTES};
        let base = e * out_dim * row_bytes;
        let xb = self.gpu.upload(bytemuck::cast_slice(x))?;
        let ob = self.gpu.alloc_host(out_dim * 4)?;
        self.gpu
            .q6k_matmul(wbuf, base, in_dim, out_dim, &xb, 1, &ob)?;
        Ok(self.gpu.read_f32(&ob, out_dim))
    }

    /// GPU Q6_K matmul of FOUR expert slices at once (single dispatch). `x` is
    /// `[4*in_dim]` (the per-expert inputs packed contiguously); `experts` are
    /// the four selected expert indices. Returns `[4*out_dim]`.
    fn mm_experts4(
        &mut self,
        wname: &str,
        experts: [u32; 4],
        in_dim: usize,
        out_dim: usize,
        x: &[f32],
    ) -> Result<Vec<f32>> {
        let wbuf = self.weight_buf(wname);
        let xb = self.gpu.upload(bytemuck::cast_slice(x))?;
        let ob = self.gpu.alloc_host(4 * out_dim * 4)?;
        self.gpu
            .q6k_matmul_4expert(wbuf, in_dim, out_dim, experts, &xb, &ob)?;
        Ok(self.gpu.read_f32(&ob, 4 * out_dim))
    }

    /// GPU f32 matmul of an F32 weight (ggml `[in_dim,out_dim]`) — used for the
    /// MoE router gate `ffn_gate_inp.weight`.
    fn mm_f32(
        &mut self,
        wname: &str,
        in_dim: usize,
        out_dim: usize,
        x: &[f32],
        n: usize,
    ) -> Result<Vec<f32>> {
        let wbuf = self.f32_buf(wname);
        let xb = self.gpu.upload(bytemuck::cast_slice(x))?;
        let ob = self.gpu.alloc_host(out_dim * n * 4)?;
        self.gpu.f32_matmul(wbuf, in_dim, out_dim, &xb, n, &ob)?;
        Ok(self.gpu.read_f32(&ob, out_dim * n))
    }

    /// GPU RMS-norm of x[dim,n] with scale[dim] into host Vec.
    fn rms(&mut self, scale_name: &str, dim: usize, x: &[f32], n: usize) -> Result<Vec<f32>> {
        if std::env::var("LFM_TRACE").is_ok() {
            eprintln!("[trace] rms {scale_name} dim={dim} n={n}");
        }
        let sbuf = self.f32_buf(scale_name);
        let xb = self.gpu.upload(bytemuck::cast_slice(x))?;
        let ob = self.gpu.alloc_host(dim * n * 4)?;
        self.gpu.rms_norm(&xb, sbuf, dim, n, self.hp.eps, &ob)?;
        Ok(self.gpu.read_f32(&ob, dim * n))
    }

    /// In-place GPU rope on q[hd*heads, n].
    fn rope(
        &mut self,
        qbuf: &GpuBuffer,
        hd: usize,
        heads: usize,
        n: usize,
        start_pos: i64,
    ) -> Result<()> {
        let tscale = self.hp.rope_theta.powf(-2.0 / hd as f32);
        self.gpu.rope(qbuf, hd, heads, n, start_pos, tscale)
    }

    /// Full forward for `ids` (n tokens) starting at absolute `start_pos`.
    /// Returns logits for the *last* token.
    pub fn forward(&mut self, ids: &[u32], start_pos: i64) -> Result<Vec<f32>> {
        self.forward_resident(ids, start_pos)
    }

    /// Preserved synchronous Rust GPU graph, available only as a developer oracle.
    pub fn forward_legacy(&mut self, ids: &[u32], start_pos: i64) -> Result<Vec<f32>> {
        let hp = self.hp.clone();
        let ctx = self.ctx;
        let n = ids.len();
        let dim = hp.dim;
        let vocab = hp.vocab;

        // ---- input embedding: look up token_embd.weight Q6_K rows ----
        let mut x = vec![0f32; dim * n];
        let emb_in = dim; // token_embd.weight ne[0]
        let emb_row_bytes = (emb_in / q6k::QK_K) * q6k::QK_K_BYTES;
        for (j, &tok) in ids.iter().enumerate() {
            let off = (tok as usize) * emb_row_bytes;
            q6k::dequantize_q6k_row(
                &self.embd_raw[off..off + emb_row_bytes],
                &mut x[j * dim..(j + 1) * dim],
            );
        }

        for l in 0..hp.n_layers {
            let prefix = format!("blk.{l}");
            let kv = hp.kv_heads[l] as usize;
            if std::env::var("LFM_TRACE").is_ok() {
                eprintln!("[trace] layer {l} kv={kv} n={n}");
            }

            // attention / shortconv
            let h = self.rms(&format!("{prefix}.attn_norm.weight"), dim, &x, n)?;

            if kv > 0 {
                // ---- attention layer ----
                let q = self.mm(&format!("{prefix}.attn_q.weight"), dim, dim, &h, n)?;
                let k = self.mm(
                    &format!("{prefix}.attn_k.weight"),
                    dim,
                    hp.head_dim * kv,
                    &h,
                    n,
                )?;
                let v = self.mm(
                    &format!("{prefix}.attn_v.weight"),
                    dim,
                    hp.head_dim * kv,
                    &h,
                    n,
                )?;

                // head norms + rope (on GPU, q/k uploaded once)
                let qn_buf = self.f32_buf(&format!("{prefix}.attn_q_norm.weight"));
                let kn_buf = self.f32_buf(&format!("{prefix}.attn_k_norm.weight"));
                let qbuf = self.gpu.upload(bytemuck::cast_slice(&q))?;
                let kbuf = self.gpu.upload(bytemuck::cast_slice(&k))?;
                let qnb = self.gpu.alloc_host(dim * n * 4)?;
                let knb = self.gpu.alloc_host(hp.head_dim * kv * n * 4)?;
                self.gpu
                    .rms_norm(&qbuf, qn_buf, hp.head_dim, hp.heads * n, hp.eps, &qnb)?;
                self.gpu
                    .rms_norm(&kbuf, kn_buf, hp.head_dim, kv * n, hp.eps, &knb)?;
                self.rope(&qnb, hp.head_dim, hp.heads, n, start_pos)?;
                self.rope(&knb, hp.head_dim, kv, n, start_pos)?;
                let k_roped = self.gpu.read_f32(&knb, hp.head_dim * kv * n);

                // write into KV cache (f16-rounded) — partial writes to only the new
                // positions, not a full read-modify-write round-trip.
                self.ensure_kv(l, kv, ctx)?;
                {
                    let keys = self.keys[l].as_ref().unwrap();
                    let values = self.values[l].as_ref().unwrap();
                    let hdkv = hp.head_dim * kv;
                    let mut kslice = vec![0f32; hdkv];
                    let mut vslice = vec![0f32; hdkv];
                    for j in 0..n {
                        let p = (start_pos + j as i64) as usize;
                        for kh in 0..kv {
                            for d in 0..hp.head_dim {
                                kslice[kh * hp.head_dim + d] =
                                    f16_round(k_roped[(kh * hp.head_dim + d) + j * hdkv]);
                                vslice[kh * hp.head_dim + d] =
                                    f16_round(v[(kh * hp.head_dim + d) + j * hdkv]);
                            }
                        }
                        keys.write(p * hdkv * 4, bytemuck::cast_slice(&kslice));
                        values.write(p * hdkv * 4, bytemuck::cast_slice(&vslice));
                    }
                }

                // attention + projection
                let keys = self.keys[l].as_ref().unwrap();
                let values = self.values[l].as_ref().unwrap();
                let abuf = self.gpu.alloc_host(dim * n * 4)?;
                let scale = 1.0 / (hp.head_dim as f32).sqrt();
                self.gpu.attention(
                    &qnb,
                    keys,
                    values,
                    hp.head_dim,
                    hp.heads,
                    kv,
                    n,
                    start_pos,
                    ctx,
                    scale,
                    &abuf,
                )?;
                let a = self.gpu.read_f32(&abuf, dim * n);

                let ax = self.mm(&format!("{prefix}.attn_output.weight"), dim, dim, &a, n)?;
                for i in 0..dim * n {
                    x[i] += ax[i];
                }
            } else {
                // ---- shortconv layer (host conv + GPU projections) ----
                let conv_l = hp.conv_l_cache; // 3
                let cache_len = conv_l - 1; // 2
                let bcx = self.mm(
                    &format!("{prefix}.shortconv.in_proj.weight"),
                    dim,
                    3 * dim,
                    &h,
                    n,
                )?;

                // bx[t, ch] = b[ch, t] * xx[ch, t], laid out [n, dim]
                let mut bx = vec![0f32; n * dim];
                for t in 0..n {
                    for ch in 0..dim {
                        let b = bcx[ch + t * (3 * dim)];
                        let xx = bcx[2 * dim + ch + t * (3 * dim)];
                        bx[t + ch * n] = b * xx;
                    }
                }

                let cache = self.conv_cache[l]
                    .clone()
                    .unwrap_or_else(|| vec![0f32; cache_len * dim]);
                let mut ci = vec![0f32; (cache_len + n) * dim];
                for ch in 0..dim {
                    for t2 in 0..cache_len {
                        ci[t2 + ch * (cache_len + n)] = cache[t2 + ch * cache_len];
                    }
                    for t2 in 0..n {
                        ci[cache_len + t2 + ch * (cache_len + n)] = bx[t2 + ch * n];
                    }
                }
                let mut newcache = vec![0f32; cache_len * dim];
                for ch in 0..dim {
                    for t2 in 0..cache_len {
                        newcache[t2 + ch * cache_len] = ci[n + t2 + ch * (cache_len + n)];
                    }
                }
                self.conv_cache[l] = Some(newcache);

                let cw = self.f32_host[&format!("{prefix}.shortconv.conv.weight")].clone();
                let mut gated = vec![0f32; dim * n];
                for t in 0..n {
                    for ch in 0..dim {
                        let mut acc = 0f32;
                        for kk in 0..conv_l {
                            acc += ci[(t + kk) + ch * (cache_len + n)] * cw[kk + ch * conv_l];
                        }
                        let cc = bcx[dim + ch + t * (3 * dim)];
                        gated[ch + t * dim] = cc * acc;
                    }
                }
                let projected = self.mm(
                    &format!("{prefix}.shortconv.out_proj.weight"),
                    dim,
                    dim,
                    &gated,
                    n,
                )?;
                for i in 0..dim * n {
                    x[i] += projected[i];
                }
            }

            // ---- FFN ----
            let nf = if l == hp.n_layers - 1 { 1 } else { n };
            let f_off = if nf == 1 { n - 1 } else { 0 };
            let mut f = vec![0f32; dim * nf];
            for j in 0..nf {
                for k in 0..dim {
                    f[k + j * dim] = x[k + (f_off + j) * dim];
                }
            }
            let fnorm = self.rms(&format!("{prefix}.ffn_norm.weight"), dim, &f, nf)?;

            if l < hp.dense_lead {
                let gate = self.mm(
                    &format!("{prefix}.ffn_gate.weight"),
                    dim,
                    hp.n_ff,
                    &fnorm,
                    nf,
                )?;
                let up = self.mm(&format!("{prefix}.ffn_up.weight"), dim, hp.n_ff, &fnorm, nf)?;
                let gb = self.gpu.upload(bytemuck::cast_slice(&gate))?;
                let ub = self.gpu.upload(bytemuck::cast_slice(&up))?;
                let swb = self.gpu.alloc_host(hp.n_ff * nf * 4)?;
                self.gpu.swiglu(&gb, &ub, hp.n_ff * nf, &swb)?;
                let swig = self.gpu.read_f32(&swb, hp.n_ff * nf);
                let ffn_out = self.mm(
                    &format!("{prefix}.ffn_down.weight"),
                    hp.n_ff,
                    dim,
                    &swig,
                    nf,
                )?;
                for j in 0..nf {
                    for k in 0..dim {
                        x[k + (f_off + j) * dim] += ffn_out[k + j * dim];
                    }
                }
            } else {
                let router = self.mm_f32(
                    &format!("{prefix}.ffn_gate_inp.weight"),
                    dim,
                    hp.n_expert,
                    &fnorm,
                    nf,
                )?;
                let bias = self.f32_host[&format!("{prefix}.exp_probs_b.bias")].clone();

                for j in 0..nf {
                    let mut probs = vec![0f32; hp.n_expert];
                    for e in 0..hp.n_expert {
                        let lg = router[e + j * hp.n_expert];
                        probs[e] = 1.0 / (1.0 + (-lg).exp());
                    }
                    let mut scored: Vec<(f32, usize)> =
                        (0..hp.n_expert).map(|e| (probs[e] + bias[e], e)).collect();
                    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                    let sel: Vec<usize> = scored[..hp.n_used].iter().map(|x| x.1).collect();
                    let mut wsel = vec![0f32; hp.n_used];
                    let mut wsum = 0f32;
                    for (i2, &e) in sel.iter().enumerate() {
                        wsel[i2] = probs[e];
                        wsum += probs[e];
                    }
                    if wsum < 6.103515625e-5 {
                        wsum = 6.103515625e-5;
                    }
                    for i2 in 0..hp.n_used {
                        wsel[i2] /= wsum;
                    }

                    let fcol: Vec<f32> = fnorm[j * dim..(j + 1) * dim].to_vec();
                    let experts: [u32; 4] =
                        [sel[0] as u32, sel[1] as u32, sel[2] as u32, sel[3] as u32];
                    // replicate fcol 4x for the batched 4-expert kernel
                    let mut x4 = vec![0f32; 4 * dim];
                    for i in 0..4 {
                        x4[i * dim..(i + 1) * dim].copy_from_slice(&fcol);
                    }
                    let gate_e = self.mm_experts4(
                        &format!("{prefix}.ffn_gate_exps.weight"),
                        experts,
                        dim,
                        hp.expert_ff,
                        &x4,
                    )?;
                    let up_e = self.mm_experts4(
                        &format!("{prefix}.ffn_up_exps.weight"),
                        experts,
                        dim,
                        hp.expert_ff,
                        &x4,
                    )?;
                    let gb = self.gpu.upload(bytemuck::cast_slice(&gate_e))?;
                    let ub = self.gpu.upload(bytemuck::cast_slice(&up_e))?;
                    let swb = self.gpu.alloc_host(4 * hp.expert_ff * 4)?;
                    self.gpu.swiglu(&gb, &ub, 4 * hp.expert_ff, &swb)?;
                    let swig = self.gpu.read_f32(&swb, 4 * hp.expert_ff);
                    let down_e = self.mm_experts4(
                        &format!("{prefix}.ffn_down_exps.weight"),
                        experts,
                        hp.expert_ff,
                        dim,
                        &swig,
                    )?;
                    let mut acc = vec![0f32; dim];
                    for i2 in 0..4 {
                        let wgt = wsel[i2];
                        for k in 0..dim {
                            acc[k] += down_e[i2 * dim + k] * wgt;
                        }
                    }
                    for k in 0..dim {
                        x[k + (f_off + j) * dim] += acc[k];
                    }
                }
            }
        }

        // ---- output norm + logits (tied embedding) ----
        let onorm = self.f32_host["token_embd_norm.weight"].clone();
        let mut xnorm = vec![0f32; dim];
        let mut ss = 0f32;
        for k in 0..dim {
            let v = x[k + (n - 1) * dim];
            ss += v * v;
        }
        let inv = 1.0 / (ss / dim as f32 + hp.eps).sqrt();
        for k in 0..dim {
            xnorm[k] = x[k + (n - 1) * dim] * inv * onorm[k];
        }
        // The final norm is computed on the host (single 2048-vector), then
        // the (large) logits matmul runs on the GPU against the tied Q6_K weight.
        let logits = self.mm("token_embd.weight", dim, vocab, &xnorm, 1)?;
        Ok(logits)
    }

    fn ensure_kv(&mut self, l: usize, kv: usize, ctx: usize) -> Result<()> {
        if self.keys[l].is_none() {
            let kbuf = self.gpu.alloc_host(self.hp.head_dim * kv * ctx * 4)?;
            let vbuf = self.gpu.alloc_host(ctx * self.hp.head_dim * kv * 4)?;
            // zero-fill
            let zkv = vec![0f32; self.hp.head_dim * kv * ctx];
            let zv = vec![0f32; ctx * self.hp.head_dim * kv];
            self.gpu.write_f32(&kbuf, &zkv);
            self.gpu.write_f32(&vbuf, &zv);
            self.keys[l] = Some(kbuf);
            self.values[l] = Some(vbuf);
        }
        Ok(())
    }
}
