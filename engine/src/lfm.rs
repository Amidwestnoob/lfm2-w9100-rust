//! LFM2.5-8B-A1B forward graph (pure-Rust reference implementation).
//!
//! Implements the exact LFM2 (hybrid MoE + short-convolution) arithmetic from
//! the retained native backend (lucebox `lfm2_backend.cpp` / llama.cpp
//! `lfm2moe`). This is the numerical reference that (a) validates the model
//! math on the CPU and (b) is the ground truth for the first-hardware Q6_K
//! comparison and full greedy-token parity.
//!
//! ## Memory layout
//!
//! Every tensor follows ggml conventions: `ne[0]` is the contiguous (fastest)
//! axis. An activation shaped `[dim, n]` therefore stores element `(k, j)` at
//! index `k + j*dim` (dim fastest, token slow). A weight `[in, out]` stores
//! element `(k, o)` at `k + o*in` (input fastest).

use crate::model_file::ModelFile;
use crate::q6k;
use anyhow::{anyhow, Result};
use std::collections::HashMap;

pub const DIM: usize = 2048;
pub const HEADS: usize = 32;
pub const HEAD_DIM: usize = 64;
pub const N_LAYERS: usize = 24;
pub const N_EXPERT: usize = 32;
pub const N_USED: usize = 4;
pub const EXPERT_FF: usize = 1792;
pub const N_FF: usize = 7168;
pub const DENSE_LEAD: usize = 2;

#[derive(Clone)]
pub struct Hyperparams {
    pub dim: usize,
    pub heads: usize,
    pub head_dim: usize,
    pub n_layers: usize,
    /// per layer: 0 = shortconv, 8 = attention (8 KV heads)
    pub kv_heads: Vec<u32>,
    pub vocab: usize,
    pub eps: f32,
    pub rope_theta: f32,
    pub eos: u32,
    pub n_expert: usize,
    pub n_used: usize,
    pub expert_ff: usize,
    pub n_ff: usize,
    pub dense_lead: usize,
    pub conv_l_cache: usize,
}

/// Raw tensor payload.
struct Tensor {
    ggml_type: u32,
    ne: Vec<u64>,
    data: Vec<u8>,
}

impl Tensor {
    fn f32_data(&self) -> &[f32] {
        assert_eq!(self.ggml_type, crate::model_file::GGML_TYPE_F32);
        let (head, body, _) = unsafe { self.data.align_to::<f32>() };
        assert!(head.is_empty(), "f32 tensor not 4-byte aligned");
        body
    }
}

pub struct Engine {
    pub hp: Hyperparams,
    tensors: HashMap<String, Tensor>,
    /// Dequantized f32 weights for every non-expert tensor (embeddings, dense,
    /// attention, shortconv, norms). Expert 3D tensors stay raw (Q6_K) and are
    /// dequantized per-slice on the fly by `mul_mat_expert`.
    f32_cache: HashMap<String, Vec<f32>>,
    /// shortconv state: per shortconv layer, l_cache-1 previous (b*xx) vectors,
    /// stored as `[cache_len, dim]` with dim fastest (ne0 = cache_len).
    conv_cache: Vec<Option<Vec<f32>>>,
    /// per attention layer KV cache.
    /// keys[l]   = [head_dim, kv_heads, ctx]  (ne0=head_dim fastest)
    /// values[l] = [ctx, head_dim, kv_heads]  (ne0=ctx fastest)
    keys: Vec<Option<Vec<f32>>>,
    values: Vec<Option<Vec<f32>>>,
    ctx: usize,
}

/// Write f32 little-endian vector to `<dir>/<name>.bin` (LFM_LAYER_DUMP hook).
fn dump_vec(dir: &str, name: &str, v: &[f32]) {
    use std::io::Write;
    let path = format!("{dir}/{name}.bin");
    let mut f = std::fs::File::create(&path).unwrap_or_else(|e| panic!("dump {path}: {e}"));
    let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
    f.write_all(&bytes).unwrap();
}

/// Round through IEEE f16 (native backend stores K/V caches as GGML_TYPE_F16).
#[inline]
fn f16_round(x: f32) -> f32 {
    half::f16::from_f32(x).to_f32()
}

/// Read and validate the LFM2.5-8B-A1B hyperparameters from model metadata.
/// Shared by the CPU reference (`Engine`) and the GPU engine (`gpu_lfm::GpuEngine`).
pub fn load_hyperparams(model: &ModelFile) -> Result<Hyperparams> {
    let n_layers = model.meta_u32("lfm2moe.block_count")? as usize;
    let dim = model.meta_u32("lfm2moe.embedding_length")? as usize;
    let heads = model.meta_u32("lfm2moe.attention.head_count")? as usize;
    let kv_heads = model.meta_u32_array("lfm2moe.attention.head_count_kv")?;
    let vocab = model.meta_u32("lfm2moe.vocab_size")? as usize;
    let n_expert = model.meta_u32("lfm2moe.expert_count")? as usize;
    let n_used = model.meta_u32("lfm2moe.expert_used_count")? as usize;
    let expert_ff = model.meta_u32("lfm2moe.expert_feed_forward_length")? as usize;
    let n_ff = model.meta_u32("lfm2moe.feed_forward_length")? as usize;
    let dense_lead = model.meta_u32_default("lfm2moe.leading_dense_block_count", 2) as usize;
    let conv_l_cache = model.meta_u32_default("lfm2moe.shortconv.l_cache", 3) as usize;

    if dim != DIM
        || heads != HEADS
        || n_layers != N_LAYERS
        || vocab != 128000
        || n_expert != N_EXPERT
        || n_used != N_USED
        || expert_ff != EXPERT_FF
    {
        return Err(anyhow!(
            "unsupported geometry: dim={dim} heads={heads} layers={n_layers} vocab={vocab} n_expert={n_expert} n_used={n_used} expert_ff={expert_ff}"
        ));
    }

    Ok(Hyperparams {
        dim,
        heads,
        head_dim: HEAD_DIM,
        n_layers,
        kv_heads: kv_heads.clone(),
        vocab,
        eps: model.meta_f32("lfm2moe.attention.layer_norm_rms_epsilon")?,
        rope_theta: model.meta_f32("lfm2moe.rope.freq_base")?,
        eos: model.meta_u32_default("tokenizer.ggml.eos_token_id", 124900),
        n_expert,
        n_used,
        expert_ff,
        n_ff,
        dense_lead,
        conv_l_cache,
    })
}

impl Engine {
    pub fn load(model: &ModelFile) -> Result<Self> {
        let hp = load_hyperparams(model)?;

        // Load every tensor's bytes.
        let mut tensors = HashMap::new();
        for t in &model.tensors {
            let bytes = model.read_tensor_bytes(&t.name)?;
            tensors.insert(
                t.name.clone(),
                Tensor {
                    ggml_type: t.ggml_type,
                    ne: t.ne.clone(),
                    data: bytes,
                },
            );
        }

        // Pre-dequantize every non-3D tensor to f32 once.
        let mut f32_cache: HashMap<String, Vec<f32>> = HashMap::new();
        for (name, t) in &tensors {
            if t.ne.len() == 3 {
                continue;
            }
            let n = t.ne.iter().map(|&d| d as usize).product();
            if t.ggml_type == crate::model_file::GGML_TYPE_F32 {
                f32_cache.insert(name.clone(), t.f32_data().to_vec());
            } else if t.ggml_type == crate::model_file::GGML_TYPE_Q6_K {
                let mut out = vec![0f32; n];
                q6k::dequantize_q6k_row(&t.data, &mut out);
                f32_cache.insert(name.clone(), out);
            }
        }

        Ok(Engine {
            hp,
            tensors,
            f32_cache,
            conv_cache: vec![None; N_LAYERS],
            keys: vec![None; N_LAYERS],
            values: vec![None; N_LAYERS],
            ctx: 4096,
        })
    }

    pub fn set_ctx(&mut self, ctx: usize) {
        self.ctx = ctx;
    }

    fn tensor(&self, name: &str) -> &Tensor {
        self.tensors
            .get(name)
            .unwrap_or_else(|| panic!("missing tensor {name}"))
    }

    /// Multiply a weight w[in_dim, out_dim] (stored `w[k + o*in_dim]`) against
    /// an activation x[in_dim, n] (stored `x[k + j*in_dim]`), writing
    /// out[out_dim, n] (stored `out[o + j*out_dim]`).
    fn mul_mat_weight(
        &self,
        wname: &str,
        in_dim: usize,
        out_dim: usize,
        x: &[f32],
        n: usize,
        out: &mut [f32],
    ) {
        let w = self
            .f32_cache
            .get(wname)
            .unwrap_or_else(|| panic!("weight {wname} not in f32 cache (expert 3D?)"));
        for o in 0..out_dim {
            let row = &w[o * in_dim..(o + 1) * in_dim];
            for j in 0..n {
                let mut acc = 0f32;
                for k in 0..in_dim {
                    acc += row[k] * x[k + j * in_dim];
                }
                out[o + j * out_dim] = acc;
            }
        }
    }

    /// RMS-norm each column of x[dim, n] using scale[dim] (dim fastest). out = x/||x||*scale.
    fn rms_norm(&self, x: &[f32], scale: &[f32], n: usize, out: &mut [f32]) {
        let dim = self.hp.dim;
        for j in 0..n {
            let mut ss = 0f32;
            for k in 0..dim {
                let v = x[k + j * dim];
                ss += v * v;
            }
            let inv = 1.0 / (ss / dim as f32 + self.hp.eps).sqrt();
            for k in 0..dim {
                out[k + j * dim] = x[k + j * dim] * inv * scale[k];
            }
        }
    }

    /// NEOX RoPE over head_dim=64 pairs. q is [head_dim*heads, n], element
    /// (d, h, j) at `d + h*hd + j*dim_all`.
    fn rope_neox(&self, q: &mut [f32], heads: usize, n: usize, positions: &[i64]) {
        let hd = self.hp.head_dim;
        let dim_all = hd * heads;
        let tscale = self.hp.rope_theta.powf(-2.0 / hd as f32);
        for j in 0..n {
            let pos = positions[j] as f32;
            for p in 0..(hd / 2) {
                let theta = pos * tscale.powi(p as i32);
                let (s, c) = theta.sin_cos();
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

    /// RMS-norm along the head_dim axis (contiguous within each head).
    fn norm_head_dim(&self, q: &mut [f32], scale: &[f32], heads: usize, n: usize) {
        let hd = self.hp.head_dim;
        let dim_all = hd * heads;
        for h in 0..heads {
            for j in 0..n {
                let mut ss = 0f32;
                for k in 0..hd {
                    let v = q[k + h * hd + j * dim_all];
                    ss += v * v;
                }
                let inv = 1.0 / (ss / hd as f32 + self.hp.eps).sqrt();
                for k in 0..hd {
                    q[k + h * hd + j * dim_all] *= inv * scale[k];
                }
            }
        }
    }

    /// Full forward for `ids` (length n) beginning at absolute position `start_pos`.
    /// Returns logits for the *last* token (vocab_len floats).
    pub fn forward(&mut self, ids: &[u32], start_pos: i64) -> Result<Vec<f32>> {
        let hp = self.hp.clone();
        let ctx = self.ctx;
        let n = ids.len();
        let dim = hp.dim;
        let vocab = hp.vocab;

        // Layer-dump hook mirroring lucebox `LUCEBOX_LAYER_DUMP`
        // (lfm2_backend.cpp@f86dba7c): only for prefill (n>1), one file per
        // layer `l_out-<l>.bin` = 2048 f32 of the LAST token row of the
        // residual stream after the layer. Extra Rust-only files: `embd.bin`
        // and `ffn_inp-<l>.bin` (after mixer residual, before FFN).
        let dump_dir = std::env::var("LFM_LAYER_DUMP").ok().filter(|_| n > 1);
        if let Some(d) = &dump_dir {
            std::fs::create_dir_all(d)?;
        }
        let kv_f16 = std::env::var("LFM_KV_F16")
            .map(|v| v == "1")
            .unwrap_or(false);

        // input embedding: x[dim, n]; token_embd.weight is [dim, vocab] (pre-dequantized).
        let mut x = vec![0f32; dim * n];
        {
            let emb = &self.f32_cache["token_embd.weight"];
            for (j, &tok) in ids.iter().enumerate() {
                let row = &emb[(tok as usize) * dim..(tok as usize + 1) * dim];
                for k in 0..dim {
                    x[k + j * dim] = row[k];
                }
            }
        }
        if let Some(d) = &dump_dir {
            dump_vec(d, "embd", &x[(n - 1) * dim..n * dim]);
        }

        for l in 0..hp.n_layers {
            let prefix = format!("blk.{l}");
            let kv = hp.kv_heads[l] as usize;

            // attention / shortconv
            let attn_norm = self
                .tensor(&format!("{prefix}.attn_norm.weight"))
                .f32_data()
                .to_vec();
            let mut h = vec![0f32; dim * n];
            self.rms_norm(&x, &attn_norm, n, &mut h);

            if kv > 0 {
                // ---- attention layer ----
                let mut q = vec![0f32; dim * n];
                self.mul_mat_weight(&format!("{prefix}.attn_q.weight"), dim, dim, &h, n, &mut q);
                let mut k = vec![0f32; hp.head_dim * kv * n];
                self.mul_mat_weight(
                    &format!("{prefix}.attn_k.weight"),
                    dim,
                    hp.head_dim * kv,
                    &h,
                    n,
                    &mut k,
                );
                let mut v = vec![0f32; hp.head_dim * kv * n];
                self.mul_mat_weight(
                    &format!("{prefix}.attn_v.weight"),
                    dim,
                    hp.head_dim * kv,
                    &h,
                    n,
                    &mut v,
                );

                let qn = self
                    .tensor(&format!("{prefix}.attn_q_norm.weight"))
                    .f32_data()
                    .to_vec();
                let kn = self
                    .tensor(&format!("{prefix}.attn_k_norm.weight"))
                    .f32_data()
                    .to_vec();
                self.norm_head_dim(&mut q, &qn, hp.heads, n);
                self.norm_head_dim(&mut k, &kn, kv, n);

                let positions: Vec<i64> = (0..n).map(|i| start_pos + i as i64).collect();
                self.rope_neox(&mut q, hp.heads, n, &positions);
                self.rope_neox(&mut k, kv, n, &positions);

                // write into KV cache
                let hdkv = hp.head_dim * kv;
                self.ensure_kv(l, kv);
                {
                    let (keys, values) = self.kv_pair_mut(l);
                    for j in 0..n {
                        let p = (start_pos + j as i64) as usize;
                        for kh in 0..kv {
                            for d in 0..hp.head_dim {
                                let kk = k[(kh * hp.head_dim + d) + j * hdkv];
                                let vv = v[(kh * hp.head_dim + d) + j * hdkv];
                                keys[d + kh * hp.head_dim + p * hdkv] =
                                    if kv_f16 { f16_round(kk) } else { kk };
                                values[p + d * ctx + kh * (ctx * hp.head_dim)] =
                                    if kv_f16 { f16_round(vv) } else { vv };
                            }
                        }
                    }
                }

                // attention + projection
                let scale = 1.0 / (hp.head_dim as f32).sqrt();
                let mut a = vec![0f32; dim * n];
                {
                    let (keys, values) = self.kv_pair(l);
                    for j in 0..n {
                        let p = (start_pos + j as i64) as usize;
                        for hh in 0..hp.heads {
                            let kh = hh * kv / hp.heads;
                            let mut s = vec![0f32; p + 1];
                            let mut m = f32::NEG_INFINITY;
                            for t in 0..=p {
                                let mut acc = 0f64;
                                for d in 0..hp.head_dim {
                                    acc += q[(hh * hp.head_dim + d) + j * dim] as f64
                                        * keys[d + kh * hp.head_dim + t * hdkv] as f64;
                                }
                                let sc = (acc as f32) * scale;
                                s[t] = sc;
                                if sc > m {
                                    m = sc;
                                }
                            }
                            let mut sum = 0f64;
                            for t in 0..=p {
                                let e = ((s[t] - m) as f64).exp();
                                s[t] = e as f32;
                                sum += e;
                            }
                            for t in 0..=p {
                                s[t] = (s[t] as f64 / sum) as f32;
                            }
                            for d in 0..hp.head_dim {
                                let mut acc = 0f64;
                                for t in 0..=p {
                                    acc += s[t] as f64
                                        * values[t + d * ctx + kh * (ctx * hp.head_dim)] as f64;
                                }
                                a[(hh * hp.head_dim + d) + j * dim] = acc as f32;
                            }
                        }
                    }
                }
                let mut ax = vec![0f32; dim * n];
                self.mul_mat_weight(
                    &format!("{prefix}.attn_output.weight"),
                    dim,
                    dim,
                    &a,
                    n,
                    &mut ax,
                );
                for i in 0..dim * n {
                    x[i] += ax[i];
                }
            } else {
                // ---- shortconv layer ----
                let conv_l = hp.conv_l_cache; // 3
                let cache_len = conv_l - 1; // 2
                let mut bcx = vec![0f32; 3 * dim * n];
                self.mul_mat_weight(
                    &format!("{prefix}.shortconv.in_proj.weight"),
                    dim,
                    3 * dim,
                    &h,
                    n,
                    &mut bcx,
                );

                // bx[t, ch] = b[ch, t] * xx[ch, t], laid out [n, dim] (ne0 = n).
                let mut bx = vec![0f32; n * dim];
                for t in 0..n {
                    for ch in 0..dim {
                        let b = bcx[ch + t * (3 * dim)];
                        let xx = bcx[2 * dim + ch + t * (3 * dim)];
                        bx[t + ch * n] = b * xx;
                    }
                }

                // ci = concat(cache [cache_len,dim], bx [n,dim], dim0) -> [cache_len+n, dim]
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

                // update cache: last cache_len vectors of ci
                let mut newcache = vec![0f32; cache_len * dim];
                for ch in 0..dim {
                    for t2 in 0..cache_len {
                        newcache[t2 + ch * cache_len] = ci[n + t2 + ch * (cache_len + n)];
                    }
                }
                self.conv_cache[l] = Some(newcache);

                // co[ch, t] = sum_k ci[t+k, ch] * w[k, ch]  ->  [dim, n]
                let cw = self
                    .tensor(&format!("{prefix}.shortconv.conv.weight"))
                    .f32_data()
                    .to_vec(); // [3, dim]
                let mut co = vec![0f32; dim * n];
                for t in 0..n {
                    for ch in 0..dim {
                        let mut acc = 0f32;
                        for kk in 0..conv_l {
                            acc += ci[(t + kk) + ch * (cache_len + n)] * cw[kk + ch * conv_l];
                        }
                        co[ch + t * dim] = acc;
                    }
                }

                // gated = cc * co ; projected = out_proj * gated ; x += projected
                let mut gated = vec![0f32; dim * n];
                for t in 0..n {
                    for ch in 0..dim {
                        let cc = bcx[dim + ch + t * (3 * dim)];
                        gated[ch + t * dim] = cc * co[ch + t * dim];
                    }
                }
                let mut projected = vec![0f32; dim * n];
                self.mul_mat_weight(
                    &format!("{prefix}.shortconv.out_proj.weight"),
                    dim,
                    dim,
                    &gated,
                    n,
                    &mut projected,
                );
                for i in 0..dim * n {
                    x[i] += projected[i];
                }
            }

            if let Some(d) = &dump_dir {
                dump_vec(d, &format!("ffn_inp-{l}"), &x[(n - 1) * dim..n * dim]);
            }

            // ---- FFN ----
            let nf = if l == hp.n_layers - 1 { 1 } else { n };
            let f_off = if nf == 1 { n - 1 } else { 0 };
            let ffn_norm = self
                .tensor(&format!("{prefix}.ffn_norm.weight"))
                .f32_data()
                .to_vec();
            // f = cols [f_off .. f_off+nf] of x, as [dim, nf]
            let mut fnorm = vec![0f32; dim * nf];
            {
                let mut f = vec![0f32; dim * nf];
                for j in 0..nf {
                    for k in 0..dim {
                        f[k + j * dim] = x[k + (f_off + j) * dim];
                    }
                }
                self.rms_norm(&f, &ffn_norm, nf, &mut fnorm);
            }

            if l < hp.dense_lead {
                // dense SwiGLU
                let mut gate = vec![0f32; hp.n_ff * nf];
                self.mul_mat_weight(
                    &format!("{prefix}.ffn_gate.weight"),
                    dim,
                    hp.n_ff,
                    &fnorm,
                    nf,
                    &mut gate,
                );
                let mut up = vec![0f32; hp.n_ff * nf];
                self.mul_mat_weight(
                    &format!("{prefix}.ffn_up.weight"),
                    dim,
                    hp.n_ff,
                    &fnorm,
                    nf,
                    &mut up,
                );
                let mut swig = vec![0f32; hp.n_ff * nf];
                for i in 0..hp.n_ff * nf {
                    let g = gate[i];
                    swig[i] = g / (1.0 + (-g).exp()) * up[i];
                }
                let mut ffn_out = vec![0f32; dim * nf];
                self.mul_mat_weight(
                    &format!("{prefix}.ffn_down.weight"),
                    hp.n_ff,
                    dim,
                    &swig,
                    nf,
                    &mut ffn_out,
                );
                for j in 0..nf {
                    for k in 0..dim {
                        x[k + (f_off + j) * dim] += ffn_out[k + j * dim];
                    }
                }
            } else {
                // MoE
                let mut router = vec![0f32; hp.n_expert * nf];
                self.mul_mat_weight(
                    &format!("{prefix}.ffn_gate_inp.weight"),
                    dim,
                    hp.n_expert,
                    &fnorm,
                    nf,
                    &mut router,
                );
                let bias = self
                    .tensor(&format!("{prefix}.exp_probs_b.bias"))
                    .f32_data()
                    .to_vec();

                for j in 0..nf {
                    // probs = sigmoid(router); selection = probs + bias; top-k
                    let mut probs = vec![0f32; hp.n_expert];
                    for e in 0..hp.n_expert {
                        let lg = router[e + j * hp.n_expert];
                        probs[e] = 1.0 / (1.0 + (-lg).exp());
                    }
                    let mut scored: Vec<(f32, usize)> =
                        (0..hp.n_expert).map(|e| (probs[e] + bias[e], e)).collect();
                    scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
                    let sel: Vec<usize> = scored[..hp.n_used].iter().map(|x| x.1).collect();

                    // weights = normalize(probs[sel])
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

                    // f column j (single activation vector)
                    let mut fcol = vec![0f32; dim];
                    for k in 0..dim {
                        fcol[k] = fnorm[k + j * dim];
                    }

                    let mut acc = vec![0f32; dim];
                    for (i2, &e) in sel.iter().enumerate() {
                        let wgt = wsel[i2];
                        let gname = format!("{prefix}.ffn_gate_exps.weight");
                        let uname = format!("{prefix}.ffn_up_exps.weight");
                        let dname = format!("{prefix}.ffn_down_exps.weight");
                        let mut gate_e = vec![0f32; hp.expert_ff];
                        self.mul_mat_expert(&gname, e, dim, hp.expert_ff, &fcol, &mut gate_e);
                        let mut up_e = vec![0f32; hp.expert_ff];
                        self.mul_mat_expert(&uname, e, dim, hp.expert_ff, &fcol, &mut up_e);
                        let mut swig = vec![0f32; hp.expert_ff];
                        for ii in 0..hp.expert_ff {
                            let g = gate_e[ii];
                            swig[ii] = g / (1.0 + (-g).exp()) * up_e[ii];
                        }
                        let mut down_e = vec![0f32; dim];
                        self.mul_mat_expert(&dname, e, hp.expert_ff, dim, &swig, &mut down_e);
                        for k in 0..dim {
                            acc[k] += down_e[k] * wgt;
                        }
                    }
                    for k in 0..dim {
                        x[k + (f_off + j) * dim] += acc[k];
                    }
                }
            }
            if let Some(d) = &dump_dir {
                dump_vec(d, &format!("l_out-{l}"), &x[(n - 1) * dim..n * dim]);
            }
        }

        // output: logits = token_embd.weight^T * norm(x_last). Last token is
        // column j = n-1 (strided by dim).
        let onorm = self.tensor("token_embd_norm.weight").f32_data().to_vec();
        let mut xnorm = vec![0f32; dim];
        {
            let mut ss = 0f32;
            for k in 0..dim {
                let v = x[k + (n - 1) * dim];
                ss += v * v;
            }
            let inv = 1.0 / (ss / dim as f32 + hp.eps).sqrt();
            for k in 0..dim {
                xnorm[k] = x[k + (n - 1) * dim] * inv * onorm[k];
            }
        }
        let mut logits = vec![0f32; vocab];
        self.mul_mat_weight("token_embd.weight", dim, vocab, &xnorm, 1, &mut logits);
        Ok(logits)
    }

    /// Multiply one expert slice of a 3D Q6_K weight [in_dim, out_dim, n_expert]
    /// against an activation vector x[in_dim], writing out[out_dim].
    fn mul_mat_expert(
        &self,
        wname: &str,
        expert: usize,
        in_dim: usize,
        out_dim: usize,
        x: &[f32],
        out: &mut [f32],
    ) {
        let t = self.tensor(wname);
        assert_eq!(t.ggml_type, crate::model_file::GGML_TYPE_Q6_K);
        let blocks_per_row = in_dim / q6k::QK_K;
        let row_bytes = blocks_per_row * q6k::QK_K_BYTES;
        let data = &t.data;
        let base = expert * out_dim * row_bytes;
        let mut wrow = vec![0f32; in_dim];
        for o in 0..out_dim {
            q6k::dequantize_q6k_row(
                &data[base + o * row_bytes..base + (o + 1) * row_bytes],
                &mut wrow,
            );
            let mut acc = 0f32;
            for k in 0..in_dim {
                acc += wrow[k] * x[k];
            }
            out[o] = acc;
        }
    }

    fn ensure_kv(&mut self, l: usize, kv: usize) {
        if self.keys[l].is_none() {
            self.keys[l] = Some(vec![0f32; self.hp.head_dim * kv * self.ctx]);
            self.values[l] = Some(vec![0f32; self.ctx * self.hp.head_dim * kv]);
        }
    }
    fn kv_pair(&self, l: usize) -> (&[f32], &[f32]) {
        (
            self.keys[l].as_ref().unwrap(),
            self.values[l].as_ref().unwrap(),
        )
    }
    fn kv_pair_mut(&mut self, l: usize) -> (&mut [f32], &mut [f32]) {
        (
            self.keys[l].as_mut().unwrap(),
            self.values[l].as_mut().unwrap(),
        )
    }
}
