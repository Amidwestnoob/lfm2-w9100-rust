//! Persistent device-local forward graph. No host tensor calculations.
use super::*;

pub(super) struct ResidentState {
    ncap: usize,
    ctx: usize,
    buffers: HashMap<&'static str, GpuBuffer>,
    keys: Vec<Option<GpuBuffer>>,
    keys_transposed: Vec<Option<GpuBuffer>>,
    values: Vec<Option<GpuBuffer>>,
    conv: Vec<Option<GpuBuffer>>,
}
impl ResidentState {
    fn new(gpu: &GpuOps, hp: &Hyperparams, n: usize, ctx: usize) -> Result<Self> {
        let mut buffers = HashMap::new();
        let wide = hp.n_ff.max(4 * hp.expert_ff);
        for (name, count) in [
            ("ids", n),
            ("position", 1),
            ("attention_scores", hp.heads * n * ctx),
            ("attention_value_parts", hp.heads * n * hp.head_dim * ctx.div_ceil(256)),
            ("attention_kt", hp.head_dim * hp.kv_heads.iter().copied().max().unwrap() as usize * ctx),
            ("argmax_partial", hp.vocab.div_ceil(512) * 2),
            ("selected", 2),
            ("expert_groups", hp.n_expert * (4 * n + 1)),
            ("x", hp.dim * n),
            ("h", hp.dim * n),
            ("q", hp.dim * n),
            ("k", hp.dim * n),
            ("v", hp.dim * n),
            ("qn", hp.dim * n),
            ("kn", hp.dim * n),
            ("a", hp.dim * n),
            ("projected", hp.dim * n),
            ("bcx", 3 * hp.dim * n),
            ("gate", wide * n),
            ("up", wide * n),
            ("sw", wide * n),
            ("down", 4 * hp.dim * n),
            ("router", hp.n_expert * n),
            ("selection", 8 * n),
            ("last", hp.dim),
            ("norm", hp.dim),
            ("logits", hp.vocab),
            ("readback", hp.vocab),
        ] {
            buffers.insert(
                name,
                gpu.ctx.alloc_buffer(
                    count * 4,
                    name == "ids" || name == "readback" || name == "position",
                )?,
            );
        }
        let mut keys = Vec::new();
        let mut keys_transposed = Vec::new();
        let mut values = Vec::new();
        let mut conv = Vec::new();
        for &kv in &hp.kv_heads {
            if kv > 0 {
                keys_transposed.push(Some(gpu.ctx.alloc_buffer(hp.head_dim * kv as usize * ctx * 4,false)?));
                keys.push(Some(
                    gpu.ctx
                        .alloc_buffer(hp.head_dim * kv as usize * ctx * 4, false)?,
                ));
                values.push(Some(
                    gpu.ctx
                        .alloc_buffer(hp.head_dim * kv as usize * ctx * 4, false)?,
                ));
                conv.push(None);
            } else {
                keys.push(None);
                keys_transposed.push(None);
                values.push(None);
                conv.push(Some(gpu.upload_to_device(&vec![0u8; hp.dim * 2 * 4])?));
            }
        }
        Ok(Self {
            ncap: n,
            ctx,
            buffers,
            keys,
            keys_transposed,
            values,
            conv,
        })
    }
}
impl GpuEngine {
    pub fn forward_token(&mut self, ids: &[u32], position: usize) -> Result<u32> {
        if !self.token_only {
            self.gpu.ctx.clear_decode_plan();
            self.token_only = true;
        }
        anyhow::ensure!(
            !ids.is_empty() && ids.len() <= crate::work_bound::PREFILL_CHUNK,
            "token batch must be 1..64"
        );
        let result = self.forward_resident(ids, position as i64)?;
        Ok(result[1] as u32)
    }

    /// Reset recurrent state in-place. Stale KV beyond the valid prefix is never
    /// attended, so no large cache clear or reallocation is necessary.
    pub fn reset_session(&mut self) -> Result<()> {
        if let Some(state) = &self.resident {
            for c in state.conv.iter().flatten() {
                let zero = self.gpu.upload(&vec![0u8; c.size as usize])?;
                self.gpu.ctx.begin_batch()?;
                self.gpu.ctx.record_copy(&zero, c, 0, 0, c.size as usize);
                self.gpu.ctx.finish_batch()?;
            }
        }
        Ok(())
    }

    /// Restore a previously saved initialized prefix into the same stable GPU
    /// buffers; command/descriptors remain valid across sequential sessions.
    pub fn restore_session(&self, valid_tokens: usize, bytes: &[u8]) -> Result<()> {
        let state = self
            .resident
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no state"))?;
        anyhow::ensure!(valid_tokens <= self.ctx, "restore context");
        let mut offset = 0;
        for l in 0..self.hp.n_layers {
            let kv = self.hp.kv_heads[l] as usize;
            let dests = if kv > 0 {
                let size = self.hp.head_dim * kv * valid_tokens * 4;
                vec![
                    (state.keys[l].as_ref().unwrap(), size),
                    (state.values[l].as_ref().unwrap(), size),
                ]
            } else {
                let c = state.conv[l].as_ref().unwrap();
                vec![(c, c.size as usize)]
            };
            for (dst, size) in dests {
                anyhow::ensure!(offset + size <= bytes.len(), "truncated snapshot");
                if size > 0 {
                    let staging = self.gpu.upload(&bytes[offset..offset + size])?;
                    self.gpu.ctx.begin_batch()?;
                    self.gpu.ctx.record_copy(&staging, dst, 0, 0, size);
                    self.gpu.ctx.finish_batch()?;
                }
                offset += size;
            }
        }
        anyhow::ensure!(offset == bytes.len(), "extra snapshot data");
        if std::env::var("LFM_ATTENTION").as_deref()==Ok("coalesced") && valid_tokens>0 {
            for l in 0..self.hp.n_layers {let kv=self.hp.kv_heads[l] as usize;if kv==0 {continue;}
                self.gpu.begin_batch()?;
                self.gpu.helper("attn_transpose",&[&state.buffers["qn"],state.keys[l].as_ref().unwrap(),state.values[l].as_ref().unwrap(),&state.buffers["a"],&state.buffers["attention_scores"],&state.buffers["position"],state.keys_transposed[l].as_ref().unwrap()],&[self.hp.head_dim as u32,self.hp.heads as u32,kv as u32,1,(valid_tokens-1) as u32,self.ctx as u32,0],(self.hp.head_dim*kv*valid_tokens).div_ceil(256) as u32,1)?;
                self.gpu.finish_batch()?;
            }
        }
        Ok(())
    }

    pub(super) fn forward_resident(&mut self, ids: &[u32], start_pos: i64) -> Result<Vec<f32>> {
        anyhow::ensure!(
            self.bounded_submissions || ids.len() <= 138,
            "unsplit long computation forbidden"
        );
        anyhow::ensure!(
            !ids.is_empty() && start_pos >= 0 && start_pos as usize + ids.len() <= self.ctx,
            "forward context range"
        );
        anyhow::ensure!(
            ids.iter().all(|&i| (i as usize) < self.hp.vocab),
            "token out of range"
        );
        if self.resident.is_none() {
            self.gpu.ctx.clear_decode_plan();
            self.resident = Some(ResidentState::new(
                &self.gpu,
                &self.hp,
                ids.len().max(crate::work_bound::PREFILL_CHUNK),
                self.ctx,
            )?);
        }
        let state = self.resident.as_ref().unwrap();
        anyhow::ensure!(
            ids.len() <= state.ncap && self.ctx == state.ctx,
            "resident graph capacity changed; create a fresh engine"
        );
        let b = |name| &state.buffers[name];
        let hp = &self.hp;
        let gpu = &self.gpu;
        let (dim, n) = (hp.dim, ids.len());
        gpu.ctx.set_work_context(start_pos as usize, n, None);
        let groups = |count: usize| count.div_ceil(256) as u32;
        let tok: Vec<f32> = ids.iter().map(|&i| i as f32).collect();
        gpu.write_f32(b("ids"), &tok);
        gpu.write_f32(b("position"), &[start_pos as f32]);
        let dynamic = self.token_only && n == 1;
        // Static full-logit forwards carry position AND context-dependent grid
        // dimensions. Never reuse their old push constants/grids. Only the
        // explicitly dynamic token-only graph is eligible for immutable replay.
        if !dynamic {
            gpu.ctx.clear_decode_plan();
        }
        if dynamic && start_pos as u32 >= gpu.ctx.decode_context_end.get() {
            gpu.ctx.clear_decode_plan();
        }
        if dynamic && !gpu.ctx.has_decode_plan() {
            gpu.ctx
                .decode_context_end
                .set(crate::work_bound::decode_end(
                    start_pos as u32,
                    self.ctx as u32,
                ));
        }
        let read_count = if self.token_only { 2 } else { hp.vocab };
        if n == 1 && gpu.ctx.has_decode_plan() {
            if dynamic {
                gpu.ctx.replay_reusable_decode()?;
                gpu.gpu_argmax_dispatches
                    .set(gpu.gpu_argmax_dispatches.get() + 2);
            } else {
                gpu.replay_decode(start_pos as u32)?;
            }
            return Ok(gpu.read_f32(b("readback"), read_count));
        }
        gpu.ctx.capture_decode_plan(dynamic);
        gpu.begin_batch()?;
        gpu.helper(
            "embedding",
            &[self.weight_buf("token_embd.weight"), b("ids"), b("x")],
            &[dim as u32, n as u32],
            groups(dim * n),
            1,
        )?;
        for l in 0..hp.n_layers {
            gpu.ctx.set_work_context(start_pos as usize, n, Some(l));
            let prefix = format!("blk.{l}");
            let w = |suffix: &str| self.weight_buf(&format!("{prefix}.{suffix}"));
            let f = |suffix: &str| self.f32_buf(&format!("{prefix}.{suffix}"));
            let kv = hp.kv_heads[l] as usize;
            gpu.rms_norm(b("x"), f("attn_norm.weight"), dim, n, hp.eps, b("h"))?;
            if kv > 0 {
                let width = hp.head_dim * kv;
                gpu.q6k_matmul(w("attn_q.weight"), 0, dim, dim, b("h"), n, b("q"))?;
                gpu.q6k_matmul(w("attn_k.weight"), 0, dim, width, b("h"), n, b("k"))?;
                gpu.q6k_matmul(w("attn_v.weight"), 0, dim, width, b("h"), n, b("v"))?;
                gpu.rms_norm(
                    b("q"),
                    f("attn_q_norm.weight"),
                    hp.head_dim,
                    hp.heads * n,
                    hp.eps,
                    b("qn"),
                )?;
                gpu.rms_norm(
                    b("k"),
                    f("attn_k_norm.weight"),
                    hp.head_dim,
                    kv * n,
                    hp.eps,
                    b("kn"),
                )?;
                let tscale = hp.rope_theta.powf(-2.0 / hp.head_dim as f32);
                if dynamic {
                    for (name, heads) in [("qn", hp.heads), ("kn", kv)] {
                        gpu.helper(
                            "rope_decode",
                            &[b(name), b("position")],
                            &[hp.head_dim as u32, heads as u32, 1, 0, tscale.to_bits()],
                            groups(hp.head_dim / 2 * heads),
                            1,
                        )?;
                    }
                } else {
                    gpu.rope(b("qn"), hp.head_dim, hp.heads, n, start_pos, tscale)?;
                    gpu.rope(b("kn"), hp.head_dim, kv, n, start_pos, tscale)?;
                }
                let keys = state.keys[l].as_ref().unwrap();
                let values = state.values[l].as_ref().unwrap();
                let mut kv_buffers = vec![b("kn"), b("v"), keys, values];
                if dynamic {
                    kv_buffers.push(b("position"));
                }
                let coalesced=std::env::var("LFM_ATTENTION").as_deref()==Ok("coalesced");
                if coalesced {
                    gpu.helper(if dynamic {"kv_write_transposed_decode"} else {"kv_write_transposed"},&[b("kn"),b("v"),keys,values,b("position"),state.keys_transposed[l].as_ref().unwrap()],&[width as u32,n as u32,start_pos as u32,self.ctx as u32],groups(width*n),1)?;
                } else {
                    gpu.helper(if dynamic {"kv_write_decode"} else {"kv_write"},&kv_buffers,&[width as u32,n as u32,start_pos as u32],groups(width*n),1)?;
                }
                if std::env::var("LFM_ATTENTION").as_deref() != Ok("reference") {
                    gpu.fast_attention(
                        &[
                            b("qn"),
                            keys,
                            values,
                            b("a"),
                            b("attention_scores"),
                            b("position"),
                            if coalesced {state.keys_transposed[l].as_ref().unwrap()} else {b("attention_kt")},
                            b("attention_value_parts"),
                        ],
                        &[
                            hp.head_dim as u32,
                            hp.heads as u32,
                            kv as u32,
                            n as u32,
                            start_pos as u32,
                            self.ctx as u32,
                            (1.0 / (hp.head_dim as f32).sqrt()).to_bits(),
                        ],
                        dynamic,
                    )?;
                } else if dynamic {
                    gpu.helper(
                        "attention_decode",
                        &[b("qn"), keys, values, b("a"), b("position")],
                        &[
                            hp.head_dim as u32,
                            hp.heads as u32,
                            kv as u32,
                            1,
                            start_pos as u32,
                            self.ctx as u32,
                            (1.0 / (hp.head_dim as f32).sqrt()).to_bits(),
                        ],
                        hp.heads as u32,
                        1,
                    )?;
                } else {
                    gpu.attention(
                        b("qn"),
                        keys,
                        values,
                        hp.head_dim,
                        hp.heads,
                        kv,
                        n,
                        start_pos,
                        self.ctx,
                        1.0 / (hp.head_dim as f32).sqrt(),
                        b("a"),
                    )?;
                }
                gpu.q6k_matmul(
                    w("attn_output.weight"),
                    0,
                    dim,
                    dim,
                    b("a"),
                    n,
                    b("projected"),
                )?;
            } else {
                gpu.q6k_matmul(
                    w("shortconv.in_proj.weight"),
                    0,
                    dim,
                    3 * dim,
                    b("h"),
                    n,
                    b("bcx"),
                )?;
                gpu.helper(
                    "shortconv",
                    &[
                        b("bcx"),
                        f("shortconv.conv.weight"),
                        state.conv[l].as_ref().unwrap(),
                        b("a"),
                    ],
                    &[dim as u32, n as u32],
                    groups(dim),
                    1,
                )?;
                gpu.q6k_matmul(
                    w("shortconv.out_proj.weight"),
                    0,
                    dim,
                    dim,
                    b("a"),
                    n,
                    b("projected"),
                )?;
            }
            gpu.helper(
                "residual",
                &[b("x"), b("projected")],
                &[(dim * n) as u32, 0],
                groups(dim * n),
                1,
            )?;
            let nf = if l == hp.n_layers - 1 { 1 } else { n };
            let offset = if nf == 1 { (n - 1) * dim } else { 0 };
            let norm_input = if nf < n {
                gpu.ctx
                    .record_copy(b("x"), b("last"), offset * 4, 0, dim * 4);
                b("last")
            } else {
                b("x")
            };
            gpu.rms_norm(norm_input, f("ffn_norm.weight"), dim, nf, hp.eps, b("h"))?;
            if l < hp.dense_lead {
                gpu.q6k_matmul(w("ffn_gate.weight"), 0, dim, hp.n_ff, b("h"), nf, b("gate"))?;
                gpu.q6k_matmul(w("ffn_up.weight"), 0, dim, hp.n_ff, b("h"), nf, b("up"))?;
                gpu.swiglu(b("gate"), b("up"), hp.n_ff * nf, b("sw"))?;
                gpu.q6k_matmul(
                    w("ffn_down.weight"),
                    0,
                    hp.n_ff,
                    dim,
                    b("sw"),
                    nf,
                    b("down"),
                )?;
                gpu.helper(
                    "residual",
                    &[b("x"), b("down")],
                    &[(dim * nf) as u32, offset as u32],
                    groups(dim * nf),
                    1,
                )?;
            } else {
                gpu.f32_matmul(
                    f("ffn_gate_inp.weight"),
                    dim,
                    hp.n_expert,
                    b("h"),
                    nf,
                    b("router"),
                )?;
                gpu.helper(
                    "route",
                    &[b("router"), f("exp_probs_b.bias"), b("selection")],
                    &[hp.n_expert as u32, nf as u32],
                    nf as u32,
                    1,
                )?;
                if nf > 1 {
                    gpu.helper(
                        "expert_groups",
                        &[b("selection"), b("expert_groups")],
                        &[nf as u32, hp.n_expert as u32],
                        groups(hp.n_expert),
                        1,
                    )?;
                }
                for (name, out) in [
                    ("ffn_gate_exps.weight", "gate"),
                    ("ffn_up_exps.weight", "up"),
                ] {
                    gpu.helper(
                        if nf > 1 { "q6k_grouped_compact" } else { "q6k_routed" },
                        &[
                            w(name),
                            b("h"),
                            b(out),
                            if nf > 1 {
                                b("expert_groups")
                            } else {
                                b("selection")
                            },
                        ],
                        &[
                            dim as u32,
                            hp.expert_ff as u32,
                            (dim / 256) as u32,
                            nf as u32,
                            0,
                        ],
                        hp.expert_ff.div_ceil(4) as u32,
                        (if nf > 1 { nf.div_ceil(4) * hp.n_expert } else { 4 }) as u32,
                    )?;
                }
                gpu.swiglu(b("gate"), b("up"), hp.expert_ff * nf * 4, b("sw"))?;
                gpu.helper(
                    if nf > 1 { "q6k_grouped_compact" } else { "q6k_routed" },
                    &[
                        w("ffn_down_exps.weight"),
                        b("sw"),
                        b("down"),
                        if nf > 1 {
                            b("expert_groups")
                        } else {
                            b("selection")
                        },
                    ],
                    &[
                        hp.expert_ff as u32,
                        dim as u32,
                        (hp.expert_ff / 256) as u32,
                        nf as u32,
                        1,
                    ],
                    dim.div_ceil(4) as u32,
                    (if nf > 1 { nf.div_ceil(4) * hp.n_expert } else { 4 }) as u32,
                )?;
                gpu.helper(
                    "combine",
                    &[b("down"), b("selection"), b("x")],
                    &[dim as u32, nf as u32, offset as u32],
                    groups(dim * nf),
                    1,
                )?;
            }
            // Keep every dispatch, token batch, buffer and arithmetic dependency intact.
            // Fence completion precedes command-buffer reuse; next dispatch retains
            // its ALL_COMMANDS memory barrier across the same-queue submission edge.
            if self.bounded_submissions && n > 1 {
                gpu.finish_batch()?;
                gpu.begin_batch()?;
            }
        }
        gpu.ctx
            .record_copy(b("x"), b("last"), (n - 1) * dim * 4, 0, dim * 4);
        gpu.rms_norm(
            b("last"),
            self.f32_buf("token_embd_norm.weight"),
            dim,
            1,
            hp.eps,
            b("norm"),
        )?;
        gpu.q6k_matmul(
            self.weight_buf("token_embd.weight"),
            0,
            dim,
            hp.vocab,
            b("norm"),
            1,
            b("logits"),
        )?;
        if self.token_only {
            gpu.helper(
                "argmax",
                &[b("logits"), b("argmax_partial")],
                &[hp.vocab as u32],
                hp.vocab.div_ceil(512) as u32,
                1,
            )?;
            gpu.helper(
                "argmax_finish",
                &[b("argmax_partial"), b("selected")],
                &[hp.vocab.div_ceil(512) as u32],
                1,
                1,
            )?;
            gpu.ctx.record_copy(b("selected"), b("readback"), 0, 0, 8);
        } else {
            gpu.ctx
                .record_copy(b("logits"), b("readback"), 0, 0, hp.vocab * 4);
        }
        gpu.finish_batch()?;
        gpu.ctx.capture_decode_plan(false);
        if dynamic {
            gpu.ctx.compile_reusable_decode()?;
        }
        Ok(gpu.read_f32(b("readback"), read_count))
    }

    /// Diagnostic only: copy the actual last full-vocabulary GPU output.
    /// Not called from production token-only inference or timing paths.
    pub fn snapshot_resident_logits(&self) -> Result<Vec<f32>> {
        let state=self.resident.as_ref().ok_or_else(|| anyhow::anyhow!("no resident state"))?;
        let out=self.gpu.alloc_host(self.hp.vocab*4)?;
        self.gpu.begin_batch()?;
        self.gpu.ctx.record_copy(&state.buffers["logits"],&out,0,0,self.hp.vocab*4);
        self.gpu.finish_batch()?;
        Ok(self.gpu.read_f32(&out,self.hp.vocab))
    }

    /// Read actual resident GPU cache bytes, not a CPU reconstruction. Only the
    /// initialized K/V prefix is meaningful; compare ALL convolution slots.
    pub fn snapshot_resident_cache(&self, valid_tokens: usize) -> Result<Vec<u8>> {
        let state = self
            .resident
            .as_ref()
            .ok_or_else(|| anyhow::anyhow!("no resident state"))?;
        anyhow::ensure!(valid_tokens <= self.ctx, "snapshot range");
        let mut bytes = Vec::new();
        for l in 0..self.hp.n_layers {
            let kv = self.hp.kv_heads[l] as usize;
            let sources: Vec<(&GpuBuffer, usize)> = if kv > 0 {
                let size = self.hp.head_dim * kv * valid_tokens * 4;
                vec![
                    (state.keys[l].as_ref().unwrap(), size),
                    (state.values[l].as_ref().unwrap(), size),
                ]
            } else {
                let conv = state.conv[l].as_ref().unwrap();
                vec![(conv, conv.size as usize)]
            };
            for (src, size) in sources {
                let readback = self.gpu.ctx.alloc_buffer(size, true)?;
                self.gpu.ctx.begin_batch()?;
                self.gpu.ctx.record_copy(src, &readback, 0, 0, size);
                self.gpu.ctx.finish_batch()?;
                let offset = bytes.len();
                bytes.resize(offset + size, 0);
                readback.read(0, &mut bytes[offset..]);
            }
        }
        Ok(bytes)
    }
}
