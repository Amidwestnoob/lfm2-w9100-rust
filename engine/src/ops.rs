//! GPU operators (ash/Vulkan compute) for the LFM2 graph.
//!
//! Kernel language: Rust typed IR for all seven operations.
//! Compiler route: Rust `kernels::build_*()` → rspirv binary builder → SPIR-V
//! → ash/Vulkan pipeline. No textual shader frontend or opaque kernel assets.
//! Tensor layouts and arithmetic order are retained from the reference host.

use crate::kernels;
use crate::vulkan::{ComputePipeline, GpuBuffer, VulkanContext, W9100_DEVICE_ID};
use anyhow::{anyhow, Result};
use std::cell::{Cell, RefCell};
use std::collections::HashMap;

// All seven kernels are now authored in Rust via the kernels::build_*() functions.
// No hand-authored GLSL/WGSL strings remain.

pub struct GpuOps {
    pub ctx: VulkanContext,
    pipelines: RefCell<
        HashMap<
            &'static str,
            (
                ash::vk::Pipeline,
                ash::vk::PipelineLayout,
                ash::vk::DescriptorSetLayout,
            ),
        >,
    >,
    dispatch_count: Cell<u64>,
    pub gpu_argmax_dispatches: Cell<u64>,
    pub expert_grouped_dispatches: Cell<u64>,
    pub prefill_weight_reuse_tokens: Cell<usize>,
    batching: Cell<bool>,
    batch_sets: RefCell<Vec<ash::vk::DescriptorSet>>,
    resident_sets: RefCell<
        HashMap<
            (
                ash::vk::DescriptorSetLayout,
                Vec<(ash::vk::Buffer, u64, u64)>,
            ),
            ash::vk::DescriptorSet,
        >,
    >,
    pub device_name: String,
    pub device_id: u32,
    pub q6_aligned: bool,
}

pub(crate) fn validate_fixed_expert(p:&[u32],gx:u32,gy:u32,grouped:bool)->Result<()> {
    anyhow::ensure!(p.len()==5,"fixed Q6 push size");
    anyhow::ensure!((p[0],p[1],p[2],p[4])==(2048,1792,8,0) || (p[0],p[1],p[2],p[4])==(1792,2048,7,1),"fixed Q6 dimensions");
    anyhow::ensure!(p[3]>0 && p[3]<=64 && gx==p[1]/4,"fixed Q6 full-row geometry");
    anyhow::ensure!(if grouped {gy==32*p[3].div_ceil(4)} else {gy==4*p[3]},"fixed Q6 column geometry");
    Ok(())
}
impl GpuOps {
    pub fn open() -> Result<Self> {
        let ctx = VulkanContext::open_w9100()?;
        let device_name = ctx.device_name();
        let device_id = ctx.props.device_id;
        if device_id != W9100_DEVICE_ID {
            return Err(anyhow!(
                "expected W9100 (id 0x{W9100_DEVICE_ID:04x}) but got {device_name} (id 0x{device_id:04x})"
            ));
        }
        Ok(GpuOps {
            ctx,
            pipelines: RefCell::new(HashMap::new()),
            dispatch_count: Cell::new(0),
            gpu_argmax_dispatches: Cell::new(0),
            expert_grouped_dispatches: Cell::new(0),
            prefill_weight_reuse_tokens: Cell::new(0),
            batching: Cell::new(false),
            batch_sets: RefCell::new(Vec::new()),
            resident_sets: RefCell::new(HashMap::new()),
            device_name,
            device_id,
            q6_aligned: std::env::var("LFM_Q6_ALIGNED").as_deref()==Ok("1"),
        })
    }

    /// Total completed compute dispatches.
    pub fn dispatch_count(&self) -> u64 {
        self.dispatch_count.get()
    }
    pub fn replay_decode(&self, position: u32) -> Result<()> {
        let count = self.ctx.replay_decode_plan(position)?;
        self.dispatch_count.set(self.dispatch_count.get() + count);
        Ok(())
    }
    pub fn begin_batch(&self) -> Result<()> {
        anyhow::ensure!(!self.batching.get(), "nested batch");
        self.ctx.begin_batch()?;
        self.batching.set(true);
        Ok(())
    }
    pub fn finish_batch(&self) -> Result<()> {
        let result = self.ctx.finish_batch();
        self.batching.set(false);
        for set in self.batch_sets.borrow_mut().drain(..) {
            self.ctx.free_descriptor_set(set);
        }
        result
    }

    fn pipeline(
        &self,
        name: &'static str,
    ) -> Result<(
        ash::vk::Pipeline,
        ash::vk::PipelineLayout,
        ash::vk::DescriptorSetLayout,
    )> {
        if let Some(e) = self.pipelines.borrow().get(name).copied() {
            return Ok(e);
        }
        // Rust-authored kernels: each build_*() returns SPIR-V words.
        let (spirv, nbind, push_size): (Vec<u32>, u32, u32) = match name {
            "q6k_matmul" => (if self.q6_aligned {kernels::build_q6k_aligned(false,false,1)} else {kernels::build_q6k_matmul()}, 3, 20),
            "q6k_prompt" => (if self.q6_aligned {kernels::build_q6k_aligned(false,false,4)} else {kernels::build_q6k_prompt()}, 3, 20),
            "q6k_prompt2" => (if self.q6_aligned {kernels::build_q6k_aligned(false,false,2)} else {kernels::build_q6k_prompt2()}, 3, 20),
            "q6k_matmul_aligned" => (kernels::build_q6k_aligned(false,false,1),3,20),
            "q6k_prompt_aligned" => (kernels::build_q6k_aligned(false,false,4),3,20),
            "q6k_routed_aligned" => (kernels::build_q6k_aligned(true,true,1),4,20),
            "q6k_grouped_aligned" => (kernels::build_q6k_aligned(true,true,4),4,20),
            "stream_copy" => (kernels::build_stream_copy(), 2, 4),
            "q6k_matmul_4expert" => (if self.q6_aligned {kernels::build_q6k_aligned(true,false,1)} else {kernels::build_q6k_matmul_4expert()}, 3, 28),
            "f32_matmul" => (kernels::build_f32_matmul(), 3, 12),
            "rms_norm" => (kernels::build_rms_norm(), 3, 12),
            "rope" => (kernels::build_rope(), 1, 20),
            "attention" => (kernels::build_attention(), 4, 28),
            "attention_unrolled" => (kernels::build_attention_unrolled(), 4, 28),
            "attention_queryhoist" => (kernels::build_attention_queryhoist(), 4, 28),
            "attention_unrolled_decode" => (kernels::build_attention_unrolled_decode(), 5, 28),
            "attn_score" => (kernels::build_attention_fast(0, false), 6, 28),
            "attn_transpose" => (kernels::build_attention_transpose(false),7,28),
            "attn_score_coalesced" => (kernels::build_attention_score_coalesced(false),7,28),
            "attn_score_coalesced_decode" => (kernels::build_attention_score_coalesced(true),7,28),
            "kv_write_transposed" => (kernels::build_kv_write_transposed(false),6,16),
            "kv_write_transposed_decode" => (kernels::build_kv_write_transposed(true),6,16),
            "attn_transpose_decode" => (kernels::build_attention_transpose(true),7,28),
            "attn_score_transposed" => (kernels::build_attention_score_transposed(false),7,28),
            "attn_score_transposed_decode" => (kernels::build_attention_score_transposed(true),7,28),
            "attn_norm" => (kernels::build_attention_fast(1, false), 6, 28),
            "attn_value" => (kernels::build_attention_fast(2, false), 6, 28),
            "attn_score_decode" => (kernels::build_attention_fast(0, true), 6, 28),
            "attn_norm_decode" => (kernels::build_attention_fast(1, true), 6, 28),
            "attn_value_decode" => (kernels::build_attention_fast(2, true), 6, 28),
            "attn_value_split" => (kernels::build_attention_value_split(false,false),8,28),
            "attn_value_split_decode" => (kernels::build_attention_value_split(true,false),8,28),
            "attn_value_merge" => (kernels::build_attention_value_split(false,true),8,28),
            "attn_value_merge_decode" => (kernels::build_attention_value_split(true,true),8,28),
            "swiglu" => (kernels::build_swiglu(), 3, 4),
            "embedding" => (kernels::build_embedding_layout(self.q6_aligned), 3, 8),
            "residual" => (kernels::build_residual(), 2, 8),
            "shortconv" => (kernels::build_shortconv(), 4, 8),
            "route" => (kernels::build_route(), 3, 8),
            "combine" => (kernels::build_combine(), 3, 12),
            "kv_write" => (kernels::build_kv_write(), 4, 12),
            "q6k_routed" => (if self.q6_aligned {kernels::build_q6k_scale_pair(true,true,1,false)} else {kernels::build_q6k_routed()}, 4, 20),
            "q6k_grouped" => (if self.q6_aligned {kernels::build_q6k_aligned(true,true,4)} else {kernels::build_q6k_grouped()}, 4, 20),
            "q6k_grouped_compact" => (if self.q6_aligned {kernels::build_q6k_scale_pair(true,true,4,true)} else {kernels::build_q6k_grouped_compact(false)},4,20),
            "q6k_grouped_compact_aligned" => (kernels::build_q6k_grouped_compact(true),4,20),
            "expert_groups" => (kernels::build_expert_groups(), 2, 8),
            "argmax" => (kernels::build_argmax(false), 2, 4),
            "argmax_finish" => (kernels::build_argmax(true), 2, 4),
            "rope_decode" => (kernels::build_rope_decode(), 2, 20),
            "kv_write_decode" => (kernels::build_kv_write_decode(), 5, 12),
            "attention_decode" => (kernels::build_attention_decode(), 5, 28),
            _ => return Err(anyhow!("unknown pipeline {name}")),
        };
        let pipe = self
            .ctx
            .create_compute_pipeline(&spirv, nbind, push_size + 8)?;
        self.ctx.register_pipeline(pipe.pipeline, name);
        let key = (
            pipe.pipeline,
            pipe.pipeline_layout,
            pipe.descriptor_set_layout,
        );
        self.pipelines.borrow_mut().insert(name, key);
        match name {
            "rope" => self.ctx.set_decode_position_offset(key.0, 12),
            "kv_write" => self.ctx.set_decode_position_offset(key.0, 8),
            "attention" => self.ctx.set_decode_position_offset(key.0, 16),
            _ => (),
        }
        Ok(key)
    }

    /// Dispatch a resident-graph helper with explicitly specified workgroups.
    pub fn helper(
        &self,
        name: &'static str,
        buffers: &[&GpuBuffer],
        push: &[u32],
        groups: u32,
        gy: u32,
    ) -> Result<()> {
        if self.q6_aligned && matches!(name,"q6k_routed"|"q6k_grouped_compact") {
            validate_fixed_expert(push,groups,gy,name=="q6k_grouped_compact")?;
        }
        // Small tails retain the byte-identical accepted compact shader.
        // It uses identical geometry and the same stripped-name work bound.
        let name=if self.q6_aligned && name=="q6k_grouped_compact" && push[3]<8 {
            "q6k_grouped_compact_aligned"
        } else {name};
        let (p, pl, dsl) = self.pipeline(name)?;
        self.dispatch(buffers, p, pl, dsl, push, groups * 256, gy)?;
        if name.starts_with("argmax") {
            self.gpu_argmax_dispatches
                .set(self.gpu_argmax_dispatches.get() + 1);
        }
        if name.starts_with("q6k_grouped") {
            self.expert_grouped_dispatches
                .set(self.expert_grouped_dispatches.get() + 1);
        }
        Ok(())
    }

    /// Allocate a host-visible storage buffer holding `bytes`.
    pub fn fast_attention(
        &self,
        buffers: &[&GpuBuffer],
        push: &[u32],
        dynamic: bool,
    ) -> Result<()> {
        anyhow::ensure!(push[0] == 64, "fast attention head dimension");
        let mode=std::env::var("LFM_ATTENTION").unwrap_or_default();
        let transposed=mode=="transposed";
        let coalesced=mode=="coalesced";
        // Preserve accepted arithmetic for the small-context region where the
        // reordered path's real-activation errors did not dominate reference.
        // This uses unchanged kernels and their original conservative bounds.
        let envelope=if dynamic {crate::work_bound::decode_end(push[4],push[5])} else {push[4]+push[3]};
        let accepted_limit=if push[3]>1 {1280} else {1024};
        if coalesced && envelope<=accepted_limit {
            if dynamic {
                self.helper("attention_unrolled_decode",&[buffers[0],buffers[1],buffers[2],buffers[3],buffers[5]],push,push[1]*push[3],1)?;
            } else {
                self.helper("attention_queryhoist",&buffers[..4],push,push[1]*push[3],1)?;
                if push[3]>1 {
                    if let Ok(dir)=std::env::var("LFM_CAPTURE_ATTENTION") {
                        if [64,138,256,512,576,768,1024,1088,1152,1280].contains(&envelope) {self.capture_attention(&dir,buffers,push)?;}
                    }
                }
            }
            return Ok(());
        }
        if dynamic && mode!="three_stage" && !transposed && !coalesced {
            return self.helper(
                "attention_unrolled_decode",
                &[buffers[0], buffers[1], buffers[2], buffers[3], buffers[5]],
                push,
                push[1] * push[3],
                1,
            );
        }
        if !dynamic && mode!="three_stage" && !transposed && !coalesced {
            return self.helper(
                "attention_queryhoist",
                &buffers[..4],
                push,
                push[1] * push[3],
                1,
            );
        }
        if transposed {
            let end=if dynamic {crate::work_bound::decode_end(push[4],push[5])} else {push[4]+push[3]};
            self.helper(if dynamic {"attn_transpose_decode"} else {"attn_transpose"},buffers,push,(push[0]*push[2]*end).div_ceil(256),1)?;
            self.helper(if dynamic {"attn_score_transposed_decode"} else {"attn_score_transposed"},buffers,push,push[1]*push[3]*push[5].div_ceil(64),1)?;
        }
        if coalesced {
            let end=if dynamic {crate::work_bound::decode_end(push[4],push[5])} else {push[4]+push[3]};
            self.helper(if dynamic {"attn_score_coalesced_decode"} else {"attn_score_coalesced"},&buffers[..7],push,push[2]*push[3]*end.div_ceil(64),1)?;
        }
        for (stage, names) in [
            (0, ["attn_score", "attn_score_decode"]),
            (1, ["attn_norm", "attn_norm_decode"]),
            (2, ["attn_value", "attn_value_decode"]),
        ] {
            if stage==0 && (transposed||coalesced) {continue;}
            let rows = push[1] * push[3];
            if stage==2 && coalesced {
                let end=if dynamic {crate::work_bound::decode_end(push[4],push[5])} else {push[4]+push[3]};
                self.helper(if dynamic {"attn_value_split_decode"} else {"attn_value_split"},buffers,push,rows*end.div_ceil(256),1)?;
                self.helper(if dynamic {"attn_value_merge_decode"} else {"attn_value_merge"},buffers,push,rows,1)?;
                continue;
            }
            self.helper(
                names[usize::from(dynamic)],
                &buffers[..6],
                push,
                rows * if stage == 0 { push[5].div_ceil(64) } else { 1 },
                1,
            )?;
        }
        if !dynamic && push[3]>1 {
            if let Ok(dir)=std::env::var("LFM_CAPTURE_ATTENTION") {
                let end=push[4]+push[3];
                if [1344,1408,1536,2048,3840,4096].contains(&end) {
                    self.capture_attention(&dir,buffers,push)?;
                }
            }
        }
        Ok(())
    }

    pub fn upload(&self, bytes: &[u8]) -> Result<GpuBuffer> {
        let size = bytes.len().max(4);
        let buf = self.ctx.alloc_buffer(size, true)?;
        if !bytes.is_empty() {
            buf.write(0, bytes);
        }
        Ok(buf)
    }

    pub fn alloc_host(&self, nbytes: usize) -> Result<GpuBuffer> {
        self.ctx.alloc_buffer(nbytes.max(4), true)
    }

    /// Upload `bytes` into a device-local buffer (weights path).
    pub fn upload_to_device(&self, bytes: &[u8]) -> Result<GpuBuffer> {
        self.ctx.upload_to_device(bytes)
    }
    pub fn upload_q6(&self, bytes: &[u8]) -> Result<GpuBuffer> {
        if self.q6_aligned {self.ctx.upload_to_device(&crate::q6k::repack_aligned(bytes))} else {self.ctx.upload_to_device(bytes)}
    }

    pub fn read_f32(&self, buf: &GpuBuffer, n: usize) -> Vec<f32> {
        let mut raw = vec![0u8; n * 4];
        buf.read(0, &mut raw);
        bytemuck::cast_slice(&raw).to_vec()
    }

    pub fn write_f32(&self, buf: &GpuBuffer, v: &[f32]) {
        buf.write(0, bytemuck::cast_slice(v));
    }

    pub fn q6k_matmul(
        &self,
        w: &GpuBuffer,
        w_base: usize,
        in_dim: usize,
        out_dim: usize,
        x: &GpuBuffer,
        n: usize,
        out: &GpuBuffer,
    ) -> Result<()> {
        let tile = if n > 1 && self.batching.get() { 4 } else { 1 };
        if tile > 1 {
            self.prefill_weight_reuse_tokens
                .set(self.prefill_weight_reuse_tokens.get().max(n.min(tile)));
        }
        let (p, pl, dsl) = self.pipeline(if tile > 1 { "q6k_prompt" } else { "q6k_matmul" })?;
        let blocks = in_dim / crate::q6k::QK_K;
        let push = [
            in_dim as u32,
            out_dim as u32,
            n as u32,
            blocks as u32,
            w_base as u32,
        ];
        // dispatch() uses 256 logical items/group; cooperative Q6 handles four rows/group.
        self.dispatch(
            &[w, x, out],
            p,
            pl,
            dsl,
            &push,
            out_dim.div_ceil(4) as u32 * 256,
            n.div_ceil(tile) as u32,
        )
    }

    pub fn q6k_matmul_4expert(
        &self,
        w: &GpuBuffer,
        in_dim: usize,
        out_dim: usize,
        experts: [u32; 4],
        x: &GpuBuffer,
        out: &GpuBuffer,
    ) -> Result<()> {
        let (p, pl, dsl) = self.pipeline("q6k_matmul_4expert")?;
        let blocks = in_dim / crate::q6k::QK_K;
        let push = [
            in_dim as u32,
            out_dim as u32,
            blocks as u32,
            experts[0],
            experts[1],
            experts[2],
            experts[3],
        ];
        self.dispatch(
            &[w, x, out],
            p,
            pl,
            dsl,
            &push,
            out_dim.div_ceil(4) as u32 * 256,
            4,
        )
    }

    pub fn f32_matmul(
        &self,
        w: &GpuBuffer,
        in_dim: usize,
        out_dim: usize,
        x: &GpuBuffer,
        n: usize,
        out: &GpuBuffer,
    ) -> Result<()> {
        let (p, pl, dsl) = self.pipeline("f32_matmul")?;
        let push = [in_dim as u32, out_dim as u32, n as u32];
        self.dispatch(
            &[w, x, out],
            p,
            pl,
            dsl,
            &push,
            out_dim as u32 * 256,
            n as u32,
        )
    }

    pub fn rms_norm(
        &self,
        x: &GpuBuffer,
        scale: &GpuBuffer,
        dim: usize,
        n: usize,
        eps: f32,
        out: &GpuBuffer,
    ) -> Result<()> {
        let (p, pl, dsl) = self.pipeline("rms_norm")?;
        let push = [dim as u32, n as u32, eps.to_bits()];
        self.dispatch(&[x, scale, out], p, pl, dsl, &push, n as u32 * 256, 1)
    }

    pub fn rope(
        &self,
        q: &GpuBuffer,
        hd: usize,
        heads: usize,
        n: usize,
        start_pos: i64,
        tscale: f32,
    ) -> Result<()> {
        let (p, pl, dsl) = self.pipeline("rope")?;
        let push = [
            hd as u32,
            heads as u32,
            n as u32,
            start_pos as u32,
            tscale.to_bits(),
        ];
        self.dispatch(&[q], p, pl, dsl, &push, (heads * n * (hd / 2)) as u32, 1)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn attention(
        &self,
        q: &GpuBuffer,
        keys: &GpuBuffer,
        values: &GpuBuffer,
        hd: usize,
        heads: usize,
        kv: usize,
        n: usize,
        start_pos: i64,
        ctx: usize,
        scale: f32,
        out: &GpuBuffer,
    ) -> Result<()> {
        let (p, pl, dsl) = self.pipeline("attention")?;
        let push = [
            hd as u32,
            heads as u32,
            kv as u32,
            n as u32,
            start_pos as u32,
            ctx as u32,
            scale.to_bits(),
        ];
        anyhow::ensure!(
            ctx <= 4096 && start_pos >= 0 && start_pos as usize + n <= ctx,
            "attention context out of range"
        );
        self.dispatch(
            &[q, keys, values, out],
            p,
            pl,
            dsl,
            &push,
            (heads * n) as u32 * 256,
            1,
        )
    }

    pub fn swiglu(
        &self,
        gate: &GpuBuffer,
        up: &GpuBuffer,
        count: usize,
        out: &GpuBuffer,
    ) -> Result<()> {
        let (p, pl, dsl) = self.pipeline("swiglu")?;
        let push = [count as u32];
        self.dispatch(&[gate, up, out], p, pl, dsl, &push, count as u32, 1)
    }

    #[allow(clippy::too_many_arguments)]
    fn dispatch(
        &self,
        bufs: &[&GpuBuffer],
        pipeline: ash::vk::Pipeline,
        pipeline_layout: ash::vk::PipelineLayout,
        dsl: ash::vk::DescriptorSetLayout,
        push: &[u32],
        gx: u32,
        gy: u32,
    ) -> Result<()> {
        let bindings: Vec<(ash::vk::Buffer, ash::vk::DeviceSize)> =
            bufs.iter().map(|b| (b.buffer, b.size)).collect();
        let set = if self.batching.get() {
            let key = (
                dsl,
                bufs.iter()
                    .map(|b| (b.buffer, b.size, b.generation))
                    .collect(),
            );
            let cached = self.resident_sets.borrow().get(&key).copied();
            if let Some(set) = cached {
                set
            } else {
                let set = self.ctx.write_descriptor_set(dsl, &bindings)?;
                self.resident_sets.borrow_mut().insert(key, set);
                set
            }
        } else {
            self.ctx.write_descriptor_set(dsl, &bindings)?
        };
        let pipe = ComputePipeline {
            pipeline,
            pipeline_layout,
            descriptor_set_layout: dsl,
        };
        let name = self.ctx.pipeline_name(pipeline);
        let tiles = crate::work_bound::tiles(name, push, gx.div_ceil(256), gy)?;
        if !self.batching.get() {
            self.ctx.begin_batch()?;
        }
        for tile in tiles {
            let mut extended = push.to_vec();
            extended.extend_from_slice(&[tile.x, tile.y]);
            self.ctx.record_compute(
                &pipe,
                set,
                bytemuck::cast_slice(&extended),
                tile.gx,
                tile.gy,
            )?;
            self.dispatch_count.set(self.dispatch_count.get() + 1);
        }
        if !self.batching.get() {
            let result = self.ctx.finish_batch();
            self.ctx.free_descriptor_set(set);
            result?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernels_compile_to_spirv() {
        // Verify all seven Rust-authored kernels produce non-empty SPIR-V.
        assert!(!kernels::build_swiglu().is_empty());
        assert!(!kernels::build_f32_matmul().is_empty());
        assert!(!kernels::build_rms_norm().is_empty());
        assert!(!kernels::build_rope().is_empty());
        assert!(!kernels::build_attention().is_empty());
        for stage in 0..3 {
            for dynamic in [false, true] {
                let words = kernels::build_attention_fast(stage, dynamic);
                let module = rspirv::dr::load_words(&words).unwrap();
                assert_eq!(words, rspirv::binary::Assemble::assemble(&module));
            }
        }
        assert!(!kernels::build_q6k_matmul().is_empty());
        for (e,r,t,c) in [(false,false,1,false),(false,false,4,false),(false,false,2,false),(true,false,1,false),(true,true,1,false),(true,true,4,false),(true,true,4,true)] {
            let words=kernels::build_q6k_scale_pair(e,r,t,c);
            let module=rspirv::dr::load_words(&words).unwrap();
            assert_eq!(words,rspirv::binary::Assemble::assemble(&module));
        }
        assert!(!kernels::build_q6k_matmul_4expert().is_empty());
    }
}
