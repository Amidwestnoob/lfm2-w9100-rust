use anyhow::{ensure, Result};
use lfm2_engine::{model_file::ModelFile, ops::GpuOps};
use serde_json::json;
fn main() -> Result<()> {
    let model = ModelFile::open(&std::env::args().nth(1).unwrap())?;
    let gpu = GpuOps::open()?;
    let count = 8 * 1024 * 1024usize;
    let values: Vec<u32> = (0..count)
        .map(|i| (i as u32).wrapping_mul(1664525))
        .collect();
    let a = gpu.upload_to_device(bytemuck::cast_slice(&values))?;
    let b = gpu.ctx.alloc_buffer(count * 4, false)?;
    let mut times = vec![];
    for iteration in 0..18 {
        let before = gpu.ctx.total_gpu_ms.get();
        gpu.begin_batch()?;
        gpu.helper(
            "stream_copy",
            &[&a, &b],
            &[count as u32],
            (count / 256) as u32,
            1,
        )?;
        gpu.finish_batch()?;
        if iteration >= 2 {
            times.push(gpu.ctx.total_gpu_ms.get() - before);
        }
    }
    let verify = gpu.alloc_host(count * 4)?;
    gpu.ctx.begin_batch()?;
    gpu.ctx.record_copy(&b, &verify, 0, 0, count * 4);
    gpu.ctx.finish_batch()?;
    let mut read = vec![0u8; count * 4];
    verify.read(0, &mut read);
    ensure!(
        read == bytemuck::cast_slice::<u32, u8>(&values),
        "stream checksum mismatch"
    );
    let total_ms: f64 = times.iter().sum();
    let bw = (count * 8 * times.len()) as f64 / (total_ms / 1000.0);
    let mut breakdown = vec![];
    let mut weight_bytes = 0u64;
    for t in &model.tensors {
        let touched = if t.name.contains("_exps.weight") {
            t.nbytes * 4 / t.ne[2]
        } else {
            t.nbytes
        };
        weight_bytes += touched;
        breakdown.push(json!({"tensor":t.name,"bytes":touched,"expert_slices":if t.name.contains("_exps.weight"){4}else{0}}));
    }
    let embedding = model.tensor("token_embd.weight")?;
    weight_bytes += embedding.nbytes / embedding.ne[1];
    println!(
        "{}",
        json!({"device":gpu.device_name,"measured_streaming_bytes_per_second":bw,"decode_weight_bytes_per_token":weight_bytes,
        "timing_source":"Vulkan timestamp queries, sum of every submission","verified_bytes":count*4,"mismatched_bytes":0,"stream_read_bytes":count*4,"stream_write_bytes":count*4,"gpu_ms":times,
        "decode_bytes_definition":"logical full weight bytes used by one decode: four equal-size expert slices per MoE weight, other tensors in full, tied embedding output plus one input row; not measured DRAM transactions or cache misses",
        "weight_breakdown":breakdown})
    );
    Ok(())
}
