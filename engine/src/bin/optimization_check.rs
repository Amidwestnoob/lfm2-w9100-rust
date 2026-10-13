//! Real Q6 prompt/grouping numerical checks and matched warm kernel timings.
use anyhow::{ensure, Result};
use lfm2_engine::{model_file::ModelFile, ops::GpuOps};
use serde_json::json;
fn main() -> Result<()> {
    let a: Vec<String> = std::env::args().collect();
    let model = ModelFile::open(&a[1])?;
    let gpu = GpuOps::open()?;
    let mut cases = vec![];
    for n in [3usize, 17, 64] {
        for expert in [false, true] {
            let tensor = model
                .tensors
                .iter()
                .find(|t| {
                    if expert {
                        t.name.ends_with("ffn_gate_exps.weight")
                    } else {
                        t.name.ends_with("attn_q.weight")
                    }
                })
                .unwrap();
            let (input, output) = (tensor.ne[0] as usize, tensor.ne[1] as usize);
            let experts = if expert { tensor.ne[2] as usize } else { 1 };
            let w = gpu.upload_q6(&model.read_tensor_bytes(&tensor.name)?)?;
            let x: Vec<f32> = (0..input * n).map(|i| ((i as f32) * 0.013).sin()).collect();
            let xb = gpu.upload_to_device(bytemuck::cast_slice(&x))?;
            let count = output * n * if expert { 4 } else { 1 };
            let ob = gpu.alloc_host(count * 4)?;
            let cb = gpu.alloc_host(count * 4)?;
            let selection: Vec<f32> = (0..n)
                .flat_map(|j| {
                    (0..8).map(move |r| {
                        if r < 4 {
                            ((j / 3 + r) % experts) as f32
                        } else {
                            0.25
                        }
                    })
                })
                .collect();
            let sb = gpu.upload_to_device(bytemuck::cast_slice(&selection))?;
            let gb = gpu.ctx.alloc_buffer(experts * (4 * n + 1) * 4, false)?;
            let mut durations = vec![];
            for candidate in [false, true] {
                let out = if candidate { &cb } else { &ob };
                let mut times = vec![];
                for rep in 0..9 {
                    let before = gpu.ctx.total_gpu_ms.get();
                    gpu.begin_batch()?;
                    if expert {
                        if candidate {
                            gpu.helper(
                                "expert_groups",
                                &[&sb, &gb],
                                &[n as u32, experts as u32],
                                experts.div_ceil(256) as u32,
                                1,
                            )?;
                        }
                        gpu.helper(
                            if candidate {
                                "q6k_grouped"
                            } else {
                                "q6k_routed"
                            },
                            &[&w, &xb, out, if candidate { &gb } else { &sb }],
                            &[
                                input as u32,
                                output as u32,
                                (input / 256) as u32,
                                n as u32,
                                0,
                            ],
                            output.div_ceil(4) as u32,
                            (if candidate { experts * n } else { n * 4 }) as u32,
                        )?;
                    } else {
                        let tile = if candidate { 4 } else { 2 };
                        gpu.helper(
                            if candidate {
                                "q6k_prompt"
                            } else {
                                "q6k_prompt2"
                            },
                            &[&w, &xb, out],
                            &[
                                input as u32,
                                output as u32,
                                n as u32,
                                (input / 256) as u32,
                                0,
                            ],
                            output.div_ceil(4) as u32,
                            n.div_ceil(tile) as u32,
                        )?;
                    }
                    gpu.finish_batch()?;
                    if rep > 0 {
                        times.push(gpu.ctx.total_gpu_ms.get() - before);
                    }
                }
                durations.push(times);
            }
            let base = gpu.read_f32(&ob, count);
            let cand = gpu.read_f32(&cb, count);
            let d: f64 = base
                .iter()
                .zip(&cand)
                .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
                .sum();
            let denom: f64 = base.iter().map(|&x| (x as f64).powi(2)).sum();
            let rel = (d / denom.max(1e-30)).sqrt();
            ensure!(
                rel < 1e-3 && cand.iter().all(|x| x.is_finite()),
                "Q6 numerical check failed"
            );
            cases.push(json!({"n":n,"expert":expert,"tensor":tensor.name,"elements":count,"relative_l2":rel,"mismatched_bits":base.iter().zip(&cand).filter(|(x,y)|x.to_bits()!=y.to_bits()).count(),"baseline_gpu_ms":durations[0],"candidate_gpu_ms":durations[1]}));
        }
    }
    // Test GPU selection independently, including ties, tails and NaNs.
    for len in [1usize, 63, 513, 128003] {
        let mut values: Vec<f32> = (0..len).map(|i| -((i % 101) as f32)).collect();
        if len > 1 {
            values[0] = f32::NAN;
            values[len - 1] = 3.0;
        }
        let x = gpu.upload(bytemuck::cast_slice(&values))?;
        let partial = gpu.ctx.alloc_buffer(len.div_ceil(512) * 8, false)?;
        let result = gpu.alloc_host(8)?;
        gpu.begin_batch()?;
        gpu.helper(
            "argmax",
            &[&x, &partial],
            &[len as u32],
            len.div_ceil(512) as u32,
            1,
        )?;
        gpu.helper(
            "argmax_finish",
            &[&partial, &result],
            &[len.div_ceil(512) as u32],
            1,
            1,
        )?;
        gpu.finish_batch()?;
        let got = gpu.read_f32(&result, 2)[1] as usize;
        ensure!(got == len - 1, "argmax wrong {len}: {got}");
    }
    println!(
        "{}",
        json!({"device":gpu.device_name,"pass":true,"cases":cases,"argmax_cases":4})
    );
    Ok(())
}
