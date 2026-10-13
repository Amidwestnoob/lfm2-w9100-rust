//! Direct numerical/timing screen. Reference uses the old arithmetic, but NEW
//! split dispatches and preventive submission planner, never the failed binary.
use anyhow::{ensure, Result};
use lfm2_engine::ops::GpuOps;
use serde_json::json;
fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let end: usize = args[1].parse()?;
    let n: usize = args[2].parse()?;
    ensure!(
        n > 0 && n <= 64 && n <= end && end <= 4096,
        "screen envelope"
    );
    let (hd, heads, kv) = (64, 32, 8);
    let gpu = GpuOps::open()?;
    let data = |len, freq: f32| -> Vec<f32> {
        (0..len)
            .map(|i| half::f16::from_f32((i as f32 * freq).sin() * 0.5).to_f32())
            .collect()
    };
    let q = data(hd * heads * n, 0.0131);
    let k = data(hd * kv * end, 0.0079);
    let v = data(hd * kv * end, 0.0173);
    let qb = gpu.upload_to_device(bytemuck::cast_slice(&q))?;
    let kb = gpu.upload_to_device(bytemuck::cast_slice(&k))?;
    let vb = gpu.upload_to_device(bytemuck::cast_slice(&v))?;
    let a = gpu.alloc_host(q.len() * 4)?;
    let b = gpu.alloc_host(q.len() * 4)?;
    let scratch = gpu.ctx.alloc_buffer(heads * n * end * 4, false)?;
    let kt = gpu.ctx.alloc_buffer(hd * kv * end * 4, false)?;
    let parts = gpu.ctx.alloc_buffer(heads*n*hd*end.div_ceil(256)*4,false)?;
    let position = gpu.upload(bytemuck::cast_slice(&[(end - n) as f32]))?;
    let p = [
        64,
        32,
        8,
        n as u32,
        (end - n) as u32,
        end as u32,
        0.125f32.to_bits(),
    ];
    let mut times = vec![];
    if std::env::var("LFM_ATTENTION").as_deref()==Ok("coalesced") {
        gpu.begin_batch()?;
        gpu.helper("attn_transpose",&[&qb,&kb,&vb,&b,&scratch,&position,&kt],&p,(hd*kv*end).div_ceil(256) as u32,1)?;
        gpu.finish_batch()?;
    }
    for candidate in [false, true, false, true] {
        gpu.ctx.set_work_context(end - n, n, None);
        gpu.ctx.set_submission_context(
            if candidate {
                "candidate_attention"
            } else {
                "reference_attention"
            },
            end,
        );
        let before = gpu.ctx.total_gpu_ms.get();
        gpu.begin_batch()?;
        if candidate {
            gpu.fast_attention(&[&qb, &kb, &vb, &b, &scratch, &position, &kt, &parts], &p, n == 1)?;
        } else {
            gpu.attention(
                &qb,
                &kb,
                &vb,
                hd,
                heads,
                kv,
                n,
                (end - n) as i64,
                end,
                0.125,
                &a,
            )?;
        }
        gpu.finish_batch()?;
        times.push(gpu.ctx.total_gpu_ms.get() - before);
    }
    let av = gpu.read_f32(&a, q.len());
    let bv = gpu.read_f32(&b, q.len());
    let numerator: f64 = av
        .iter()
        .zip(&bv)
        .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
        .sum();
    let denom: f64 = av.iter().map(|&x| (x as f64).powi(2)).sum();
    let rel = (numerator / denom.max(1e-30)).sqrt();
    let max_abs = av
        .iter()
        .zip(&bv)
        .map(|(&x, &y)| (x - y).abs())
        .fold(0.0f32, f32::max);
    // Independent f64 reference for selected heads/queries (not a token oracle).
    let mut cpu_max = 0.0f64;
    for hh in [0, 7, 31] {
        for j in [0, n - 1] {
            let nt = end - n + j + 1;
            let kh = hh * kv / heads;
            let qr = hh * hd + j * hd * heads;
            let scores: Vec<f64> = (0..nt)
                .map(|t| {
                    (0..hd)
                        .map(|d| q[qr + d] as f64 * k[t * hd * kv + kh * hd + d] as f64)
                        .sum::<f64>()
                        * 0.125
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let sum: f64 = weights.iter().sum();
            for d in 0..hd {
                let y = (0..nt)
                    .map(|t| weights[t] * v[t * hd * kv + kh * hd + d] as f64)
                    .sum::<f64>()
                    / sum;
                cpu_max = cpu_max.max((bv[qr + d] as f64 - y).abs());
            }
        }
    }
    println!(
        "{}",
        json!({"device":gpu.device_name,"end":end,"n":n,"relative_l2":rel,"max_abs":max_abs,"cpu_f64_max_abs":cpu_max,"elements":q.len(),"gpu_ms_A1_B1_A2_B2":times,"pass":rel<1e-3&&cpu_max<1e-4})
    );
    ensure!(
        rel < 1e-3 && cpu_max < 1e-4 && bv.iter().all(|v| v.is_finite()),
        "attention numerical mismatch"
    );
    Ok(())
}
