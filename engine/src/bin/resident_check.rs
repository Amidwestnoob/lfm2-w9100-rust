//! Additional W9100 regressions for optimized paths beyond the frozen seven-op harness.
//! CPU arithmetic below is a test oracle only; inference never calls this binary.
use anyhow::Result;
use lfm2_engine::{model_file::ModelFile, ops::GpuOps};

fn main() -> Result<()> {
    let path = std::env::args().nth(1).expect("model path");
    let model = ModelFile::open(&path)?;
    let input = model.tensor("token_embd.weight")?.ne[0] as usize;
    let output = 7usize; // Partial four-row group and odd column tiles.
    let bytes =
        model.read_tensor_range("token_embd.weight", 0, (input / 256 * 210 * output) as u64)?;
    let mut rows = Vec::new();
    for n in [2usize, 3, 5, 8] {
        let gpu = GpuOps::open()?;
        let w = gpu.upload_to_device(&bytes)?;
        let values: Vec<f32> = (0..input * n)
            .map(|i| ((i * 17 % 101) as f32 - 50.0) / 71.0)
            .collect();
        let x = gpu.upload(bytemuck::cast_slice(&values))?;
        let reference = gpu.alloc_host(output * n * 4)?;
        let tiled = gpu.alloc_host(output * n * 4)?;
        gpu.q6k_matmul(&w, 0, input, output, &x, n, &reference)?;
        gpu.begin_batch()?;
        gpu.q6k_matmul(&w, 0, input, output, &x, n, &tiled)?;
        gpu.finish_batch()?;
        let a = gpu.read_f32(&reference, output * n);
        let b = gpu.read_f32(&tiled, output * n);
        anyhow::ensure!(
            a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits()),
            "tiled Q6 changed arithmetic at n={n}"
        );
        rows.push(serde_json::json!({"operation":"q6k_prompt","columns":n,"rows":output,"bit_exact_gpu_reference":true}));
    }
    for experts in [4usize, 32, 64] {
        let gpu = GpuOps::open()?;
        let n = 3usize;
        let logits: Vec<f32> = (0..n * experts)
            .map(|i| {
                if i < experts {
                    0.0
                } else {
                    ((i * 13 % 19) as f32 - 9.0) / 8.0
                }
            })
            .collect();
        let bias = vec![0f32; experts];
        let scores = gpu.upload(bytemuck::cast_slice(&logits))?;
        let biases = gpu.upload(bytemuck::cast_slice(&bias))?;
        let out = gpu.alloc_host(n * 8 * 4)?;
        gpu.begin_batch()?;
        gpu.helper(
            "route",
            &[&scores, &biases, &out],
            &[experts as u32, n as u32],
            n as u32,
            1,
        )?;
        gpu.finish_batch()?;
        let actual = gpu.read_f32(&out, n * 8);
        for j in 0..n {
            let p: Vec<f32> = logits[j * experts..(j + 1) * experts]
                .iter()
                .map(|x| 1.0 / (1.0 + (-x).exp()))
                .collect();
            let mut ids: Vec<usize> = (0..experts).collect();
            ids.sort_by(|&a, &b| p[b].total_cmp(&p[a]).then(a.cmp(&b)));
            let sum = ids[..4]
                .iter()
                .fold(0f32, |s, &i| s + p[i])
                .max(6.103515625e-5);
            for rank in 0..4 {
                anyhow::ensure!(
                    actual[j * 8 + rank] as usize == ids[rank],
                    "routing order/tie mismatch"
                );
                anyhow::ensure!(
                    (actual[j * 8 + 4 + rank] - p[ids[rank]] / sum).abs() < 1e-6,
                    "routing probability mismatch"
                );
            }
        }
        rows.push(serde_json::json!({"operation":"route","experts":experts,"tokens":n,"stable_ties":true,"probabilities_pass":true}));
    }
    println!("{}", serde_json::json!({"status":"PASS","cases":rows}));
    Ok(())
}
