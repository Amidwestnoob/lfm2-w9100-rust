use lfm2_engine::kernels;
use sha2::Digest;
fn main() {
    for (name, spirv) in [
        ("swiglu", kernels::build_swiglu()),
        ("f32_matmul", kernels::build_f32_matmul()),
        ("rms_norm", kernels::build_rms_norm()),
        ("rope", kernels::build_rope()),
        ("attention", kernels::build_attention()),
        ("attention_unrolled", kernels::build_attention_unrolled()),
        ("attention_queryhoist", kernels::build_attention_queryhoist()),
        ("attention_unrolled_decode", kernels::build_attention_unrolled_decode()),
        ("q6k_aligned_matmul", kernels::build_q6k_aligned(false,false,1)),
        ("q6k_aligned_prompt", kernels::build_q6k_aligned(false,false,4)),
        ("q6k_aligned_grouped", kernels::build_q6k_aligned(true,true,4)),
        ("q6k_matmul", kernels::build_q6k_matmul()),
        ("q6k_matmul_4expert", kernels::build_q6k_matmul_4expert()),
        ("embedding", kernels::build_embedding()),
        ("residual", kernels::build_residual()),
        ("shortconv", kernels::build_shortconv()),
        ("route", kernels::build_route()),
        ("combine", kernels::build_combine()),
        ("kv_write", kernels::build_kv_write()),
        ("q6k_routed", kernels::build_q6k_routed()),
        ("q6k_prompt", kernels::build_q6k_prompt()),
    ] {
        let bytes: Vec<u8> = spirv.iter().flat_map(|w| w.to_le_bytes()).collect();
        if let Some(dir) = std::env::args().nth(1) {
            std::fs::create_dir_all(&dir).unwrap();
            std::fs::write(
                std::path::Path::new(&dir).join(format!("{name}.spv")),
                &bytes,
            )
            .unwrap();
        }
        let h = sha2::Sha256::digest(&bytes);
        println!("{} {}", name, hex::encode(h));
    }
}
