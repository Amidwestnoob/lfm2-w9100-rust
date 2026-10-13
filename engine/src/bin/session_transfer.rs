//! Time real host-visible session round trips separately from forward/decode.
use anyhow::{Result,ensure};
use lfm2_engine::{gpu_lfm::GpuEngine,model_file::ModelFile};
use sha2::{Digest,Sha256};
use serde_json::json;
use std::time::Instant;
fn main()->Result<()> {
 let a:Vec<String>=std::env::args().collect();let m=ModelFile::open(&a[1])?;
 let mut e=GpuEngine::load(&m)?;e.set_ctx(4096);
 let request:serde_json::Value=serde_json::from_str(&std::fs::read_to_string(&a[2])?)?;
 let ids:Vec<u32>=request["tokens"].as_array().unwrap().iter().map(|x|x.as_u64().unwrap() as u32).collect();
 ensure!(ids.len()==138,"frozen short transfer measurement");
 for (i,chunk) in ids.chunks(64).enumerate(){e.forward_token(chunk,i*64)?;}
 let mut cases=vec![];
 for _ in 0..3 {
  let start=Instant::now();let before=e.gpu.ctx.total_gpu_ms.get();let bytes=e.snapshot_resident_cache(ids.len())?;
  let snapshot_ms=start.elapsed().as_secs_f64()*1000.;let snapshot_gpu_ms=e.gpu.ctx.total_gpu_ms.get()-before;
  let start=Instant::now();let before=e.gpu.ctx.total_gpu_ms.get();e.reset_session()?;
  let reset_ms=start.elapsed().as_secs_f64()*1000.;let reset_gpu_ms=e.gpu.ctx.total_gpu_ms.get()-before;
  let start=Instant::now();let before=e.gpu.ctx.total_gpu_ms.get();e.restore_session(ids.len(),&bytes)?;
  let restore_ms=start.elapsed().as_secs_f64()*1000.;let restore_gpu_ms=e.gpu.ctx.total_gpu_ms.get()-before;
  let check=e.snapshot_resident_cache(ids.len())?;ensure!(bytes==check,"restored state differs");
  cases.push(json!({"valid_tokens":ids.len(),"transfer_bytes_each_direction":bytes.len(),"snapshot_ms":snapshot_ms,"snapshot_gpu_ms":snapshot_gpu_ms,"restore_ms":restore_ms,"restore_gpu_ms":restore_gpu_ms,"reset_ms":reset_ms,"reset_gpu_ms":reset_gpu_ms,"restored_bytes_exact":true,"state_sha256":hex::encode(Sha256::digest(&bytes))}));
 }
 println!("{}",json!({"device":e.device_name(),"aligned":e.gpu.q6_aligned,"cases":cases}));Ok(())
}
