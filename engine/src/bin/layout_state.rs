//! Real initialized KV/recurrent bytes and complete logits, including final token forward.
use anyhow::{Result,ensure};
use lfm2_engine::{gpu_lfm::GpuEngine,model_file::ModelFile};
use serde_json::json;
use sha2::{Digest,Sha256};
fn main()->Result<()> {
 let args:Vec<String>=std::env::args().collect();
 let model=ModelFile::open(&args[1])?;let mut engine=GpuEngine::load(&model)?;engine.set_ctx(4096);
 let input:serde_json::Value=serde_json::from_str(&std::fs::read_to_string(&args[2])?)?;
 let ids:Vec<u32>=input["tokens"].as_array().unwrap().iter().map(|x|x.as_u64().unwrap() as u32).collect();
 ensure!(ids.len()<=138,"short state validation only");
 let mut cases=vec![];let mut all=vec![];
 for len in [2usize,13,138] {
  engine.reset_session()?;let mut logits=vec![];let mut pos=0;
  for chunk in ids[..len].chunks(64) {logits=engine.forward(chunk,pos as i64)?;pos+=chunk.len();}
  let mut tokens=vec![];let mut forwards=vec![];
  for step in 0..=32 {
   let cache=engine.snapshot_resident_cache(pos)?;
   let bytes=bytemuck::cast_slice::<f32,u8>(&logits);
   forwards.push(json!({"step":step,"position":pos,"logits_elements":logits.len(),"logits_sha256":hex::encode(Sha256::digest(bytes)),"cache_bytes":cache.len(),"cache_sha256":hex::encode(Sha256::digest(&cache))}));
   if step==32 {break;}
   let (mut best,mut value)=(0,f32::NEG_INFINITY);
   for (i,&v) in logits.iter().enumerate() {if v>value {best=i;value=v;}}
   tokens.push(best as u32);logits=engine.forward(&[best as u32],pos as i64)?;pos+=1;
  }
  cases.push(json!({"prompt_tokens":len,"tokens":tokens,"forwards":forwards,"final_forward_compared":true}));all.extend(tokens);
 }
 println!("{}",json!({"device":engine.device_name(),"aligned":engine.gpu.q6_aligned,"cases":cases}));Ok(())
}
