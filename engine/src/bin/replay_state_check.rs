//! Full real logits/cache equality: freshly recorded forwards versus immutable replay.
use anyhow::{Result,ensure};
use lfm2_engine::{gpu_lfm::GpuEngine,model_file::ModelFile};
use sha2::{Digest,Sha256};use serde_json::json;
fn top(v:&[f32])->u32 {let(mut i,mut m)=(0,f32::NEG_INFINITY);for(j,&x)in v.iter().enumerate(){if x>m{i=j;m=x;}}i as u32}
fn hash(v:&[u8])->String {hex::encode(Sha256::digest(v))}
fn main()->Result<()> {
 let a:Vec<_>=std::env::args().collect();let m=ModelFile::open(&a[1])?;let q:serde_json::Value=serde_json::from_str(&std::fs::read_to_string(&a[2])?)?;let ids:Vec<u32>=serde_json::from_value(q["tokens"].clone())?;let n=q["n"].as_u64().unwrap() as usize;
 let mut e=GpuEngine::load(&m)?;e.set_ctx(4096);let mut reference=vec![];let mut next_ids=vec![];let mut reports=vec![];
 for dynamic in [false,true] {
  e.reset_session()?;let mut pos=0;let mut logits=vec![];let mut next=0;
  for chunk in ids.chunks(64) {if dynamic {next=e.forward_token(chunk,pos)?;} else {logits=e.forward(chunk,pos as i64)?;next=top(&logits);}pos+=chunk.len();}
  for step in 0..=n {
   if dynamic {logits=e.snapshot_resident_logits()?;ensure!(top(&logits)==next,"GPU argmax mismatch");}
   ensure!(logits.iter().all(|x|x.is_finite()),"nonfinite replay logits");
   let cache=e.snapshot_resident_cache(pos)?;let identity=(hash(bytemuck::cast_slice(&logits)),hash(&cache),next);
   if dynamic {ensure!(identity==reference[step],"full logits/cache replay bit mismatch step={step} position={pos}");} else {reference.push(identity.clone());next_ids.push(next);}
   reports.push(json!({"dynamic":dynamic,"step":step,"position":pos,"logits_elements":logits.len(),"cache_bytes":cache.len(),"logits_sha256":identity.0,"cache_sha256":identity.1,"next":next}));
   if step<n {let token=next_ids[step];if dynamic {next=e.forward_token(&[token],pos)?;} else {logits=e.forward(&[token],pos as i64)?;next=top(&logits);}pos+=1;}
  }
 }
 ensure!(e.gpu.ctx.decode_command_buffer_reuses.get()>0,"no replay exercised");
 println!("{}",json!({"pass":true,"input_tokens":ids.len(),"output_tokens":n,"final_forward_included":true,"full_logits_and_cache_bit_equal":true,"model_loads":1,"reuses":e.gpu.ctx.decode_command_buffer_reuses.get(),"steps":reports}));Ok(())
}
