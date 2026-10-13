//! Exact old/new attention screen with unchanged planner and full f32-bit arrays.
use anyhow::{Result,ensure};
use lfm2_engine::{ops::GpuOps,kernels};
use serde_json::json;
use sha2::{Digest,Sha256};
fn main()->Result<()> {
 let args:Vec<String>=std::env::args().collect();
 let end:usize=args[1].parse()?; let n:usize=args[2].parse()?;
 ensure!(n>0&&n<=64&&n<=end&&end<=4096,"bounded geometry");
 let gpu=GpuOps::open()?;
 let data=|len:usize,freq:f32|->Vec<f32>{(0..len).map(|i|half::f16::from_f32((i as f32*freq).sin()*0.5).to_f32()).collect()};
 let q=data(64*32*n,0.0131);let k=data(64*8*end,0.0079);let v=data(64*8*end,0.0173);
 let qb=gpu.upload_to_device(bytemuck::cast_slice(&q))?;let kb=gpu.upload_to_device(bytemuck::cast_slice(&k))?;let vb=gpu.upload_to_device(bytemuck::cast_slice(&v))?;
 let a=gpu.alloc_host(q.len()*4)?;let b=gpu.alloc_host(q.len()*4)?;
 let push=[64,32,8,n as u32,(end-n) as u32,4096,0.125f32.to_bits()];
 let mut times=vec![];
 for (name,out) in [("attention_unrolled",&a),("attention_queryhoist",&b),("attention_unrolled",&a)] {
  gpu.ctx.set_work_context(end-n,n,None);gpu.ctx.set_submission_context("direct_attention",end);
  let before=gpu.ctx.total_gpu_ms.get();gpu.begin_batch()?;
  gpu.helper(name,&[&qb,&kb,&vb,out],&push,(32*n) as u32,1)?;gpu.finish_batch()?;
  times.push(gpu.ctx.total_gpu_ms.get()-before);
 }
 let av=gpu.read_f32(&a,q.len());let bv=gpu.read_f32(&b,q.len());
 let mismatch=av.iter().zip(&bv).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
 let hash=|words:Vec<u32>|hex::encode(Sha256::digest(bytemuck::cast_slice(&words)));
 println!("{}",json!({"pass":mismatch==0,"end":end,"n":n,"elements":q.len(),"bit_mismatches":mismatch,"gpu_ms_A1_B_A2":times,"baseline_spirv_sha256":hash(kernels::build_attention_unrolled()),"candidate_spirv_sha256":hash(kernels::build_attention_queryhoist()),"finite":bv.iter().all(|v|v.is_finite())}));
 ensure!(mismatch==0&&bv.iter().all(|v|v.is_finite()),"exact attention mismatch");Ok(())
}
