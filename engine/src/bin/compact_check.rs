use anyhow::{ensure,Result};
use lfm2_engine::{model_file::ModelFile,ops::GpuOps};
use serde_json::json;
fn main()->Result<()> {
 let a:Vec<_>=std::env::args().collect();let model=ModelFile::open(&a[1])?;let gpu=GpuOps::open()?;let mut cases=vec![];
 for suffix in ["ffn_gate_exps.weight","ffn_down_exps.weight"] {
  let t=model.tensors.iter().find(|t|t.name.ends_with(suffix)).unwrap();let (input,output,experts)=(t.ne[0] as usize,t.ne[1] as usize,t.ne[2] as usize);
  let w=gpu.upload_q6(&model.read_tensor_bytes(&t.name)?)?;
  for n in [1usize,2,3,4,17,63,64] {for pattern in [0usize,1] {
   let x:Vec<f32>=(0..input*n*if suffix.starts_with("ffn_down") {4} else {1}).map(|i|(i as f32*0.013).sin()).collect();let xb=gpu.upload_to_device(bytemuck::cast_slice(&x))?;
   let selection:Vec<f32>=(0..n).flat_map(|j|(0..8).map(move |r|(if r<4 {((if pattern==0 {0} else {j/3})+r)%experts} else {0}) as f32)).collect();
   let sb=gpu.upload_to_device(bytemuck::cast_slice(&selection))?;let gb=gpu.ctx.alloc_buffer(experts*(4*n+1)*4,false)?;
   let oa=gpu.alloc_host(output*n*4*4)?;let ob=gpu.alloc_host(output*n*4*4)?;let mut times=vec![];
   gpu.begin_batch()?;gpu.helper("expert_groups",&[&sb,&gb],&[n as u32,experts as u32],experts.div_ceil(256) as u32,1)?;gpu.finish_batch()?;
   for compact in [false,true,false,true] {
    let before=gpu.ctx.total_gpu_ms.get();gpu.begin_batch()?;
    gpu.helper(if compact {"q6k_grouped_compact"} else {"q6k_grouped"},&[&w,&xb,if compact {&ob} else {&oa},&gb],&[input as u32,output as u32,(input/256) as u32,n as u32,if suffix.starts_with("ffn_down") {1} else {0}],output.div_ceil(4) as u32,(experts*if compact {n.div_ceil(4)} else {n}) as u32)?;
    gpu.finish_batch()?;times.push(gpu.ctx.total_gpu_ms.get()-before);
   }
   let av=gpu.read_f32(&oa,output*n*4);let bv=gpu.read_f32(&ob,output*n*4);let mismatch=av.iter().zip(&bv).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
   ensure!(mismatch==0 && bv.iter().all(|v|v.is_finite()),"compact bit equality failed");
   cases.push(json!({"tensor":t.name,"n":n,"pattern":pattern,"elements":av.len(),"bit_mismatches":mismatch,"gpu_ms_A_B_A_B":times}));
  }}
 }
 println!("{}",json!({"pass":true,"cases":cases}));Ok(())
}
