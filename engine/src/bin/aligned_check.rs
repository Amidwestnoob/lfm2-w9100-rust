//! Same-context original versus aligned Q6 numerical/time check, with real model data.
use anyhow::{ensure,Result};
use lfm2_engine::{model_file::ModelFile,ops::GpuOps,q6k};
use serde_json::json;
fn main()->Result<()> {
 let model=ModelFile::open(&std::env::args().nth(1).unwrap())?;
 let gpu=GpuOps::open()?; ensure!(!gpu.q6_aligned,"checker requires original default; aligned pipelines explicit");
 let mut cases=vec![];
 for n in [1usize,3,17,64] {
  for expert in [false,true] {
   let tensor=model.tensors.iter().find(|t|t.name.ends_with(if expert {"ffn_gate_exps.weight"} else {"attn_q.weight"})).unwrap();
   let (input,output)=(tensor.ne[0] as usize,tensor.ne[1] as usize);
   let experts=if expert {tensor.ne[2] as usize} else {1};
   let raw=model.read_tensor_bytes(&tensor.name)?;
   let w=gpu.upload_to_device(&raw)?; let wa=gpu.upload_to_device(&q6k::repack_aligned(&raw))?;
   let x:Vec<f32>=(0..input*n).map(|i|((i as f32)*0.013).sin()).collect();
   let xb=gpu.upload_to_device(bytemuck::cast_slice(&x))?;
   let count=output*n*if expert {4} else {1};
   let out=gpu.alloc_host(count*4)?; let aligned=gpu.alloc_host(count*4)?;
   let selection:Vec<f32>=(0..n).flat_map(|j|(0..8).map(move |r|if r<4 {((j/3+r)%experts) as f32} else {0.25})).collect();
   let sb=gpu.upload_to_device(bytemuck::cast_slice(&selection))?;
   let gb=gpu.ctx.alloc_buffer(experts*(4*n+1)*4,false)?;
   let mut timings=vec![]; let mut counters=vec![];
   for phase in ["A1","B","A2"] {
    let candidate=phase=="B";
    let name=match (expert,n>1,candidate) {
      (false,false,false)=>"q6k_matmul",(false,false,true)=>"q6k_matmul_aligned",
      (false,true,false)=>"q6k_prompt",(false,true,true)=>"q6k_prompt_aligned",
      (true,false,false)=>"q6k_routed",(true,false,true)=>"q6k_routed_aligned",
      (true,true,false)=>"q6k_grouped",(true,true,true)=>"q6k_grouped_aligned",
    };
    let mut times=vec![]; let mut actual=0;
    for rep in 0..9 {
     let start=gpu.ctx.total_gpu_ms.get(); let dc=gpu.dispatch_count(); gpu.begin_batch()?;
     if expert && n>1 {gpu.helper("expert_groups",&[&sb,&gb],&[n as u32,experts as u32],experts.div_ceil(256) as u32,1)?;}
     let p=if expert {vec![input as u32,output as u32,(input/256) as u32,n as u32,0]} else {vec![input as u32,output as u32,n as u32,(input/256) as u32,0]};
     let buffers=if expert {vec![if candidate {&wa} else {&w},&xb,if candidate {&aligned} else {&out},if n>1 {&gb} else {&sb}]} else {vec![if candidate {&wa} else {&w},&xb,if candidate {&aligned} else {&out}]};
     let gy=if expert {if n>1 {experts*n} else {n*4}} else {n.div_ceil(if n>1 {4} else {1})};
     gpu.helper(name,&buffers,&p,output.div_ceil(4) as u32,gy as u32)?;gpu.finish_batch()?;
     actual+=gpu.dispatch_count()-dc;
     if rep>0 {times.push(gpu.ctx.total_gpu_ms.get()-start);}
    }
    timings.push(json!({"phase":phase,"gpu_ms":times})); counters.push(json!({"phase":phase,"pipeline":name,"actual_compute_dispatches":actual}));
   }
   let a=gpu.read_f32(&out,count);let b=gpu.read_f32(&aligned,count);
   let mismatches=a.iter().zip(&b).filter(|(x,y)|x.to_bits()!=y.to_bits()).count();
   ensure!(mismatches==0,"aligned Q6 bits differ n={n} expert={expert}: {mismatches}");
   cases.push(json!({"n":n,"expert":expert,"tensor":tensor.name,"elements":count,"mismatched_bits":mismatches,"timings":timings,"counters":counters,"raw_weight_bytes":raw.len(),"aligned_weight_bytes":wa.size}));
  }
 }
 println!("{}",json!({"pass":true,"device":gpu.device_name,"cases":cases}));Ok(())
}
