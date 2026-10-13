//! Isolated real-weight bit test plus naturally idle-clock first-use evidence.
//! Kernel-level timings are diagnostic only, never product acceptance.
use anyhow::{ensure,Result};
use lfm2_engine::{model_file::ModelFile,ops::GpuOps};
use serde_json::json;
fn idle_clock()->Result<String> {
    let path="/sys/class/drm/card0/device/pp_dpm_sclk";
    for _ in 0..100 {
        let s=std::fs::read_to_string(path)?;
        if s.lines().any(|l|l.contains("300Mhz") && l.contains('*')) {return Ok(s);}
        std::thread::sleep(std::time::Duration::from_millis(100));
    }
    anyhow::bail!("natural idle 300MHz clock not observed; no changed-kernel cold claim")
}
fn main()->Result<()> {
    let model=ModelFile::open(&std::env::args().nth(1).unwrap())?;
    let gpu=GpuOps::open()?;ensure!(gpu.q6_aligned,"aligned production mode required");
    let mut cases=vec![];
    for suffix in ["ffn_gate_exps.weight","ffn_down_exps.weight"] {
        let t=model.tensors.iter().find(|t|t.name.ends_with(suffix)).unwrap();
        let (input,output,experts)=(t.ne[0] as usize,t.ne[1] as usize,t.ne[2] as usize);
        let down=suffix.starts_with("ffn_down");
        let w=gpu.upload_q6(&model.read_tensor_bytes(&t.name)?)?;
        for n in [64usize,1,2,3,4,17,63] {for pattern in [0usize,1] {
            let x:Vec<f32>=(0..input*n*if down {4} else {1}).map(|i|(i as f32*0.013).sin()).collect();
            let xb=gpu.upload_to_device(bytemuck::cast_slice(&x))?;
            let selection:Vec<f32>=(0..n).flat_map(|j|(0..8).map(move |r| (if r<4 {((if pattern==0 {0} else {j/3})+r)%experts} else {0}) as f32)).collect();
            let sb=gpu.upload_to_device(bytemuck::cast_slice(&selection))?;
            let gb=gpu.ctx.alloc_buffer(experts*(4*n+1)*4,false)?;
            let oa=gpu.alloc_host(output*n*4*4)?;let ob=gpu.alloc_host(output*n*4*4)?;
            gpu.ctx.set_work_context(0,n,None);
            if n>1 {gpu.begin_batch()?;gpu.helper("expert_groups",&[&sb,&gb],&[n as u32,experts as u32],experts.div_ceil(256) as u32,1)?;gpu.finish_batch()?;}
            let mut timings=vec![];
            for (rep,candidate) in [true,false,true,false].into_iter().enumerate() {
                let cold=if rep==0 && pattern==0 && (n==64 || n==1) {Some(idle_clock()?)} else {None};
                let before=gpu.ctx.total_gpu_ms.get();gpu.begin_batch()?;
                let name=match (n>1,candidate) {(true,true)=>"q6k_grouped_compact",(true,false)=>"q6k_grouped_aligned",(false,true)=>"q6k_routed",(false,false)=>"q6k_routed_aligned"};
                let gy=if n==1 {4} else if candidate {experts*n.div_ceil(4)} else {experts*n};
                gpu.helper(name,&[&w,&xb,if candidate {&ob} else {&oa},if n>1 {&gb} else {&sb}],&[input as u32,output as u32,(input/256) as u32,n as u32,u32::from(down)],output.div_ceil(4) as u32,gy as u32)?;
                gpu.finish_batch()?;
                timings.push(json!({"candidate":candidate,"pipeline":name,"gpu_ms":gpu.ctx.total_gpu_ms.get()-before,"cold_sclk_before_recording":cold,"last_submission_ms":gpu.ctx.last_gpu_ms.get()}));
            }
            let a=gpu.read_f32(&oa,output*n*4);let b=gpu.read_f32(&ob,output*n*4);
            let mismatches=a.iter().zip(&b).filter(|(a,b)|a.to_bits()!=b.to_bits()).count();
            ensure!(mismatches==0 && b.iter().all(|v|v.is_finite()),"row schedule bit mismatch");
            cases.push(json!({"tensor":t.name,"n":n,"pattern":pattern,"elements":a.len(),"bit_mismatches":mismatches,"timings":timings}));
        }}
    }
    println!("{}",json!({"pass":true,"cases":cases,"reference_scope":"unchanged four-row Q6, expanded grouped grid; NOT schedule-v3 whole-request timing","cold_scope":"300MHz observed immediately before recording first changed helper at n64/n1; timestamp screens include all resulting split submissions; no clock/power changes"}));Ok(())
}
