//! CPU f64 independent real-activation audit plus bounded accepted GPU attention.
use anyhow::{Result,ensure};use lfm2_engine::ops::GpuOps;use serde_json::json;
fn load(p:std::path::PathBuf)->Result<Vec<f32>> {let b=std::fs::read(p)?;Ok(b.chunks_exact(4).map(|x|f32::from_le_bytes(x.try_into().unwrap())).collect())}
fn main()->Result<()> {
 let a:Vec<_>=std::env::args().collect();let dir=std::path::Path::new(&a[1]);let gpu=GpuOps::open()?;
 let mut metas:Vec<_>=std::fs::read_dir(dir)?.map(|e|e.unwrap().path()).filter(|p|p.extension().is_some_and(|s|s=="json")).collect();metas.sort();
 let mut rows=vec![];let(mut ce,mut re,mut count)=(0f64,0f64,0usize);
 for meta in metas {
  let m:serde_json::Value=serde_json::from_slice(&std::fs::read(&meta)?)?;let id=m["id"].as_u64().unwrap();let p:Vec<u32>=serde_json::from_value(m["push"].clone())?;
  let q=load(dir.join(format!("{id}-q.bin")))?;let k=load(dir.join(format!("{id}-k.bin")))?;let v=load(dir.join(format!("{id}-v.bin")))?;let c=load(dir.join(format!("{id}-candidate.bin")))?;
  ensure!(q.iter().chain(&k).chain(&v).chain(&c).all(|v|v.is_finite()),"nonfinite activation");
  let qb=gpu.upload_to_device(bytemuck::cast_slice(&q))?;let kb=gpu.upload_to_device(bytemuck::cast_slice(&k))?;let vb=gpu.upload_to_device(bytemuck::cast_slice(&v))?;let rb=gpu.alloc_host(q.len()*4)?;
  gpu.begin_batch()?;gpu.helper("attention_queryhoist",&[&qb,&kb,&vb,&rb],&p,p[1]*p[3],1)?;gpu.finish_batch()?;
  let r=gpu.read_f32(&rb,q.len());ensure!(r.iter().all(|v|v.is_finite()),"nonfinite reference");
  let(hd,heads,kv,n,start)=(p[0] as usize,p[1] as usize,p[2] as usize,p[3] as usize,p[4] as usize);
  let(mut cse,mut rse,mut cm,mut rm,mut norm,mut nc)=(0f64,0f64,0f64,0f64,0f64,0usize);
  for j in 0..n {for h in 0..heads {let nt=start+j+1;let kh=h*kv/heads;let base=(j*heads+h)*hd;
   let scores:Vec<f64>=(0..nt).map(|t|(0..hd).map(|d|q[base+d] as f64*k[(t*kv+kh)*hd+d] as f64).sum::<f64>()*f32::from_bits(p[6]) as f64).collect();
   let max=scores.iter().copied().fold(f64::NEG_INFINITY,f64::max);let weights:Vec<f64>=scores.iter().map(|s|(s-max).exp()).collect();let sum=weights.iter().sum::<f64>();
   for d in 0..hd {let y=(0..nt).map(|t|weights[t]*v[(t*kv+kh)*hd+d] as f64).sum::<f64>()/sum;
    let e=c[base+d] as f64-y;let b=r[base+d] as f64-y;cse+=e*e;rse+=b*b;cm=cm.max(e.abs());rm=rm.max(b.abs());norm+=y*y;nc+=1;
   }
  }}
  ce+=cse;re+=rse;count+=nc;rows.push(json!({"capture":id,"push":p,"elements":nc,"candidate_squared_error":cse,"reference_squared_error":rse,"candidate_max_abs":cm,"reference_max_abs":rm,"candidate_relative_l2":(cse/norm.max(1e-30)).sqrt()}));
 }
 let pass=ce<=re && rows.iter().all(|r|r["candidate_relative_l2"].as_f64().unwrap()<1e-3 && r["candidate_squared_error"].as_f64().unwrap()<=r["reference_squared_error"].as_f64().unwrap() && r["candidate_max_abs"].as_f64().unwrap()<=r["reference_max_abs"].as_f64().unwrap());
 println!("{}",json!({"pass":pass,"captures":rows,"f64_elements":count,"all_captured_queries_checked":true,"candidate_squared_error":ce,"reference_squared_error":re,"aggregate_error_nonregression":ce<=re,"reference":"accepted query-hoist kernels; frozen conservative bounds"}));ensure!(pass,"real attention error regression");Ok(())
}
