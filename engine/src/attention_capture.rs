//! Opt-in correctness captures only. No changes to kernels or timed paths.
use crate::{ops::GpuOps,vulkan::GpuBuffer};
use anyhow::Result;
impl GpuOps {
 pub(crate) fn capture_attention(&self,dir:&str,b:&[&GpuBuffer],p:&[u32])->Result<()> {
  use std::sync::atomic::{AtomicUsize,Ordering};static NEXT:AtomicUsize=AtomicUsize::new(0);
  self.finish_batch()?;
  let id=NEXT.fetch_add(1,Ordering::Relaxed);let root=std::path::Path::new(dir);std::fs::create_dir_all(root)?;
  let nq=(p[0]*p[1]*p[3]) as usize;let nk=(p[0]*p[2]*(p[4]+p[3])) as usize;
  for (name,index,count) in [("q",0,nq),("k",1,nk),("v",2,nk),("candidate",3,nq)] {
   let host=self.alloc_host(count*4)?;self.begin_batch()?;self.ctx.record_copy(b[index],&host,0,0,count*4);self.finish_batch()?;
   let mut raw=vec![0;count*4];host.read(0,&mut raw);std::fs::write(root.join(format!("{id}-{name}.bin")),raw)?;
  }
  std::fs::write(root.join(format!("{id}.json")),serde_json::to_vec(&serde_json::json!({"id":id,"push":p,"source":"live post-rope Q and f16-rounded canonical K/V, candidate output"}))?)?;
  self.begin_batch()?;Ok(())
 }
}
