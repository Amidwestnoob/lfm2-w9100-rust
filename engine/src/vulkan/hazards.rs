//! Whole-buffer RAW/WAR/WAW tracking. Unknown operations retain full barriers.
//! No range inference, cross-submit elision, stage narrowing, or kernel change.
use ash::vk;
#[derive(Default)]
pub(super) struct Hazards { reads:Vec<vk::Buffer>, writes:Vec<vk::Buffer>, initialized:bool }
impl Hazards {
    pub fn reset(&mut self) {*self=Self::default();}
    pub fn before(&mut self,name:&str,buffers:&[vk::Buffer])->bool {
        // Proven from these Rust IR builders: each reads inputs and only writes
        // the designated output (rope is read/modify/write binding zero).
        let output=match name {
            "q6k_matmul"|"q6k_prompt"|"q6k_prompt2"|"q6k_matmul_aligned"|"q6k_prompt_aligned"|"q6k_routed_aligned"|"q6k_grouped_aligned"|"q6k_matmul_4expert"|"q6k_routed"|"q6k_grouped"|"q6k_grouped_compact"|"f32_matmul"|"rms_norm"|"swiglu"|"embedding"=>2,
            "rope"|"rope_decode"=>0,
            _=>{self.reset();return true;}
        };
        if buffers.len()<=output {self.reset();return true;}
        // Output also treated as a read: conservative for in-place/aliasing.
        let writes=[buffers[output]];
        let barrier=!self.initialized || buffers.iter().any(|b|self.writes.contains(b)) || writes.iter().any(|b|self.reads.contains(b));
        if barrier {self.reads.clear();self.writes.clear();}
        self.initialized=true;
        for b in buffers {if !self.reads.contains(b){self.reads.push(*b);}}
        for b in writes {if !self.writes.contains(&b){self.writes.push(b);}}
        barrier
    }
}
#[cfg(test)]
mod tests {
 use super::*;use ash::vk::Handle;
 fn b(x:u64)->vk::Buffer {vk::Buffer::from_raw(x)}
 #[test] fn whole_buffer_dependencies_preserved() {
  let mut h=Hazards::default();
  assert!(h.before("q6k_matmul",&[b(1),b(2),b(3)]));
  assert!(!h.before("q6k_matmul",&[b(4),b(2),b(5)]));
  assert!(!h.before("q6k_matmul",&[b(6),b(2),b(7)]));
  assert!(h.before("rms_norm",&[b(3),b(8),b(9)])); // RAW
  assert!(!h.before("rms_norm",&[b(5),b(10),b(11)]));
  assert!(h.before("rope",&[b(9)])); // in-place
  assert!(!h.before("rope",&[b(11)]));
  assert!(h.before("unknown",&[]));
  assert!(h.before("q6k_matmul",&[b(1),b(2),b(3)]));
  assert!(h.before("q6k_matmul",&[b(4),b(5),b(2)])); // WAR
  assert!(h.before("q6k_matmul",&[b(4),b(5),b(2)])); // WAW
  h.reset();assert!(h.before("rope",&[b(100)]));
 }
 #[test] fn exhaustive_two_dispatch_aliases() {
  for a in 1..=4 {for b0 in 1..=4 {for c in 1..=4 {for d in 1..=4 {for e in 1..=4 {for f in 1..=4 {
   let mut h=Hazards::default();assert!(h.before("q6k_matmul",&[b(a),b(b0),b(c)]));
   let barrier=h.before("q6k_matmul",&[b(d),b(e),b(f)]);
   let hazard=[d,e,f].contains(&c)||[a,b0,c].contains(&f);
   assert_eq!(barrier,hazard);
  }}}}}}
 }
}
