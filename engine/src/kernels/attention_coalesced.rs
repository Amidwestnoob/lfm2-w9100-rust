//! Four GQA heads share each coalesced K load; canonical K cache is preserved.
use super::*;
use super::resident::half_round;
pub fn build_attention_score_coalesced(dynamic:bool)->Vec<u32> {
 let ir=Ir::configured(false,7,7,64);
 let (hd,heads,kv,n,ctx)=(ir.pc(0),ir.pc(1),ir.pc(2),ir.pc(3),ir.pc(5));
 let start=if dynamic {ir.load(5,ir.uint(0)).uint()} else {ir.pc(4)};
 let end=if dynamic {let bound=(ir.pc(4)/ir.uint(256)+ir.uint(1))*ir.uint(256);bound.lt(ctx).choose_u(bound,ctx)} else {start+n};
 let group=ir.gid(0)/ir.uint(64);let tid=ir.gid(0)%ir.uint(64);let tiles=(end+ir.uint(63))/ir.uint(64);
 let q=group/tiles;let (kh,j)=(q/n,q%n);ir.guard(kh.ge(kv));
 let t=(group%tiles)*ir.uint(64)+tid;let nt=start+j+ir.uint(1);ir.guard(t.ge(nt));
 let mut dots=vec![ir.float(0.0);4];
 for d in 0..64 {
  let key=ir.load(6,(kh*hd+ir.uint(d))*ctx+t);
  for h in 0..4 {let qr=(kh*ir.uint(4)+ir.uint(h))*hd+j*hd*heads;dots[h as usize]=dots[h as usize]+ir.load(0,qr+ir.uint(d))*key;}
 }
 for h in 0..4 {let row=(kh*ir.uint(4)+ir.uint(h))*n+j;ir.store(4,row*ctx+t,dots[h as usize]*ir.pc(6).float_bits());}
 ir.finish()
}
pub fn build_kv_write_transposed(dynamic:bool)->Vec<u32> {
 let ir=Ir::configured(false,6,4,256);
 let (width,n,ctx)=(ir.pc(0),ir.pc(1),ir.pc(3));
 let start=if dynamic {ir.load(4,ir.uint(0)).uint()} else {ir.pc(2)};
 let i=ir.gid(0);ir.guard(i.ge(width*n));let j=i/width;let ch=i%width;
 let key=half_round(&ir,ir.load(0,i));let value=half_round(&ir,ir.load(1,i));
 ir.store(2,start*width+i,key);ir.store(3,start*width+i,value);ir.store(5,ch*ctx+start+j,key);
 ir.finish()
}
