//! Coalesced K transpose plus one-lane ordered 64-term score dot.
//! K/V canonical cache layout remains unchanged; scratch is reconstructed per layer.
use super::*;
pub fn build_attention_transpose(dynamic:bool)->Vec<u32> {
 let ir=Ir::configured(false,7,7,256);
 let (hd,kv,n,ctx)=(ir.pc(0),ir.pc(2),ir.pc(3),ir.pc(5));
 let start=if dynamic {ir.load(5,ir.uint(0)).uint()} else {ir.pc(4)};
 let end=start+n;let i=ir.gid(0);ir.guard(i.ge(hd*kv*end));
 let t=i%end;let d=(i/end)%hd;let kh=(i/end)/hd;
 ir.store(6,(kh*hd+d)*ctx+t,ir.load(1,t*hd*kv+kh*hd+d));
 ir.finish()
}
pub fn build_attention_score_transposed(dynamic:bool)->Vec<u32> {
 let ir=Ir::configured(false,7,7,64);
 let (hd,heads,kv,n,ctx)=(ir.pc(0),ir.pc(1),ir.pc(2),ir.pc(3),ir.pc(5));
 let start=if dynamic {ir.load(5,ir.uint(0)).uint()} else {ir.pc(4)};
 let group=ir.gid(0)/ir.uint(64);let tid=ir.gid(0)%ir.uint(64);let tiles=(ctx+ir.uint(63))/ir.uint(64);
 let row=group/tiles;let (hh,j)=(row/n,row%n);ir.guard(hh.ge(heads));
 let t=(group%tiles)*ir.uint(64)+tid;let nt=start+j+ir.uint(1);ir.guard(t.ge(nt));
 let kh=hh*kv/heads;let qr=hh*hd+j*hd*heads;
 let mut dot=ir.float(0.0);
 for d in 0..64 {dot=dot+ir.load(0,qr+ir.uint(d))*ir.load(6,(kh*hd+ir.uint(d))*ctx+t);}
 ir.store(4,row*ctx+t,dot*ir.pc(6).float_bits());ir.finish()
}
