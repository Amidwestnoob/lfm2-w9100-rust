//! Bounded 256-token value partitions; finite f64/reference audit required.
use super::*;
pub fn build_attention_value_split(dynamic: bool, merge: bool) -> Vec<u32> {
    let local=if merge {64} else {256};
    let ir=Ir::configured(false,8,7,local);
    let (hd,heads,kv,n,ctx)=(ir.pc(0),ir.pc(1),ir.pc(2),ir.pc(3),ir.pc(5));
    let start=if dynamic {ir.load(5,ir.uint(0)).uint()} else {ir.pc(4)};
    let end=if dynamic {let bound=(ir.pc(4)/ir.uint(256)+ir.uint(1))*ir.uint(256);bound.lt(ctx).choose_u(bound,ctx)} else {start+n};
    let parts=(end+ir.uint(255))/ir.uint(256);
    let group=ir.gid(0)/ir.uint(local);let tid=ir.gid(0)%ir.uint(local);
    let row=if merge {group} else {group/parts};
    let hh=row/n;let j=row%n;ir.guard(hh.ge(heads));
    let nt=start+j+ir.uint(1);let kh=hh*kv/heads;
    if merge {
        let sum=ir.accumulator(ir.float(0.0));
        ir.range(parts,|p|sum.set(sum.get()+ir.load(7,(row*parts+p)*hd+tid)));
        ir.store(3,j*heads*hd+hh*hd+tid,sum.get());
    } else {
        let part=group%parts;let wave=tid/ir.uint(64);let d=tid%ir.uint(64);
        let shared=ir.shared(256);let sum=ir.accumulator(ir.float(0.0));
        ir.range(ir.uint(64),|i|{
            let t=part*ir.uint(256)+wave*ir.uint(64)+i;
            ir.when(t.lt(nt),||sum.set(sum.get()+ir.load(4,row*ctx+t)*ir.load(2,t*hd*kv+kh*hd+d)));
        });
        ir.shared_store(shared,tid,sum.get());ir.barrier();
        ir.when(wave.eq(ir.uint(0)),||{
            let a=ir.shared_load(shared,d)+ir.shared_load(shared,d+ir.uint(64));
            let b=ir.shared_load(shared,d+ir.uint(128))+ir.shared_load(shared,d+ir.uint(192));
            ir.store(7,(row*parts+part)*hd+d,a+b);
        });
    }
    ir.finish()
}
