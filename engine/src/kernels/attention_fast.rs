//! Three-stage attention experiment. QK and V reductions deliberately reorder
//! arithmetic; promotion requires numerical error checks AND exact native tokens.
use super::*;
pub fn build_attention_fast(stage: u32, dynamic: bool) -> Vec<u32> {
    let local = if stage == 1 { 64 } else { 256 };
    let ir = Ir::configured(false, 6, 7, local);
    let (hd, heads, kv, n, ctx) = (ir.pc(0), ir.pc(1), ir.pc(2), ir.pc(3), ir.pc(5));
    let start = if dynamic {
        ir.load(5, ir.uint(0)).uint()
    } else {
        ir.pc(4)
    };
    let group = ir.gid(0) / ir.uint(local);
    let tid = ir.gid(0) % ir.uint(local);
    let tiles = (ctx + ir.uint(63)) / ir.uint(64);
    let row = if stage == 0 { group / tiles } else { group };
    let (hh, j) = (row / n, row % n);
    ir.guard(hh.ge(heads));
    let nt = start + j + ir.uint(1);
    let kh = hh * kv / heads;
    let qrow = hh * hd + j * hd * heads;
    let scorebase = row * ctx;
    if stage == 0 {
        // Sixteen independent 16-lane dots, four positions per lane group.
        let lane = tid % ir.uint(16);
        let query: Vec<_> = (0..4)
            .map(|i| ir.load(0, qrow + ir.uint(i * 16) + lane))
            .collect();
        for t in 0..4 {
            let tt = (group % tiles) * ir.uint(64) + tid / ir.uint(16) + ir.uint(t * 16);
            ir.when(tt.lt(nt), || {
                let mut dot = ir.float(0.0);
                for i in 0..4 {
                    let d = ir.uint(i * 16) + lane;
                    dot = dot + query[i as usize] * ir.load(1, tt * hd * kv + kh * hd + d);
                }
                for shift in [8, 4, 2, 1] {
                    let other: F<'_> = ir.emit(
                        Op::GroupNonUniformShuffleXor,
                        ir.f,
                        &[ir.uint(3).id, dot.id, ir.uint(shift).id],
                    );
                    dot = dot + other;
                }
                ir.when(lane.eq(ir.uint(0)), || {
                    ir.store(4, scorebase + tt, dot * ir.pc(6).float_bits())
                });
            });
        }
    } else if stage == 1 {
        let count = (nt + ir.uint(63)) / ir.uint(64);
        let m = ir.accumulator(ir.float(-1.0e30));
        ir.range(count, |i| {
            let t = i * ir.uint(64) + tid;
            ir.when(t.lt(nt), || m.set(m.get().max(ir.load(4, scorebase + t))));
        });
        let max = ir.subgroup_max(m.get());
        let sum = ir.accumulator(ir.float(0.0));
        ir.range(count, |i| {
            let t = i * ir.uint(64) + tid;
            ir.when(t.lt(nt), || {
                let v = (ir.load(4, scorebase + t) - max).exp();
                ir.store(4, scorebase + t, v);
                sum.set(sum.get() + v);
            });
        });
        let denom = ir.subgroup_sum(sum.get());
        ir.range(count, |i| {
            let t = i * ir.uint(64) + tid;
            ir.when(t.lt(nt), || {
                ir.store(4, scorebase + t, ir.load(4, scorebase + t) / denom)
            });
        });
    } else {
        let shared = ir.shared(256);
        let d = tid % ir.uint(64);
        let wave = tid / ir.uint(64);
        let count = (nt + ir.uint(3)) / ir.uint(4);
        let sum = ir.accumulator(ir.float(0.0));
        ir.range(count, |i| {
            let t = wave * count + i;
            ir.when(t.lt(nt), || {
                sum.set(
                    sum.get() + ir.load(4, scorebase + t) * ir.load(2, t * hd * kv + kh * hd + d),
                );
            });
        });
        ir.shared_store(shared, tid, sum.get());
        ir.barrier();
        ir.when(wave.eq(ir.uint(0)), || {
            let a = ir.shared_load(shared, d) + ir.shared_load(shared, d + ir.uint(64));
            let b =
                ir.shared_load(shared, d + ir.uint(128)) + ir.shared_load(shared, d + ir.uint(192));
            ir.store(3, qrow + d, a + b);
        });
    }
    ir.finish()
}
