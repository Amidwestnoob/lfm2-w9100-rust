use super::*;

pub fn build_stream_copy() -> Vec<u32> {
    let ir = Ir::new(false, 2, 1);
    let i = ir.gid(0);
    ir.guard(i.ge(ir.pc(0)));
    ir.store(1, i, ir.load(0, i));
    ir.finish()
}

/// Two-stage deterministic argmax, ignoring NaNs and preferring lowest IDs.
pub fn build_argmax(pairs: bool) -> Vec<u32> {
    let ir = Ir::configured(false, 2, 1, 64);
    let tid = ir.gid(0) % ir.uint(64);
    let group = ir.gid(0) / ir.uint(64);
    let n = ir.pc(0);
    let best = ir.accumulator(ir.float(f32::NEG_INFINITY));
    let index = ir.accumulator(ir.float(0.0));
    let count = if pairs {
        (n + ir.uint(63)) / ir.uint(64)
    } else {
        ir.uint(8)
    };
    ir.range(count, |k| {
        let i = if pairs {
            k * ir.uint(64) + tid
        } else {
            group * ir.uint(512) + k * ir.uint(64) + tid
        };
        ir.when(i.lt(n), || {
            let v = ir.load(0, if pairs { i * ir.uint(2) } else { i });
            let id = if pairs {
                ir.load(0, i * ir.uint(2) + ir.uint(1))
            } else {
                i.float()
            };
            let win = v.gt(best.get()) | (v.eq(best.get()) & index.get().gt(id));
            index.set(win.choose_f(id, index.get()));
            best.set(win.choose_f(v, best.get()));
        });
    });
    let value = ir.subgroup_max(best.get());
    let id = -ir.subgroup_max(
        best.get()
            .eq(value)
            .choose_f(-index.get(), ir.float(f32::NEG_INFINITY)),
    );
    ir.when(tid.eq(ir.uint(0)), || {
        ir.store(1, group * ir.uint(2), value);
        ir.store(1, group * ir.uint(2) + ir.uint(1), id);
    });
    ir.finish()
}

/// Stable expert grouping in device memory. A lane per expert walks the actual
/// top-four routing result and records token/rank slots; Q6 tiles reuse a weight
/// block across four such slots and scatter to the original combine layout.
pub fn build_expert_groups() -> Vec<u32> {
    let ir = Ir::new(false, 2, 2);
    let e = ir.gid(0);
    let (n, experts) = (ir.pc(0), ir.pc(1));
    ir.guard(e.ge(experts));
    let base = e * (n * ir.uint(4) + ir.uint(1));
    let count = ir.local(ir.uint(0), ir.u);
    ir.range(n * ir.uint(4), |slot| {
        let chosen = ir
            .load(0, slot / ir.uint(4) * ir.uint(8) + slot % ir.uint(4))
            .uint();
        ir.when(chosen.eq(e), || {
            ir.store(1, base + ir.uint(1) + count.get(), slot.float());
            count.set(count.get() + ir.uint(1));
        });
    });
    ir.store(1, base, count.get().float());
    ir.finish()
}
