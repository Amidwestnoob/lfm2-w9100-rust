//! Resident graph support, authored in the same typed Rust instruction IR.
use super::*;

pub fn build_embedding() -> Vec<u32> {
    build_embedding_layout(false)
}
pub fn build_embedding_layout(aligned: bool) -> Vec<u32> {
    let ir = Ir::new(true, 3, 2);
    let (dim, n) = (ir.pc(0), ir.pc(1));
    let i = ir.gid(0);
    ir.guard(i.ge(dim * n));
    let j = i / dim;
    let k = i % dim;
    let token = ir.load(1, j).uint();
    let base = (token * (dim / ir.uint(256)) + k / ir.uint(256)) * ir.uint(if aligned {212} else {210});
    ir.store(2, i, dequant(&ir, base, k % ir.uint(256)));
    ir.finish()
}

pub fn build_residual() -> Vec<u32> {
    let ir = Ir::new(false, 2, 2);
    let (count, offset) = (ir.pc(0), ir.pc(1));
    let i = ir.gid(0);
    ir.guard(i.ge(count));
    ir.store(
        0,
        i + offset,
        (ir.load(0, i + offset) + ir.load(1, i)).precise(),
    );
    ir.finish()
}

pub fn build_shortconv() -> Vec<u32> {
    // One channel per lane; serial time preserves the three-tap host order.
    // bindings: projected B/C/X, convolution weights, two-slot cache, output.
    let ir = Ir::new(false, 4, 2);
    let (dim, n) = (ir.pc(0), ir.pc(1));
    let ch = ir.gid(0);
    ir.guard(ch.ge(dim));
    let a = ir.accumulator(ir.load(2, ch * ir.uint(2)));
    let b = ir.accumulator(ir.load(2, ch * ir.uint(2) + ir.uint(1)));
    ir.range(n, |j| {
        let base = j * dim * ir.uint(3) + ch;
        let next = (ir.load(0, base) * ir.load(0, base + ir.uint(2) * dim)).precise();
        let mut sum = ir.float(0.0);
        for (k, value) in [a.get(), b.get(), next].into_iter().enumerate() {
            sum = (sum + value * ir.load(1, ch * ir.uint(3) + ir.uint(k as u32))).precise();
        }
        ir.store(3, j * dim + ch, (ir.load(0, base + dim) * sum).precise());
        a.set(b.get());
        b.set(next);
    });
    ir.store(2, ch * ir.uint(2), a.get());
    ir.store(2, ch * ir.uint(2) + ir.uint(1), b.get());
    ir.finish()
}

pub fn build_route() -> Vec<u32> {
    // One Hawaii subgroup per token. Each lane evaluates one expert exactly
    // once. Max score then max negative ID preserves stable smaller-ID ties.
    let ir = Ir::configured(false, 3, 2, 64);
    let (experts, n) = (ir.pc(0), ir.pc(1));
    let j = ir.gid(0) / ir.uint(64);
    let e = ir.gid(0) % ir.uint(64);
    ir.guard(j.ge(n));
    let prob = ir.accumulator(ir.float(0.0));
    let score = ir.accumulator(ir.float(f32::NEG_INFINITY));
    ir.when(e.lt(experts), || {
        prob.set(ir.float(1.0) / (ir.float(1.0) + (-ir.load(0, j * experts + e)).exp()));
        score.set(prob.get() + ir.load(1, e));
    });
    let mut probabilities = Vec::new();
    for rank in 0..4 {
        let best = ir.subgroup_max(score.get());
        let tied = score.get().eq(best) & e.lt(experts);
        let id = (-ir.subgroup_max(tied.choose_f(-e.float(), ir.float(f32::NEG_INFINITY)))).uint();
        let selected = e.eq(id);
        probabilities.push(ir.subgroup_sum(selected.choose_f(prob.get(), ir.float(0.0))));
        ir.when(e.eq(ir.uint(0)), || {
            ir.store(2, j * ir.uint(8) + ir.uint(rank), id.float())
        });
        score.set(selected.choose_f(ir.float(f32::NEG_INFINITY), score.get()));
    }
    let mut sum = ir.float(0.0);
    for &p in &probabilities {
        sum = (sum + p).precise();
    }
    sum = sum.max(ir.float(6.103515625e-5));
    ir.when(e.eq(ir.uint(0)), || {
        for rank in 0..4 {
            ir.store(
                2,
                j * ir.uint(8) + ir.uint(4 + rank),
                probabilities[rank as usize] / sum,
            );
        }
    });
    ir.finish()
}

pub fn build_combine() -> Vec<u32> {
    // bindings: expert output [token,expert,dim], routing, residual.
    let ir = Ir::new(false, 3, 3);
    let (dim, n, offset) = (ir.pc(0), ir.pc(1), ir.pc(2));
    let i = ir.gid(0);
    ir.guard(i.ge(dim * n));
    let (j, k) = (i / dim, i % dim);
    let mut sum = ir.float(0.0);
    for e in 0..4 {
        sum = (sum
            + ir.load(0, (j * ir.uint(4) + ir.uint(e)) * dim + k)
                * ir.load(1, j * ir.uint(8) + ir.uint(4 + e)))
        .precise();
    }
    ir.store(2, i + offset, (ir.load(2, i + offset) + sum).precise());
    ir.finish()
}

pub(super) fn half_round<'a>(ir: &'a Ir, value: F<'a>) -> F<'a> {
    // Integer RNE conversion, including gradual underflow. Unpack only widens
    // the already rounded binary16 representation; it does not choose rounding.
    let bits = value.bits();
    let sign = (bits >> ir.uint(16)) & ir.uint(0x8000);
    let mag = bits & ir.uint(0x7fffffff);
    let exp = mag >> ir.uint(23);
    let rounded = mag + ir.uint(0xfff) + ((mag >> ir.uint(13)) & ir.uint(1));
    let normal = (rounded - ir.uint(0x38000000)) >> ir.uint(13);
    let shift = (ir.uint(126) - exp) & ir.uint(31);
    let mant = (mag & ir.uint(0x7fffff)) | ir.uint(0x800000);
    let base = mant >> shift;
    let rem = mant & ((ir.uint(1) << shift) - ir.uint(1));
    let halfway = ir.uint(1) << ((shift - ir.uint(1)) & ir.uint(31));
    let inc = (halfway.lt(rem) | (rem.eq(halfway) & (base & ir.uint(1)).eq(ir.uint(1))))
        .choose_u(ir.uint(1), ir.uint(0));
    let sub = exp.lt(ir.uint(102)).choose_u(ir.uint(0), base + inc);
    let finite = exp.lt(ir.uint(113)).choose_u(
        sub,
        rounded
            .ge(ir.uint(0x47800000))
            .choose_u(ir.uint(0x7c00), normal),
    );
    let nan = ir.uint(0x7e00) | ((mag >> ir.uint(13)) & ir.uint(0x3ff));
    let result = ir.uint(0x7f800000).lt(mag).choose_u(nan, finite);
    ir.half_low(sign | result)
}

pub fn build_kv_write() -> Vec<u32> {
    kv_write_impl(false)
}
pub fn build_kv_write_decode() -> Vec<u32> {
    kv_write_impl(true)
}
fn kv_write_impl(dynamic: bool) -> Vec<u32> {
    let ir = Ir::new(false, if dynamic { 5 } else { 4 }, 3);
    let (width, n, start) = (
        ir.pc(0),
        ir.pc(1),
        if dynamic {
            ir.load(4, ir.uint(0)).uint()
        } else {
            ir.pc(2)
        },
    );
    let i = ir.gid(0);
    ir.guard(i.ge(width * n));
    ir.store(2, start * width + i, half_round(&ir, ir.load(0, i)));
    ir.store(3, start * width + i, half_round(&ir, ir.load(1, i)));
    ir.finish()
}
