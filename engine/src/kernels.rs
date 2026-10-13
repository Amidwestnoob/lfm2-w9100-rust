//! Seven compute operations authored in typed Rust IR, lowered directly to SPIR-V.
//!
//! No shader-text frontend or precompiled modules are used. `Value<T>` keeps
//! floating-point, unsigned and boolean arithmetic distinct at Rust compile time.
//! Rust closures build structured GPU loops; they do not execute the tensor math
//! on the CPU. The production host and numerical harness call these same builders.
use rspirv::{
    binary::Assemble,
    dr::{Builder, Instruction, Operand},
};
use spirv::{self as s, GlslStd450Op as GLOp, Op};
use std::{cell::RefCell, marker::PhantomData, ops};

struct Ir {
    b: RefCell<Builder>,
    locals: RefCell<Vec<Instruction>>,
    globals: RefCell<Vec<Instruction>>,
    u: u32,
    f: u32,
    boolean: u32,
    f2: u32,
    ext: u32,
    gid: u32,
    pc: u32,
    offset_pc: u32,
    local_size: u32,
    buffers: Vec<(u32, u32)>,
}
#[derive(Clone, Copy)]
struct Value<'a, T: Copy> {
    ir: &'a Ir,
    id: u32,
    marker: PhantomData<T>,
}
type U<'a> = Value<'a, u32>;
type F<'a> = Value<'a, f32>;
type B<'a> = Value<'a, bool>;
#[derive(Clone, Copy)]
struct Local<'a, T: Copy> {
    ptr: Value<'a, T>,
    ty: u32,
}
impl<'a, T: Copy> Value<'a, T> {
    fn new(ir: &'a Ir, id: u32) -> Self {
        Self {
            ir,
            id,
            marker: PhantomData,
        }
    }
}
impl Ir {
    fn new(integer_weights: bool, bindings: usize, push_fields: u32) -> Self {
        Self::configured(integer_weights, bindings, push_fields, 256)
    }
    fn configured(
        integer_weights: bool,
        bindings: usize,
        push_fields: u32,
        local_size: u32,
    ) -> Self {
        let mut b = Builder::new();
        b.set_version(1, 3);
        b.capability(s::Capability::Shader);
        b.capability(s::Capability::GroupNonUniform);
        b.capability(s::Capability::GroupNonUniformArithmetic);
        b.capability(s::Capability::GroupNonUniformShuffle);
        let ext = b.ext_inst_import("GLSL.std.450"); // Standard binary instruction set, not shader source.
        b.memory_model(s::AddressingModel::Logical, s::MemoryModel::GLSL450);
        let u = b.type_int(32, 0);
        let f = b.type_float(32, None);
        let boolean = b.type_bool();
        let f2 = b.type_vector(f, 2);
        let u3 = b.type_vector(u, 3);
        let gid_ptr = b.type_pointer(None, s::StorageClass::Input, u3);
        let gid = b.variable(gid_ptr, None, s::StorageClass::Input, None);
        b.decorate(
            gid,
            s::Decoration::BuiltIn,
            [Operand::BuiltIn(s::BuiltIn::GlobalInvocationId)],
        );
        let mut buffers = Vec::new();
        for binding in 0..bindings {
            let scalar = if integer_weights && binding == 0 {
                u
            } else {
                f
            };
            let array = b.type_runtime_array(scalar);
            b.decorate(
                array,
                s::Decoration::ArrayStride,
                [Operand::LiteralBit32(4)],
            );
            let st = b.type_struct([array]);
            b.decorate(st, s::Decoration::BufferBlock, []);
            b.member_decorate(st, 0, s::Decoration::Offset, [Operand::LiteralBit32(0)]);
            let ptr = b.type_pointer(None, s::StorageClass::Uniform, st);
            let var = b.variable(ptr, None, s::StorageClass::Uniform, None);
            b.decorate(
                var,
                s::Decoration::DescriptorSet,
                [Operand::LiteralBit32(0)],
            );
            b.decorate(
                var,
                s::Decoration::Binding,
                [Operand::LiteralBit32(binding as u32)],
            );
            buffers.push((var, scalar));
        }
        let st = b.type_struct(vec![u; push_fields as usize + 2]);
        b.decorate(st, s::Decoration::Block, []);
        for i in 0..push_fields + 2 {
            b.member_decorate(st, i, s::Decoration::Offset, [Operand::LiteralBit32(i * 4)]);
        }
        let ptr = b.type_pointer(None, s::StorageClass::PushConstant, st);
        let pc = b.variable(ptr, None, s::StorageClass::PushConstant, None);
        let void = b.type_void();
        let ft = b.type_function(void, []);
        let main = b
            .begin_function(void, None, s::FunctionControl::NONE, ft)
            .unwrap();
        b.entry_point(s::ExecutionModel::GLCompute, main, "main", [gid]);
        b.execution_mode(main, s::ExecutionMode::LocalSize, [local_size, 1, 1]);
        b.begin_block(None).unwrap();
        Self {
            b: RefCell::new(b),
            locals: RefCell::new(Vec::new()),
            globals: RefCell::new(Vec::new()),
            u,
            f,
            boolean,
            f2,
            ext,
            gid,
            pc,
            offset_pc: push_fields,
            local_size,
            buffers,
        }
    }
    fn emit<T: Copy>(&self, op: Op, ty: u32, ids: &[u32]) -> Value<'_, T> {
        let mut b = self.b.borrow_mut();
        let id = b.id();
        b.insert_into_block(
            rspirv::dr::InsertPoint::End,
            Instruction::new(
                op,
                Some(ty),
                Some(id),
                ids.iter().map(|&v| Operand::IdRef(v)).collect(),
            ),
        )
        .unwrap();
        Value::new(self, id)
    }
    fn uint(&self, n: u32) -> U<'_> {
        Value::new(self, self.b.borrow_mut().constant_bit32(self.u, n))
    }
    fn float(&self, n: f32) -> F<'_> {
        Value::new(
            self,
            self.b.borrow_mut().constant_bit32(self.f, n.to_bits()),
        )
    }
    fn pc(&self, field: u32) -> U<'_> {
        let index = self.uint(field);
        let mut b = self.b.borrow_mut();
        let ptr_ty = b.type_pointer(None, s::StorageClass::PushConstant, self.u);
        let ptr = b.access_chain(ptr_ty, None, self.pc, [index.id]).unwrap();
        Value::new(self, b.load(self.u, None, ptr, None, []).unwrap())
    }
    fn gid(&self, axis: u32) -> U<'_> {
        let mut b = self.b.borrow_mut();
        let ty = b.type_vector(self.u, 3);
        let v = b.load(ty, None, self.gid, None, []).unwrap();
        let raw: U<'_> = Value::new(self, b.composite_extract(self.u, None, v, [axis]).unwrap());
        drop(b);
        if axis < 2 {
            raw + self.pc(self.offset_pc + axis)
                * self.uint(if axis == 0 { self.local_size } else { 1 })
        } else {
            raw
        }
    }
    fn address(&self, binding: usize, index: U<'_>) -> u32 {
        let zero = self.uint(0);
        let (buffer, ty) = self.buffers[binding];
        let mut b = self.b.borrow_mut();
        let ptr = b.type_pointer(None, s::StorageClass::Uniform, ty);
        b.access_chain(ptr, None, buffer, [zero.id, index.id])
            .unwrap()
    }
    fn load(&self, binding: usize, index: U<'_>) -> F<'_> {
        assert_eq!(self.buffers[binding].1, self.f);
        let ptr = self.address(binding, index);
        Value::new(
            self,
            self.b
                .borrow_mut()
                .load(self.f, None, ptr, None, [])
                .unwrap(),
        )
    }
    fn word(&self, index: U<'_>) -> U<'_> {
        assert_eq!(self.buffers[0].1, self.u);
        let ptr = self.address(0, index);
        Value::new(
            self,
            self.b
                .borrow_mut()
                .load(self.u, None, ptr, None, [])
                .unwrap(),
        )
    }
    fn signed_scale(&self, word: U<'_>, shift: U<'_>) -> F<'_> {
        // Exact int8 decoding, not a quantization or floating arithmetic change.
        let ty=self.b.borrow_mut().type_int(32,1);
        let signed:Value<'_,i32>=self.emit(Op::Bitcast,ty,&[word.id]);
        let byte:Value<'_,i32>=self.emit(Op::BitFieldSExtract,ty,&[signed.id,shift.id,self.uint(8).id]);
        self.emit(Op::ConvertSToF,self.f,&[byte.id])
    }
    fn store(&self, binding: usize, index: U<'_>, value: F<'_>) {
        assert_eq!(self.buffers[binding].1, self.f);
        let ptr = self.address(binding, index);
        self.b.borrow_mut().store(ptr, value.id, None, []).unwrap();
    }
    fn local<T: Copy>(&self, init: Value<'_, T>, ty: u32) -> Local<'_, T> {
        let mut b = self.b.borrow_mut();
        let ptr_ty = b.type_pointer(None, s::StorageClass::Function, ty);
        let ptr = b.variable(ptr_ty, None, s::StorageClass::Function, None);
        self.locals.borrow_mut().push(b.pop_instruction().unwrap());
        b.store(ptr, init.id, None, []).unwrap();
        Local {
            ptr: Value::new(self, ptr),
            ty,
        }
    }
    fn accumulator(&self, init: F<'_>) -> Local<'_, f32> {
        self.local(init, self.f)
    }
    fn when(&self, cond: B<'_>, body: impl FnOnce()) {
        let (work, merge) = {
            let mut b = self.b.borrow_mut();
            let work = b.id();
            let merge = b.id();
            b.selection_merge(merge, s::SelectionControl::NONE).unwrap();
            b.branch_conditional(cond.id, work, merge, []).unwrap();
            b.begin_block(Some(work)).unwrap();
            (work, merge)
        };
        let _ = work;
        body();
        let mut b = self.b.borrow_mut();
        b.branch(merge).unwrap();
        b.begin_block(Some(merge)).unwrap();
    }
    fn subgroup_sum(&self, v: F<'_>) -> F<'_> {
        let scope = self.uint(3);
        let id = self
            .b
            .borrow_mut()
            .group_non_uniform_f_add(
                self.f,
                None,
                scope.id,
                s::GroupOperation::Reduce,
                v.id,
                None,
            )
            .unwrap();
        Value::new(self, id)
    }
    fn subgroup_max(&self, v: F<'_>) -> F<'_> {
        let scope = self.uint(3);
        let id = self
            .b
            .borrow_mut()
            .group_non_uniform_f_max(
                self.f,
                None,
                scope.id,
                s::GroupOperation::Reduce,
                v.id,
                None,
            )
            .unwrap();
        Value::new(self, id)
    }
    fn shared(&self, len: u32) -> u32 {
        let len = self.uint(len);
        let mut b = self.b.borrow_mut();
        let array = b.type_array(self.f, len.id);
        let ptr = b.type_pointer(None, s::StorageClass::Workgroup, array);
        let id = b.variable(ptr, None, s::StorageClass::Workgroup, None);
        self.globals.borrow_mut().push(b.pop_instruction().unwrap());
        id
    }
    fn shared_ptr(&self, array: u32, i: U<'_>) -> u32 {
        let mut b = self.b.borrow_mut();
        let ptr = b.type_pointer(None, s::StorageClass::Workgroup, self.f);
        b.access_chain(ptr, None, array, [i.id]).unwrap()
    }
    fn shared_load(&self, array: u32, i: U<'_>) -> F<'_> {
        let ptr = self.shared_ptr(array, i);
        Value::new(
            self,
            self.b
                .borrow_mut()
                .load(self.f, None, ptr, None, [])
                .unwrap(),
        )
    }
    fn shared_store(&self, array: u32, i: U<'_>, v: F<'_>) {
        let ptr = self.shared_ptr(array, i);
        self.b.borrow_mut().store(ptr, v.id, None, []).unwrap();
    }
    fn barrier(&self) {
        let scope = self.uint(2);
        let semantics = self.uint(0x108); // AcquireRelease | WorkgroupMemory
        self.b
            .borrow_mut()
            .control_barrier(scope.id, scope.id, semantics.id)
            .unwrap();
    }
    fn guard(&self, invalid: B<'_>) {
        let mut b = self.b.borrow_mut();
        let ret = b.id();
        let merge = b.id();
        b.selection_merge(merge, s::SelectionControl::NONE).unwrap();
        b.branch_conditional(invalid.id, ret, merge, []).unwrap();
        b.begin_block(Some(ret)).unwrap();
        b.ret().unwrap();
        b.begin_block(Some(merge)).unwrap();
    }
    fn range<'a>(&'a self, end: U<'a>, body: impl FnOnce(U<'a>)) {
        let index = self.local(self.uint(0), self.u);
        let (header, work, cont, merge) = {
            let mut b = self.b.borrow_mut();
            let ids = (b.id(), b.id(), b.id(), b.id());
            b.branch(ids.0).unwrap();
            b.begin_block(Some(ids.0)).unwrap();
            ids
        };
        let i = index.get();
        let cond = i.lt(end);
        {
            let mut b = self.b.borrow_mut();
            b.loop_merge(merge, cont, s::LoopControl::NONE, []).unwrap();
            b.branch_conditional(cond.id, work, merge, []).unwrap();
            b.begin_block(Some(work)).unwrap();
        }
        body(i);
        {
            let mut b = self.b.borrow_mut();
            b.branch(cont).unwrap();
            b.begin_block(Some(cont)).unwrap();
        }
        index.set(i + self.uint(1));
        let mut b = self.b.borrow_mut();
        b.branch(header).unwrap();
        b.begin_block(Some(merge)).unwrap();
    }
    fn math(&self, op: GLOp, args: &[F<'_>]) -> F<'_> {
        Value::new(
            self,
            self.b
                .borrow_mut()
                .ext_inst(
                    self.f,
                    None,
                    self.ext,
                    op as u32,
                    args.iter().map(|v| Operand::IdRef(v.id)),
                )
                .unwrap(),
        )
    }
    fn half_low(&self, bits: U<'_>) -> F<'_> {
        let mut b = self.b.borrow_mut();
        let pair = b
            .ext_inst(
                self.f2,
                None,
                self.ext,
                GLOp::UnpackHalf2x16 as u32,
                [Operand::IdRef(bits.id)],
            )
            .unwrap();
        Value::new(self, b.composite_extract(self.f, None, pair, [0]).unwrap())
    }
    fn finish(self) -> Vec<u32> {
        let mut b = self.b.into_inner();
        b.ret().unwrap();
        b.end_function().unwrap();
        let mut module = b.module();
        module.types_global_values.extend(self.globals.into_inner());
        // rspirv interns identical types: shared buffer types need each layout
        // decoration only once, even when several bindings use that type.
        let mut seen = Vec::new();
        module.annotations.retain(|instruction| {
            if seen.contains(instruction) {
                false
            } else {
                seen.push(instruction.clone());
                true
            }
        });
        let first = &mut module.functions[0].blocks[0].instructions;
        first.splice(0..0, self.locals.into_inner());
        module.assemble()
    }
}
impl<'a, T: Copy> Local<'a, T> {
    fn get(self) -> Value<'a, T> {
        Value::new(
            self.ptr.ir,
            self.ptr
                .ir
                .b
                .borrow_mut()
                .load(self.ty, None, self.ptr.id, None, [])
                .unwrap(),
        )
    }
    fn set(self, value: Value<'_, T>) {
        self.ptr
            .ir
            .b
            .borrow_mut()
            .store(self.ptr.id, value.id, None, [])
            .unwrap();
    }
}
macro_rules! binary {
    ($ty:ty, $trait:ident, $method:ident, $op:ident, $field:ident) => {
        impl<'a> ops::$trait for Value<'a, $ty> {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                assert!(std::ptr::eq(self.ir, rhs.ir));
                self.ir.emit(Op::$op, self.ir.$field, &[self.id, rhs.id])
            }
        }
    };
}
binary!(u32, Add, add, IAdd, u);
binary!(u32, Sub, sub, ISub, u);
binary!(u32, Mul, mul, IMul, u);
binary!(u32, Div, div, UDiv, u);
binary!(u32, Rem, rem, UMod, u);
binary!(u32, BitAnd, bitand, BitwiseAnd, u);
binary!(u32, BitOr, bitor, BitwiseOr, u);
binary!(u32, Shr, shr, ShiftRightLogical, u);
binary!(u32, Shl, shl, ShiftLeftLogical, u);
binary!(f32, Add, add, FAdd, f);
binary!(f32, Sub, sub, FSub, f);
binary!(f32, Mul, mul, FMul, f);
binary!(f32, Div, div, FDiv, f);
binary!(bool, BitOr, bitor, LogicalOr, boolean);
binary!(bool, BitAnd, bitand, LogicalAnd, boolean);
impl<'a> U<'a> {
    fn lt(self, rhs: Self) -> B<'a> {
        self.ir
            .emit(Op::ULessThan, self.ir.boolean, &[self.id, rhs.id])
    }
    fn ge(self, rhs: Self) -> B<'a> {
        self.ir
            .emit(Op::UGreaterThanEqual, self.ir.boolean, &[self.id, rhs.id])
    }
    fn eq(self, rhs: Self) -> B<'a> {
        self.ir
            .emit(Op::IEqual, self.ir.boolean, &[self.id, rhs.id])
    }
    fn float(self) -> F<'a> {
        self.ir.emit(Op::ConvertUToF, self.ir.f, &[self.id])
    }
    fn float_bits(self) -> F<'a> {
        self.ir.emit(Op::Bitcast, self.ir.f, &[self.id])
    }
}
impl<'a> B<'a> {
    fn choose_u(self, yes: U<'a>, no: U<'a>) -> U<'a> {
        self.ir
            .emit(Op::Select, self.ir.u, &[self.id, yes.id, no.id])
    }
    fn choose_f(self, yes: F<'a>, no: F<'a>) -> F<'a> {
        self.ir
            .emit(Op::Select, self.ir.f, &[self.id, yes.id, no.id])
    }
}
impl<'a> F<'a> {
    fn eq(self, rhs: Self) -> B<'a> {
        self.ir
            .emit(Op::FOrdEqual, self.ir.boolean, &[self.id, rhs.id])
    }
    fn uint(self) -> U<'a> {
        self.ir.emit(Op::ConvertFToU, self.ir.u, &[self.id])
    }
    fn bits(self) -> U<'a> {
        self.ir.emit(Op::Bitcast, self.ir.u, &[self.id])
    }
    fn gt(self, rhs: Self) -> B<'a> {
        self.ir
            .emit(Op::FOrdGreaterThan, self.ir.boolean, &[self.id, rhs.id])
    }
    fn precise(self) -> Self {
        self.ir
            .b
            .borrow_mut()
            .decorate(self.id, s::Decoration::NoContraction, []);
        self
    }
    fn fma(self, b: Self, c: Self) -> Self {
        self.ir.math(GLOp::Fma, &[self, b, c])
    }
    fn exp(self) -> Self {
        self.ir.math(GLOp::Exp, &[self])
    }
    fn invsqrt(self) -> Self {
        self.ir.math(GLOp::InverseSqrt, &[self])
    }
    fn pow(self, rhs: Self) -> Self {
        self.ir.math(GLOp::Pow, &[self, rhs])
    }
    fn sin(self) -> Self {
        self.ir.math(GLOp::Sin, &[self])
    }
    fn cos(self) -> Self {
        self.ir.math(GLOp::Cos, &[self])
    }
    fn max(self, rhs: Self) -> Self {
        self.ir.math(GLOp::FMax, &[self, rhs])
    }
}
impl<'a> ops::Neg for F<'a> {
    type Output = Self;
    fn neg(self) -> Self {
        self.ir.emit(Op::FNegate, self.ir.f, &[self.id])
    }
}

pub fn build_swiglu() -> Vec<u32> {
    let ir = Ir::new(false, 3, 1);
    let i = ir.gid(0);
    ir.guard(i.ge(ir.pc(0)));
    let g = ir.load(0, i);
    ir.store(2, i, g / (ir.float(1.0) + (-g).exp()) * ir.load(1, i));
    ir.finish()
}

pub fn build_f32_matmul() -> Vec<u32> {
    let ir = Ir::configured(false, 3, 3, 64);
    let (input, output, n) = (ir.pc(0), ir.pc(1), ir.pc(2));
    let (o, j) = (ir.gid(0) / ir.uint(64), ir.gid(1));
    let tid = ir.gid(0) % ir.uint(64);
    ir.guard(o.ge(output) | j.ge(n));
    let acc = ir.accumulator(ir.float(0.0));
    ir.range((input + ir.uint(255)) / ir.uint(256), |i| {
        let mut dot = ir.float(0.0);
        for v in 0..4 {
            let k = i * ir.uint(256) + tid * ir.uint(4) + ir.uint(v);
            let a = ir.accumulator(ir.float(0.0));
            ir.when(k.lt(input), || {
                a.set(ir.load(0, k + o * input) * ir.load(1, k + j * input))
            });
            dot = dot + a.get();
        }
        acc.set(acc.get() + dot);
    });
    let sum = ir.subgroup_sum(acc.get());
    ir.when(tid.eq(ir.uint(0)), || ir.store(2, o + j * output, sum));
    ir.finish()
}

pub fn build_rms_norm() -> Vec<u32> {
    // Native 512-lane, descending shared-memory reduction tree.
    let ir = Ir::configured(false, 3, 3, 512);
    let shared = ir.shared(512);
    let (dim, n, eps) = (ir.pc(0), ir.pc(1), ir.pc(2).float_bits());
    let j = ir.gid(0) / ir.uint(512);
    let tid = ir.gid(0) % ir.uint(512);
    ir.guard(j.ge(n));
    let base = j * dim;
    let ss = ir.accumulator(ir.float(0.0));
    let count = (dim + ir.uint(511)) / ir.uint(512);
    ir.range(count, |i| {
        let k = i * ir.uint(512) + tid;
        ir.when(k.lt(dim), || {
            let v = ir.load(0, k + base);
            ss.set(ss.get() + v * v);
        });
    });
    ir.shared_store(shared, tid, ss.get());
    ir.barrier();
    for shift in (6..9).rev() {
        let step = ir.uint(1 << shift);
        ir.when(tid.lt(step), || {
            ir.shared_store(
                shared,
                tid,
                ir.shared_load(shared, tid) + ir.shared_load(shared, tid + step),
            )
        });
        ir.barrier();
    }
    // The final 64 lanes belong to one Hawaii subgroup. XOR steps reproduce
    // the same descending tree for lane zero without six workgroup barriers.
    ir.when(tid.lt(ir.uint(64)), || {
        let mut sum = ir.shared_load(shared, tid);
        for shift in (0..6).rev() {
            let scope = ir.uint(3);
            let mask = ir.uint(1 << shift);
            let other: F<'_> = ir.emit(
                Op::GroupNonUniformShuffleXor,
                ir.f,
                &[scope.id, sum.id, mask.id],
            );
            sum = sum + other;
        }
        ir.when(tid.eq(ir.uint(0)), || {
            ir.shared_store(shared, ir.uint(0), sum)
        });
    });
    ir.barrier();
    let inv = (ir.shared_load(shared, ir.uint(0)) / dim.float() + eps).invsqrt();
    ir.range(count, |i| {
        let k = i * ir.uint(512) + tid;
        ir.when(k.lt(dim), || {
            ir.store(2, k + base, ir.load(0, k + base) * inv * ir.load(1, k))
        });
    });
    ir.finish()
}

pub fn build_rope() -> Vec<u32> {
    rope_impl(false)
}
pub fn build_rope_decode() -> Vec<u32> {
    rope_impl(true)
}
fn rope_impl(dynamic: bool) -> Vec<u32> {
    let ir = Ir::new(false, if dynamic { 2 } else { 1 }, 5);
    let (hd, heads, n, start, tscale) = (
        ir.pc(0),
        ir.pc(1),
        ir.pc(2),
        if dynamic {
            ir.load(1, ir.uint(0)).uint()
        } else {
            ir.pc(3)
        },
        ir.pc(4).float_bits(),
    );
    let idx = ir.gid(0);
    let hd2 = hd / ir.uint(2);
    let p = idx % hd2;
    let row = idx / hd2;
    let (h, j) = (row / n, row % n);
    ir.guard(h.ge(heads) | j.ge(n));
    let pos = (start + j).float();
    let theta = pos * tscale.pow(p.float());
    let (c, s) = (theta.cos(), theta.sin());
    let base = h * hd + j * hd * heads;
    let (i0, i1) = (p + base, p + hd2 + base);
    let (x0, x1) = (ir.load(0, i0), ir.load(0, i1));
    ir.store(0, i0, x0 * c - x1 * s);
    ir.store(0, i1, x0 * s + x1 * c);
    ir.finish()
}

pub fn build_attention() -> Vec<u32> {
    attention_impl(false, false, false)
}
pub fn build_attention_decode() -> Vec<u32> {
    attention_impl(true, false, false)
}
pub fn build_attention_unrolled() -> Vec<u32> {
    attention_impl(false, true, false)
}
pub fn build_attention_unrolled_decode() -> Vec<u32> {
    attention_impl(true, true, false)
}
/// Prefill-only query reuse; all ordered arithmetic and dependencies unchanged.
pub fn build_attention_queryhoist() -> Vec<u32> {
    attention_impl(false, true, true)
}
fn attention_impl(dynamic: bool, unroll: bool, hoist_query: bool) -> Vec<u32> {
    let ir = Ir::configured(false, if dynamic { 5 } else { 4 }, 7, 64);
    let scores = ir.shared(4096);
    let (hd, heads, kv, n) = (ir.pc(0), ir.pc(1), ir.pc(2), ir.pc(3));
    let (start, scale) = (
        if dynamic {
            ir.load(4, ir.uint(0)).uint()
        } else {
            ir.pc(4)
        },
        ir.pc(6).float_bits(),
    );
    let idx = ir.gid(0) / ir.uint(64);
    let tid = ir.gid(0) % ir.uint(64);
    let (hh, j) = (idx / n, idx % n);
    ir.guard(hh.ge(heads) | j.ge(n));
    let nt = start + j + ir.uint(1);
    let kh = hh * kv / heads;
    let qrow = hh * hd + j * hd * heads;
    let count = (nt + ir.uint(63)) / ir.uint(64);
    // These storage values are immutable for this entire dispatch. Do not alter
    // the 64-term dot order, reduction, scores or V accumulation below.
    let query: Vec<_> = if hoist_query {
        (0..64).map(|d| ir.load(0, ir.uint(d) + qrow)).collect()
    } else { Vec::new() };
    let m = ir.accumulator(ir.float(-1.0e30));
    ir.range(count, |i| {
        let tt = i * ir.uint(64) + tid;
        ir.when(tt.lt(nt), || {
            let dot = ir.accumulator(ir.float(0.0));
            if unroll {
                for d in 0..64 {
                    let q = if hoist_query {query[d as usize]} else {ir.load(0, ir.uint(d) + qrow)};
                    let d = ir.uint(d);
                    dot.set(
                        dot.get() + q * ir.load(1, d + kh * hd + tt * hd * kv),
                    );
                }
            } else {
                ir.range(hd, |d| {
                    dot.set(
                        dot.get() + ir.load(0, d + qrow) * ir.load(1, d + kh * hd + tt * hd * kv),
                    )
                });
            }
            let value = dot.get() * scale;
            ir.shared_store(scores, tt, value);
            m.set(m.get().max(value));
        });
    });
    let max = ir.subgroup_max(m.get());
    let sum = ir.accumulator(ir.float(0.0));
    ir.range(count, |i| {
        let tt = i * ir.uint(64) + tid;
        ir.when(tt.lt(nt), || {
            let value = (ir.shared_load(scores, tt) - max).exp();
            ir.shared_store(scores, tt, value);
            sum.set(sum.get() + value);
        });
    });
    let denom = ir.subgroup_sum(sum.get());
    ir.barrier();
    ir.range((hd + ir.uint(63)) / ir.uint(64), |i| {
        let d = i * ir.uint(64) + tid;
        ir.when(d.lt(hd), || {
            let acc = ir.accumulator(ir.float(0.0));
            ir.range(nt, |tt| {
                acc.set(
                    acc.get() + ir.shared_load(scores, tt) * ir.load(2, tt * hd * kv + kh * hd + d),
                )
            });
            ir.store(3, d + qrow, acc.get() / denom);
        });
    });
    ir.finish()
}

/// Byte addressing handles Q6_K's 210-byte blocks, including unaligned halves.
fn byte<'a>(ir: &'a Ir, index: U<'a>) -> U<'a> {
    (ir.word(index >> ir.uint(2)) >> (ir.uint(8) * (index & ir.uint(3)))) & ir.uint(255)
}
fn dequant<'a>(ir: &'a Ir, base: U<'a>, k: U<'a>) -> F<'a> {
    let half = k >> ir.uint(7);
    let jj = k & ir.uint(127);
    let sub = jj >> ir.uint(5);
    let l = jj & ir.uint(31);
    let is = l >> ir.uint(4);
    let ql = byte(
        ir,
        base + half * ir.uint(64) + l + (sub & ir.uint(1)) * ir.uint(32),
    );
    let qh = byte(ir, base + ir.uint(128) + half * ir.uint(32) + l);
    let sc = byte(
        ir,
        base + ir.uint(192) + half * ir.uint(8) + is + ir.uint(2) * sub,
    );
    let nibble = (sub >> ir.uint(1))
        .eq(ir.uint(0))
        .choose_u(ql & ir.uint(15), ql >> ir.uint(4));
    let field = (qh >> (ir.uint(2) * sub)) & ir.uint(3);
    let q = (nibble | (field << ir.uint(4))).float() - ir.float(32.0);
    let signed_scale = sc
        .ge(ir.uint(128))
        .choose_f(sc.float() - ir.float(256.0), sc.float());
    let dh = byte(ir, base + ir.uint(208)) | (byte(ir, base + ir.uint(209)) << ir.uint(8));
    ir.half_low(dh) * signed_scale * q
}
// Cooperative translation of llama.cpp mul_mat_vec_q6_k (MIT; see
// LICENSE-llama.cpp). A Hawaii wave handles four rows, 16 lanes/block.
// Packed loads, four independent dot products, nested scale FMAs and the
// subgroup reduction follow the native algorithm. Scales are loaded directly
// instead of exchanged in shared memory; no cross-lane scale dependency remains.
fn packed_word<'a>(ir: &'a Ir, address: U<'a>) -> U<'a> {
    let shift = (address & ir.uint(3)) * ir.uint(8);
    let lo = ir.word(address >> ir.uint(2));
    let hi = ir.word((address >> ir.uint(2)) + ir.uint(1));
    let upper = shift
        .eq(ir.uint(0))
        .choose_u(ir.uint(0), hi << ((ir.uint(32) - shift) & ir.uint(31)));
    (lo >> shift) | upper
}
fn signed_byte<'a>(ir: &'a Ir, address: U<'a>) -> F<'a> {
    let v = byte(ir, address);
    v.ge(ir.uint(128))
        .choose_f(v.float() - ir.float(256.0), v.float())
}
fn q6(expert: bool, resident: bool, tile: usize) -> Vec<u32> {
    q6_layout(expert, resident, tile, false)
}
pub fn build_q6k_aligned(expert: bool, resident: bool, tile: usize) -> Vec<u32> {
    q6_layout(expert, resident, tile, true)
}
fn q6_layout(expert: bool, resident: bool, tile: usize, aligned: bool) -> Vec<u32> {
    q6_layout_compact(expert,resident,tile,aligned,false)
}
pub fn build_q6k_grouped_compact(aligned: bool) -> Vec<u32> {
    q6_layout_compact(true,true,4,aligned,true)
}
fn q6_layout_compact(expert: bool, resident: bool, tile: usize, aligned: bool, compact: bool) -> Vec<u32> {
    q6_scale_layout(expert,resident,tile,aligned,compact,false)
}
pub fn build_q6k_scale_pair(expert: bool, resident: bool, tile: usize, compact: bool) -> Vec<u32> {
    q6_scale_layout(expert,resident,tile,true,compact,true)
}
fn q6_scale_layout(expert: bool, resident: bool, tile: usize, aligned: bool, compact: bool, scale_pair: bool) -> Vec<u32> {
    let stride = if aligned {212} else {210};
    let grouped = resident && tile > 1;
    let ir = Ir::configured(
        true,
        if resident { 4 } else { 3 },
        if expert && !resident { 7 } else { 5 },
        64,
    );
    let (input, output) = (ir.pc(0), ir.pc(1));
    let blocks = ir.pc(if expert { 2 } else { 3 });
    let tid = ir.gid(0) % ir.uint(64);
    let first = ir.gid(0) / ir.uint(64) * ir.uint(4);
    let tiles = if grouped {
        if compact {(ir.pc(3)+ir.uint(3))/ir.uint(4)} else {ir.pc(3)}
    } else { ir.uint(1) };
    let expert_id = ir.gid(1) / tiles;
    let j = if grouped {
        (ir.gid(1) % tiles) * ir.uint(tile as u32)
    } else {
        ir.gid(1) * ir.uint(tile as u32)
    };
    let group_base = expert_id * (ir.pc(3) * ir.uint(4) + ir.uint(1));
    let group_count = if grouped {
        ir.load(3, group_base).uint()
    } else {
        ir.uint(0)
    };
    if grouped {
        ir.guard(j.ge(group_count));
    }
    let group_slot = |t: usize| {
        let col = j + ir.uint(t as u32);
        ir.load(
            3,
            group_base + ir.uint(1) + col.lt(group_count).choose_u(col, ir.uint(0)),
        )
        .uint()
    };
    let lane = tid % ir.uint(16);
    let half = lane / ir.uint(8);
    let l = lane % ir.uint(8) * ir.uint(4);
    let scale_offset = half * ir.uint(8) + l / ir.uint(16);
    let input_j = if resident {
        ir.pc(4).eq(ir.uint(0)).choose_u(j / ir.uint(4), j)
    } else {
        j
    };
    let wbase = if grouped {
        expert_id * output * blocks * ir.uint(stride)
    } else if resident {
        ir.load(3, (j / ir.uint(4)) * ir.uint(8) + j % ir.uint(4))
            .uint()
            * output
            * blocks
            * ir.uint(stride)
    } else if expert {
        let ei = j.eq(ir.uint(0)).choose_u(
            ir.pc(3),
            j.eq(ir.uint(1))
                .choose_u(ir.pc(4), j.eq(ir.uint(2)).choose_u(ir.pc(5), ir.pc(6))),
        );
        ei * output * blocks * ir.uint(stride)
    } else {
        ir.pc(4)
    };
    let acc: Vec<_> = (0..4 * tile)
        .map(|_| ir.accumulator(ir.float(0.0)))
        .collect();
    let fixed_expert = scale_pair && expert && resident && grouped;
    let block_body = |iteration: U<'_>| {
        let bi = iteration * ir.uint(4) + tid / ir.uint(16);
        ir.when(bi.lt(blocks), || {
            let xs: Vec<Vec<Vec<_>>> = (0..tile)
                .map(|t| {
                    // Tail columns use a valid input but are never stored.
                    let col = if grouped {
                        let slot = group_slot(t);
                        ir.pc(4).eq(ir.uint(0)).choose_u(slot / ir.uint(4), slot)
                    } else if tile > 1 {
                        let col = input_j + ir.uint(t as u32);
                        col.lt(ir.pc(2)).choose_u(col, ir.uint(0))
                    } else {
                        input_j
                    };
                    (0..4)
                        .map(|s| {
                            (0..4)
                                .map(|v| {
                                    ir.load(
                                        1,
                                        col * input
                                            + bi * ir.uint(256)
                                            + half * ir.uint(128)
                                            + l
                                            + ir.uint(s * 32 + v),
                                    )
                                })
                                .collect()
                        })
                        .collect()
                })
                .collect();
            for row in 0..4 {
                let o = first + ir.uint(row);
                let row_body = || {
                    let base = wbase + (o * blocks + bi) * ir.uint(stride);
                    let load_word = |addr| if aligned {ir.word(addr >> ir.uint(2))} else {packed_word(&ir, addr)};
                    let ql0 = load_word(base + half * ir.uint(64) + l);
                    let ql1 = load_word(base + half * ir.uint(64) + l + ir.uint(32));
                    let qh = load_word(base + ir.uint(128) + half * ir.uint(32) + l);
                    let qs = [
                        (ql0 & ir.uint(0x0f0f0f0f)) | ((qh & ir.uint(0x03030303)) << ir.uint(4)),
                        (ql1 & ir.uint(0x0f0f0f0f)) | ((qh & ir.uint(0x0c0c0c0c)) << ir.uint(2)),
                        ((ql0 >> ir.uint(4)) & ir.uint(0x0f0f0f0f)) | (qh & ir.uint(0x30303030)),
                        ((ql1 >> ir.uint(4)) & ir.uint(0x0f0f0f0f))
                            | ((qh & ir.uint(0xc0c0c0c0)) >> ir.uint(2)),
                    ];
                    let scales: Vec<_> = if scale_pair {
                        // base is four-byte aligned. Each half-wave needs two
                        // words, not four generic byte-addressed word loads.
                        // Extract the identical bytes; signed float conversion
                        // and every subsequent FMA remain verbatim unchanged.
                        let words = [
                            ir.word((base + ir.uint(192) + half * ir.uint(8)) >> ir.uint(2)),
                            ir.word((base + ir.uint(196) + half * ir.uint(8)) >> ir.uint(2)),
                        ];
                        (0..4usize).map(|s| {
                            let shift=(l / ir.uint(16) + ir.uint((s%2) as u32*2))*ir.uint(8);
                            ir.signed_scale(words[s/2],shift)
                        }).collect()
                    } else {
                        (0..4).map(|s| signed_byte(&ir, base + ir.uint(192 + s * 2) + scale_offset)).collect()
                    };
                    let d = if aligned {ir.half_low(ir.word((base + ir.uint(208)) >> ir.uint(2)))} else {ir.half_low(
                        byte(&ir, base + ir.uint(208))
                            | (byte(&ir, base + ir.uint(209)) << ir.uint(8)),
                    )};
                    let quants: Vec<Vec<_>> = (0..4)
                        .map(|s| {
                            (0..4)
                                .map(|v| {
                                    ((qs[s] >> ir.uint(v * 8)) & ir.uint(255)).float()
                                        - ir.float(32.0)
                                })
                                .collect()
                        })
                        .collect();
                    for t in 0..tile {
                        let sums: Vec<_> = (0..4)
                            .map(|s| {
                                let mut sum = ir.float(0.0);
                                for v in 0..4 {
                                    sum = xs[t][s][v].fma(quants[s][v], sum);
                                }
                                sum
                            })
                            .collect();
                        let sum = sums[0].fma(
                            scales[0],
                            sums[1].fma(scales[1], sums[2].fma(scales[2], sums[3] * scales[3])),
                        );
                        let a = &acc[t * 4 + row as usize];
                        a.set(sum.fma(d, a.get()));
                    }
                };
                if fixed_expert { row_body(); } else { ir.when(o.lt(output),row_body); }
            }
        });
    };
    ir.range((blocks+ir.uint(3))/ir.uint(4),block_body);
    for t in 0..tile {
        for row in 0..4 {
            let result = ir.subgroup_sum(acc[t * 4 + row].get());
            let o = first + ir.uint(row as u32);
            let col = if grouped {
                group_slot(t)
            } else {
                j + ir.uint(t as u32)
            };
            let valid = if grouped {
                tid.eq(ir.uint(0)) & (j + ir.uint(t as u32)).lt(group_count)
            } else if tile > 1 {
                tid.eq(ir.uint(0)) & col.lt(ir.pc(2))
            } else {
                tid.eq(ir.uint(0))
            };
            ir.when(valid, || {
                ir.when(o.lt(output), || ir.store(2, o + col * output, result))
            });
        }
    }
    ir.finish()
}
pub fn build_q6k_matmul() -> Vec<u32> {
    q6(false, false, 1)
}
pub fn build_q6k_matmul_4expert() -> Vec<u32> {
    q6(true, false, 1)
}
pub fn build_q6k_routed() -> Vec<u32> {
    q6(true, true, 1)
}
pub fn build_q6k_grouped() -> Vec<u32> {
    q6(true, true, 4)
}

pub fn build_q6k_prompt() -> Vec<u32> {
    q6(false, false, 4)
}
pub fn build_q6k_prompt2() -> Vec<u32> {
    q6(false, false, 2)
}

mod attention_fast;
mod attention_transposed;
mod attention_coalesced;
mod attention_split;
pub use attention_split::*;
pub use attention_coalesced::*;
pub use attention_transposed::*;
mod resident;
pub use attention_fast::*;
pub use resident::*;
mod persistent;
pub use persistent::*;
