//! Q6_K dequantization and tile math (CPU reference implementation).
//!
//! This is the exact arithmetic behind the Vulkan GLSL kernels in `shaders/`.
//! It exists as an independent, human-checkable reference used to (a) verify
//! the first real Q6_K multiply against the GPU, and (b) run a pure-Rust CPU
//! forward pass for correctness testing.
//!
//! Q6_K block on disk (256 weights, 210 bytes), per ggml's `block_q6_K`:
//!   ql[128]     — 4-bit low nibbles, 2 weights per byte
//!   qh[64]      — 2-bit high nibbles, 4 weights per byte
//!   scales[16]  — int8, one per 16-weight group
//!   d[2]        — fp16 super-block scale
//!
//! Weight j (0..=255) in a block:
//!   group  g = j / 16            -> scale `scales[g]`
//!   within the block, elements are reordered as described by ggml's
//!   `dequantize_row_q6_K` (see comments in the dequant function).

/// Q6_K block size.
pub const QK_K: usize = 256;
/// Q6_K bytes per block.
pub const QK_K_BYTES: usize = 210;
/// Isolated GPU representation; every original byte retained plus two zero pad bytes.
pub const ALIGNED_BYTES: usize = 212;
pub fn repack_aligned(raw: &[u8]) -> Vec<u8> {
    assert_eq!(raw.len()%QK_K_BYTES,0);
    let mut out = vec![0u8; raw.len()/QK_K_BYTES*ALIGNED_BYTES];
    for (a,b) in raw.chunks_exact(QK_K_BYTES).zip(out.chunks_exact_mut(ALIGNED_BYTES)) {b[..QK_K_BYTES].copy_from_slice(a);}
    out
}
pub fn repack_aligned_in_place(bytes: &mut [u8], raw_len: usize) {
    assert_eq!(raw_len%QK_K_BYTES,0);
    let blocks = raw_len/QK_K_BYTES;
    assert!(bytes.len()>=blocks*ALIGNED_BYTES);
    for i in (0..blocks).rev() {
        bytes.copy_within(i*QK_K_BYTES..(i+1)*QK_K_BYTES,i*ALIGNED_BYTES);
        bytes[i*ALIGNED_BYTES+QK_K_BYTES..(i+1)*ALIGNED_BYTES].fill(0);
    }
}

use std::sync::OnceLock;

/// Diagnostic switch: `LFM_Q6K_LEGACY_UNSIGNED=1` restores the snapshot's
/// (buggy) u8 interpretation of the Q6_K scales, so the Rig experiment can show
/// old-vs-fixed against the native layer dumps in one session.
pub fn legacy_unsigned_scales() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| {
        std::env::var("LFM_Q6K_LEGACY_UNSIGNED")
            .map(|v| v == "1")
            .unwrap_or(false)
    })
}

#[inline]
pub fn fp16_to_f32(bits: u16) -> f32 {
    half::f16::from_bits(bits).to_f32()
}

/// Dequantize one Q6_K block (210 bytes) into 256 f32 weights.
///
/// Mirrors ggml's `dequantize_row_q6_K` exactly (see
/// ggml/src/ggml-quants.c). For each half (128 elements):
///   - `is = l/16` selects between scale pairs.
///   - low nibbles of ql[0..64] give elements 0..63 (two per byte),
///     high nibbles give elements 64..127.
///   - qh's 2-bit fields supply the top 2 bits (values -32 .. 31).
pub fn dequantize_q6k_block(block: &[u8], out: &mut [f32]) {
    dequantize_q6k_block_mode(block, out, legacy_unsigned_scales())
}

/// Same as [`dequantize_q6k_block`] with an explicit scale interpretation.
/// `legacy_unsigned = true` reproduces the snapshot bug (u8 scales).
pub fn dequantize_q6k_block_mode(block: &[u8], out: &mut [f32], legacy_unsigned: bool) {
    debug_assert_eq!(block.len(), QK_K_BYTES);
    debug_assert_eq!(out.len(), QK_K);

    let d = fp16_to_f32(u16::from_le_bytes([block[208], block[209]]));
    let ql = &block[0..128];
    let qh = &block[128..192];
    let sc = &block[192..208];

    // First half (elements 0..127) uses ql[0..64], qh[0..32], sc[0..8].
    dequant_half(
        d,
        &ql[0..64],
        &qh[0..32],
        &sc[0..8],
        &mut out[0..128],
        legacy_unsigned,
    );
    // Second half (elements 128..255) uses ql[64..128], qh[32..64], sc[8..16].
    dequant_half(
        d,
        &ql[64..128],
        &qh[32..64],
        &sc[8..16],
        &mut out[128..256],
        legacy_unsigned,
    );
}

#[inline]
fn dequant_half(d: f32, ql: &[u8], qh: &[u8], sc: &[u8], y: &mut [f32], legacy_unsigned: bool) {
    // ql: 64 bytes = 128 nibbles. qh: 32 bytes = 128 2-bit fields.
    for l in 0..32usize {
        let is = (l / 16) as usize;
        let qh_l = qh[l];
        // element l       : low  nibble of ql[l],     qh bits 0-1
        let q1 = ((ql[l] & 0x0F) as i8 | (((qh_l & 0x03) as i8) << 4)) - 32;
        // element l + 32  : low  nibble of ql[l + 32], qh bits 2-3
        let q2 = ((ql[l + 32] & 0x0F) as i8 | ((((qh_l >> 2) & 0x03) as i8) << 4)) - 32;
        // element l + 64  : high nibble of ql[l],      qh bits 4-5
        let q3 = ((ql[l] >> 4) as i8) | ((((qh_l >> 4) & 0x03) as i8) << 4);
        let q3 = q3 - 32;
        // element l + 96  : high nibble of ql[l + 32], qh bits 6-7
        let q4 = ((ql[l + 32] >> 4) as i8) | ((((qh_l >> 6) & 0x03) as i8) << 4);
        let q4 = q4 - 32;

        // ggml `block_q6_K::scales` is `int8_t` (ggml-common.h). The quantizer
        // (quantize_row_q6_K_ref) maps the largest-|scale| group to exactly
        // -128, so negative scales occur in essentially every block. The
        // snapshot read these bytes as u8 (0x80 -> +128), corrupting every
        // dequantized weight in a negative-scale group.
        let (s0, s2, s4, s6) = if legacy_unsigned {
            (
                sc[is] as f32,
                sc[is + 2] as f32,
                sc[is + 4] as f32,
                sc[is + 6] as f32,
            )
        } else {
            (
                sc[is] as i8 as f32,
                sc[is + 2] as i8 as f32,
                sc[is + 4] as i8 as f32,
                sc[is + 6] as i8 as f32,
            )
        };

        y[l] = d * s0 * q1 as f32;
        y[l + 32] = d * s2 * q2 as f32;
        y[l + 64] = d * s4 * q3 as f32;
        y[l + 96] = d * s6 * q4 as f32;
    }
}

/// Dequantize a whole row-vector of `n` weights (n a multiple of 256) which is
/// stored as `n/256` contiguous Q6_K blocks, into `f32`.
pub fn dequantize_q6k_row(blocks: &[u8], out: &mut [f32]) {
    let nblocks = blocks.len() / QK_K_BYTES;
    debug_assert_eq!(nblocks * 256, out.len());
    for b in 0..nblocks {
        dequantize_q6k_block(
            &blocks[b * QK_K_BYTES..(b + 1) * QK_K_BYTES],
            &mut out[b * 256..(b + 1) * 256],
        );
    }
}

/// CPU-reference matrix-vector-style multiply used by the first-hardware check.
///
/// `weight` is a Q6_K tensor with `in_dim` inputs (contiguous) x `out_dim`
/// outputs; each output row is `in_dim/256` contiguous Q6_K blocks. `x` is
/// `in_dim` (a single activation vector). Returns `out_dim` dot products, by
/// dequantizing each row and taking an f32 dot product.
pub fn q6k_matvec(weight: &[u8], in_dim: usize, out_dim: usize, x: &[f32], out: &mut [f32]) {
    let blocks_per_row = in_dim / QK_K;
    let row_bytes = blocks_per_row * QK_K_BYTES;
    debug_assert_eq!(weight.len(), row_bytes * out_dim);
    debug_assert_eq!(x.len(), in_dim);
    debug_assert_eq!(out.len(), out_dim);

    let mut wrow = vec![0f32; in_dim];
    for o in 0..out_dim {
        dequantize_q6k_row(&weight[o * row_bytes..(o + 1) * row_bytes], &mut wrow);
        let mut acc = 0f64;
        for k in 0..in_dim {
            acc += wrow[k] as f64 * x[k] as f64;
        }
        out[o] = acc as f32;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn aligned_repack_preserves_all_bytes_and_padding() {
        for blocks in [1,2,7,128] {
            let raw:Vec<u8>=(0..blocks*210).map(|x|x as u8).collect();
            let mut inplace=vec![0;blocks*212]; inplace[..raw.len()].copy_from_slice(&raw);
            repack_aligned_in_place(&mut inplace,raw.len());
            assert_eq!(inplace,repack_aligned(&raw));
            for (a,b) in raw.chunks_exact(210).zip(inplace.chunks_exact(212)) {assert_eq!(a,&b[..210]);assert_eq!(&b[210..],&[0,0]);}
        }
    }

    #[test]
    fn fp16_roundtrip() {
        assert_eq!(fp16_to_f32(half::f16::from_f32(1.0).to_bits()), 1.0);
        assert_eq!(fp16_to_f32(half::f16::from_f32(-2.5).to_bits()), -2.5);
    }

    #[test]
    fn dequant_single_known_block() {
        // d = 1.0, all scales = 1.
        let mut block = [0u8; QK_K_BYTES];
        block[208] = 0x00; // fp16 1.0 = 0x3C00
        block[209] = 0x3C;
        for i in 192..208 {
            block[i] = 1; // all scales = +1
        }
        // Encode element 0 with value q1: set ql[0] low nibble = 5, qh[0]=0
        // -> q1 = 5 - 32 = -27, y[0] = 1 * sc[0] * -27 = -27.
        block[0] = 0x05;
        block[128] = 0x00;
        let mut out = [0f32; QK_K];
        dequantize_q6k_block(&block, &mut out);
        assert_eq!(out[0], -27.0);
        // element 64 = high nibble of ql[0]=0 -> q3 = -32, scale sc[4]=1
        assert_eq!(out[64], -32.0);
        // element 32 = low nibble ql[32]=0 -> q2 = -32, scale sc[2]=1
        assert_eq!(out[32], -32.0);
        // element 96 = high nibble ql[32]=0 -> q4 = -32, scale sc[6]=1
        assert_eq!(out[96], -32.0);
        // element 16..31 use sc[1]; element 16 = low nibble ql[16]=0 -> -32*1
        assert_eq!(out[16], -32.0);
        // element 48..63 use sc[3]; element 48 = low nibble ql[48]=0 -> -32
        assert_eq!(out[48], -32.0);
        // element 80..95 use sc[5]; element 80 = high nibble ql[16]=0 -> -32
        assert_eq!(out[80], -32.0);
        // element 112..127 use sc[7]; element 112 = high nibble ql[48]=0 -> -32
        assert_eq!(out[112], -32.0);
    }

    #[test]
    fn dequant_scale_and_d() {
        let mut block = [0u8; QK_K_BYTES];
        block[208] = 0x00;
        block[209] = 0x40; // fp16 2.0
        for i in 192..208 {
            block[i] = 1;
        }
        block[0] = 0x0F; // low nibble 15 -> q1 = 15-32 = -17
        block[128] = 0x00;
        let mut out = [0f32; QK_K];
        dequantize_q6k_block(&block, &mut out);
        assert_eq!(out[0], 2.0 * 1.0 * -17.0);
    }

    #[test]
    fn matvec_consistency() {
        // Build a tiny weight of 256x4 (1 block per row), unquantized-ish.
        let in_dim = 256usize;
        let out_dim = 4usize;
        let mut weight = vec![0u8; (in_dim / QK_K) * QK_K_BYTES * out_dim];
        // fill each row's block so that dequant yields a simple pattern
        for o in 0..out_dim {
            let off = o * QK_K_BYTES;
            weight[off + 208] = 0x00;
            weight[off + 209] = 0x3C; // d = 1.0
            for s in 0..16 {
                weight[off + 192 + s] = 1;
            }
            // set element 0's low nibble = (o+1) -> q1 = (o+1)-32
            weight[off] = (o + 1) as u8;
            weight[off + 128] = 0;
        }
        let x: Vec<f32> = (0..in_dim).map(|i| (i % 7) as f32 * 0.5).collect();
        let mut out = vec![0f32; out_dim];
        q6k_matvec(&weight, in_dim, out_dim, &x, &mut out);

        // Independently: full dequant + f32 dot.
        let mut expect = vec![0f32; out_dim];
        let mut wrow = vec![0f32; in_dim];
        for o in 0..out_dim {
            dequantize_q6k_row(&weight[o * QK_K_BYTES..(o + 1) * QK_K_BYTES], &mut wrow);
            let acc: f64 = wrow
                .iter()
                .zip(&x)
                .map(|(w, x)| (*w as f64) * (*x as f64))
                .sum();
            expect[o] = acc as f32;
        }
        assert_eq!(out, expect);
        // and a spot check vs hand: row 0 has only element0 nonzero-additive
        // (all q = -32 except element0 = -31 ... actually many -32*x nonzero)
        // so just trust the independent dot above.
    }
}
