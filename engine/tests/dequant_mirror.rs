//! Verify the Q6_K byte layout used by `kernels::dequant` is bit-exact
//! with the pure-Rust reference `q6k::dequantize_q6k_block`, by mirroring the
//! shader's byte-granular extraction + arithmetic in Rust and running it over
//! many random blocks (including the ubiquitous negative `-128` group scale).
//! This is the offline gate before spending a real W9100 dispatch.

use lfm2_engine::q6k::{dequantize_q6k_block, QK_K, QK_K_BYTES};

/// Scalar mirror of the GPU byte extraction / dequantization layout.
fn shader_dequant(block: &[u8], k: usize) -> f32 {
    fn gb(block: &[u8], byteidx: usize) -> u8 {
        block[byteidx]
    }

    let base: usize = 0;
    let k = k as u32;
    let half = k >> 7;
    let jj = k & 127;
    let sub = jj >> 5;
    let l = jj & 31;
    let is = l >> 4;

    let ql_idx = base + (half * 64) as usize + (l + (sub & 1) * 32) as usize;
    let qh_idx = base + 128 + (half * 32) as usize + l as usize;
    let sc_idx = base + 192 + (half * 8) as usize + (is + 2 * sub) as usize;
    let dh_idx = base + 208;

    let qlbyte = gb(block, ql_idx);
    let nibble = if (sub >> 1) != 0 {
        qlbyte >> 4
    } else {
        qlbyte & 0x0F
    };
    let qhbyte = gb(block, qh_idx);
    let field = (qhbyte >> (2 * sub)) & 3;
    let sixbit = nibble | (field << 4);
    let q = sixbit as i32 - 32;

    let scbyte = gb(block, sc_idx) as i32;
    let sc = if scbyte >= 128 { scbyte - 256 } else { scbyte };

    let dh = (gb(block, dh_idx) as u32) | ((gb(block, dh_idx + 1) as u32) << 8);
    let d = half::f16::from_bits(dh as u16).to_f32();

    d * sc as f32 * q as f32
}

#[test]
fn shader_decode_matches_reference_bit_exact() {
    // Deterministic PRNG.
    let mut s: u64 = 0x9E3779B97F4A7C15;
    let mut rng = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as u8
    };

    let nblocks = 512;
    let mut mismatches = 0;
    for _ in 0..nblocks {
        let mut block = vec![0u8; QK_K_BYTES];
        for b in block.iter_mut() {
            *b = rng();
        }
        // ensure d is a sane fp16 (avoid NaN/inf); overwrite d with 1.0-ish range.
        block[208] = 0x00;
        block[209] = 0x3B; // ~0.54
                           // Force some negative scales explicitly (rng already produces them
                           // often, but guarantee coverage of the -128 case).
        block[192] = 0x80; // -128
        block[193] = 0xFF; // -1
        block[200] = 0x9C; // -100

        let mut ref_v = vec![0f32; QK_K];
        dequantize_q6k_block(&block, &mut ref_v);
        for k in 0..QK_K {
            let m = shader_dequant(&block, k);
            if m.to_bits() != ref_v[k].to_bits() {
                mismatches += 1;
                if mismatches <= 5 {
                    eprintln!("k={k}: shader={m} ref={} (block[{k}])", ref_v[k]);
                }
            }
        }
    }
    assert_eq!(
        mismatches, 0,
        "shader decode diverged from reference on {mismatches} values"
    );
}
