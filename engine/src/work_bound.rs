//! Preventive work bounds. Estimates are conservative screening ceilings, not
//! timing guarantees. Every dispatch is independently split before recording.
use anyhow::{ensure, Result};
pub const TARGET_MS: f64 = 80.0;
pub const DISPATCH_MS: f64 = 40.0;
pub const PREFILL_CHUNK: usize = 64;
pub fn decode_end(position: u32, capacity: u32) -> u32 {
    ((position / 256 + 1) * 256).min(capacity)
}
#[derive(Clone, Debug)]
pub struct Tile {
    pub x: u32,
    pub y: u32,
    pub gx: u32,
    pub gy: u32,
    pub ms: f64,
}
pub fn estimate(name: &str, p: &[u32], gx: u32, gy: u32) -> Result<f64> {
    let name = name.strip_suffix("_aligned").unwrap_or(name);
    let name = name.strip_suffix("_compact").unwrap_or(name);
    // Query reuse retains the exact original attention geometry/work. This is
    // an alias, not a reduced screening coefficient or increased ceiling.
    let name = if name == "attention_queryhoist" {"attention_unrolled"} else {name};
    ensure!(gx > 0 && gy > 0, "empty dispatch");
    let g = gx as f64 * gy as f64;
    let ms = if name.starts_with("kv_write_transposed") {
        ensure!(p.len()>=4 && p[0]<=2048 && p[1]<=64 && p[3]<=4096,"transposed KV bounds");
        g*0.008
    } else if name.starts_with("attn_") {
        ensure!(
            p.len() >= 7 && p[0] == 64 && p[5] <= 4096 && p[3] > 0,
            "fast attention bounds"
        );
        let end = if name.ends_with("decode") {
            decode_end(p[4], p[5])
        } else {
            p[4] + p[3]
        };
        ensure!(end <= p[5], "fast attention context bound");
        if name.starts_with("attn_value_split") {
            ensure!(p[0]==64 && p[1]==32 && p[2]==8,"value split geometry");
            // <=16,384 terms/group plus reduction/stores, 64-term/lane loops.
            g*0.08
        } else if name.starts_with("attn_value_merge") {
            g*0.005
        } else if name.starts_with("attn_score_coalesced") {
            ensure!(p[1]==32 && p[2]==8,"four-head coalesced geometry");
            g*0.08
        } else if name.starts_with("attn_transpose") {
            // 256 elements/group, <=2048 bytes traffic/group. 0.008 ms/group
            // is 256 MB/s effective traffic plus guard/address instruction margin.
            g * 0.008
        } else if name.starts_with("attn_score_transposed") {
            // 64 lanes x 64 ordered terms. 50k terms/ms conservative cold ceiling;
            // five times the existing three-stage score per-group coefficient.
            g * 4096.0 / 50_000.0
        } else if name.starts_with("attn_score") {
            g * 4096.0 / 250_000.0
        } else if name.starts_with("attn_norm") {
            g * end as f64 / 100_000.0
        } else {
            g * 64.0 * end as f64 / 250_000.0
        }
    } else if name.starts_with("attention") {
        ensure!(
            p.len() >= 7 && p[0] <= 256 && p[1] <= 64 && p[3] > 0 && p[5] <= 4096,
            "attention geometry bound"
        );
        let end = if name.ends_with("decode") {
            decode_end(p[4], p[5])
        } else {
            p[4].checked_add(p[3]).unwrap_or(u32::MAX)
        };
        ensure!(end <= p[5], "attention envelope bound");
        g * p[0] as f64 * end as f64 / 250_000.0
    } else if name.starts_with("q6k") {
        ensure!(
            p.len() >= 5 && p[0] <= 8192 && p[1] <= 128000,
            "Q6 geometry bound"
        );
        let expert = name == "q6k_grouped" || name == "q6k_routed" || name == "q6k_matmul_4expert";
        let blocks = p[if expert { 2 } else { 3 }] as f64;
        let tile = match name {
            "q6k_prompt" => 4.0,
            "q6k_prompt2" => 2.0,
            _ => 1.0,
        };
        let active = if name == "q6k_grouped" {
            (gy as f64 * 4.0).min(p[3] as f64 * 4.0)
        } else {
            gy as f64 * tile
        };
        let coefficient = match name {
            // Cold-clock direct screens of the ungrouped reference needed a
            // larger ceiling than grouped/promoted model work.
            "q6k_routed" | "q6k_matmul_4expert" => 0.00005,
            "q6k_prompt2" => 0.000025,
            _ => 0.0000125,
        };
        gx as f64 * active * blocks * coefficient + g * 0.000001
    } else if name == "f32_matmul" {
        ensure!(p[0] <= 8192, "f32 input bound");
        g * p[0] as f64 / 500_000.0
    } else if name == "shortconv" {
        g * p[1] as f64 * 0.02
    } else if name == "stream_copy" {
        g * 256.0 * 8.0 / 5_000_000.0
    } else {
        ensure!(g <= 16_777_216.0, "helper grid bound");
        g * 0.005
    };
    Ok(0.1 + ms)
}
pub fn tiles(name: &str, p: &[u32], gx: u32, gy: u32) -> Result<Vec<Tile>> {
    fn split(
        out: &mut Vec<Tile>,
        name: &str,
        p: &[u32],
        x: u32,
        y: u32,
        gx: u32,
        gy: u32,
    ) -> Result<()> {
        let ms = estimate(name, p, gx, gy)?;
        if ms <= DISPATCH_MS {
            out.push(Tile { x, y, gx, gy, ms });
            return Ok(());
        }
        ensure!(
            gx > 1 || gy > 1,
            "indivisible dispatch predicted over ceiling"
        );
        if gx >= gy && gx > 1 {
            let a = gx / 2;
            split(out, name, p, x, y, a, gy)?;
            split(out, name, p, x + a, y, gx - a, gy)?
        } else {
            let a = gy / 2;
            split(out, name, p, x, y, gx, a)?;
            split(out, name, p, x, y + a, gx, gy - a)?
        }
        Ok(())
    }
    let mut out = vec![];
    split(&mut out, name, p, 0, 0, gx, gy)?;
    Ok(out)
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn compact_group_exact_surviving_work_and_bounds() {
        for n in 1..=64u32 {
            let compact=n.div_ceil(4);
            for count in 0..=n {
                let old:Vec<_>=(0..n).filter(|i|4*i<count).collect();
                let new:Vec<_>=(0..compact).filter(|i|4*i<count).collect();
                assert_eq!(old,new);
            }
            for output in [1792,2048] {
                let p=[2048,output,8,n,0];
                let gx=output/4;let gy=32*compact;
                assert_eq!(estimate("q6k_grouped_compact",&p,gx,gy).unwrap().to_bits(),estimate("q6k_grouped",&p,gx,gy).unwrap().to_bits());
                let ts=tiles("q6k_grouped_compact",&p,gx,gy).unwrap();
                let mut seen=vec![0u8;(gx*gy) as usize];
                for t in ts {assert!(t.ms<=40.0);for y in t.y..t.y+t.gy {for x in t.x..t.x+t.gx {seen[(y*gx+x) as usize]+=1;}}}
                assert!(seen.iter().all(|v|*v==1));
            }
        }
    }
    #[test]
    fn three_stage_full_envelope_coverage() {
        for end in 1..=4096u32 {
            for n in [1, 17, 64] {
                if n > end {continue;}
                let p=[64,32,8,n,end-n,4096,0];
                for name in ["attn_score", "attn_score_transposed", "attn_score_coalesced", "attn_value_split", "attn_value_merge", "attn_norm", "attn_value", "attn_transpose"] {
                    let gx=if name=="attn_value_split" {32*n*end.div_ceil(256)} else if name=="attn_transpose" {(64*8*end).div_ceil(256)} else {if name=="attn_score_coalesced" {8*n*end.div_ceil(64)} else {32*n*if name.starts_with("attn_score") {4096u32.div_ceil(64)} else {1}}};
                    let ts=tiles(name,&p,gx,1).unwrap();
                    let mut next=0;
                    let mut total=0.0;
                    for t in ts {
                        assert_eq!((t.x,t.y,t.gy),(next,0,1));
                        assert!(t.ms.is_finite() && t.ms<=DISPATCH_MS);
                        next+=t.gx;
                        if total+t.ms>TARGET_MS {total=0.0;}
                        total+=t.ms;assert!(total<=TARGET_MS);
                    }
                    assert_eq!(next,gx);
                }
            }
            let p=[64,32,8,1,end-1,4096,0];
            for (name,gx) in [("attn_score_decode",2048),("attn_score_transposed_decode",2048),("attn_score_coalesced_decode",8*decode_end(end-1,4096).div_ceil(64)),("attn_value_split_decode",32*decode_end(end-1,4096).div_ceil(256)),("attn_value_merge_decode",32),("attn_norm_decode",32),("attn_value_decode",32),("attn_transpose_decode",8192)] {
                let ts=tiles(name,&p,gx,1).unwrap();
                assert_eq!(ts.iter().map(|t|t.gx).sum::<u32>(),gx);
                assert!(ts.iter().all(|t|t.ms<=DISPATCH_MS));
            }
        }
    }
    #[test]
    fn queryhoist_identical_conservative_bounds() {
        for end in 1..=4096u32 {
            for n in [1, 17, 64] {
                if n > end {continue;}
                let p = [64, 32, 8, n, end-n, 4096, 0];
                let a=tiles("attention_unrolled", &p, 32*n, 1).unwrap();
                let b=tiles("attention_queryhoist", &p, 32*n, 1).unwrap();
                assert_eq!(a.len(), b.len());
                for (a,b) in a.iter().zip(&b) {
                    assert_eq!((a.x,a.y,a.gx,a.gy,a.ms.to_bits()), (b.x,b.y,b.gx,b.gy,b.ms.to_bits()));
                }
            }
        }
    }
    #[test]
    fn full_attention_envelope() {
        for end in 1..=4096u32 {
            for n in [1, 17, 64, 128, 256] {
                if n > end {
                    continue;
                }
                let p = [64, 32, 8, n, end - n, 4096, 0];
                let ts = tiles("attention", &p, 32 * n, 1).unwrap();
                let mut seen = vec![0; 32 * n as usize];
                for t in ts {
                    assert!(t.ms <= DISPATCH_MS);
                    for i in t.x..t.x + t.gx {
                        seen[i as usize] += 1;
                    }
                }
                assert!(seen.iter().all(|&x| x == 1));
            }
        }
    }
    #[test]
    fn q6_coverage_and_submission_partition() {
        for name in ["q6k_prompt", "q6k_grouped", "q6k_routed"] {
            for n in [1, 17, 64, 128, 256] {
                let p = [2048, 1792, 8, n, 0];
                let (gx, gy) = (448, n * 32);
                let ts = tiles(name, &p, gx, gy).unwrap();
                let mut seen = vec![0; gx as usize * gy as usize];
                let mut total = 0.;
                for t in ts {
                    if total + t.ms > TARGET_MS {
                        total = 0.;
                    }
                    total += t.ms;
                    assert!(total <= TARGET_MS);
                    for y in t.y..t.y + t.gy {
                        for x in t.x..t.x + t.gx {
                            seen[(y * gx + x) as usize] += 1;
                        }
                    }
                }
                assert!(seen.iter().all(|&x| x == 1));
            }
        }
    }
    #[test]
    fn decode_context_tiers() {
        for position in 0..4096u32 {
            let bound = decode_end(position, 4096);
            assert!(bound > position && bound <= 4096);
            for p in position..bound {
                assert!(p < bound);
            }
            let push = [64, 32, 8, 1, position, 4096, 0];
            assert!(estimate("attention_decode", &push, 32, 1).unwrap() <= DISPATCH_MS);
        }
    }
    #[test]
    fn reject_out_of_envelope() {
        assert!(tiles("attention", &[64, 32, 8, 64, 4090, 4096, 0], 32, 1).is_err());
        assert!(tiles("attention_decode", &[64, 32, 8, 1, 0, 8192, 0], 32, 1).is_err());
    }
}
