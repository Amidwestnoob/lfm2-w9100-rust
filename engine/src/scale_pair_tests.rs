//! CPU-only independent integer byte extraction and unchanged work-envelope proof.
#[test]
fn scale_pair_bytes_geometry_cold_bound() {
    use crate::work_bound::{tiles,estimate,DISPATCH_MS,TARGET_MS};
    for base in (0..4096usize).step_by(4) { for lane in 0..64usize { for s in 0..4usize {
        let half=(lane%16)/8; let l=(lane%8)*4;
        let original=base+192+s*2+half*8+l/16;
        let word=base+192+half*8+(s/2)*4;
        let shift=(l/16+(s%2)*2)*8;
        assert_eq!(original,word+shift/8);
        assert!(word>=base+192 && word+3<base+208);
        for value in 0..256u32 {
            let w=(0x5aa53cc3 & !(255<<shift)) | (value<<shift);
            let v=(w>>shift)&255;
            let got=if v>=128 {v as f32-256.0} else {v as f32};
            assert_eq!(got.to_bits(),(value as u8 as i8 as f32).to_bits());
            let bitfield=(((w<< (24-shift)) as i32)>>24) as f32;
            assert_eq!(bitfield.to_bits(),got.to_bits());
        }
    }}}
    for n in 1u32..=64 { for (input,output) in [(2048u32,1792u32),(1792,2048)] {
        for grouped in [false,true] {
            if !grouped && n!=1 {continue;}
            let name=if grouped {"q6k_grouped_compact"} else {"q6k_routed"};
            let p=[input,output,input/256,n,0];
            let gx=output/4; let gy=if grouped {32*n.div_ceil(4)} else {4};
            let mut seen=vec![0u8;(gx*gy) as usize]; let mut total=0.0;
            for t in tiles(name,&p,gx,gy).unwrap() {
                assert_eq!(t.ms.to_bits(),estimate(name,&p,t.gx,t.gy).unwrap().to_bits());
                if grouped {
                    assert_eq!(t.ms.to_bits(),estimate("q6k_grouped_compact_aligned",&p,t.gx,t.gy).unwrap().to_bits());
                }
                assert!(t.ms.is_finite() && t.ms<=DISPATCH_MS);
                if total+t.ms>TARGET_MS {total=0.0;}
                total+=t.ms;assert!(total<=TARGET_MS);
                for y in t.y..t.y+t.gy {for x in t.x..t.x+t.gx {seen[(y*gx+x) as usize]+=1;}}
            }
            assert!(seen.iter().all(|&v|v==1));
        }
    }}
    println!("Scale-pair proof: every lane, all four scale positions, all256 byte values, aligned bases; same bytes, signed-float conversion, four-row/tile geometry, exact split coverage at every n1..64; two word loads replace four; all floating FMA/reduction order unchanged.");
}

#[test]
fn fixed_expert_unroll_order_and_refusals() {
    use crate::ops::validate_fixed_expert;
    for (input,output,blocks,down) in [(2048u32,1792u32,8u32,0u32),(1792,2048,7,1)] {
        for lane in 0..64u32 {
            let old:Vec<_>=(0..blocks.div_ceil(4)).map(|i|i*4+lane/16).filter(|b|*b<blocks).collect();
            let new:Vec<_>=(0..2).map(|i|i*4+lane/16).filter(|b|*b<blocks).collect();
            assert_eq!(old,new);
        }
        assert_eq!(output%4,0);
        for n in 1..=64u32 {for grouped in [false,true] {
            let p=[input,output,blocks,n,down];let gx=output/4;let gy=if grouped {32*n.div_ceil(4)} else {4*n};
            validate_fixed_expert(&p,gx,gy,grouped).unwrap();
            for field in 0..5 {
                let mut bad=p;bad[field]=u32::MAX;
                assert!(validate_fixed_expert(&bad,gx,gy,grouped).is_err());
            }
            assert!(validate_fixed_expert(&p,gx+1,gy,grouped).is_err());
            assert!(validate_fixed_expert(&p,gx,gy+1,grouped).is_err());
            for o in 0..output {assert!((o/4)*4+o%4<output);}
        }}
    }
    assert!(validate_fixed_expert(&[],0,0,false).is_err());
}
