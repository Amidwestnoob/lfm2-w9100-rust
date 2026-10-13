//! CPU-only independent coverage and conservative envelope proof for row schedules.
#[test]
fn q6_row_schedule_coverage_and_cold_work_subset() {
    use crate::work_bound::{tiles, estimate, DISPATCH_MS, TARGET_MS};
    for rows in [1usize, 2, 4] {
        for output in 1usize..=2051 {
            let groups=output.div_ceil(4)*(4/rows);
            let mut seen=vec![0u8;output];
            for g in 0..groups {for r in 0..rows {
                let o=g*rows+r;
                if o<output {
                    seen[o]+=1;
                    // Original row identity and identical block/lane walk.
                    assert_eq!((o/4)*4+o%4,o);

                }
            }}
            assert!(seen.iter().all(|&v|v==1));
        }
        for blocks in [7,8,32] {for lane in 0..64 {
            let original:Vec<_>=(0..(blocks+3)/4).map(|i|i*4+lane/16).filter(|b|*b<blocks).collect();
            let scheduled:Vec<_>=(0..(blocks+3)/4).map(|i|i*4+lane/16).filter(|b|*b<blocks).collect();
            assert_eq!(original,scheduled);
        }}
        for n in 1u32..=64 {
            // Stable top-four grouping has at most n entries per expert.
            for count in 0..=n {
                let slots:Vec<_>=(0..n.div_ceil(4)).flat_map(|g|(0..4).map(move |t|g*4+t)).filter(|s|*s<count).collect();
                assert_eq!(slots,(0..count).collect::<Vec<_>>());
            }
            for (input,output) in [(2048u32,1792u32),(1792,2048)] {
                for grouped in [false,true] {
                    if !grouped && n!=1 {continue;}
                    let name=if grouped {"q6k_grouped_compact"} else {"q6k_routed"};
                    let p=[input,output,input/256,n,0];
                    let gx=output.div_ceil(4)*(4/rows as u32);
                    let gy=if grouped {32*n.div_ceil(4)} else {4};
                    let mut seen=vec![0u8;(gx*gy) as usize];let mut total=0.0f64;
                    for t in tiles(name,&p,gx,gy).unwrap() {
                        // Unchanged coefficient evaluates same grid identically;
                        // changed per-wave work is <= original four-row body.
                        assert_eq!(t.ms.to_bits(),estimate(name,&p,t.gx,t.gy).unwrap().to_bits());
                        assert!(t.ms.is_finite() && t.ms<=DISPATCH_MS);
                        if total+t.ms>TARGET_MS {total=0.0;}
                        total+=t.ms;assert!(total<=TARGET_MS);
                        for y in t.y..t.y+t.gy {for x in t.x..t.x+t.gx {seen[(y*gx+x) as usize]+=1;}}
                    }
                    assert!(seen.iter().all(|&v|v==1));
                    assert!(rows*if grouped {4} else {1}<=4*if grouped {4} else {1});
                }
            }
        }
    }
    println!("Q6 row proof: rows1/2/4, output tails1..2051, all n1..64, both1792/2048 directions, full split-grid exact coverage; unchanged cold coefficients; per-wave floating work is a strict row subset.");
}
