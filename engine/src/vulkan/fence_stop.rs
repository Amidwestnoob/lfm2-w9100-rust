//! Common immediate fence-completion timing latch, independent of GPU APIs.
//! A rejected measurement permanently refuses every subsequent submission.
use std::cell::Cell;
#[derive(Default)]
pub(super) struct FenceStop { stopped: Cell<bool> }
impl FenceStop {
    pub(super) fn before_submission(&self) -> Result<(), &'static str> {
        if self.stopped.get() { Err("immediate fence timing stop is latched") } else { Ok(()) }
    }
    pub(super) fn completed(&self, gpu_ms: f64, host_elapsed_ns: u64) -> Result<(), &'static str> {
        self.before_submission()?;
        if !gpu_ms.is_finite() || gpu_ms <= 0.0 || gpu_ms >= 40.0 || host_elapsed_ns >= 1_000_000_000 {
            self.stopped.set(true);
            Err("immediate fence timing stop: nonfinite/nonpositive or actual >=40ms or host >=1s")
        } else { Ok(()) }
    }
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn fence_stop_oversized_nonfinite_boundary_refuse_next_submission() {
        for timing in [40.0, 40.000001, 59.755407, 250.0, f64::INFINITY, f64::NEG_INFINITY, f64::NAN, 0.0, -1.0] {
            let guard=FenceStop::default();
            let mut submissions=0;
            assert!(guard.before_submission().is_ok()); submissions+=1;
            assert!(guard.completed(timing, 1).is_err());
            for _ in 0..3 {
                if guard.before_submission().is_ok() {submissions+=1;}
                assert!(guard.before_submission().is_err());
                assert!(guard.completed(1.0, 1).is_err());
            }
            assert_eq!(submissions,1);
        }
    }
    #[test]
    fn fence_stop_strict_below_boundary_passes() {
        for timing in [f64::MIN_POSITIVE, 0.001, 39.99999999999999] {
            let guard=FenceStop::default();
            for _ in 0..5 {assert!(guard.before_submission().is_ok());assert!(guard.completed(timing,999_999_999).is_ok());}
        }
    }
    #[test]
    fn fence_stop_host_boundary_latches() {
        for elapsed in [1_000_000_000, u64::MAX] {
            let guard=FenceStop::default();assert!(guard.completed(1.0,elapsed).is_err());assert!(guard.before_submission().is_err());
        }
    }
}
