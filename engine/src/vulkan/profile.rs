//! Optional timestamps inside existing submissions; no repartitioning or arithmetic changes.
use super::*;
impl VulkanContext {
    pub(super) fn profile_write(&self, row: serde_json::Value) -> Result<()> {
        if let Some(f) = &self.profile {
            let mut f = f.lock().unwrap();
            writeln!(f, "{}", row)?;
            f.flush()?;
        }
        Ok(())
    }
    pub fn profiling(&self) -> bool { self.profile.is_some() }
    pub fn profile_forward(&self, start_ns: u64, end_ns: u64, position: usize, n: usize) -> Result<()> {
        self.profile_write(serde_json::json!({"event":"forward", "start_ns":start_ns,
            "end_ns":end_ns,"position":position,"n":n,"phase":self.phase.get(),
            "prompt_tokens":self.prompt_tokens.get()}))
    }
    pub(super) unsafe fn profile_begin_command(&self, cb: vk::CommandBuffer, index: u32) {
        if self.profile_dispatch {
            assert!(index + 1 < 8192, "profile query capacity");
            self.device.cmd_write_timestamp(cb, vk::PipelineStageFlags::TOP_OF_PIPE, self.query_pool, index);
        }
    }
    pub(super) unsafe fn profile_end_command(&self, cb: vk::CommandBuffer, index: u32) {
        if self.profile_dispatch {
            self.device.cmd_write_timestamp(cb, vk::PipelineStageFlags::BOTTOM_OF_PIPE, self.query_pool, index+1);
        }
    }
    pub(super) fn profile_submission(&self, sequence: u64, times: [u64; 8], gpu_ms: f64) -> Result<()> {
        if self.profile.is_none() { return Ok(()); }
        let commands = self.pending_work.borrow();
        let mut durations = Vec::new();
        if self.profile_dispatch && !commands.is_empty() {
            let mut q = vec![0u64; commands.len()*2];
            unsafe { self.device.get_query_pool_results(self.query_pool, 2, &mut q,
                vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT)?; }
            let mask = u64::MAX >> (64-self.timestamp_bits);
            for (i, c) in commands.iter().enumerate() {
                let ns = (q[2*i+1].wrapping_sub(q[2*i]) & mask) as f64 * self.props.limits.timestamp_period as f64;
                durations.push(serde_json::json!({"work":c,"ns":ns,"ticks":[q[2*i],q[2*i+1]]}));
            }
        }
        self.profile_write(serde_json::json!({"event":"submission_profile","sequence":sequence,
            "phase":self.phase.get(),"prompt_tokens":self.prompt_tokens.get(),"work_context":self.work_context.get(),
            "gpu_frame_ns":gpu_ms*1e6,"predicted_ms":self.predicted_ms.get(),"command_count":commands.len(),
            "batch_begin_ns":times[0],"intent_begin_ns":times[1],"intent_end_ns":times[2],
            "monitor_end_ns":times[3],"queue_submit_end_ns":times[4],"fence_end_ns":times[5],
            "frame_query_end_ns":times[6],"trace_end_ns":times[7],
            "exclusive_record_ns":times[1]-times[0],"exclusive_intent_json_ns":times[2]-times[1],
            "exclusive_monitor_ns":times[3]-times[2],"queue_submit_ns":times[4]-times[3],
            "fence_wait_ns":times[5]-times[4],"exclusive_frame_query_ns":times[6]-times[5],
            "exclusive_completion_json_ns":times[7]-times[6],"dispatches":durations}))?;
        Ok(())
    }
}
/// Public monotonic clock for exact forward boundaries in the request harness.
pub fn profile_clock_ns() -> u64 { monotonic_ns() }
