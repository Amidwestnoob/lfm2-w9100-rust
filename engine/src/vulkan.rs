//! ash/Vulkan device + buffer + compute-pipeline ownership.
//!
//! Opens a Vulkan device (explicitly selecting the AMD FirePro W9100),
//! allocates storage buffers and dispatches Rust-authored SPIR-V via ash.
//! Typed Rust instruction builders in `kernels.rs` author the GPU computation;
//! no shader-text frontend or opaque precompiled compute modules are used.

use anyhow::{anyhow, Result};
use ash::vk;
use std::cell::{Cell, RefCell};
use std::ffi::CStr;
use std::io::Write;
use std::sync::Arc;
use std::sync::Mutex;

// Linux CLOCK_MONOTONIC: paired markers in the same clock domain as the supervisor.
fn monotonic_ns() -> u64 {
    #[repr(C)]
    struct Timespec {
        sec: i64,
        nsec: i64,
    }
    unsafe extern "C" {
        fn clock_gettime(clock: i32, ts: *mut Timespec) -> i32;
    }
    let mut ts = Timespec { sec: 0, nsec: 0 };
    assert_eq!(unsafe { clock_gettime(1, &mut ts) }, 0);
    ts.sec as u64 * 1_000_000_000 + ts.nsec as u64
}

pub const W9100_DEVICE_ID: u32 = 0x67a0;
mod reusable;
mod profile;
mod fence_stop;
mod intent_cache;
mod hazards;
pub use profile::profile_clock_ns;

/// A device-local (or host-visible) buffer.
pub struct GpuBuffer {
    pub generation: u64,
    pub device: Arc<ash::Device>,
    pub buffer: vk::Buffer,
    pub memory: vk::DeviceMemory,
    pub size: vk::DeviceSize,
    /// true if the memory is host-visible (mappable).
    pub host_visible: bool,
    pub mapped: Option<std::ptr::NonNull<u8>>,
}

unsafe impl Send for GpuBuffer {}
unsafe impl Sync for GpuBuffer {}

impl GpuBuffer {
    /// Write bytes into a host-visible buffer at `offset`.
    pub fn write(&self, offset: usize, data: &[u8]) {
        assert!(self.host_visible, "buffer not host-visible");
        let ptr = self.mapped.expect("buffer not mapped");
        unsafe {
            let dst = ptr.as_ptr().add(offset);
            std::ptr::copy_nonoverlapping(data.as_ptr(), dst, data.len());
        }
    }

    /// Read bytes from a host-visible buffer at `offset`.
    pub fn read(&self, offset: usize, out: &mut [u8]) {
        assert!(self.host_visible, "buffer not host-visible");
        let ptr = self.mapped.expect("buffer not mapped");
        unsafe {
            let src = ptr.as_ptr().add(offset);
            std::ptr::copy_nonoverlapping(src, out.as_mut_ptr(), out.len());
        }
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        unsafe {
            if let Some(p) = self.mapped {
                self.device.unmap_memory(self.memory);
                let _ = p;
            }
            self.device.destroy_buffer(self.buffer, None);
            self.device.free_memory(self.memory, None);
        }
    }
}

/// An owned compute pipeline (pipeline + layout + descriptor-set layout).
#[derive(Clone)]
pub struct ComputePipeline {
    pub pipeline: vk::Pipeline,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
}

#[derive(Clone)]
enum DecodeCommand {
    Compute {
        work: serde_json::Value,
        pipe: ComputePipeline,
        set: vk::DescriptorSet,
        push: Vec<u8>,
        gx: u32,
        gy: u32,
    },
    Copy {
        work: serde_json::Value,
        src: vk::Buffer,
        dst: vk::Buffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    },
}

pub struct VulkanContext {
    pub entry: ash::Entry,
    pub instance: ash::Instance,
    pub device: Arc<ash::Device>,
    pub physical_device: vk::PhysicalDevice,
    pub props: vk::PhysicalDeviceProperties,
    pub queue: vk::Queue,
    pub queue_family: u32,
    pub command_pool: vk::CommandPool,
    pub command_buffer: vk::CommandBuffer,
    pub descriptor_pool: vk::DescriptorPool,
    fence: vk::Fence,
    query_pool: vk::QueryPool,
    timestamp_bits: u32,
    profile: Option<Mutex<std::fs::File>>,
    profile_dispatch: bool,
    profile_next: Cell<u32>,
    profile_batch_start: Cell<u64>,
    pub last_gpu_ms: Cell<f64>,
    pub total_gpu_ms: Cell<f64>,
    pub decode_context_end: Cell<u32>,
    trace: Option<Mutex<std::fs::File>>,
    phase: Cell<&'static str>,
    prompt_tokens: Cell<usize>,
    capture_decode: Cell<bool>,
    decode_commands: RefCell<Vec<DecodeCommand>>,
    decode_secondary: RefCell<Vec<(vk::CommandBuffer, Vec<serde_json::Value>, intent_cache::CommandTemplate)>>,
    cached_intent_commands: RefCell<Option<Vec<u8>>>,
    descriptor_buffers: RefCell<std::collections::HashMap<vk::DescriptorSet,Vec<vk::Buffer>>>,
    hazards: RefCell<hazards::Hazards>,
    pipeline_names: RefCell<std::collections::HashMap<vk::Pipeline, &'static str>>,
    pending_work: RefCell<Vec<serde_json::Value>>,
    predicted_ms: Cell<f64>,
    submission_sequence: Cell<u64>,
    fence_stop: fence_stop::FenceStop,
    work_context: Cell<(usize, usize, Option<usize>)>,
    pub decode_command_buffer_reuses: Cell<u64>,
    decode_position_offsets: RefCell<std::collections::HashMap<vk::Pipeline, usize>>,
}

impl VulkanContext {
    /// Find and open the W9100. Fails loudly if it is not present.
    pub fn open_w9100() -> Result<Self> {
        let entry = unsafe { ash::Entry::load() }.map_err(|e| anyhow!("load vulkan: {e}"))?;

        let app_name = CStr::from_bytes_with_nul(b"lfm2-engine\0").unwrap();
        let engine_name = CStr::from_bytes_with_nul(b"lfm2\0").unwrap();
        let app_info = vk::ApplicationInfo::default()
            .application_name(app_name)
            .application_version(1)
            .engine_name(engine_name)
            .api_version(vk::API_VERSION_1_3);

        let instance_create_info = vk::InstanceCreateInfo::default().application_info(&app_info);
        let instance = unsafe { entry.create_instance(&instance_create_info, None) }
            .map_err(|e| anyhow!("create instance: {e}"))?;

        let pdevices = unsafe { instance.enumerate_physical_devices() }
            .map_err(|e| anyhow!("enumerate physical devices: {e}"))?;

        let mut target = None;
        let mut all = Vec::new();
        for pd in &pdevices {
            let props = unsafe { instance.get_physical_device_properties(*pd) };
            let name = cstr_props(&props.device_name);
            all.push(format!("{} (id {:04x})", name, props.device_id));
            if props.device_id == W9100_DEVICE_ID {
                target = Some((*pd, props));
            }
        }

        let (physical_device, props) = match target {
            Some(t) => t,
            None => {
                // Fall back to the first device but record that it is not the W9100.
                let (pd, props0) = pdevices
                    .first()
                    .map(|pd| {
                        let p = unsafe { instance.get_physical_device_properties(*pd) };
                        (*pd, p)
                    })
                    .ok_or_else(|| anyhow!("no vulkan physical devices found ({all:?})"))?;
                (pd, props0)
            }
        };

        let (queue_family, queue, device) = {
            let families =
                unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
            let compute_family = families
                .iter()
                .enumerate()
                .find(|(_, f)| f.queue_flags.contains(vk::QueueFlags::COMPUTE))
                .map(|(i, _)| i as u32)
                .ok_or_else(|| anyhow!("no compute queue family"))?;
            let priorities = [1.0f32];
            let qci = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(compute_family)
                .queue_priorities(&priorities);
            let features = vk::PhysicalDeviceFeatures::default();
            let dci = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&qci))
                .enabled_features(&features);
            let device = unsafe { instance.create_device(physical_device, &dci, None) }
                .map_err(|e| anyhow!("create device: {e}"))?;
            let queue = unsafe { device.get_device_queue(compute_family, 0) };
            (compute_family, queue, device)
        };

        let pool_ci = vk::CommandPoolCreateInfo::default()
            .queue_family_index(queue_family)
            .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER);
        let command_pool = unsafe { device.create_command_pool(&pool_ci, None) }
            .map_err(|e| anyhow!("create command pool: {e}"))?;

        let cb_ci = vk::CommandBufferAllocateInfo::default()
            .command_pool(command_pool)
            .level(vk::CommandBufferLevel::PRIMARY)
            .command_buffer_count(1);
        let command_buffer = unsafe { device.allocate_command_buffers(&cb_ci) }
            .map_err(|e| anyhow!("alloc command buffer: {e}"))?[0];

        // One descriptor pool for all compute passes (recycled sets).
        let ds_pool_size = vk::DescriptorPoolSize::default()
            .ty(vk::DescriptorType::STORAGE_BUFFER)
            .descriptor_count(4096);
        let pool_ci = vk::DescriptorPoolCreateInfo::default()
            .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET)
            .max_sets(2048)
            .pool_sizes(std::slice::from_ref(&ds_pool_size));
        let descriptor_pool = unsafe { device.create_descriptor_pool(&pool_ci, None) }
            .map_err(|e| anyhow!("create descriptor pool: {e}"))?;

        let fence_ci = vk::FenceCreateInfo::default();
        let fence = unsafe { device.create_fence(&fence_ci, None) }
            .map_err(|e| anyhow!("create fence: {e}"))?;

        let timestamp_bits =
            unsafe { instance.get_physical_device_queue_family_properties(physical_device) }
                [queue_family as usize]
                .timestamp_valid_bits;
        anyhow::ensure!(
            timestamp_bits > 0 && props.limits.timestamp_period > 0.0,
            "timestamp queries unavailable"
        );
        let query_pool = unsafe {
            device.create_query_pool(
                &vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(if std::env::var("LFM_PROFILE_DISPATCH").as_deref()==Ok("1") {8192} else {2}),
                None,
            )
        }?;
        let trace = std::env::var("LFM_SUBMISSION_TRACE")
            .ok()
            .map(|p| {
                std::fs::OpenOptions::new()
                    .create(true)
                    .append(true)
                    .open(p)
            })
            .transpose()?
            .map(Mutex::new);
        let profile = std::env::var("LFM_PROFILE").ok().map(|p| std::fs::OpenOptions::new().create(true).append(true).open(p)).transpose()?.map(Mutex::new);
        if let Some(f) = &profile {
            writeln!(f.lock().unwrap(), "{}", serde_json::json!({"event":"device", "device_id":props.device_id,
                "device_name":cstr_props(&props.device_name),"timestamp_valid_bits":timestamp_bits,
                "timestamp_period_ns":props.limits.timestamp_period,"queue_family":queue_family,
                "driver_version":props.driver_version,"api_version":props.api_version}))?;
        }
        Ok(VulkanContext {
            entry,
            instance,
            device: Arc::new(device),
            physical_device,
            props,
            queue,
            queue_family,
            command_pool,
            command_buffer,
            descriptor_pool,
            fence,
            query_pool,
            timestamp_bits,
            profile,
            profile_dispatch: std::env::var("LFM_PROFILE_DISPATCH").as_deref()==Ok("1"),
            profile_next: Cell::new(2),
            profile_batch_start: Cell::new(0),
            last_gpu_ms: Cell::new(0.0),
            total_gpu_ms: Cell::new(0.0),
            decode_context_end: Cell::new(0),
            trace,
            phase: Cell::new("setup"),
            prompt_tokens: Cell::new(0),
            capture_decode: Cell::new(false),
            decode_commands: RefCell::new(Vec::new()),
            decode_secondary: RefCell::new(Vec::new()),
            cached_intent_commands: RefCell::new(None),
            descriptor_buffers: RefCell::new(std::collections::HashMap::new()),
            hazards: RefCell::new(hazards::Hazards::default()),
            pipeline_names: RefCell::new(std::collections::HashMap::new()),
            pending_work: RefCell::new(Vec::new()),
            predicted_ms: Cell::new(0.0),
            submission_sequence: Cell::new(0),
            fence_stop: fence_stop::FenceStop::default(),
            work_context: Cell::new((0, 0, None)),
            decode_command_buffer_reuses: Cell::new(0),
            decode_position_offsets: RefCell::new(std::collections::HashMap::new()),
        })
    }

    pub fn device_name(&self) -> String {
        cstr_props(&self.props.device_name)
    }
    pub fn register_pipeline(&self, pipeline: vk::Pipeline, name: &'static str) {
        self.pipeline_names.borrow_mut().insert(pipeline, name);
    }
    pub fn pipeline_name(&self, pipeline: vk::Pipeline) -> &'static str {
        self.pipeline_names.borrow()[&pipeline]
    }
    pub fn set_work_context(&self, start: usize, n: usize, layer: Option<usize>) {
        self.work_context.set((start, n, layer));
    }
    fn prepare_work(&self, work: serde_json::Value) -> Result<()> {
        let ms = work["predicted_ms"].as_f64().unwrap();
        anyhow::ensure!(
            ms.is_finite() && ms > 0.0 && ms <= crate::work_bound::DISPATCH_MS,
            "command work ceiling"
        );
        if self.predicted_ms.get() + ms > crate::work_bound::TARGET_MS {
            self.finish_batch()?;
            self.begin_batch()?;
        }
        self.predicted_ms.set(self.predicted_ms.get() + ms);
        self.pending_work.borrow_mut().push(work);
        Ok(())
    }

    pub fn is_w9100(&self) -> bool {
        self.props.device_id == W9100_DEVICE_ID
    }

    pub fn set_submission_context(&self, phase: &'static str, prompt_tokens: usize) {
        self.phase.set(phase);
        self.prompt_tokens.set(prompt_tokens);
    }

    pub fn clear_decode_plan(&self) {
        self.capture_decode.set(false);
        self.decode_commands.borrow_mut().clear();
        for (cb, _, _) in self.decode_secondary.borrow_mut().drain(..) {
            unsafe {
                self.device.free_command_buffers(self.command_pool, &[cb]);
            }
        }
    }
    pub fn has_decode_plan(&self) -> bool {
        !self.decode_commands.borrow().is_empty()
    }
    pub fn capture_decode_plan(&self, active: bool) {
        self.capture_decode.set(active);
    }
    /// Replay identical dispatch/copy order and barriers. Only the explicitly
    /// registered position fields change; resident resource ownership is stable.
    pub fn replay_decode_plan(&self, position: u32) -> Result<u64> {
        anyhow::ensure!(!self.capture_decode.get(), "cannot replay while capturing");
        self.begin_batch()?;
        let mut dispatches = 0;
        for command in self.decode_commands.borrow().iter() {
            match command {
                DecodeCommand::Compute {
                    work: _,
                    pipe,
                    set,
                    push,
                    gx,
                    gy,
                } => {
                    let mut updated = push.clone();
                    if let Some(offset) = self.decode_position_offsets.borrow().get(&pipe.pipeline)
                    {
                        updated[*offset..*offset + 4].copy_from_slice(&position.to_le_bytes());
                    }
                    self.record_compute(pipe, *set, &updated, *gx, *gy)?;
                    dispatches += 1;
                }
                DecodeCommand::Copy {
                    work: _,
                    src,
                    dst,
                    src_offset,
                    dst_offset,
                    size,
                } => {
                    self.record_copy_raw(*src, *dst, *src_offset, *dst_offset, *size);
                }
            }
        }
        self.finish_batch()?;
        Ok(dispatches)
    }
    pub fn set_decode_position_offset(&self, pipeline: vk::Pipeline, offset: usize) {
        self.decode_position_offsets
            .borrow_mut()
            .insert(pipeline, offset);
    }

    unsafe fn timestamp_begin(&self) {
        self.device
            .cmd_reset_query_pool(self.command_buffer, self.query_pool, 0, if self.profile_dispatch {8192} else {2});
        self.device.cmd_write_timestamp(
            self.command_buffer,
            vk::PipelineStageFlags::TOP_OF_PIPE,
            self.query_pool,
            0,
        );
    }

    unsafe fn submit_and_wait(&self) -> Result<()> {
        self.fence_stop.before_submission().map_err(|e| anyhow!(e))?;
        anyhow::ensure!(
            self.predicted_ms.get() <= crate::work_bound::TARGET_MS,
            "pre-submit work ceiling"
        );
        let sequence = self.submission_sequence.get();
        self.submission_sequence.set(sequence + 1);
        self.device.cmd_write_timestamp(
            self.command_buffer,
            vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            self.query_pool,
            1,
        );
        self.device.end_command_buffer(self.command_buffer)?;
        let submit =
            vk::SubmitInfo::default().command_buffers(std::slice::from_ref(&self.command_buffer));
        let host_submit_monotonic_ns = monotonic_ns();
        if let Some(file) = &self.trace {
            let mut file = file.lock().unwrap();
            #[derive(serde::Serialize)]
            struct Intent<'a> {
                event: &'static str, sequence:u64, phase:&'static str, prompt_tokens:usize,
                work_context:(usize,usize,Option<usize>), predicted_ms:f64,
                commands:&'a [serde_json::Value], host_submit_monotonic_ns:u64, queue_family:u32,
            }
            let commands=self.pending_work.borrow();
            let intent=Intent {event:"submission_intent",sequence,phase:self.phase.get(),prompt_tokens:self.prompt_tokens.get(),work_context:self.work_context.get(),predicted_ms:self.predicted_ms.get(),commands:&commands,host_submit_monotonic_ns,queue_family:self.queue_family};
            let bytes=if let Some(cached)=self.cached_intent_commands.borrow().as_ref() {
                // Serialize live metadata on every submission; the immutable
                // command packet still carries each dispatch and live context.
                #[derive(serde::Serialize)]
                struct Header { event:&'static str, sequence:u64, phase:&'static str,
                    prompt_tokens:usize, work_context:(usize,usize,Option<usize>),
                    predicted_ms:f64, host_submit_monotonic_ns:u64, queue_family:u32 }
                let header=Header{event:intent.event,sequence,phase:intent.phase,prompt_tokens:intent.prompt_tokens,
                    work_context:intent.work_context,predicted_ms:intent.predicted_ms,host_submit_monotonic_ns,queue_family:self.queue_family};
                let mut b=serde_json::to_vec(&header)?;b.pop();b.extend_from_slice(b",\"commands\":");b.extend_from_slice(cached);b.push(b'}');b
            } else {serde_json::to_vec(&intent)?};
            file.write_all(&bytes)?;file.write_all(b"\n")?;
            // Flush before submission so an owned-process kill cannot lose its
            // breadcrumb. The supervisor's fault latch is separately fsynced.
            // No machine/power-loss durability is claimed for each trace row.
            file.flush()?;
        }
        let profile_intent_end = if self.profiling() {monotonic_ns()} else {0};
        if let Ok(path) = std::env::var("LFM_MONITOR_STATUS") {
            let monitor: serde_json::Value = serde_json::from_slice(&std::fs::read(path)?)?;
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)?
                .as_nanos();
            let stamp = monitor["time_ns"]
                .as_u64()
                .ok_or_else(|| anyhow!("missing monitor timestamp"))?
                as u128;
            anyhow::ensure!(
                now >= stamp
                    && now - stamp < 5_000_000_000
                    && monitor["new_gpu_faults"] == 0
                    && monitor["protected_health_failures"] == 0,
                "monitor missing/stale/unhealthy"
            );
        }
        let profile_monitor_end = if self.profiling() {monotonic_ns()} else {0};
        self.device
            .queue_submit(self.queue, &[submit], self.fence)?;
        let profile_submit_end = if self.profiling() {monotonic_ns()} else {0};
        self.device.wait_for_fences(&[self.fence], true, u64::MAX)?;
        let host_wait_complete_monotonic_ns = monotonic_ns();
        let mut timestamps = [0u64; 2];
        self.device.get_query_pool_results(
            self.query_pool,
            0,
            &mut timestamps,
            vk::QueryResultFlags::TYPE_64 | vk::QueryResultFlags::WAIT,
        )?;
        let mask = u64::MAX >> (64 - self.timestamp_bits);
        let ticks = timestamps[1].wrapping_sub(timestamps[0]) & mask;
        let gpu_ms = ticks as f64 * self.props.limits.timestamp_period as f64 / 1e6;
        // Latch immediately at common fence completion, before logging and before
        // any possible next queue submission, even if a caller catches the error.
        let fence_result = self.fence_stop.completed(gpu_ms,
            host_wait_complete_monotonic_ns.saturating_sub(host_submit_monotonic_ns));
        self.last_gpu_ms.set(gpu_ms);
        self.total_gpu_ms.set(self.total_gpu_ms.get() + gpu_ms);
        let profile_query_end = if self.profiling() {monotonic_ns()} else {0};
        let within_budget = fence_result.is_ok();
        if let Some(file) = &self.trace {
            let mut file = file.lock().unwrap();
            #[derive(serde::Serialize)]
            struct Completion { event:&'static str, sequence:u64, predicted_ms:f64,
                status:&'static str, phase:&'static str, prompt_tokens:usize,
                host_submit_monotonic_ns:u64, host_wait_complete_monotonic_ns:u64,
                gpu_ms:f64, timing_source:&'static str, timestamp_ticks:[u64;2],
                timestamp_valid_bits:u32, timestamp_period_ns:f32 }
            let row=Completion{event:"gpu_submission",sequence,predicted_ms:self.predicted_ms.get(),
                status:if within_budget {"ok"} else {"budget_exceeded"},phase:self.phase.get(),
                prompt_tokens:self.prompt_tokens.get(),host_submit_monotonic_ns,host_wait_complete_monotonic_ns,
                gpu_ms,timing_source:"vulkan_timestamp_queries",timestamp_ticks:timestamps,
                timestamp_valid_bits:self.timestamp_bits,timestamp_period_ns:self.props.limits.timestamp_period};
            let mut bytes=serde_json::to_vec(&row)?;bytes.push(b'\n');file.write_all(&bytes)?;
            file.flush()?;
        }
        if self.profiling() {
            self.profile_submission(sequence, [self.profile_batch_start.get(),host_submit_monotonic_ns,
                profile_intent_end,profile_monitor_end,profile_submit_end,host_wait_complete_monotonic_ns,
                profile_query_end,monotonic_ns()], gpu_ms)?;
        }
        fence_result.map_err(|e| anyhow!(e))?;
        Ok(())
    }

    /// Allocate and write one descriptor set binding `buffers` (in order) as
    /// STORAGE_BUFFERs for `layout`. Free with [`Self::free_descriptor_set`].
    pub fn write_descriptor_set(
        &self,
        layout: vk::DescriptorSetLayout,
        buffers: &[(vk::Buffer, vk::DeviceSize)],
    ) -> Result<vk::DescriptorSet> {
        let alloc = vk::DescriptorSetAllocateInfo::default()
            .descriptor_pool(self.descriptor_pool)
            .set_layouts(std::slice::from_ref(&layout));
        let set = unsafe { self.device.allocate_descriptor_sets(&alloc) }
            .map_err(|e| anyhow!("allocate descriptor set: {e}"))?[0];

        let binfo: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|(buf, size)| {
                vk::DescriptorBufferInfo::default()
                    .buffer(*buf)
                    .offset(0)
                    .range(*size)
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet> = buffers
            .iter()
            .enumerate()
            .map(|(i, _)| {
                vk::WriteDescriptorSet::default()
                    .dst_set(set)
                    .dst_binding(i as u32)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(&binfo[i]))
            })
            .collect();
        unsafe { self.device.update_descriptor_sets(&writes, &[]) };
        self.descriptor_buffers.borrow_mut().insert(set,buffers.iter().map(|x|x.0).collect());
        Ok(set)
    }

    pub fn free_descriptor_set(&self, set: vk::DescriptorSet) {
        self.descriptor_buffers.borrow_mut().remove(&set);
        unsafe {
            self.device
                .free_descriptor_sets(self.descriptor_pool, &[set])
        }
        .ok();
    }

    /// Begin an isolated command batch. All referenced resources must remain
    /// alive and host writes must finish before submission.
    pub fn begin_batch(&self) -> Result<()> {
        self.hazards.borrow_mut().reset();
        if self.profiling() {self.profile_batch_start.set(monotonic_ns());}
        self.profile_next.set(2);
        self.pending_work.borrow_mut().clear();
        self.predicted_ms.set(0.0);
        unsafe {
            self.device.reset_fences(&[self.fence])?;
            self.device
                .reset_command_buffer(self.command_buffer, vk::CommandBufferResetFlags::empty())?;
            self.device.begin_command_buffer(
                self.command_buffer,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            self.timestamp_begin();
        }
        Ok(())
    }
    pub fn batch_barrier(&self) {
        unsafe {
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE)],
                &[],
                &[],
            );
        }
    }
    pub fn record_compute(
        &self,
        pipe: &ComputePipeline,
        set: vk::DescriptorSet,
        push: &[u8],
        gx: u32,
        gy: u32,
    ) -> Result<()> {
        let p: Vec<u32> = push
            .chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let name = self.pipeline_name(pipe.pipeline);
        let ms = crate::work_bound::estimate(name, &p, gx, gy)?;
        anyhow::ensure!(
            ms <= crate::work_bound::DISPATCH_MS,
            "unsplit dispatch over ceiling"
        );
        let work = serde_json::json!({"pipeline":name,"push":p,"grid":[gx,gy],"predicted_ms":ms,"context":self.work_context.get()});
        if self.capture_decode.get() {
            self.prepare_work(work.clone())?;
            self.decode_commands
                .borrow_mut()
                .push(DecodeCommand::Compute {
                    work,
                    pipe: pipe.clone(),
                    set,
                    push: push.to_vec(),
                    gx,
                    gy,
                });
        } else {self.prepare_work(work)?;}
        let barrier=self.hazards.borrow_mut().before(name,
            self.descriptor_buffers.borrow().get(&set).map(Vec::as_slice).unwrap_or(&[]));
        if barrier {self.batch_barrier();}
        unsafe {
            self.device.cmd_bind_pipeline(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipe.pipeline,
            );
            self.device.cmd_bind_descriptor_sets(
                self.command_buffer,
                vk::PipelineBindPoint::COMPUTE,
                pipe.pipeline_layout,
                0,
                &[set],
                &[],
            );
            self.device.cmd_push_constants(
                self.command_buffer,
                pipe.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push,
            );
            let index = self.profile_next.get();
            self.profile_begin_command(self.command_buffer, index);
            self.device.cmd_dispatch(self.command_buffer, gx, gy, 1);
            self.profile_end_command(self.command_buffer, index);
            self.profile_next.set(index+2);
        }
        Ok(())
    }
    pub fn record_copy(
        &self,
        src: &GpuBuffer,
        dst: &GpuBuffer,
        src_offset: usize,
        dst_offset: usize,
        size: usize,
    ) {
        assert!(src_offset + size <= src.size as usize && dst_offset + size <= dst.size as usize);
        self.record_copy_raw(
            src.buffer,
            dst.buffer,
            src_offset as u64,
            dst_offset as u64,
            size as u64,
        );
    }
    fn record_copy_raw(
        &self,
        src: vk::Buffer,
        dst: vk::Buffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    ) {
        if size > 16 * 1024 * 1024 {
            for offset in (0..size).step_by(16 * 1024 * 1024) {
                self.record_copy_raw(
                    src,
                    dst,
                    src_offset + offset,
                    dst_offset + offset,
                    (size - offset).min(16 * 1024 * 1024),
                );
            }
            return;
        }
        let work = serde_json::json!({"pipeline":"copy","src":format!("{:?}",src),"dst":format!("{:?}",dst),"bytes":size,"predicted_ms":0.05+size as f64/2_000_000.0,"context":self.work_context.get()});
        if self.capture_decode.get() {
            self.prepare_work(work.clone()).expect("copy work planner");
            self.decode_commands.borrow_mut().push(DecodeCommand::Copy {
                work,
                src,
                dst,
                src_offset,
                dst_offset,
                size,
            });
        } else {self.prepare_work(work).expect("copy work planner");}
        self.batch_barrier();
        self.hazards.borrow_mut().reset();
        unsafe {
            let index = self.profile_next.get();
            self.profile_begin_command(self.command_buffer, index);
            self.device.cmd_copy_buffer(
                self.command_buffer,
                src,
                dst,
                &[vk::BufferCopy::default()
                    .src_offset(src_offset as u64)
                    .dst_offset(dst_offset as u64)
                    .size(size as u64)],
            );
            self.profile_end_command(self.command_buffer, index);
            self.profile_next.set(index+2);
        }
    }
    pub fn finish_batch(&self) -> Result<()> {
        self.batch_barrier();
        unsafe {
            // Coherent mapping avoids invalidate calls, not the device-to-host
            // memory dependency. The fence below completes this visibility step.
            self.device.cmd_pipeline_barrier(
                self.command_buffer,
                vk::PipelineStageFlags::ALL_COMMANDS,
                vk::PipelineStageFlags::HOST,
                vk::DependencyFlags::empty(),
                &[vk::MemoryBarrier::default()
                    .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                    .dst_access_mask(vk::AccessFlags::HOST_READ)],
                &[],
                &[],
            );
            self.submit_and_wait()?;
        }
        Ok(())
    }

    /// Record and run one compute dispatch to completion. `push` are the push
    /// constant bytes (may be empty). Blocks until the dispatch finishes.
    pub fn run_compute(
        &self,
        pipe: &ComputePipeline,
        set: vk::DescriptorSet,
        push: &[u8],
        gx: u32,
        gy: u32,
        gz: u32,
    ) -> Result<()> {
        anyhow::ensure!(gz == 1, "bounded dispatch supports two axes");
        self.begin_batch()?;
        self.record_compute(pipe, set, push, gx, gy)?;
        self.finish_batch()?;
        Ok(())
    }

    /// Allocate a storage buffer, optionally host-visible (mappable).
    pub fn alloc_buffer(&self, size: usize, host_visible: bool) -> Result<GpuBuffer> {
        let flags = vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::TRANSFER_SRC
            | vk::BufferUsageFlags::TRANSFER_DST;
        let buf_ci = vk::BufferCreateInfo::default()
            .size(size as vk::DeviceSize)
            .usage(flags)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { self.device.create_buffer(&buf_ci, None) }
            .map_err(|e| anyhow!("create buffer: {e}"))?;

        let req = unsafe { self.device.get_buffer_memory_requirements(buffer) };
        let mem_props = unsafe {
            self.instance
                .get_physical_device_memory_properties(self.physical_device)
        };

        let desired = if host_visible {
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT
        } else {
            vk::MemoryPropertyFlags::DEVICE_LOCAL
        };
        let mem_type = if host_visible {
            find_memory_type(
                &mem_props,
                req.memory_type_bits,
                desired | vk::MemoryPropertyFlags::HOST_CACHED,
            )
            .or_else(|| find_memory_type(&mem_props, req.memory_type_bits, desired))
        } else {
            find_memory_type(&mem_props, req.memory_type_bits, desired)
        }
        .ok_or_else(|| anyhow!("no suitable memory type (host_visible={host_visible})"))?;

        let alloc = vk::MemoryAllocateInfo::default()
            .allocation_size(req.size)
            .memory_type_index(mem_type);
        let memory = unsafe { self.device.allocate_memory(&alloc, None) }
            .map_err(|e| anyhow!("alloc memory: {e}"))?;
        unsafe {
            self.device
                .bind_buffer_memory(buffer, memory, 0)
                .map_err(|e| anyhow!("bind buffer memory: {e}"))?;
        }

        let mapped = if host_visible {
            let p = unsafe {
                self.device
                    .map_memory(memory, 0, req.size, vk::MemoryMapFlags::empty())
                    .map_err(|e| anyhow!("map memory: {e}"))?
            };
            Some(std::ptr::NonNull::new(p as *mut u8).unwrap())
        } else {
            None
        };

        self.profile_write(serde_json::json!({"event":"allocation", "logical_bytes":size,
            "allocation_bytes":req.size,"memory_type":mem_type,"host_visible":host_visible,
            "memory_flags":mem_props.memory_types[mem_type as usize].property_flags.as_raw()}))?;
        Ok(GpuBuffer {
            generation: {
                static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
                NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
            },
            device: self.device.clone(),
            buffer,
            memory,
            size: size as vk::DeviceSize,
            host_visible,
            mapped,
        })
    }

    /// Upload `bytes` into a device-local (VRAM) storage buffer via a host-visible
    /// staging buffer and a `vkCmdCopyBuffer`. Weights use this path (they are
    /// read-only on the GPU and too large for the host-visible GTT/GART aperture).
    pub fn upload_to_device(&self, bytes: &[u8]) -> Result<GpuBuffer> {
        let size = bytes.len().max(4);
        let dst = self.alloc_buffer(size, false)?;
        let staging = self.alloc_buffer(size, true)?;
        if !bytes.is_empty() {
            staging.write(0, bytes);
        }
        for offset in (0..size).step_by(16 * 1024 * 1024) {
            let count = (size - offset).min(16 * 1024 * 1024);
            self.begin_batch()?;
            self.record_copy(&staging, &dst, offset, offset, count);
            self.finish_batch()?;
        }
        // staging is dropped here (host-visible, freed).
        Ok(dst)
    }

    /// Build a compute pipeline for a SPIR-V module using a descriptor-set
    /// layout of `n_bindings` storage buffers and a small push-constant block.
    pub fn create_compute_pipeline(
        &self,
        spirv: &[u32],
        n_bindings: u32,
        push_constant_size: u32,
    ) -> Result<ComputePipeline> {
        let module_ci = vk::ShaderModuleCreateInfo::default().code(spirv);
        let module = unsafe { self.device.create_shader_module(&module_ci, None) }
            .map_err(|e| anyhow!("create shader module: {e}"))?;

        let mut bindings: Vec<vk::DescriptorSetLayoutBinding> = (0..n_bindings)
            .map(|i| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(i)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();

        let layout_ci = vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
        let dsl = unsafe { self.device.create_descriptor_set_layout(&layout_ci, None) }
            .map_err(|e| anyhow!("create descriptor set layout: {e}"))?;

        let pc_range = vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(push_constant_size);
        let pc_ranges = if push_constant_size > 0 {
            std::slice::from_ref(&pc_range)
        } else {
            &[]
        };

        let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::default()
            .set_layouts(std::slice::from_ref(&dsl))
            .push_constant_ranges(pc_ranges);
        let pipeline_layout = unsafe {
            self.device
                .create_pipeline_layout(&pipeline_layout_ci, None)
        }
        .map_err(|e| anyhow!("create pipeline layout: {e}"))?;

        let stage_ci = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(module)
            .name(CStr::from_bytes_with_nul(b"main\0").unwrap());
        let pipe_ci = vk::ComputePipelineCreateInfo::default()
            .stage(stage_ci)
            .layout(pipeline_layout);
        let pipeline = unsafe {
            self.device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipe_ci),
                    None,
                )
                .map_err(|(_partial, err)| anyhow!("create compute pipeline: {err:?}"))?
        }[0];

        unsafe {
            self.device.destroy_shader_module(module, None);
        }
        let _ = &mut bindings;

        Ok(ComputePipeline {
            pipeline,
            pipeline_layout,
            descriptor_set_layout: dsl,
        })
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_fence(self.fence, None);
            self.device.destroy_query_pool(self.query_pool, None);
            self.device
                .destroy_descriptor_pool(self.descriptor_pool, None);
            self.device
                .free_command_buffers(self.command_pool, &[self.command_buffer]);
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            self.instance.destroy_instance(None);
        }
    }
}

fn cstr_props(bytes: &[std::os::raw::c_char]) -> String {
    let end = bytes.iter().position(|&b| b == 0).unwrap_or(bytes.len());
    let u8s: &[u8] = unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const u8, end) };
    String::from_utf8_lossy(u8s).into_owned()
}

fn find_memory_type(
    mem_props: &vk::PhysicalDeviceMemoryProperties,
    type_bits: u32,
    desired: vk::MemoryPropertyFlags,
) -> Option<u32> {
    for i in 0..mem_props.memory_type_count {
        if type_bits & (1 << i) != 0
            && mem_props.memory_types[i as usize]
                .property_flags
                .contains(desired)
        {
            return Some(i);
        }
    }
    None
}
