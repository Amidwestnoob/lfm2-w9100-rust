//! Immutable secondary decode graphs, partitioned using worst-envelope work.
use super::*;
impl VulkanContext {
    pub fn compile_reusable_decode(&self) -> Result<()> {
        anyhow::ensure!(!self.capture_decode.get(), "capture still active");
        anyhow::ensure!(
            self.decode_secondary.borrow().is_empty(),
            "already compiled"
        );
        let commands = self.decode_commands.borrow();
        let mut groups = vec![Vec::new()];
        let mut total = 0.0;
        for cmd in commands.iter() {
            let work = match cmd {
                DecodeCommand::Compute { work, .. } | DecodeCommand::Copy { work, .. } => work,
            };
            let ms = work["predicted_ms"].as_f64().unwrap();
            anyhow::ensure!(
                ms <= crate::work_bound::DISPATCH_MS,
                "decode dispatch bound"
            );
            if total + ms > crate::work_bound::TARGET_MS {
                groups.push(Vec::new());
                total = 0.0;
            }
            groups.last_mut().unwrap().push(cmd);
            total += ms;
        }
        for group in groups {
            unsafe {
                let cb = self.device.allocate_command_buffers(
                    &vk::CommandBufferAllocateInfo::default()
                        .command_pool(self.command_pool)
                        .level(vk::CommandBufferLevel::SECONDARY)
                        .command_buffer_count(1),
                )?[0];
                let inheritance = vk::CommandBufferInheritanceInfo::default();
                self.device.begin_command_buffer(
                    cb,
                    &vk::CommandBufferBeginInfo::default().inheritance_info(&inheritance),
                )?;
                let mut identities = Vec::new();
                let mut hazards=hazards::Hazards::default();
                for cmd in group {
                    let barrier=match cmd {
                        DecodeCommand::Compute{pipe,set,..}=>hazards.before(self.pipeline_name(pipe.pipeline),
                            self.descriptor_buffers.borrow().get(set).map(Vec::as_slice).unwrap_or(&[])),
                        DecodeCommand::Copy{..}=>{hazards.reset();true}
                    };
                    if barrier {
                    self.device.cmd_pipeline_barrier(
                        cb,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::PipelineStageFlags::ALL_COMMANDS,
                        vk::DependencyFlags::empty(),
                        &[vk::MemoryBarrier::default()
                            .src_access_mask(vk::AccessFlags::MEMORY_WRITE)
                            .dst_access_mask(
                                vk::AccessFlags::MEMORY_READ | vk::AccessFlags::MEMORY_WRITE,
                            )],
                        &[],
                        &[],
                    );
                    }
                    let index = 2 + identities.len() as u32 * 2;
                    self.profile_begin_command(cb, index);
                    match cmd {
                        DecodeCommand::Compute {
                            work,
                            pipe,
                            set,
                            push,
                            gx,
                            gy,
                        } => {
                            identities.push(work.clone());
                            anyhow::ensure!(
                                !self
                                    .decode_position_offsets
                                    .borrow()
                                    .contains_key(&pipe.pipeline),
                                "dynamic position required"
                            );
                            self.device.cmd_bind_pipeline(
                                cb,
                                vk::PipelineBindPoint::COMPUTE,
                                pipe.pipeline,
                            );
                            self.device.cmd_bind_descriptor_sets(
                                cb,
                                vk::PipelineBindPoint::COMPUTE,
                                pipe.pipeline_layout,
                                0,
                                &[*set],
                                &[],
                            );
                            self.device.cmd_push_constants(
                                cb,
                                pipe.pipeline_layout,
                                vk::ShaderStageFlags::COMPUTE,
                                0,
                                push,
                            );
                            self.device.cmd_dispatch(cb, *gx, *gy, 1);
                        }
                        DecodeCommand::Copy {
                            work,
                            src,
                            dst,
                            src_offset,
                            dst_offset,
                            size,
                        } => {
                            identities.push(work.clone());
                            self.device.cmd_copy_buffer(
                                cb,
                                *src,
                                *dst,
                                &[vk::BufferCopy::default()
                                    .src_offset(*src_offset)
                                    .dst_offset(*dst_offset)
                                    .size(*size)],
                            );
                        }
                    }
                    self.profile_end_command(cb, index);
                }
                self.device.end_command_buffer(cb)?;
                let template=intent_cache::CommandTemplate::new(&identities)?;
                self.decode_secondary.borrow_mut().push((cb, identities, template));
            }
        }
        Ok(())
    }
    pub fn replay_reusable_decode(&self) -> Result<u64> {
        anyhow::ensure!(
            self.work_context.get().0 < self.decode_context_end.get() as usize,
            "captured decode context bound"
        );
        let mut secondaries = self.decode_secondary.borrow_mut();
        anyhow::ensure!(!secondaries.is_empty(), "no reusable decode");
        for (cb, identities, template) in secondaries.iter_mut() {
            self.begin_batch()?;
            let mut total=0.0;
            for work in identities.iter_mut() {

                let ms=work["predicted_ms"].as_f64().unwrap();
                anyhow::ensure!(ms.is_finite() && ms>0.0 && ms<=crate::work_bound::DISPATCH_MS,"cached dispatch bound");
                total+=ms;
            }
            anyhow::ensure!(total<=crate::work_bound::TARGET_MS,"cached submission bound");
            self.predicted_ms.set(total);
            *self.cached_intent_commands.borrow_mut()=Some(template.render(self.work_context.get().0));
            std::mem::swap(&mut *self.pending_work.borrow_mut(),identities);
            self.batch_barrier();
            unsafe {
                self.device
                    .cmd_execute_commands(self.command_buffer, &[*cb]);
            }
            let result=self.finish_batch();
            self.cached_intent_commands.borrow_mut().take();
            std::mem::swap(&mut *self.pending_work.borrow_mut(),identities);
            result?;
        }
        self.decode_command_buffer_reuses
            .set(self.decode_command_buffer_reuses.get() + 1);
        Ok(self
            .decode_commands
            .borrow()
            .iter()
            .filter(|c| matches!(c, DecodeCommand::Compute { .. }))
            .count() as u64)
    }
}
