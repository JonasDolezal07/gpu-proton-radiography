//! Renderer - ties together swapchain, compute, and presentation

use ash::vk;
use anyhow::{Result, Context};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;
use std::sync::{Arc, Mutex};

use super::{VulkanContext, Swapchain, GpuBuffer, FieldTexture, StagingBuffer, ComputePipeline, SimParams};
use crate::loaders::{FieldData, ParticleData, Particle};

const MAX_FRAMES_IN_FLIGHT: usize = 2;
const MAX_DETECTOR_HITS: usize = 1_000_000;

/// Detector hit record (matches shader)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DetectorHit {
    pub position: [f32; 2],
    pub energy: f32,
    pub particle_id: u32,
}

/// Per-frame synchronization
struct FrameData {
    image_available: vk::Semaphore,
    render_finished: vk::Semaphore,
    in_flight_fence: vk::Fence,
    command_buffer: vk::CommandBuffer,
}

pub struct Renderer {
    allocator: Arc<Mutex<Allocator>>,
    swapchain: Swapchain,

    // Compute resources
    compute_pipeline: Option<ComputePipeline>,
    particle_buffer: Option<GpuBuffer>,
    field_texture: Option<FieldTexture>,
    detector_buffer: Option<GpuBuffer>,

    // Simulation state
    sim_params: SimParams,
    particle_count: u32,
    is_running: bool,

    // Synchronization
    frames: Vec<FrameData>,
    current_frame: usize,

    // Cached references
    ctx: Arc<VulkanContext>,
}

impl Renderer {
    pub fn new(ctx: Arc<VulkanContext>, width: u32, height: u32) -> Result<Self> {
        // Create memory allocator
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: ctx.instance().clone(),
            device: ctx.device().clone(),
            physical_device: ctx.physical_device(),
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        }).context("Failed to create GPU allocator")?;

        let allocator = Arc::new(Mutex::new(allocator));

        // Create swapchain
        let swapchain = Swapchain::new(
            ctx.instance(),
            ctx.device(),
            ctx.physical_device(),
            ctx.surface_loader(),
            ctx.surface(),
            ctx.graphics_queue_family(),
            width,
            height,
        )?;

        // Create frame synchronization objects
        let frames = Self::create_frame_data(ctx.device(), ctx.command_pool())?;

        let sim_params = SimParams {
            dt: 1e-12,
            q_over_m: 9.58e7,  // Proton: e/m_p
            n_particles: 0,
            _pad: 0,
            field_min: [0.0; 4],
            field_max: [1.0, 1.0, 1.0, 0.0],
            detector_pos: [0.0, 0.0, 1.0, 0.0],
            detector_normal: [0.0, 0.0, -1.0, 0.0],
        };

        Ok(Self {
            allocator,
            swapchain,
            compute_pipeline: None,
            particle_buffer: None,
            field_texture: None,
            detector_buffer: None,
            sim_params,
            particle_count: 0,
            is_running: false,
            frames,
            current_frame: 0,
            ctx,
        })
    }

    fn create_frame_data(device: &ash::Device, command_pool: vk::CommandPool) -> Result<Vec<FrameData>> {
        let mut frames = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);

        unsafe {
            let semaphore_info = vk::SemaphoreCreateInfo::default();
            let fence_info = vk::FenceCreateInfo {
                flags: vk::FenceCreateFlags::SIGNALED,
                ..Default::default()
            };

            let alloc_info = vk::CommandBufferAllocateInfo {
                command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: MAX_FRAMES_IN_FLIGHT as u32,
                ..Default::default()
            };

            let command_buffers = device.allocate_command_buffers(&alloc_info)?;

            for i in 0..MAX_FRAMES_IN_FLIGHT {
                frames.push(FrameData {
                    image_available: device.create_semaphore(&semaphore_info, None)?,
                    render_finished: device.create_semaphore(&semaphore_info, None)?,
                    in_flight_fence: device.create_fence(&fence_info, None)?,
                    command_buffer: command_buffers[i],
                });
            }
        }

        Ok(frames)
    }

    pub fn load_compute_shader(&mut self, shader_spirv: &[u8]) -> Result<()> {
        self.compute_pipeline = Some(ComputePipeline::new(self.ctx.device(), shader_spirv)?);
        log::info!("Loaded compute shader");
        Ok(())
    }

    pub fn upload_field(&mut self, field: &FieldData) -> Result<()> {
        let device = self.ctx.device();

        // Create field texture
        let mut texture = FieldTexture::new(
            device,
            &self.allocator,
            field.nx,
            field.ny,
            field.nz,
        )?;

        // Convert field data to RGBA format (xyz + padding)
        let num_voxels = (field.nx * field.ny * field.nz) as usize;
        let mut rgba_data = vec![0.0f32; num_voxels * 4];
        for i in 0..num_voxels {
            rgba_data[i * 4 + 0] = field.data[i * 3 + 0];
            rgba_data[i * 4 + 1] = field.data[i * 3 + 1];
            rgba_data[i * 4 + 2] = field.data[i * 3 + 2];
            rgba_data[i * 4 + 3] = 0.0;
        }

        // Create staging buffer and upload
        let byte_size = (rgba_data.len() * std::mem::size_of::<f32>()) as vk::DeviceSize;
        let mut staging = StagingBuffer::new(device, &self.allocator, byte_size)?;
        staging.write(&rgba_data)?;

        // Record and submit transfer commands
        self.upload_texture_data(&mut texture, &staging, field.nx, field.ny, field.nz)?;

        staging.cleanup(device, &self.allocator);

        // Update sim params with field bounds
        self.sim_params.field_min = [field.bounds.x_min, field.bounds.y_min, field.bounds.z_min, 0.0];
        self.sim_params.field_max = [field.bounds.x_max, field.bounds.y_max, field.bounds.z_max, 0.0];

        if let Some(old) = self.field_texture.take() {
            let mut old = old;
            old.cleanup(device, &self.allocator);
        }
        self.field_texture = Some(texture);

        log::info!("Uploaded field texture: {}x{}x{}", field.nx, field.ny, field.nz);
        Ok(())
    }

    fn upload_texture_data(
        &self,
        texture: &mut FieldTexture,
        staging: &StagingBuffer,
        nx: u32,
        ny: u32,
        nz: u32,
    ) -> Result<()> {
        let device = self.ctx.device();
        let queue = self.ctx.compute_queue();

        unsafe {
            // Allocate one-shot command buffer
            let alloc_info = vk::CommandBufferAllocateInfo {
                command_pool: self.ctx.command_pool(),
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };
            let cmd = device.allocate_command_buffers(&alloc_info)?[0];

            let begin_info = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };
            device.begin_command_buffer(cmd, &begin_info)?;

            // Transition image to transfer dst
            let barrier = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                image: texture.image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };

            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            // Copy buffer to image
            let copy_region = vk::BufferImageCopy {
                buffer_offset: 0,
                buffer_row_length: 0,
                buffer_image_height: 0,
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    mip_level: 0,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                image_offset: vk::Offset3D { x: 0, y: 0, z: 0 },
                image_extent: vk::Extent3D { width: nx, height: ny, depth: nz },
            };

            device.cmd_copy_buffer_to_image(
                cmd,
                staging.buffer(),
                texture.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[copy_region],
            );

            // Transition to shader read
            let barrier = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                image: texture.image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                ..Default::default()
            };

            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::COMPUTE_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            device.end_command_buffer(cmd)?;

            // Submit and wait
            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd,
                ..Default::default()
            };

            device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
            device.queue_wait_idle(queue)?;
            device.free_command_buffers(self.ctx.command_pool(), &[cmd]);
        }

        Ok(())
    }

    pub fn upload_particles(&mut self, particles: &ParticleData) -> Result<()> {
        let device = self.ctx.device();
        let byte_size = particles.size_bytes() as vk::DeviceSize;

        // Create particle buffer (GPU-visible, mappable for readback)
        let buffer = GpuBuffer::new(
            device,
            &self.allocator,
            byte_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "particles",
        )?;

        buffer.write(&particles.particles)?;

        self.particle_count = particles.count;
        self.sim_params.n_particles = particles.count;

        if let Some(old) = self.particle_buffer.take() {
            let mut old = old;
            old.cleanup(device, &self.allocator);
        }
        self.particle_buffer = Some(buffer);

        // Create detector buffer
        let detector_size = (std::mem::size_of::<u32>() * 4  // header: count + padding
            + std::mem::size_of::<DetectorHit>() * MAX_DETECTOR_HITS) as vk::DeviceSize;

        let detector_buffer = GpuBuffer::new(
            device,
            &self.allocator,
            detector_size,
            vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "detector_hits",
        )?;

        // Zero the hit count
        let zeros = [0u32; 4];
        detector_buffer.write(&zeros)?;

        if let Some(old) = self.detector_buffer.take() {
            let mut old = old;
            old.cleanup(device, &self.allocator);
        }
        self.detector_buffer = Some(detector_buffer);

        log::info!("Uploaded {} particles", particles.count);
        Ok(())
    }

    pub fn update_descriptors(&mut self) -> Result<()> {
        let pipeline = self.compute_pipeline.as_ref().context("No compute pipeline")?;
        let particles = self.particle_buffer.as_ref().context("No particle buffer")?;
        let field = self.field_texture.as_ref().context("No field texture")?;
        let detector = self.detector_buffer.as_ref().context("No detector buffer")?;

        pipeline.update_descriptors(
            self.ctx.device(),
            particles.buffer,
            particles.size,
            field.view,
            field.sampler,
            detector.buffer,
            detector.size,
        );

        Ok(())
    }

    pub fn set_sim_params(&mut self, dt: f32, detector_pos: [f32; 3], detector_normal: [f32; 3]) {
        self.sim_params.dt = dt;
        self.sim_params.detector_pos = [detector_pos[0], detector_pos[1], detector_pos[2], 0.0];
        self.sim_params.detector_normal = [detector_normal[0], detector_normal[1], detector_normal[2], 0.0];
    }

    pub fn start_simulation(&mut self) {
        self.is_running = true;
        log::info!("Simulation started");
    }

    pub fn stop_simulation(&mut self) {
        self.is_running = false;
        log::info!("Simulation stopped");
    }

    pub fn toggle_simulation(&mut self) {
        if self.is_running {
            self.stop_simulation();
        } else {
            self.start_simulation();
        }
    }

    pub fn step_simulation(&mut self) -> Result<()> {
        if self.compute_pipeline.is_none() || self.particle_buffer.is_none() {
            return Ok(());
        }

        let device = self.ctx.device();
        let frame = &self.frames[self.current_frame];

        unsafe {
            // Wait for previous frame
            device.wait_for_fences(&[frame.in_flight_fence], true, u64::MAX)?;
            device.reset_fences(&[frame.in_flight_fence])?;

            // Record compute commands
            let cmd = frame.command_buffer;
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(cmd, &begin_info)?;

            // Dispatch compute shader
            let pipeline = self.compute_pipeline.as_ref().unwrap();
            pipeline.record_dispatch(device, cmd, &self.sim_params, 256);

            device.end_command_buffer(cmd)?;

            // Submit
            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd,
                ..Default::default()
            };

            device.queue_submit(
                self.ctx.compute_queue(),
                &[submit_info],
                frame.in_flight_fence,
            )?;
        }

        self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
        Ok(())
    }

    pub fn render_frame(&mut self) -> Result<bool> {
        // Run simulation step if active
        if self.is_running {
            self.step_simulation()?;
        }

        let device = self.ctx.device();
        let frame = &self.frames[self.current_frame];

        unsafe {
            // Acquire next swapchain image
            let (image_index, suboptimal) = match self.swapchain.acquire_next_image(frame.image_available) {
                Ok(result) => result,
                Err(_) => return Ok(true), // Need resize
            };

            if suboptimal {
                return Ok(true);
            }

            // For now, just clear the image
            // TODO: Render particles as points, field visualization

            // Wait for previous use of this image
            device.wait_for_fences(&[frame.in_flight_fence], true, u64::MAX)?;
            device.reset_fences(&[frame.in_flight_fence])?;

            let cmd = frame.command_buffer;
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(cmd, &begin_info)?;

            // Transition image for clear
            let barrier = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                image: self.swapchain.image(image_index as usize),
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };

            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TOP_OF_PIPE,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            // Clear to dark blue
            let clear_color = vk::ClearColorValue {
                float32: [0.02, 0.02, 0.08, 1.0],
            };

            device.cmd_clear_color_image(
                cmd,
                self.swapchain.image(image_index as usize),
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_color,
                &[vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }],
            );

            // Transition for present
            let barrier = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                image: self.swapchain.image(image_index as usize),
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                ..Default::default()
            };

            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            device.end_command_buffer(cmd)?;

            // Submit
            let wait_stages = [vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT];
            let submit_info = vk::SubmitInfo {
                wait_semaphore_count: 1,
                p_wait_semaphores: &frame.image_available,
                p_wait_dst_stage_mask: wait_stages.as_ptr(),
                command_buffer_count: 1,
                p_command_buffers: &cmd,
                signal_semaphore_count: 1,
                p_signal_semaphores: &frame.render_finished,
                ..Default::default()
            };

            device.queue_submit(
                self.ctx.graphics_queue(),
                &[submit_info],
                frame.in_flight_fence,
            )?;

            // Present
            let suboptimal = self.swapchain.present(
                self.ctx.graphics_queue(),
                image_index,
                frame.render_finished,
            )?;

            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;

            Ok(suboptimal)
        }
    }

    pub fn resize(&mut self, width: u32, height: u32) -> Result<()> {
        if width == 0 || height == 0 {
            return Ok(());
        }

        unsafe {
            self.ctx.device().device_wait_idle()?;
        }

        self.swapchain.recreate(
            self.ctx.instance(),
            self.ctx.device(),
            self.ctx.physical_device(),
            self.ctx.surface_loader(),
            self.ctx.surface(),
            self.ctx.graphics_queue_family(),
            width,
            height,
        )?;

        log::info!("Swapchain resized to {}x{}", width, height);
        Ok(())
    }

    pub fn read_detector_hits(&self) -> Result<Vec<DetectorHit>> {
        let detector = self.detector_buffer.as_ref().context("No detector buffer")?;

        // Read hit count
        let mut header = [0u32; 4];
        detector.read(&mut header)?;
        let hit_count = header[0] as usize;

        if hit_count == 0 {
            return Ok(Vec::new());
        }

        let hit_count = hit_count.min(MAX_DETECTOR_HITS);

        // Read hits
        let mut hits = vec![DetectorHit { position: [0.0; 2], energy: 0.0, particle_id: 0 }; hit_count];

        // Read the buffer data manually
        let allocation = detector.allocation.as_ref().context("No allocation")?;
        let ptr = allocation.mapped_ptr().context("Buffer not mapped")?;

        unsafe {
            let hits_ptr = (ptr.as_ptr() as *const u8).add(16) as *const DetectorHit; // Skip 16-byte header
            std::ptr::copy_nonoverlapping(hits_ptr, hits.as_mut_ptr(), hit_count);
        }

        Ok(hits)
    }

    pub fn cleanup(&mut self) {
        let device = self.ctx.device();

        unsafe {
            device.device_wait_idle().ok();
        }

        // Clean up buffers
        if let Some(mut buf) = self.particle_buffer.take() {
            buf.cleanup(device, &self.allocator);
        }
        if let Some(mut tex) = self.field_texture.take() {
            tex.cleanup(device, &self.allocator);
        }
        if let Some(mut buf) = self.detector_buffer.take() {
            buf.cleanup(device, &self.allocator);
        }
        if let Some(mut pipe) = self.compute_pipeline.take() {
            pipe.cleanup(device);
        }

        // Clean up frames
        unsafe {
            for frame in &self.frames {
                device.destroy_semaphore(frame.image_available, None);
                device.destroy_semaphore(frame.render_finished, None);
                device.destroy_fence(frame.in_flight_fence, None);
            }
        }

        self.swapchain.cleanup(device);
    }
}
