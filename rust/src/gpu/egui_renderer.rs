//! Custom egui Vulkan renderer for ash 0.38
//!
//! Renders egui output to Vulkan using our own pipeline.

use ash::vk;
use anyhow::{Result, Context};
use std::ffi::CString;
use std::sync::{Arc, Mutex};
use gpu_allocator::vulkan::{Allocator, Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;
use bytemuck::{Pod, Zeroable};

/// Push constants for egui rendering
#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
struct EguiPushConstants {
    screen_size: [f32; 2],
}

/// Egui Vulkan renderer
pub struct EguiRenderer {
    device: ash::Device,
    allocator: Arc<Mutex<Allocator>>,

    // Pipeline
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,

    // Font texture
    font_image: vk::Image,
    font_allocation: Option<Allocation>,
    font_image_view: vk::ImageView,
    font_sampler: vk::Sampler,
    font_version: u64,

    // Vertex/index buffers (per frame)
    vertex_buffer: vk::Buffer,
    vertex_allocation: Option<Allocation>,
    vertex_capacity: usize,
    index_buffer: vk::Buffer,
    index_allocation: Option<Allocation>,
    index_capacity: usize,

    // Command pool for uploads
    command_pool: vk::CommandPool,
    queue: vk::Queue,
}

impl EguiRenderer {
    pub fn new(
        device: ash::Device,
        allocator: Arc<Mutex<Allocator>>,
        render_pass: vk::RenderPass,
        queue: vk::Queue,
        queue_family: u32,
        vert_shader: &[u8],
        frag_shader: &[u8],
    ) -> Result<Self> {
        unsafe {
            // Create descriptor set layout for font texture
            let binding = vk::DescriptorSetLayoutBinding {
                binding: 0,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                ..Default::default()
            };

            let layout_info = vk::DescriptorSetLayoutCreateInfo {
                binding_count: 1,
                p_bindings: &binding,
                ..Default::default()
            };

            let descriptor_set_layout = device
                .create_descriptor_set_layout(&layout_info, None)
                .context("Failed to create egui descriptor set layout")?;

            // Create pipeline layout with push constants
            let push_constant_range = vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX,
                offset: 0,
                size: std::mem::size_of::<EguiPushConstants>() as u32,
            };

            let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
                set_layout_count: 1,
                p_set_layouts: &descriptor_set_layout,
                push_constant_range_count: 1,
                p_push_constant_ranges: &push_constant_range,
                ..Default::default()
            };

            let pipeline_layout = device
                .create_pipeline_layout(&pipeline_layout_info, None)
                .context("Failed to create egui pipeline layout")?;

            // Create pipeline
            let pipeline = Self::create_pipeline(
                &device,
                render_pass,
                pipeline_layout,
                vert_shader,
                frag_shader,
            )?;

            // Create descriptor pool and set
            let pool_size = vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 1,
            };

            let pool_info = vk::DescriptorPoolCreateInfo {
                max_sets: 1,
                pool_size_count: 1,
                p_pool_sizes: &pool_size,
                ..Default::default()
            };

            let descriptor_pool = device
                .create_descriptor_pool(&pool_info, None)
                .context("Failed to create egui descriptor pool")?;

            let alloc_info = vk::DescriptorSetAllocateInfo {
                descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: &descriptor_set_layout,
                ..Default::default()
            };

            let descriptor_set = device.allocate_descriptor_sets(&alloc_info)?[0];

            // Create font sampler
            let sampler_info = vk::SamplerCreateInfo {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                max_anisotropy: 1.0,
                ..Default::default()
            };

            let font_sampler = device
                .create_sampler(&sampler_info, None)
                .context("Failed to create egui font sampler")?;

            // Create command pool for uploads
            let command_pool_info = vk::CommandPoolCreateInfo {
                queue_family_index: queue_family,
                flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
                ..Default::default()
            };

            let command_pool = device
                .create_command_pool(&command_pool_info, None)
                .context("Failed to create egui command pool")?;

            // Create placeholder font texture (will be updated on first frame)
            let (font_image, font_allocation, font_image_view) =
                Self::create_font_texture(&device, &allocator, queue, command_pool, 1, 1, &[255u8; 4])?;

            // Update descriptor set
            let image_info = vk::DescriptorImageInfo {
                sampler: font_sampler,
                image_view: font_image_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };

            let write = vk::WriteDescriptorSet {
                dst_set: descriptor_set,
                dst_binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: &image_info,
                ..Default::default()
            };

            device.update_descriptor_sets(&[write], &[]);

            // Create initial vertex/index buffers
            // egui::epaint::Vertex is 20 bytes: pos [f32;2], uv [f32;2], color [u8;4]
            let vertex_capacity = 64 * 1024;
            let index_capacity = 128 * 1024;

            let (vertex_buffer, vertex_allocation) = Self::create_buffer(
                &device,
                &allocator,
                vertex_capacity * 20, // 20 bytes per egui vertex
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?;

            let (index_buffer, index_allocation) = Self::create_buffer(
                &device,
                &allocator,
                index_capacity * std::mem::size_of::<u32>(),
                vk::BufferUsageFlags::INDEX_BUFFER,
            )?;

            Ok(Self {
                device,
                allocator,
                pipeline,
                pipeline_layout,
                descriptor_set_layout,
                descriptor_pool,
                descriptor_set,
                font_image,
                font_allocation: Some(font_allocation),
                font_image_view,
                font_sampler,
                font_version: 0,
                vertex_buffer,
                vertex_allocation: Some(vertex_allocation),
                vertex_capacity,
                index_buffer,
                index_allocation: Some(index_allocation),
                index_capacity,
                command_pool,
                queue,
            })
        }
    }

    fn create_pipeline(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        pipeline_layout: vk::PipelineLayout,
        vert_shader: &[u8],
        frag_shader: &[u8],
    ) -> Result<vk::Pipeline> {
        unsafe {
            let vert_module = Self::create_shader_module(device, vert_shader)?;
            let frag_module = Self::create_shader_module(device, frag_shader)?;

            let entry_point = CString::new("main").unwrap();

            let shader_stages = [
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::VERTEX,
                    module: vert_module,
                    p_name: entry_point.as_ptr(),
                    ..Default::default()
                },
                vk::PipelineShaderStageCreateInfo {
                    stage: vk::ShaderStageFlags::FRAGMENT,
                    module: frag_module,
                    p_name: entry_point.as_ptr(),
                    ..Default::default()
                },
            ];

            // Vertex input: match egui::epaint::Vertex exactly (20 bytes)
            // pos: [f32; 2] at offset 0
            // uv: [f32; 2] at offset 8
            // color: [u8; 4] at offset 16 (as R8G8B8A8_UNORM)
            let binding_desc = vk::VertexInputBindingDescription {
                binding: 0,
                stride: 20, // egui vertex is exactly 20 bytes
                input_rate: vk::VertexInputRate::VERTEX,
            };

            let attr_descs = [
                vk::VertexInputAttributeDescription {
                    location: 0,
                    binding: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: 0,
                },
                vk::VertexInputAttributeDescription {
                    location: 1,
                    binding: 0,
                    format: vk::Format::R32G32_SFLOAT,
                    offset: 8,
                },
                vk::VertexInputAttributeDescription {
                    location: 2,
                    binding: 0,
                    format: vk::Format::R8G8B8A8_UNORM, // Color as normalized RGBA
                    offset: 16,
                },
            ];

            let vertex_input = vk::PipelineVertexInputStateCreateInfo {
                vertex_binding_description_count: 1,
                p_vertex_binding_descriptions: &binding_desc,
                vertex_attribute_description_count: attr_descs.len() as u32,
                p_vertex_attribute_descriptions: attr_descs.as_ptr(),
                ..Default::default()
            };

            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            };

            let viewport_state = vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                scissor_count: 1,
                ..Default::default()
            };

            let rasterization = vk::PipelineRasterizationStateCreateInfo {
                polygon_mode: vk::PolygonMode::FILL,
                cull_mode: vk::CullModeFlags::NONE,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                ..Default::default()
            };

            let multisample = vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };

            // Premultiplied alpha blending
            let blend_attachment = vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::TRUE,
                src_color_blend_factor: vk::BlendFactor::ONE,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::RGBA,
            };

            let color_blend = vk::PipelineColorBlendStateCreateInfo {
                attachment_count: 1,
                p_attachments: &blend_attachment,
                ..Default::default()
            };

            // Disable depth testing for UI overlay
            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: vk::FALSE,
                depth_write_enable: vk::FALSE,
                ..Default::default()
            };

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state = vk::PipelineDynamicStateCreateInfo {
                dynamic_state_count: dynamic_states.len() as u32,
                p_dynamic_states: dynamic_states.as_ptr(),
                ..Default::default()
            };

            let pipeline_info = vk::GraphicsPipelineCreateInfo {
                stage_count: shader_stages.len() as u32,
                p_stages: shader_stages.as_ptr(),
                p_vertex_input_state: &vertex_input,
                p_input_assembly_state: &input_assembly,
                p_viewport_state: &viewport_state,
                p_rasterization_state: &rasterization,
                p_multisample_state: &multisample,
                p_depth_stencil_state: &depth_stencil,
                p_color_blend_state: &color_blend,
                p_dynamic_state: &dynamic_state,
                layout: pipeline_layout,
                render_pass,
                subpass: 0,
                ..Default::default()
            };

            let pipeline = device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|e| anyhow::anyhow!("Failed to create egui pipeline: {:?}", e.1))?[0];

            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);

            Ok(pipeline)
        }
    }

    fn create_shader_module(device: &ash::Device, code: &[u8]) -> Result<vk::ShaderModule> {
        let code_aligned: Vec<u32> = code
            .chunks(4)
            .map(|chunk| {
                u32::from_le_bytes([
                    chunk.get(0).copied().unwrap_or(0),
                    chunk.get(1).copied().unwrap_or(0),
                    chunk.get(2).copied().unwrap_or(0),
                    chunk.get(3).copied().unwrap_or(0),
                ])
            })
            .collect();

        let create_info = vk::ShaderModuleCreateInfo {
            code_size: code.len(),
            p_code: code_aligned.as_ptr(),
            ..Default::default()
        };

        unsafe {
            device
                .create_shader_module(&create_info, None)
                .context("Failed to create shader module")
        }
    }

    fn create_buffer(
        device: &ash::Device,
        allocator: &Arc<Mutex<Allocator>>,
        size: usize,
        usage: vk::BufferUsageFlags,
    ) -> Result<(vk::Buffer, Allocation)> {
        let buffer_info = vk::BufferCreateInfo {
            size: size as u64,
            usage,
            sharing_mode: vk::SharingMode::EXCLUSIVE,
            ..Default::default()
        };

        let buffer = unsafe { device.create_buffer(&buffer_info, None)? };
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };

        let allocation = allocator
            .lock()
            .unwrap()
            .allocate(&AllocationCreateDesc {
                name: "egui_buffer",
                requirements,
                location: MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;

        unsafe {
            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
        }

        Ok((buffer, allocation))
    }

    fn create_font_texture(
        device: &ash::Device,
        allocator: &Arc<Mutex<Allocator>>,
        queue: vk::Queue,
        command_pool: vk::CommandPool,
        width: u32,
        height: u32,
        pixels: &[u8],
    ) -> Result<(vk::Image, Allocation, vk::ImageView)> {
        unsafe {
            // Create image
            let image_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UNORM,
                extent: vk::Extent3D { width, height, depth: 1 },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };

            let image = device.create_image(&image_info, None)?;
            let requirements = device.get_image_memory_requirements(image);

            let allocation = allocator
                .lock()
                .unwrap()
                .allocate(&AllocationCreateDesc {
                    name: "egui_font_texture",
                    requirements,
                    location: MemoryLocation::GpuOnly,
                    linear: false,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })?;

            device.bind_image_memory(image, allocation.memory(), allocation.offset())?;

            // Create staging buffer
            let staging_size = (width * height * 4) as usize;
            let staging_info = vk::BufferCreateInfo {
                size: staging_size as u64,
                usage: vk::BufferUsageFlags::TRANSFER_SRC,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };

            let staging_buffer = device.create_buffer(&staging_info, None)?;
            let staging_req = device.get_buffer_memory_requirements(staging_buffer);

            let staging_alloc = allocator
                .lock()
                .unwrap()
                .allocate(&AllocationCreateDesc {
                    name: "egui_staging",
                    requirements: staging_req,
                    location: MemoryLocation::CpuToGpu,
                    linear: true,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })?;

            device.bind_buffer_memory(staging_buffer, staging_alloc.memory(), staging_alloc.offset())?;

            // Copy data to staging
            let ptr = staging_alloc.mapped_ptr().unwrap().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), ptr, pixels.len().min(staging_size));

            // Create command buffer for transfer
            let cmd_alloc_info = vk::CommandBufferAllocateInfo {
                command_pool,
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };

            let cmd = device.allocate_command_buffers(&cmd_alloc_info)?[0];

            let begin_info = vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            };

            device.begin_command_buffer(cmd, &begin_info)?;

            // Transition to transfer dst
            let barrier = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::UNDEFINED,
                new_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                src_access_mask: vk::AccessFlags::empty(),
                dst_access_mask: vk::AccessFlags::TRANSFER_WRITE,
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
            let region = vk::BufferImageCopy {
                image_subresource: vk::ImageSubresourceLayers {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    layer_count: 1,
                    ..Default::default()
                },
                image_extent: vk::Extent3D { width, height, depth: 1 },
                ..Default::default()
            };

            device.cmd_copy_buffer_to_image(
                cmd,
                staging_buffer,
                image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[region],
            );

            // Transition to shader read
            let barrier = vk::ImageMemoryBarrier {
                old_layout: vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                image,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                src_access_mask: vk::AccessFlags::TRANSFER_WRITE,
                dst_access_mask: vk::AccessFlags::SHADER_READ,
                ..Default::default()
            };

            device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
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

            // Cleanup staging
            device.free_command_buffers(command_pool, &[cmd]);
            device.destroy_buffer(staging_buffer, None);
            allocator.lock().unwrap().free(staging_alloc)?;

            // Create image view
            let view_info = vk::ImageViewCreateInfo {
                image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: vk::Format::R8G8B8A8_UNORM,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    level_count: 1,
                    layer_count: 1,
                    ..Default::default()
                },
                ..Default::default()
            };

            let image_view = device.create_image_view(&view_info, None)?;

            Ok((image, allocation, image_view))
        }
    }

    /// Apply a `TexturesDelta` produced by `egui_ctx.run()`.
    ///
    /// egui 0.29 puts all pending texture updates (including the font atlas) into
    /// `output.textures_delta` inside `ctx.run()`. Calling `ctx.fonts()` afterward
    /// returns nothing because the delta has already been consumed.
    ///
    /// Each `ImageDelta` has two cases:
    ///   pos == None   → full atlas rebuild  → recreate the GPU texture
    ///   pos == Some   → partial glyph patch → blit new pixels at offset
    pub fn apply_textures_delta(&mut self, delta: &egui::TexturesDelta) -> Result<()> {
        for (_id, image_delta) in &delta.set {
            self.apply_image_delta(image_delta)?;
        }
        Ok(())
    }

    fn apply_image_delta(&mut self, delta: &egui::epaint::ImageDelta) -> Result<()> {
        let pixels: Vec<u8> = match &delta.image {
            egui::ImageData::Color(img) => {
                img.pixels.iter().flat_map(|c| c.to_array()).collect()
            }
            egui::ImageData::Font(img) => {
                img.srgba_pixels(None).flat_map(|c| c.to_array()).collect()
            }
        };

        let patch_w = delta.image.width()  as u32;
        let patch_h = delta.image.height() as u32;

        if let Some([ox, oy]) = delta.pos {
            // Partial update: blit new glyph pixels into the existing texture.
            self.blit_font_patch(ox as u32, oy as u32, patch_w, patch_h, &pixels)?;
        } else {
            // Full update: destroy old texture, allocate new one at the new size.
            unsafe {
                self.device.device_wait_idle()?;
                self.device.destroy_image_view(self.font_image_view, None);
                self.device.destroy_image(self.font_image, None);
                if let Some(alloc) = self.font_allocation.take() {
                    self.allocator.lock().unwrap().free(alloc)?;
                }
            }

            let (img, alloc, view) = Self::create_font_texture(
                &self.device, &self.allocator, self.queue, self.command_pool,
                patch_w, patch_h, &pixels,
            )?;
            self.font_image      = img;
            self.font_allocation = Some(alloc);
            self.font_image_view = view;

            let image_info = vk::DescriptorImageInfo {
                sampler:      self.font_sampler,
                image_view:   self.font_image_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            };
            let write = vk::WriteDescriptorSet {
                dst_set:          self.descriptor_set,
                dst_binding:      0,
                descriptor_count: 1,
                descriptor_type:  vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info:     &image_info,
                ..Default::default()
            };
            unsafe { self.device.update_descriptor_sets(&[write], &[]); }

            self.font_version += 1;
        }

        Ok(())
    }

    /// Blit a rectangular patch of pixels into the existing font texture at (offset_x, offset_y).
    fn blit_font_patch(
        &self,
        offset_x: u32, offset_y: u32,
        width: u32,    height: u32,
        pixels: &[u8],
    ) -> Result<()> {
        unsafe {
            // Staging buffer
            let staging_size = (width * height * 4) as usize;
            let staging_buf = self.device.create_buffer(
                &vk::BufferCreateInfo {
                    size:         staging_size as u64,
                    usage:        vk::BufferUsageFlags::TRANSFER_SRC,
                    sharing_mode: vk::SharingMode::EXCLUSIVE,
                    ..Default::default()
                },
                None,
            )?;
            let staging_req = self.device.get_buffer_memory_requirements(staging_buf);
            let staging_alloc = self.allocator.lock().unwrap().allocate(&AllocationCreateDesc {
                name:              "egui_font_patch_staging",
                requirements:      staging_req,
                location:          MemoryLocation::CpuToGpu,
                linear:            true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })?;
            self.device.bind_buffer_memory(staging_buf, staging_alloc.memory(), staging_alloc.offset())?;

            let ptr = staging_alloc.mapped_ptr().unwrap().as_ptr() as *mut u8;
            std::ptr::copy_nonoverlapping(pixels.as_ptr(), ptr, pixels.len().min(staging_size));

            // One-shot command buffer
            let cmd = self.device.allocate_command_buffers(&vk::CommandBufferAllocateInfo {
                command_pool:        self.command_pool,
                level:               vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            })?[0];
            self.device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            })?;

            let subresource = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                level_count: 1,
                layer_count: 1,
                ..Default::default()
            };

            // SHADER_READ_ONLY → TRANSFER_DST
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(), &[], &[],
                &[vk::ImageMemoryBarrier {
                    old_layout:            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    new_layout:            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image:                 self.font_image,
                    subresource_range:     subresource,
                    src_access_mask:       vk::AccessFlags::SHADER_READ,
                    dst_access_mask:       vk::AccessFlags::TRANSFER_WRITE,
                    ..Default::default()
                }],
            );

            // Copy patch at offset
            self.device.cmd_copy_buffer_to_image(
                cmd, staging_buf, self.font_image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &[vk::BufferImageCopy {
                    image_subresource: vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        layer_count: 1,
                        ..Default::default()
                    },
                    image_offset: vk::Offset3D { x: offset_x as i32, y: offset_y as i32, z: 0 },
                    image_extent: vk::Extent3D { width, height, depth: 1 },
                    ..Default::default()
                }],
            );

            // TRANSFER_DST → SHADER_READ_ONLY
            self.device.cmd_pipeline_barrier(
                cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(), &[], &[],
                &[vk::ImageMemoryBarrier {
                    old_layout:            vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                    new_layout:            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    src_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    dst_queue_family_index: vk::QUEUE_FAMILY_IGNORED,
                    image:                 self.font_image,
                    subresource_range:     subresource,
                    src_access_mask:       vk::AccessFlags::TRANSFER_WRITE,
                    dst_access_mask:       vk::AccessFlags::SHADER_READ,
                    ..Default::default()
                }],
            );

            self.device.end_command_buffer(cmd)?;
            self.device.queue_submit(
                self.queue,
                &[vk::SubmitInfo {
                    command_buffer_count: 1,
                    p_command_buffers:    &cmd,
                    ..Default::default()
                }],
                vk::Fence::null(),
            )?;
            self.device.queue_wait_idle(self.queue)?;
            self.device.free_command_buffers(self.command_pool, &[cmd]);

            // Free staging resources
            self.device.destroy_buffer(staging_buf, None);
            self.allocator.lock().unwrap().free(staging_alloc)?;
        }
        Ok(())
    }

    /// Render egui output
    pub fn render(
        &mut self,
        command_buffer: vk::CommandBuffer,
        clipped_primitives: &[egui::ClippedPrimitive],
        screen_size: [f32; 2],
        pixels_per_point: f32,
    ) -> Result<()> {
        if clipped_primitives.is_empty() {
            return Ok(());
        }

        // Collect all vertices and indices directly from egui
        let mut all_vertices: Vec<u8> = Vec::new();
        let mut all_indices: Vec<u32> = Vec::new();
        let mut mesh_ranges: Vec<(u32, u32, egui::Rect)> = Vec::new(); // (index_offset, index_count, clip_rect)

        for primitive in clipped_primitives {
            if let egui::epaint::Primitive::Mesh(mesh) = &primitive.primitive {
                if mesh.vertices.is_empty() || mesh.indices.is_empty() {
                    continue;
                }

                let base_vertex = (all_vertices.len() / 20) as u32;
                let index_offset = all_indices.len() as u32;

                // Copy vertices directly as raw bytes (egui::epaint::Vertex is 20 bytes)
                for v in &mesh.vertices {
                    all_vertices.extend_from_slice(&v.pos.x.to_le_bytes());
                    all_vertices.extend_from_slice(&v.pos.y.to_le_bytes());
                    all_vertices.extend_from_slice(&v.uv.x.to_le_bytes());
                    all_vertices.extend_from_slice(&v.uv.y.to_le_bytes());
                    all_vertices.extend_from_slice(&v.color.to_array());
                }

                // Copy indices with base vertex offset
                for &idx in &mesh.indices {
                    all_indices.push(base_vertex + idx);
                }

                mesh_ranges.push((index_offset, mesh.indices.len() as u32, primitive.clip_rect));
            }
        }

        if all_vertices.is_empty() {
            return Ok(());
        }

        // Ensure buffers are large enough
        let vertex_count = all_vertices.len() / 20;
        self.ensure_buffer_capacity(vertex_count, all_indices.len())?;

        // Upload vertex data
        {
            let alloc = self.vertex_allocation.as_ref().unwrap();
            let ptr = alloc.mapped_ptr().unwrap().as_ptr() as *mut u8;
            unsafe {
                std::ptr::copy_nonoverlapping(all_vertices.as_ptr(), ptr, all_vertices.len());
            }
        }

        // Upload index data
        {
            let alloc = self.index_allocation.as_ref().unwrap();
            let ptr = alloc.mapped_ptr().unwrap().as_ptr() as *mut u32;
            unsafe {
                std::ptr::copy_nonoverlapping(all_indices.as_ptr(), ptr, all_indices.len());
            }
        }

        unsafe {
            // Bind pipeline
            self.device.cmd_bind_pipeline(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline,
            );

            // Bind descriptor set
            self.device.cmd_bind_descriptor_sets(
                command_buffer,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            // egui tessellate() outputs vertex positions in logical pixels (points),
            // not physical pixels. The shader does NDC = 2*pos/screen_size - 1,
            // so screen_size must match: logical = physical / pixels_per_point.
            // Viewport and scissor stay in physical pixels (Vulkan convention).
            let logical_size = [
                screen_size[0] / pixels_per_point,
                screen_size[1] / pixels_per_point,
            ];
            log::debug!(
                "egui render: physical={:?} ppp={:.2} push_screen_size={:?}",
                screen_size, pixels_per_point, logical_size
            );
            let push_constants = EguiPushConstants { screen_size: logical_size };
            let pc_bytes = bytemuck::bytes_of(&push_constants);
            self.device.cmd_push_constants(
                command_buffer,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX,
                0,
                pc_bytes,
            );

            // Viewport in physical pixels
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: screen_size[0],
                height: screen_size[1],
                min_depth: 0.0,
                max_depth: 1.0,
            };
            self.device.cmd_set_viewport(command_buffer, 0, &[viewport]);

            // Bind buffers
            self.device.cmd_bind_vertex_buffers(command_buffer, 0, &[self.vertex_buffer], &[0]);
            self.device.cmd_bind_index_buffer(command_buffer, self.index_buffer, 0, vk::IndexType::UINT32);

            // Draw each mesh with its scissor rect (physical pixels)
            let screen_w = screen_size[0] as i32;
            let screen_h = screen_size[1] as i32;

            for (index_offset, index_count, clip_rect) in mesh_ranges {
                // clip_rect is in points; convert to physical pixels for Vulkan scissor
                let min_x = (clip_rect.min.x * pixels_per_point).round() as i32;
                let min_y = (clip_rect.min.y * pixels_per_point).round() as i32;
                let max_x = (clip_rect.max.x * pixels_per_point).round() as i32;
                let max_y = (clip_rect.max.y * pixels_per_point).round() as i32;

                // Clamp to screen
                let x = min_x.max(0).min(screen_w);
                let y = min_y.max(0).min(screen_h);
                let w = (max_x - x).max(0).min(screen_w - x) as u32;
                let h = (max_y - y).max(0).min(screen_h - y) as u32;

                if w == 0 || h == 0 {
                    continue;
                }

                let scissor = vk::Rect2D {
                    offset: vk::Offset2D { x, y },
                    extent: vk::Extent2D { width: w, height: h },
                };
                self.device.cmd_set_scissor(command_buffer, 0, &[scissor]);

                // Draw
                self.device.cmd_draw_indexed(
                    command_buffer,
                    index_count,
                    1,
                    index_offset,
                    0,
                    0,
                );
            }
        }

        Ok(())
    }

    fn ensure_buffer_capacity(&mut self, vertex_count: usize, index_count: usize) -> Result<()> {
        // Grow vertex buffer if needed
        if vertex_count > self.vertex_capacity {
            let new_capacity = (vertex_count * 2).max(64 * 1024);

            unsafe {
                self.device.device_wait_idle()?;
                self.device.destroy_buffer(self.vertex_buffer, None);
                if let Some(alloc) = self.vertex_allocation.take() {
                    self.allocator.lock().unwrap().free(alloc)?;
                }
            }

            let (buffer, allocation) = Self::create_buffer(
                &self.device,
                &self.allocator,
                new_capacity * 20, // 20 bytes per egui vertex
                vk::BufferUsageFlags::VERTEX_BUFFER,
            )?;

            self.vertex_buffer = buffer;
            self.vertex_allocation = Some(allocation);
            self.vertex_capacity = new_capacity;
        }

        // Grow index buffer if needed
        if index_count > self.index_capacity {
            let new_capacity = (index_count * 2).max(128 * 1024);

            unsafe {
                self.device.device_wait_idle()?;
                self.device.destroy_buffer(self.index_buffer, None);
                if let Some(alloc) = self.index_allocation.take() {
                    self.allocator.lock().unwrap().free(alloc)?;
                }
            }

            let (buffer, allocation) = Self::create_buffer(
                &self.device,
                &self.allocator,
                new_capacity * std::mem::size_of::<u32>(),
                vk::BufferUsageFlags::INDEX_BUFFER,
            )?;

            self.index_buffer = buffer;
            self.index_allocation = Some(allocation);
            self.index_capacity = new_capacity;
        }

        Ok(())
    }
}

impl Drop for EguiRenderer {
    fn drop(&mut self) {
        unsafe {
            let _ = self.device.device_wait_idle();

            self.device.destroy_command_pool(self.command_pool, None);

            self.device.destroy_buffer(self.vertex_buffer, None);
            if let Some(alloc) = self.vertex_allocation.take() {
                let _ = self.allocator.lock().unwrap().free(alloc);
            }

            self.device.destroy_buffer(self.index_buffer, None);
            if let Some(alloc) = self.index_allocation.take() {
                let _ = self.allocator.lock().unwrap().free(alloc);
            }

            self.device.destroy_image_view(self.font_image_view, None);
            self.device.destroy_image(self.font_image, None);
            if let Some(alloc) = self.font_allocation.take() {
                let _ = self.allocator.lock().unwrap().free(alloc);
            }

            self.device.destroy_sampler(self.font_sampler, None);
            self.device.destroy_descriptor_pool(self.descriptor_pool, None);
            self.device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);
            self.device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
