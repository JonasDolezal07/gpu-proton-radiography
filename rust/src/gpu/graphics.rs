//! Graphics pipeline for rendering detector and volume

use ash::vk;
use anyhow::{Result, Context};
use std::ffi::CString;
use std::sync::{Arc, Mutex};
use gpu_allocator::vulkan::{Allocator, Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;

/// Volume rendering parameters (push constants)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VolumeParams {
    pub inv_view_proj: [[f32; 4]; 4],  // Inverse view-projection matrix
    pub view_proj: [[f32; 4]; 4],      // View-projection matrix (for depth calculation)
    pub camera_pos: [f32; 4],           // Camera position
    pub volume_min: [f32; 4],           // Volume AABB min
    pub volume_max: [f32; 4],           // Volume AABB max
    pub step_size: f32,
    pub density_scale: f32,
    pub brightness: f32,
    pub num_steps: u32,
}

impl Default for VolumeParams {
    fn default() -> Self {
        Self {
            inv_view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            camera_pos: [0.0, 0.0, -0.5, 1.0],
            volume_min: [-0.1, -0.1, -0.1, 0.0],
            volume_max: [0.1, 0.1, 0.2, 0.0],
            step_size: 0.005,
            density_scale: 0.5,
            brightness: 10.0,
            num_steps: 128,
        }
    }
}

/// Marker parameters for source/target visualization (push constants)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MarkerParams {
    pub view_proj: [[f32; 4]; 4],
    pub position: [f32; 4],  // xyz = position, w = size
    pub color: [f32; 4],     // rgba
}

impl Default for MarkerParams {
    fn default() -> Self {
        Self {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            position: [0.0, 0.0, 0.0, 0.01],  // 1cm default size
            color: [1.0, 0.2, 0.1, 1.0],      // Red
        }
    }
}

/// Display parameters for detector shader (push constants)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DisplayParams {
    pub max_count: f32,
    pub gamma: f32,
    pub exposure: f32,
    pub use_log_scale: u32,
    pub colormap_mode: u32,  // 0 = RCF film (realistic), 1 = scientific (dark->light)
}

impl Default for DisplayParams {
    fn default() -> Self {
        Self {
            max_count: 1000.0,
            gamma: 0.5,  // sqrt for better dynamic range
            exposure: 1.0,
            use_log_scale: 0,
            colormap_mode: 0,  // Default to realistic RCF film
        }
    }
}

/// 3D detector parameters (push constants for 3D detector rendering)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Detector3DParams {
    pub view_proj: [[f32; 4]; 4],      // Camera view-projection matrix
    pub detector_pos: [f32; 4],         // Detector center position
    pub detector_normal: [f32; 4],      // Detector facing direction
    pub detector_extent: [f32; 4],      // Half-size (x, y, unused, unused)
    // Display params
    pub max_count: f32,
    pub gamma: f32,
    pub exposure: f32,
    pub use_log_scale: u32,
    pub colormap_mode: u32,            // 0 = RCF film (realistic), 1 = scientific
}

impl Default for Detector3DParams {
    fn default() -> Self {
        Self {
            view_proj: [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            detector_pos: [0.0, 0.0, 0.15, 1.0],
            detector_normal: [0.0, 0.0, 1.0, 0.0],
            detector_extent: [0.05, 0.05, 0.0, 0.0],
            max_count: 1000.0,
            gamma: 0.5,
            exposure: 1.0,
            use_log_scale: 0,
            colormap_mode: 0,  // Default to RCF film
        }
    }
}

/// Graphics pipeline for detector display
pub struct DetectorPipeline {
    render_pass: vk::RenderPass,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    framebuffers: Vec<vk::Framebuffer>,
    sampler: vk::Sampler,
    // Depth buffer resources
    depth_image: vk::Image,
    depth_allocation: Option<Allocation>,
    depth_image_view: vk::ImageView,
    depth_format: vk::Format,
}

impl DetectorPipeline {
    pub fn new(
        device: &ash::Device,
        allocator: &Arc<Mutex<Allocator>>,
        swapchain_format: vk::Format,
        swapchain_extent: vk::Extent2D,
        swapchain_image_views: &[vk::ImageView],
        vert_shader: &[u8],
        frag_shader: &[u8],
    ) -> Result<Self> {
        unsafe {
            // Choose depth format
            let depth_format = vk::Format::D32_SFLOAT;

            // Create depth image
            let depth_image_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: depth_format,
                extent: vk::Extent3D {
                    width: swapchain_extent.width,
                    height: swapchain_extent.height,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };

            let depth_image = device.create_image(&depth_image_info, None)
                .context("Failed to create depth image")?;

            let mem_requirements = device.get_image_memory_requirements(depth_image);

            let depth_allocation = allocator
                .lock()
                .unwrap()
                .allocate(&AllocationCreateDesc {
                    name: "depth_buffer",
                    requirements: mem_requirements,
                    location: MemoryLocation::GpuOnly,
                    linear: false,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .context("Failed to allocate depth buffer memory")?;

            device.bind_image_memory(depth_image, depth_allocation.memory(), depth_allocation.offset())
                .context("Failed to bind depth image memory")?;

            // Create depth image view
            let depth_view_info = vk::ImageViewCreateInfo {
                image: depth_image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: depth_format,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };

            let depth_image_view = device.create_image_view(&depth_view_info, None)
                .context("Failed to create depth image view")?;

            // Create render pass with depth attachment
            let attachments = [
                vk::AttachmentDescription {
                    format: swapchain_format,
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::STORE,
                    stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                    stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: vk::ImageLayout::PRESENT_SRC_KHR,
                    ..Default::default()
                },
                vk::AttachmentDescription {
                    format: depth_format,
                    samples: vk::SampleCountFlags::TYPE_1,
                    load_op: vk::AttachmentLoadOp::CLEAR,
                    store_op: vk::AttachmentStoreOp::DONT_CARE,
                    stencil_load_op: vk::AttachmentLoadOp::DONT_CARE,
                    stencil_store_op: vk::AttachmentStoreOp::DONT_CARE,
                    initial_layout: vk::ImageLayout::UNDEFINED,
                    final_layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
                    ..Default::default()
                },
            ];

            let color_ref = vk::AttachmentReference {
                attachment: 0,
                layout: vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            };

            let depth_ref = vk::AttachmentReference {
                attachment: 1,
                layout: vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            };

            let subpass = vk::SubpassDescription {
                pipeline_bind_point: vk::PipelineBindPoint::GRAPHICS,
                color_attachment_count: 1,
                p_color_attachments: &color_ref,
                p_depth_stencil_attachment: &depth_ref,
                ..Default::default()
            };

            let dependencies = [
                vk::SubpassDependency {
                    src_subpass: vk::SUBPASS_EXTERNAL,
                    dst_subpass: 0,
                    src_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    dst_stage_mask: vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT
                        | vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
                    src_access_mask: vk::AccessFlags::empty(),
                    dst_access_mask: vk::AccessFlags::COLOR_ATTACHMENT_WRITE
                        | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                    ..Default::default()
                },
            ];

            let render_pass_info = vk::RenderPassCreateInfo {
                attachment_count: attachments.len() as u32,
                p_attachments: attachments.as_ptr(),
                subpass_count: 1,
                p_subpasses: &subpass,
                dependency_count: dependencies.len() as u32,
                p_dependencies: dependencies.as_ptr(),
                ..Default::default()
            };

            let render_pass = device.create_render_pass(&render_pass_info, None)
                .context("Failed to create render pass")?;

            // Create framebuffers with depth attachment
            let framebuffers = Self::create_framebuffers(
                device,
                render_pass,
                swapchain_image_views,
                depth_image_view,
                swapchain_extent,
            )?;

            // Create sampler for detector texture
            let sampler_info = vk::SamplerCreateInfo {
                mag_filter: vk::Filter::NEAREST,  // No interpolation for counts
                min_filter: vk::Filter::NEAREST,
                address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                ..Default::default()
            };
            let sampler = device.create_sampler(&sampler_info, None)?;

            // Descriptor set layout: binding 0 = detector texture
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

            let descriptor_set_layout = device.create_descriptor_set_layout(&layout_info, None)?;

            // Push constants for 3D detector params (used by both vertex and fragment)
            let push_constant_range = vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                offset: 0,
                size: std::mem::size_of::<Detector3DParams>() as u32,
            };

            let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
                set_layout_count: 1,
                p_set_layouts: &descriptor_set_layout,
                push_constant_range_count: 1,
                p_push_constant_ranges: &push_constant_range,
                ..Default::default()
            };

            let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)?;

            // Create shader modules
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

            // No vertex input (full-screen triangle generated in shader)
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            };

            // Dynamic viewport/scissor
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state = vk::PipelineDynamicStateCreateInfo {
                dynamic_state_count: dynamic_states.len() as u32,
                p_dynamic_states: dynamic_states.as_ptr(),
                ..Default::default()
            };

            let viewport_state = vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                scissor_count: 1,
                ..Default::default()
            };

            let rasterizer = vk::PipelineRasterizationStateCreateInfo {
                polygon_mode: vk::PolygonMode::FILL,
                cull_mode: vk::CullModeFlags::NONE,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                ..Default::default()
            };

            let multisampling = vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };

            // Alpha blending for transparent detector overlay
            let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::TRUE,
                src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            };

            let color_blending = vk::PipelineColorBlendStateCreateInfo {
                logic_op_enable: vk::FALSE,
                attachment_count: 1,
                p_attachments: &color_blend_attachment,
                ..Default::default()
            };

            // Depth-stencil state: enable depth testing and writing
            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: vk::TRUE,
                depth_write_enable: vk::TRUE,
                depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
                depth_bounds_test_enable: vk::FALSE,
                stencil_test_enable: vk::FALSE,
                ..Default::default()
            };

            let pipeline_info = vk::GraphicsPipelineCreateInfo {
                stage_count: 2,
                p_stages: shader_stages.as_ptr(),
                p_vertex_input_state: &vertex_input,
                p_input_assembly_state: &input_assembly,
                p_viewport_state: &viewport_state,
                p_rasterization_state: &rasterizer,
                p_multisample_state: &multisampling,
                p_depth_stencil_state: &depth_stencil,
                p_color_blend_state: &color_blending,
                p_dynamic_state: &dynamic_state,
                layout: pipeline_layout,
                render_pass,
                subpass: 0,
                ..Default::default()
            };

            let pipelines = device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_pipes, e)| e)
                .context("Failed to create graphics pipeline")?;

            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);

            // Create descriptor pool
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

            let descriptor_pool = device.create_descriptor_pool(&pool_info, None)?;

            // Allocate descriptor set
            let alloc_info = vk::DescriptorSetAllocateInfo {
                descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: &descriptor_set_layout,
                ..Default::default()
            };

            let descriptor_sets = device.allocate_descriptor_sets(&alloc_info)?;

            log::info!("Created detector graphics pipeline");

            Ok(Self {
                render_pass,
                pipeline: pipelines[0],
                pipeline_layout,
                descriptor_set_layout,
                descriptor_pool,
                descriptor_set: descriptor_sets[0],
                framebuffers,
                sampler,
                depth_image,
                depth_allocation: Some(depth_allocation),
                depth_image_view,
                depth_format,
            })
        }
    }

    fn create_framebuffers(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        image_views: &[vk::ImageView],
        depth_view: vk::ImageView,
        extent: vk::Extent2D,
    ) -> Result<Vec<vk::Framebuffer>> {
        image_views
            .iter()
            .map(|&view| {
                // Color attachment + depth attachment
                let attachments = [view, depth_view];
                let info = vk::FramebufferCreateInfo {
                    render_pass,
                    attachment_count: 2,
                    p_attachments: attachments.as_ptr(),
                    width: extent.width,
                    height: extent.height,
                    layers: 1,
                    ..Default::default()
                };
                unsafe { device.create_framebuffer(&info, None) }
                    .context("Failed to create framebuffer")
            })
            .collect()
    }

    unsafe fn create_shader_module(device: &ash::Device, code: &[u8]) -> Result<vk::ShaderModule> {
        let code_aligned: Vec<u32> = code
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let create_info = vk::ShaderModuleCreateInfo {
            code_size: code.len(),
            p_code: code_aligned.as_ptr(),
            ..Default::default()
        };

        device.create_shader_module(&create_info, None)
            .context("Failed to create shader module")
    }

    pub fn update_descriptor(&self, device: &ash::Device, detector_view: vk::ImageView) {
        let image_info = vk::DescriptorImageInfo {
            sampler: self.sampler,
            image_view: detector_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        let write = vk::WriteDescriptorSet {
            dst_set: self.descriptor_set,
            dst_binding: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info: &image_info,
            ..Default::default()
        };

        unsafe {
            device.update_descriptor_sets(&[write], &[]);
        }
    }

    pub fn recreate_framebuffers(
        &mut self,
        device: &ash::Device,
        allocator: &Arc<Mutex<Allocator>>,
        image_views: &[vk::ImageView],
        extent: vk::Extent2D,
    ) -> Result<()> {
        unsafe {
            // Destroy old framebuffers
            for fb in &self.framebuffers {
                device.destroy_framebuffer(*fb, None);
            }

            // Destroy old depth buffer
            device.destroy_image_view(self.depth_image_view, None);
            if let Some(alloc) = self.depth_allocation.take() {
                allocator.lock().unwrap().free(alloc).ok();
            }
            device.destroy_image(self.depth_image, None);

            // Create new depth buffer
            let depth_image_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_2D,
                format: self.depth_format,
                extent: vk::Extent3D {
                    width: extent.width,
                    height: extent.height,
                    depth: 1,
                },
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };

            self.depth_image = device.create_image(&depth_image_info, None)
                .context("Failed to create depth image")?;

            let mem_requirements = device.get_image_memory_requirements(self.depth_image);

            let depth_allocation = allocator
                .lock()
                .unwrap()
                .allocate(&AllocationCreateDesc {
                    name: "depth_buffer",
                    requirements: mem_requirements,
                    location: MemoryLocation::GpuOnly,
                    linear: false,
                    allocation_scheme: AllocationScheme::GpuAllocatorManaged,
                })
                .context("Failed to allocate depth buffer memory")?;

            device.bind_image_memory(self.depth_image, depth_allocation.memory(), depth_allocation.offset())
                .context("Failed to bind depth image memory")?;

            let depth_view_info = vk::ImageViewCreateInfo {
                image: self.depth_image,
                view_type: vk::ImageViewType::TYPE_2D,
                format: self.depth_format,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::DEPTH,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };

            self.depth_image_view = device.create_image_view(&depth_view_info, None)
                .context("Failed to create depth image view")?;

            self.depth_allocation = Some(depth_allocation);
        }

        self.framebuffers = Self::create_framebuffers(
            device,
            self.render_pass,
            image_views,
            self.depth_image_view,
            extent,
        )?;
        Ok(())
    }

    pub fn render_pass(&self) -> vk::RenderPass {
        self.render_pass
    }

    pub fn begin_render_pass(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        image_index: usize,
        extent: vk::Extent2D,
    ) {
        unsafe {
            // Clear values for color and depth attachments
            let clear_values = [
                vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.02, 0.02, 0.08, 1.0],
                    },
                },
                vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                },
            ];

            let render_pass_info = vk::RenderPassBeginInfo {
                render_pass: self.render_pass,
                framebuffer: self.framebuffers[image_index],
                render_area: vk::Rect2D {
                    offset: vk::Offset2D { x: 0, y: 0 },
                    extent,
                },
                clear_value_count: clear_values.len() as u32,
                p_clear_values: clear_values.as_ptr(),
                ..Default::default()
            };

            device.cmd_begin_render_pass(cmd, &render_pass_info, vk::SubpassContents::INLINE);
        }
    }

    pub fn end_render_pass(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        unsafe {
            device.cmd_end_render_pass(cmd);
        }
    }

    pub fn draw(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        extent: vk::Extent2D,
        params: &Detector3DParams,
    ) {
        unsafe {
            // Set viewport and scissor dynamically
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(cmd, 0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            };
            device.cmd_set_scissor(cmd, 0, &[scissor]);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(params),
            );

            // Draw 3D detector quad (6 vertices for 2 triangles)
            device.cmd_draw(cmd, 6, 1, 0, 0);
        }
    }

    pub fn record_commands(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        image_index: usize,
        extent: vk::Extent2D,
        params: &Detector3DParams,
    ) {
        self.begin_render_pass(device, cmd, image_index, extent);
        self.draw(device, cmd, extent, params);
        self.end_render_pass(device, cmd);
    }

    pub fn cleanup(&mut self, device: &ash::Device, allocator: &Arc<Mutex<Allocator>>) {
        unsafe {
            for fb in &self.framebuffers {
                device.destroy_framebuffer(*fb, None);
            }
            device.destroy_sampler(self.sampler, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            device.destroy_render_pass(self.render_pass, None);

            // Clean up depth buffer
            device.destroy_image_view(self.depth_image_view, None);
            if let Some(alloc) = self.depth_allocation.take() {
                allocator.lock().unwrap().free(alloc).ok();
            }
            device.destroy_image(self.depth_image, None);
        }
    }
}

/// Graphics pipeline for volume rendering
pub struct VolumePipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
    sampler: vk::Sampler,
}

impl VolumePipeline {
    pub fn new(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        vert_shader: &[u8],
        frag_shader: &[u8],
    ) -> Result<Self> {
        unsafe {
            // Create sampler for field texture (trilinear filtering)
            let sampler_info = vk::SamplerCreateInfo {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                ..Default::default()
            };
            let sampler = device.create_sampler(&sampler_info, None)?;

            // Descriptor set layout: binding 0 = 3D field texture
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

            let descriptor_set_layout = device.create_descriptor_set_layout(&layout_info, None)?;

            // Push constants for volume params
            let push_constant_range = vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::FRAGMENT,
                offset: 0,
                size: std::mem::size_of::<VolumeParams>() as u32,
            };

            let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
                set_layout_count: 1,
                p_set_layouts: &descriptor_set_layout,
                push_constant_range_count: 1,
                p_push_constant_ranges: &push_constant_range,
                ..Default::default()
            };

            let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)?;

            // Create shader modules
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

            // No vertex input (full-screen triangle)
            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            };

            // Dynamic viewport/scissor
            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state = vk::PipelineDynamicStateCreateInfo {
                dynamic_state_count: dynamic_states.len() as u32,
                p_dynamic_states: dynamic_states.as_ptr(),
                ..Default::default()
            };

            let viewport_state = vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                scissor_count: 1,
                ..Default::default()
            };

            let rasterizer = vk::PipelineRasterizationStateCreateInfo {
                polygon_mode: vk::PolygonMode::FILL,
                cull_mode: vk::CullModeFlags::NONE,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                ..Default::default()
            };

            let multisampling = vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };

            // Alpha blending for transparent volume rendering
            let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::TRUE,
                src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            };

            let color_blending = vk::PipelineColorBlendStateCreateInfo {
                logic_op_enable: vk::FALSE,
                attachment_count: 1,
                p_attachments: &color_blend_attachment,
                ..Default::default()
            };

            // Depth-stencil state: test and write depth
            // Volume shader writes gl_FragDepth at volume entry point
            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: vk::TRUE,
                depth_write_enable: vk::TRUE,  // Write depth from gl_FragDepth
                depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
                depth_bounds_test_enable: vk::FALSE,
                stencil_test_enable: vk::FALSE,
                ..Default::default()
            };

            let pipeline_info = vk::GraphicsPipelineCreateInfo {
                stage_count: 2,
                p_stages: shader_stages.as_ptr(),
                p_vertex_input_state: &vertex_input,
                p_input_assembly_state: &input_assembly,
                p_viewport_state: &viewport_state,
                p_rasterization_state: &rasterizer,
                p_multisample_state: &multisampling,
                p_depth_stencil_state: &depth_stencil,
                p_color_blend_state: &color_blending,
                p_dynamic_state: &dynamic_state,
                layout: pipeline_layout,
                render_pass,
                subpass: 0,
                ..Default::default()
            };

            let pipelines = device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_pipes, e)| e)
                .context("Failed to create volume pipeline")?;

            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);

            // Create descriptor pool
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

            let descriptor_pool = device.create_descriptor_pool(&pool_info, None)?;

            // Allocate descriptor set
            let alloc_info = vk::DescriptorSetAllocateInfo {
                descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: &descriptor_set_layout,
                ..Default::default()
            };

            let descriptor_sets = device.allocate_descriptor_sets(&alloc_info)?;

            log::info!("Created volume graphics pipeline");

            Ok(Self {
                pipeline: pipelines[0],
                pipeline_layout,
                descriptor_set_layout,
                descriptor_pool,
                descriptor_set: descriptor_sets[0],
                sampler,
            })
        }
    }

    unsafe fn create_shader_module(device: &ash::Device, code: &[u8]) -> Result<vk::ShaderModule> {
        let code_aligned: Vec<u32> = code
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let create_info = vk::ShaderModuleCreateInfo {
            code_size: code.len(),
            p_code: code_aligned.as_ptr(),
            ..Default::default()
        };

        device.create_shader_module(&create_info, None)
            .context("Failed to create shader module")
    }

    pub fn update_descriptor(&self, device: &ash::Device, field_view: vk::ImageView) {
        let image_info = vk::DescriptorImageInfo {
            sampler: self.sampler,
            image_view: field_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        let write = vk::WriteDescriptorSet {
            dst_set: self.descriptor_set,
            dst_binding: 0,
            descriptor_count: 1,
            descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
            p_image_info: &image_info,
            ..Default::default()
        };

        unsafe {
            device.update_descriptor_sets(&[write], &[]);
        }
    }

    pub fn record_commands(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        extent: vk::Extent2D,
        params: &VolumeParams,
    ) {
        unsafe {
            // Set dynamic viewport and scissor
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(cmd, 0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            };
            device.cmd_set_scissor(cmd, 0, &[scissor]);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::GRAPHICS,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );

            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(params),
            );

            // Draw full-screen triangle
            device.cmd_draw(cmd, 3, 1, 0, 0);
        }
    }

    pub fn cleanup(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}

/// Graphics pipeline for rendering source/target markers
pub struct MarkerPipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
}

impl MarkerPipeline {
    pub fn new(
        device: &ash::Device,
        render_pass: vk::RenderPass,
        vert_shader: &[u8],
        frag_shader: &[u8],
    ) -> Result<Self> {
        unsafe {
            // Push constants only (no descriptors needed)
            let push_constant_range = vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                offset: 0,
                size: std::mem::size_of::<MarkerParams>() as u32,
            };

            let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
                push_constant_range_count: 1,
                p_push_constant_ranges: &push_constant_range,
                ..Default::default()
            };

            let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)?;

            // Create shader modules
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

            let vertex_input = vk::PipelineVertexInputStateCreateInfo::default();

            let input_assembly = vk::PipelineInputAssemblyStateCreateInfo {
                topology: vk::PrimitiveTopology::TRIANGLE_LIST,
                ..Default::default()
            };

            let dynamic_states = [vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR];
            let dynamic_state = vk::PipelineDynamicStateCreateInfo {
                dynamic_state_count: dynamic_states.len() as u32,
                p_dynamic_states: dynamic_states.as_ptr(),
                ..Default::default()
            };

            let viewport_state = vk::PipelineViewportStateCreateInfo {
                viewport_count: 1,
                scissor_count: 1,
                ..Default::default()
            };

            let rasterizer = vk::PipelineRasterizationStateCreateInfo {
                polygon_mode: vk::PolygonMode::FILL,
                cull_mode: vk::CullModeFlags::NONE,
                front_face: vk::FrontFace::COUNTER_CLOCKWISE,
                line_width: 1.0,
                ..Default::default()
            };

            let multisampling = vk::PipelineMultisampleStateCreateInfo {
                rasterization_samples: vk::SampleCountFlags::TYPE_1,
                ..Default::default()
            };

            // Alpha blending for glowing marker
            let color_blend_attachment = vk::PipelineColorBlendAttachmentState {
                blend_enable: vk::TRUE,
                src_color_blend_factor: vk::BlendFactor::SRC_ALPHA,
                dst_color_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                color_blend_op: vk::BlendOp::ADD,
                src_alpha_blend_factor: vk::BlendFactor::ONE,
                dst_alpha_blend_factor: vk::BlendFactor::ONE_MINUS_SRC_ALPHA,
                alpha_blend_op: vk::BlendOp::ADD,
                color_write_mask: vk::ColorComponentFlags::R
                    | vk::ColorComponentFlags::G
                    | vk::ColorComponentFlags::B
                    | vk::ColorComponentFlags::A,
            };

            let color_blending = vk::PipelineColorBlendStateCreateInfo {
                logic_op_enable: vk::FALSE,
                attachment_count: 1,
                p_attachments: &color_blend_attachment,
                ..Default::default()
            };

            // Depth test but no write (marker renders on top of volume but behind detector)
            let depth_stencil = vk::PipelineDepthStencilStateCreateInfo {
                depth_test_enable: vk::TRUE,
                depth_write_enable: vk::TRUE,
                depth_compare_op: vk::CompareOp::LESS_OR_EQUAL,
                depth_bounds_test_enable: vk::FALSE,
                stencil_test_enable: vk::FALSE,
                ..Default::default()
            };

            let pipeline_info = vk::GraphicsPipelineCreateInfo {
                stage_count: 2,
                p_stages: shader_stages.as_ptr(),
                p_vertex_input_state: &vertex_input,
                p_input_assembly_state: &input_assembly,
                p_viewport_state: &viewport_state,
                p_rasterization_state: &rasterizer,
                p_multisample_state: &multisampling,
                p_depth_stencil_state: &depth_stencil,
                p_color_blend_state: &color_blending,
                p_dynamic_state: &dynamic_state,
                layout: pipeline_layout,
                render_pass,
                subpass: 0,
                ..Default::default()
            };

            let pipelines = device
                .create_graphics_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_pipes, e)| e)
                .context("Failed to create marker pipeline")?;

            device.destroy_shader_module(vert_module, None);
            device.destroy_shader_module(frag_module, None);

            log::info!("Created marker graphics pipeline");

            Ok(Self {
                pipeline: pipelines[0],
                pipeline_layout,
            })
        }
    }

    unsafe fn create_shader_module(device: &ash::Device, code: &[u8]) -> Result<vk::ShaderModule> {
        let code_aligned: Vec<u32> = code
            .chunks_exact(4)
            .map(|chunk| u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]))
            .collect();

        let create_info = vk::ShaderModuleCreateInfo {
            code_size: code.len(),
            p_code: code_aligned.as_ptr(),
            ..Default::default()
        };

        device.create_shader_module(&create_info, None)
            .context("Failed to create shader module")
    }

    pub fn draw(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        extent: vk::Extent2D,
        params: &MarkerParams,
    ) {
        unsafe {
            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: extent.width as f32,
                height: extent.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            device.cmd_set_viewport(cmd, 0, &[viewport]);

            let scissor = vk::Rect2D {
                offset: vk::Offset2D { x: 0, y: 0 },
                extent,
            };
            device.cmd_set_scissor(cmd, 0, &[scissor]);

            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::GRAPHICS, self.pipeline);

            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::FRAGMENT,
                0,
                bytemuck::bytes_of(params),
            );

            // Draw billboard quad (6 vertices)
            device.cmd_draw(cmd, 6, 1, 0, 0);
        }
    }

    pub fn cleanup(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
        }
    }
}
