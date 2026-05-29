//! Compute pipeline for Boris integrator

use ash::vk;
use anyhow::{Result, Context};
use std::ffi::CString;

/// Simulation parameters pushed to compute shader
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct SimParams {
    pub dt: f32,
    pub q_over_m: f32,  // charge/mass ratio (for protons: e/m_p)
    pub n_particles: u32,
    pub steps_per_dispatch: u32,  // Number of integration steps per dispatch (batching)
    pub max_steps: u32,           // Hard per-particle step budget (exits when exhausted)
    pub density_mode: u32,        // 0 = CSDA energy loss, 1 = opaque absorber
    pub opaque_threshold: f32,    // density [g/cm³] above which opaque mode kills particle
    pub _pad_c: u32,              // pad to reach 32-byte boundary before field_min vec4

    // Field bounds for texture sampling
    pub field_min: [f32; 4],  // xyz + pad
    pub field_max: [f32; 4],  // xyz + pad

    // Detector plane
    pub detector_pos: [f32; 4],    // point on plane
    pub detector_normal: [f32; 4], // normal vector (default: [1,0,0])
    pub detector_extent: [f32; 4], // half-width (y-axis), half-height (z-axis)
    pub detector_up: [f32; 4],     // detector y-axis in world space (default: [0,1,0])
}

/// Compute pipeline for particle integration
pub struct ComputePipeline {
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_pool: vk::DescriptorPool,
    descriptor_set: vk::DescriptorSet,
}

impl ComputePipeline {
    pub fn new(device: &ash::Device, shader_code: &[u8]) -> Result<Self> {
        unsafe {
            // Create shader module
            let shader_module = Self::create_shader_module(device, shader_code)?;

            // Descriptor set layout:
            // binding 0: particle buffer (storage)
            // binding 1: B-field texture (sampled image)
            // binding 2: detector hits buffer (storage)
            // binding 3: detector image (storage image for atomic writes)
            // binding 4: E-field texture (sampled image)
            // binding 5: density texture (sampled image, R32_SFLOAT)
            // binding 6: stopping power table (storage buffer, std430)
            let bindings = [
                vk::DescriptorSetLayoutBinding {
                    binding: 0,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    ..Default::default()
                },
                vk::DescriptorSetLayoutBinding {
                    binding: 1,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    ..Default::default()
                },
                vk::DescriptorSetLayoutBinding {
                    binding: 2,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    ..Default::default()
                },
                vk::DescriptorSetLayoutBinding {
                    binding: 3,
                    descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    ..Default::default()
                },
                vk::DescriptorSetLayoutBinding {
                    binding: 4,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    ..Default::default()
                },
                vk::DescriptorSetLayoutBinding {
                    binding: 5,
                    descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    ..Default::default()
                },
                vk::DescriptorSetLayoutBinding {
                    binding: 6,
                    descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 1,
                    stage_flags: vk::ShaderStageFlags::COMPUTE,
                    ..Default::default()
                },
            ];

            let layout_info = vk::DescriptorSetLayoutCreateInfo {
                binding_count: bindings.len() as u32,
                p_bindings: bindings.as_ptr(),
                ..Default::default()
            };

            let descriptor_set_layout = device.create_descriptor_set_layout(&layout_info, None)
                .context("Failed to create descriptor set layout")?;

            // Push constants for simulation parameters
            let push_constant_range = vk::PushConstantRange {
                stage_flags: vk::ShaderStageFlags::COMPUTE,
                offset: 0,
                size: std::mem::size_of::<SimParams>() as u32,
            };

            let pipeline_layout_info = vk::PipelineLayoutCreateInfo {
                set_layout_count: 1,
                p_set_layouts: &descriptor_set_layout,
                push_constant_range_count: 1,
                p_push_constant_ranges: &push_constant_range,
                ..Default::default()
            };

            let pipeline_layout = device.create_pipeline_layout(&pipeline_layout_info, None)
                .context("Failed to create pipeline layout")?;

            // Create compute pipeline
            let entry_point = CString::new("main").unwrap();
            let stage_info = vk::PipelineShaderStageCreateInfo {
                stage: vk::ShaderStageFlags::COMPUTE,
                module: shader_module,
                p_name: entry_point.as_ptr(),
                ..Default::default()
            };

            let pipeline_info = vk::ComputePipelineCreateInfo {
                stage: stage_info,
                layout: pipeline_layout,
                ..Default::default()
            };

            let pipelines = device.create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_info], None)
                .map_err(|(_pipelines, e)| e)
                .context("Failed to create compute pipeline")?;

            device.destroy_shader_module(shader_module, None);

            // Create descriptor pool
            let pool_sizes = [
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_BUFFER,
                    descriptor_count: 3,  // particle + detector + stopping power
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                    descriptor_count: 3,  // B field + E field + density
                },
                vk::DescriptorPoolSize {
                    ty: vk::DescriptorType::STORAGE_IMAGE,
                    descriptor_count: 1,
                },
            ];

            let pool_info = vk::DescriptorPoolCreateInfo {
                max_sets: 1,
                pool_size_count: pool_sizes.len() as u32,
                p_pool_sizes: pool_sizes.as_ptr(),
                ..Default::default()
            };

            let descriptor_pool = device.create_descriptor_pool(&pool_info, None)
                .context("Failed to create descriptor pool")?;

            // Allocate descriptor set
            let alloc_info = vk::DescriptorSetAllocateInfo {
                descriptor_pool,
                descriptor_set_count: 1,
                p_set_layouts: &descriptor_set_layout,
                ..Default::default()
            };

            let descriptor_sets = device.allocate_descriptor_sets(&alloc_info)
                .context("Failed to allocate descriptor set")?;

            Ok(Self {
                pipeline: pipelines[0],
                pipeline_layout,
                descriptor_set_layout,
                descriptor_pool,
                descriptor_set: descriptor_sets[0],
            })
        }
    }

    unsafe fn create_shader_module(device: &ash::Device, code: &[u8]) -> Result<vk::ShaderModule> {
        // Ensure alignment for SPIR-V
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

    pub fn update_descriptors(
        &self,
        device: &ash::Device,
        particle_buffer: vk::Buffer,
        particle_size: vk::DeviceSize,
        b_field_view: vk::ImageView,
        b_field_sampler: vk::Sampler,
        detector_buffer: vk::Buffer,
        detector_size: vk::DeviceSize,
        detector_image_view: vk::ImageView,
        e_field_view: vk::ImageView,
        e_field_sampler: vk::Sampler,
        density_view: vk::ImageView,
        density_sampler: vk::Sampler,
        stopping_buffer: vk::Buffer,
        stopping_size: vk::DeviceSize,
    ) {
        let particle_info = vk::DescriptorBufferInfo {
            buffer: particle_buffer,
            offset: 0,
            range: particle_size,
        };

        let b_field_info = vk::DescriptorImageInfo {
            sampler: b_field_sampler,
            image_view: b_field_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        let detector_info = vk::DescriptorBufferInfo {
            buffer: detector_buffer,
            offset: 0,
            range: detector_size,
        };

        let detector_image_info = vk::DescriptorImageInfo {
            sampler: vk::Sampler::null(),
            image_view: detector_image_view,
            image_layout: vk::ImageLayout::GENERAL,
        };

        let e_field_info = vk::DescriptorImageInfo {
            sampler: e_field_sampler,
            image_view: e_field_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        let density_info = vk::DescriptorImageInfo {
            sampler: density_sampler,
            image_view: density_view,
            image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
        };

        let stopping_info = vk::DescriptorBufferInfo {
            buffer: stopping_buffer,
            offset: 0,
            range: stopping_size,
        };

        let writes = [
            vk::WriteDescriptorSet {
                dst_set: self.descriptor_set,
                dst_binding: 0,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &particle_info,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: self.descriptor_set,
                dst_binding: 1,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: &b_field_info,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: self.descriptor_set,
                dst_binding: 2,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &detector_info,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: self.descriptor_set,
                dst_binding: 3,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_IMAGE,
                p_image_info: &detector_image_info,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: self.descriptor_set,
                dst_binding: 4,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: &e_field_info,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: self.descriptor_set,
                dst_binding: 5,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                p_image_info: &density_info,
                ..Default::default()
            },
            vk::WriteDescriptorSet {
                dst_set: self.descriptor_set,
                dst_binding: 6,
                descriptor_count: 1,
                descriptor_type: vk::DescriptorType::STORAGE_BUFFER,
                p_buffer_info: &stopping_info,
                ..Default::default()
            },
        ];

        unsafe {
            device.update_descriptor_sets(&writes, &[]);
        }
    }

    pub fn record_dispatch(
        &self,
        device: &ash::Device,
        cmd: vk::CommandBuffer,
        params: &SimParams,
        workgroup_size: u32,
    ) {
        unsafe {
            device.cmd_bind_pipeline(cmd, vk::PipelineBindPoint::COMPUTE, self.pipeline);
            device.cmd_bind_descriptor_sets(
                cmd,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                &[self.descriptor_set],
                &[],
            );
            device.cmd_push_constants(
                cmd,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                bytemuck::bytes_of(params),
            );

            let num_workgroups = (params.n_particles + workgroup_size - 1) / workgroup_size;
            device.cmd_dispatch(cmd, num_workgroups, 1, 1);
        }
    }

    pub fn cleanup(&mut self, device: &ash::Device) {
        unsafe {
            device.destroy_pipeline(self.pipeline, None);
            device.destroy_pipeline_layout(self.pipeline_layout, None);
            device.destroy_descriptor_pool(self.descriptor_pool, None);
            device.destroy_descriptor_set_layout(self.descriptor_set_layout, None);
        }
    }
}
