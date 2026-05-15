//! GPU buffer and texture management using gpu-allocator

use ash::vk;
use anyhow::{Result, Context};
use gpu_allocator::vulkan::{Allocator, Allocation, AllocationCreateDesc, AllocationScheme};
use gpu_allocator::MemoryLocation;
use std::sync::{Arc, Mutex};

/// GPU buffer with automatic memory management
pub struct GpuBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Option<Allocation>,
    pub size: vk::DeviceSize,
}

impl GpuBuffer {
    pub fn new(
        device: &ash::Device,
        allocator: &Arc<Mutex<Allocator>>,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        location: MemoryLocation,
        name: &str,
    ) -> Result<Self> {
        unsafe {
            let buffer_info = vk::BufferCreateInfo {
                size,
                usage,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                ..Default::default()
            };

            let buffer = device.create_buffer(&buffer_info, None)
                .context("Failed to create buffer")?;

            let requirements = device.get_buffer_memory_requirements(buffer);

            let allocation = allocator.lock().unwrap().allocate(&AllocationCreateDesc {
                name,
                requirements,
                location,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            }).context("Failed to allocate buffer memory")?;

            device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())
                .context("Failed to bind buffer memory")?;

            Ok(Self {
                buffer,
                allocation: Some(allocation),
                size,
            })
        }
    }

    /// Write data to buffer (must be CPU-visible)
    pub fn write<T: Copy>(&self, data: &[T]) -> Result<()> {
        let allocation = self.allocation.as_ref().context("Buffer has no allocation")?;
        let ptr = allocation.mapped_ptr().context("Buffer is not mapped")?;

        unsafe {
            let byte_size = std::mem::size_of_val(data);
            if byte_size as vk::DeviceSize > self.size {
                anyhow::bail!("Data size {} exceeds buffer size {}", byte_size, self.size);
            }
            std::ptr::copy_nonoverlapping(data.as_ptr() as *const u8, ptr.as_ptr() as *mut u8, byte_size);
        }
        Ok(())
    }

    /// Read data from buffer (must be CPU-visible)
    pub fn read<T: Copy>(&self, data: &mut [T]) -> Result<()> {
        let allocation = self.allocation.as_ref().context("Buffer has no allocation")?;
        let ptr = allocation.mapped_ptr().context("Buffer is not mapped")?;

        unsafe {
            let byte_size = std::mem::size_of_val(data);
            if byte_size as vk::DeviceSize > self.size {
                anyhow::bail!("Data size {} exceeds buffer size {}", byte_size, self.size);
            }
            std::ptr::copy_nonoverlapping(ptr.as_ptr() as *const u8, data.as_mut_ptr() as *mut u8, byte_size);
        }
        Ok(())
    }

    pub fn cleanup(&mut self, device: &ash::Device, allocator: &Arc<Mutex<Allocator>>) {
        unsafe {
            device.destroy_buffer(self.buffer, None);
        }
        if let Some(allocation) = self.allocation.take() {
            allocator.lock().unwrap().free(allocation).ok();
        }
    }
}

/// 3D texture for field data
pub struct FieldTexture {
    pub image: vk::Image,
    pub view: vk::ImageView,
    pub sampler: vk::Sampler,
    allocation: Option<Allocation>,
    pub extent: vk::Extent3D,
}

impl FieldTexture {
    pub fn new(
        device: &ash::Device,
        allocator: &Arc<Mutex<Allocator>>,
        nx: u32,
        ny: u32,
        nz: u32,
    ) -> Result<Self> {
        unsafe {
            let extent = vk::Extent3D { width: nx, height: ny, depth: nz };

            // Create 3D image for vector field (RGBA32F - xyz + padding)
            let image_info = vk::ImageCreateInfo {
                image_type: vk::ImageType::TYPE_3D,
                format: vk::Format::R32G32B32A32_SFLOAT,
                extent,
                mip_levels: 1,
                array_layers: 1,
                samples: vk::SampleCountFlags::TYPE_1,
                tiling: vk::ImageTiling::OPTIMAL,
                usage: vk::ImageUsageFlags::SAMPLED | vk::ImageUsageFlags::TRANSFER_DST,
                sharing_mode: vk::SharingMode::EXCLUSIVE,
                initial_layout: vk::ImageLayout::UNDEFINED,
                ..Default::default()
            };

            let image = device.create_image(&image_info, None)
                .context("Failed to create field image")?;

            let requirements = device.get_image_memory_requirements(image);

            let allocation = allocator.lock().unwrap().allocate(&AllocationCreateDesc {
                name: "field_texture",
                requirements,
                location: MemoryLocation::GpuOnly,
                linear: false,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            }).context("Failed to allocate field image memory")?;

            device.bind_image_memory(image, allocation.memory(), allocation.offset())
                .context("Failed to bind field image memory")?;

            // Create image view
            let view_info = vk::ImageViewCreateInfo {
                image,
                view_type: vk::ImageViewType::TYPE_3D,
                format: vk::Format::R32G32B32A32_SFLOAT,
                subresource_range: vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                },
                ..Default::default()
            };

            let view = device.create_image_view(&view_info, None)
                .context("Failed to create field image view")?;

            // Create sampler with trilinear interpolation
            let sampler_info = vk::SamplerCreateInfo {
                mag_filter: vk::Filter::LINEAR,
                min_filter: vk::Filter::LINEAR,
                mipmap_mode: vk::SamplerMipmapMode::LINEAR,
                address_mode_u: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_v: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                address_mode_w: vk::SamplerAddressMode::CLAMP_TO_EDGE,
                ..Default::default()
            };

            let sampler = device.create_sampler(&sampler_info, None)
                .context("Failed to create field sampler")?;

            log::info!("Created field texture: {}x{}x{}", nx, ny, nz);

            Ok(Self {
                image,
                view,
                sampler,
                allocation: Some(allocation),
                extent,
            })
        }
    }

    pub fn cleanup(&mut self, device: &ash::Device, allocator: &Arc<Mutex<Allocator>>) {
        unsafe {
            device.destroy_sampler(self.sampler, None);
            device.destroy_image_view(self.view, None);
            device.destroy_image(self.image, None);
        }
        if let Some(allocation) = self.allocation.take() {
            allocator.lock().unwrap().free(allocation).ok();
        }
    }
}

/// Staging buffer for uploads
pub struct StagingBuffer {
    buffer: GpuBuffer,
}

impl StagingBuffer {
    pub fn new(
        device: &ash::Device,
        allocator: &Arc<Mutex<Allocator>>,
        size: vk::DeviceSize,
    ) -> Result<Self> {
        let buffer = GpuBuffer::new(
            device,
            allocator,
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            MemoryLocation::CpuToGpu,
            "staging",
        )?;
        Ok(Self { buffer })
    }

    pub fn write<T: Copy>(&self, data: &[T]) -> Result<()> {
        self.buffer.write(data)
    }

    pub fn buffer(&self) -> vk::Buffer {
        self.buffer.buffer
    }

    pub fn cleanup(&mut self, device: &ash::Device, allocator: &Arc<Mutex<Allocator>>) {
        self.buffer.cleanup(device, allocator);
    }
}
