//! Swapchain management for rendering

use ash::{vk, khr};
use anyhow::{Result, Context};

pub struct Swapchain {
    loader: khr::swapchain::Device,
    swapchain: vk::SwapchainKHR,
    images: Vec<vk::Image>,
    image_views: Vec<vk::ImageView>,
    pub format: vk::Format,
    pub extent: vk::Extent2D,
}

impl Swapchain {
    pub fn new(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface_loader: &khr::surface::Instance,
        surface: vk::SurfaceKHR,
        graphics_family: u32,
        width: u32,
        height: u32,
    ) -> Result<Self> {
        unsafe { Self::create(instance, device, physical_device, surface_loader, surface, graphics_family, width, height, None) }
    }

    pub fn recreate(
        &mut self,
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface_loader: &khr::surface::Instance,
        surface: vk::SurfaceKHR,
        graphics_family: u32,
        width: u32,
        height: u32,
    ) -> Result<()> {
        unsafe {
            device.device_wait_idle()?;
            let old_swapchain = self.swapchain;

            // Clean up old image views
            for &view in &self.image_views {
                device.destroy_image_view(view, None);
            }
            self.image_views.clear();
            self.images.clear();

            // Create new swapchain with old as predecessor
            let caps = surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?;
            let formats = surface_loader.get_physical_device_surface_formats(physical_device, surface)?;
            let present_modes = surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?;

            let format = formats
                .iter()
                .find(|f| f.format == vk::Format::B8G8R8A8_SRGB && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
                .or_else(|| formats.iter().find(|f| f.format == vk::Format::B8G8R8A8_UNORM))
                .unwrap_or(&formats[0]);

            let present_mode = present_modes
                .iter()
                .copied()
                .find(|&m| m == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let extent = if caps.current_extent.width != u32::MAX {
                caps.current_extent
            } else {
                vk::Extent2D {
                    width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                    height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
                }
            };

            let image_count = (caps.min_image_count + 1).min(
                if caps.max_image_count == 0 { u32::MAX } else { caps.max_image_count }
            );

            let create_info = vk::SwapchainCreateInfoKHR {
                surface,
                min_image_count: image_count,
                image_format: format.format,
                image_color_space: format.color_space,
                image_extent: extent,
                image_array_layers: 1,
                image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
                image_sharing_mode: vk::SharingMode::EXCLUSIVE,
                queue_family_index_count: 1,
                p_queue_family_indices: &graphics_family,
                pre_transform: caps.current_transform,
                composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
                present_mode,
                clipped: vk::TRUE,
                old_swapchain,
                ..Default::default()
            };

            let new_swapchain = self.loader.create_swapchain(&create_info, None)
                .context("Failed to recreate swapchain")?;

            // Destroy old swapchain
            self.loader.destroy_swapchain(old_swapchain, None);

            self.swapchain = new_swapchain;
            self.images = self.loader.get_swapchain_images(new_swapchain)?;
            self.format = format.format;
            self.extent = extent;

            // Create new image views
            for &image in &self.images {
                let view_info = vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: self.format,
                    components: vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    },
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                };
                self.image_views.push(device.create_image_view(&view_info, None)?);
            }

            log::info!("Recreated swapchain: {}x{}", extent.width, extent.height);
        }
        Ok(())
    }

    unsafe fn create(
        instance: &ash::Instance,
        device: &ash::Device,
        physical_device: vk::PhysicalDevice,
        surface_loader: &khr::surface::Instance,
        surface: vk::SurfaceKHR,
        graphics_family: u32,
        width: u32,
        height: u32,
        old_swapchain: Option<vk::SwapchainKHR>,
    ) -> Result<Self> {
        // Query surface capabilities
        let caps = surface_loader.get_physical_device_surface_capabilities(physical_device, surface)?;
        let formats = surface_loader.get_physical_device_surface_formats(physical_device, surface)?;
        let present_modes = surface_loader.get_physical_device_surface_present_modes(physical_device, surface)?;

        // Choose format (prefer BGRA8 SRGB)
        let format = formats
            .iter()
            .find(|f| f.format == vk::Format::B8G8R8A8_SRGB && f.color_space == vk::ColorSpaceKHR::SRGB_NONLINEAR)
            .or_else(|| formats.iter().find(|f| f.format == vk::Format::B8G8R8A8_UNORM))
            .unwrap_or(&formats[0]);

        // Choose present mode (prefer mailbox for low latency, fallback to FIFO)
        let present_mode = present_modes
            .iter()
            .copied()
            .find(|&m| m == vk::PresentModeKHR::MAILBOX)
            .unwrap_or(vk::PresentModeKHR::FIFO);

        // Choose extent
        let extent = if caps.current_extent.width != u32::MAX {
            caps.current_extent
        } else {
            vk::Extent2D {
                width: width.clamp(caps.min_image_extent.width, caps.max_image_extent.width),
                height: height.clamp(caps.min_image_extent.height, caps.max_image_extent.height),
            }
        };

        // Image count (prefer triple buffering)
        let image_count = (caps.min_image_count + 1).min(
            if caps.max_image_count == 0 { u32::MAX } else { caps.max_image_count }
        );

        let create_info = vk::SwapchainCreateInfoKHR {
            surface,
            min_image_count: image_count,
            image_format: format.format,
            image_color_space: format.color_space,
            image_extent: extent,
            image_array_layers: 1,
            image_usage: vk::ImageUsageFlags::COLOR_ATTACHMENT | vk::ImageUsageFlags::TRANSFER_DST,
            image_sharing_mode: vk::SharingMode::EXCLUSIVE,
            queue_family_index_count: 1,
            p_queue_family_indices: &graphics_family,
            pre_transform: caps.current_transform,
            composite_alpha: vk::CompositeAlphaFlagsKHR::OPAQUE,
            present_mode,
            clipped: vk::TRUE,
            old_swapchain: old_swapchain.unwrap_or(vk::SwapchainKHR::null()),
            ..Default::default()
        };

        let loader = khr::swapchain::Device::new(instance, device);
        let swapchain = loader.create_swapchain(&create_info, None)
            .context("Failed to create swapchain")?;

        let images = loader.get_swapchain_images(swapchain)?;

        // Create image views
        let image_views: Vec<vk::ImageView> = images
            .iter()
            .map(|&image| {
                let view_info = vk::ImageViewCreateInfo {
                    image,
                    view_type: vk::ImageViewType::TYPE_2D,
                    format: format.format,
                    components: vk::ComponentMapping {
                        r: vk::ComponentSwizzle::IDENTITY,
                        g: vk::ComponentSwizzle::IDENTITY,
                        b: vk::ComponentSwizzle::IDENTITY,
                        a: vk::ComponentSwizzle::IDENTITY,
                    },
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    ..Default::default()
                };
                device.create_image_view(&view_info, None)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;

        log::info!("Created swapchain: {}x{}, {} images, {:?}",
            extent.width, extent.height, images.len(), format.format);

        Ok(Self {
            loader,
            swapchain,
            images,
            image_views,
            format: format.format,
            extent,
        })
    }

    pub fn acquire_next_image(&self, semaphore: vk::Semaphore) -> Result<(u32, bool)> {
        unsafe {
            match self.loader.acquire_next_image(self.swapchain, u64::MAX, semaphore, vk::Fence::null()) {
                Ok((index, suboptimal)) => Ok((index, suboptimal)),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok((0, true)),
                Err(e) => Err(e.into()),
            }
        }
    }

    pub fn present(&self, queue: vk::Queue, image_index: u32, wait_semaphore: vk::Semaphore) -> Result<bool> {
        let swapchains = [self.swapchain];
        let image_indices = [image_index];
        let wait_semaphores = [wait_semaphore];

        let present_info = vk::PresentInfoKHR {
            wait_semaphore_count: 1,
            p_wait_semaphores: wait_semaphores.as_ptr(),
            swapchain_count: 1,
            p_swapchains: swapchains.as_ptr(),
            p_image_indices: image_indices.as_ptr(),
            ..Default::default()
        };

        unsafe {
            match self.loader.queue_present(queue, &present_info) {
                Ok(suboptimal) => Ok(suboptimal),
                Err(vk::Result::ERROR_OUT_OF_DATE_KHR) => Ok(true),
                Err(e) => Err(e.into()),
            }
        }
    }

    pub fn image_count(&self) -> usize {
        self.images.len()
    }

    pub fn image(&self, index: usize) -> vk::Image {
        self.images[index]
    }

    pub fn image_view(&self, index: usize) -> vk::ImageView {
        self.image_views[index]
    }

    pub fn cleanup(&mut self, device: &ash::Device) {
        unsafe {
            for &view in &self.image_views {
                device.destroy_image_view(view, None);
            }
            self.loader.destroy_swapchain(self.swapchain, None);
        }
    }
}
