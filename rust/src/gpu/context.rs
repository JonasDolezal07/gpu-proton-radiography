//! Vulkan context - device, queues, and core infrastructure

use ash::{vk, Entry};
use ash::khr;
use anyhow::{Result, Context};
use std::ffi::{CStr, CString};
use std::sync::Arc;
use winit::window::Window;
use raw_window_handle::{HasDisplayHandle, HasWindowHandle};

/// Core Vulkan objects
pub struct VulkanContext {
    _entry: Entry,
    instance: ash::Instance,
    surface_loader: khr::surface::Instance,
    surface: vk::SurfaceKHR,
    physical_device: vk::PhysicalDevice,
    device: ash::Device,
    compute_queue: vk::Queue,
    graphics_queue: vk::Queue,
    compute_queue_family: u32,
    graphics_queue_family: u32,
    device_name: String,
    command_pool: vk::CommandPool,
}

impl VulkanContext {
    pub fn new(window: &Arc<Window>) -> Result<Self> {
        unsafe { Self::init(window) }
    }

    /// Create a Vulkan context without any surface or display server.
    /// Safe to call on headless Linux servers (no DISPLAY/WAYLAND_DISPLAY needed).
    pub fn new_headless() -> Result<Self> {
        unsafe { Self::init_headless() }
    }

    unsafe fn init_headless() -> Result<Self> {
        let entry = Entry::load()
            .context("Failed to load Vulkan. Is the Vulkan SDK installed?")?;

        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 3, 0),
            ..Default::default()
        };

        let layer_names: Vec<CString> = vec![];
        let layer_ptrs: Vec<*const i8> = layer_names.iter().map(|n| n.as_ptr()).collect();

        // Headless: no surface extensions. On macOS/MoltenVK, portability
        // enumeration is still required or vkCreateInstance returns
        // VK_ERROR_INCOMPATIBLE_DRIVER.
        #[allow(unused_mut)]
        let mut extension_names: Vec<*const i8> = vec![];
        #[cfg(target_os = "macos")]
        extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());

        let create_flags = {
            #[cfg(target_os = "macos")]
            { vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR }
            #[cfg(not(target_os = "macos"))]
            { vk::InstanceCreateFlags::empty() }
        };

        let instance_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_layer_count: layer_ptrs.len() as u32,
            pp_enabled_layer_names: layer_ptrs.as_ptr(),
            enabled_extension_count: extension_names.len() as u32,
            pp_enabled_extension_names: extension_names.as_ptr(),
            flags: create_flags,
            ..Default::default()
        };

        let instance = entry
            .create_instance(&instance_info, None)
            .context("Failed to create Vulkan instance (headless)")?;

        // Create surface_loader even though surface is null — callers may use the loader
        let surface_loader = khr::surface::Instance::new(&entry, &instance);
        let surface = vk::SurfaceKHR::null();

        let (physical_device, device_name) = Self::pick_physical_device(&instance)?;
        log::info!("Selected GPU (headless): {}", device_name);

        let (graphics_family, compute_family) =
            Self::find_queue_families_headless(&instance, physical_device)?;

        let (device, graphics_queue, compute_queue) =
            Self::create_device_headless(&instance, physical_device, graphics_family, compute_family)?;

        let pool_info = vk::CommandPoolCreateInfo {
            queue_family_index: compute_family,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };
        let command_pool = device.create_command_pool(&pool_info, None)?;

        Ok(Self {
            _entry: entry,
            instance,
            surface_loader,
            surface,
            physical_device,
            device,
            compute_queue,
            graphics_queue,
            compute_queue_family: compute_family,
            graphics_queue_family: graphics_family,
            device_name,
            command_pool,
        })
    }

    unsafe fn find_queue_families_headless(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
    ) -> Result<(u32, u32)> {
        let families = instance.get_physical_device_queue_family_properties(device);

        let mut graphics: Option<u32> = None;
        let mut compute:  Option<u32> = None;

        for (i, family) in families.iter().enumerate() {
            let i = i as u32;
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                graphics = Some(i);
            }
            if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                compute = Some(i);
            }
        }

        let compute = compute.context("No compute queue family found")?;
        // Use compute queue for both if no graphics queue is available
        let graphics = graphics.unwrap_or(compute);

        Ok((graphics, compute))
    }

    unsafe fn create_device_headless(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        graphics_family: u32,
        compute_family: u32,
    ) -> Result<(ash::Device, vk::Queue, vk::Queue)> {
        let priorities = [1.0f32];

        let mut queue_infos = vec![vk::DeviceQueueCreateInfo {
            queue_family_index: compute_family,
            queue_count: 1,
            p_queue_priorities: priorities.as_ptr(),
            ..Default::default()
        }];

        // Add graphics queue if it is a different family
        if graphics_family != compute_family {
            queue_infos.push(vk::DeviceQueueCreateInfo {
                queue_family_index: graphics_family,
                queue_count: 1,
                p_queue_priorities: priorities.as_ptr(),
                ..Default::default()
            });
        }

        // Headless: no VK_KHR_swapchain needed.
        // On macOS MoltenVK, portability_subset is still required; on Linux it is not.
        let device_extensions: Vec<*const i8> = {
            #[cfg(target_os = "macos")]
            { vec![ash::khr::portability_subset::NAME.as_ptr()] }
            #[cfg(not(target_os = "macos"))]
            { vec![] }
        };

        let device_info = vk::DeviceCreateInfo {
            queue_create_info_count: queue_infos.len() as u32,
            p_queue_create_infos: queue_infos.as_ptr(),
            enabled_extension_count: device_extensions.len() as u32,
            pp_enabled_extension_names: if device_extensions.is_empty() {
                std::ptr::null()
            } else {
                device_extensions.as_ptr()
            },
            ..Default::default()
        };

        let device = instance.create_device(physical_device, &device_info, None)?;

        let graphics_queue = device.get_device_queue(graphics_family, 0);
        let compute_queue  = device.get_device_queue(compute_family, 0);

        Ok((device, graphics_queue, compute_queue))
    }

    unsafe fn init(window: &Arc<Window>) -> Result<Self> {
        // Load Vulkan dynamically
        let entry = Entry::load()
            .context("Failed to load Vulkan. Is the Vulkan SDK installed?")?;

        // Create instance with surface extensions
        let app_info = vk::ApplicationInfo {
            api_version: vk::make_api_version(0, 1, 3, 0),
            ..Default::default()
        };

        let layer_names: Vec<CString> = vec![];
        let layer_ptrs: Vec<*const i8> = layer_names.iter().map(|n| n.as_ptr()).collect();

        // Required extensions for windowing
        let mut extension_names = vec![
            khr::surface::NAME.as_ptr(),
        ];

        #[cfg(target_os = "macos")]
        {
            extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
            extension_names.push(ash::ext::metal_surface::NAME.as_ptr());
        }

        #[cfg(target_os = "windows")]
        {
            extension_names.push(ash::khr::win32_surface::NAME.as_ptr());
        }

        #[cfg(target_os = "linux")]
        {
            extension_names.push(ash::khr::xlib_surface::NAME.as_ptr());
        }

        let create_flags = {
            #[cfg(target_os = "macos")]
            { vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR }
            #[cfg(not(target_os = "macos"))]
            { vk::InstanceCreateFlags::empty() }
        };

        let instance_info = vk::InstanceCreateInfo {
            p_application_info: &app_info,
            enabled_layer_count: layer_ptrs.len() as u32,
            pp_enabled_layer_names: layer_ptrs.as_ptr(),
            enabled_extension_count: extension_names.len() as u32,
            pp_enabled_extension_names: extension_names.as_ptr(),
            flags: create_flags,
            ..Default::default()
        };

        let instance = entry
            .create_instance(&instance_info, None)
            .context("Failed to create Vulkan instance")?;

        // Create surface
        let surface_loader = khr::surface::Instance::new(&entry, &instance);

        let surface = Self::create_surface(&entry, &instance, window)?;

        // Pick physical device
        let (physical_device, device_name) = Self::pick_physical_device(&instance)?;
        log::info!("Selected GPU: {}", device_name);

        // Find queue families
        let (graphics_family, compute_family) =
            Self::find_queue_families(&instance, physical_device, &surface_loader, surface)?;

        // Create logical device
        let (device, graphics_queue, compute_queue) =
            Self::create_device(&instance, physical_device, graphics_family, compute_family)?;

        // Create command pool
        let pool_info = vk::CommandPoolCreateInfo {
            queue_family_index: compute_family,
            flags: vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            ..Default::default()
        };
        let command_pool = device.create_command_pool(&pool_info, None)?;

        Ok(Self {
            _entry: entry,
            instance,
            surface_loader,
            surface,
            physical_device,
            device,
            compute_queue,
            graphics_queue,
            compute_queue_family: compute_family,
            graphics_queue_family: graphics_family,
            device_name,
            command_pool,
        })
    }

    #[cfg(target_os = "macos")]
    unsafe fn create_surface(
        entry: &Entry,
        instance: &ash::Instance,
        window: &Arc<Window>,
    ) -> Result<vk::SurfaceKHR> {
        use ash::ext::metal_surface;
        use cocoa::base::id as cocoa_id;
        use objc::runtime::YES;
        use raw_window_handle::{HasWindowHandle, RawWindowHandle};
        use std::os::raw::c_void;

        let metal_loader = metal_surface::Instance::new(entry, instance);

        let handle = window.window_handle()?.as_raw();
        if let RawWindowHandle::AppKit(handle) = handle {
            let view = handle.ns_view.as_ptr() as cocoa_id;

            // Create CAMetalLayer and attach to view
            let layer: cocoa_id = msg_send![class!(CAMetalLayer), layer];
            let _: () = msg_send![view, setWantsLayer: YES];
            let _: () = msg_send![view, setLayer: layer];
            // Match contentsScale to the window backing scale so MoltenVK
            // reports physical pixels in vkGetPhysicalDeviceSurfaceCapabilitiesKHR.
            // Without this, currentExtent is always the logical size (e.g. 1280×720)
            // and we get a low-res swapchain that the OS upscales 2× (blurry).
            let scale: f64 = window.scale_factor();
            let _: () = msg_send![layer, setContentsScale: scale];

            let info = vk::MetalSurfaceCreateInfoEXT {
                p_layer: layer as *const c_void,
                ..Default::default()
            };
            metal_loader
                .create_metal_surface(&info, None)
                .context("Failed to create Metal surface")
        } else {
            anyhow::bail!("Expected AppKit window handle on macOS")
        }
    }

    #[cfg(target_os = "windows")]
    unsafe fn create_surface(
        entry: &Entry,
        instance: &ash::Instance,
        window: &Arc<Window>,
    ) -> Result<vk::SurfaceKHR> {
        use ash::khr::win32_surface;
        use raw_window_handle::RawWindowHandle;

        let win32_loader = win32_surface::Instance::new(entry, instance);

        let handle = window.window_handle()?.as_raw();
        if let RawWindowHandle::Win32(handle) = handle {
            let info = vk::Win32SurfaceCreateInfoKHR {
                hinstance: handle.hinstance.unwrap().get() as *const _,
                hwnd: handle.hwnd.get() as *const _,
                ..Default::default()
            };
            win32_loader
                .create_win32_surface(&info, None)
                .context("Failed to create Win32 surface")
        } else {
            anyhow::bail!("Expected Win32 window handle")
        }
    }

    #[cfg(target_os = "linux")]
    unsafe fn create_surface(
        entry: &Entry,
        instance: &ash::Instance,
        window: &Arc<Window>,
    ) -> Result<vk::SurfaceKHR> {
        // Simplified - would need proper X11/Wayland handling
        anyhow::bail!("Linux surface creation not implemented")
    }

    unsafe fn pick_physical_device(instance: &ash::Instance) -> Result<(vk::PhysicalDevice, String)> {
        let devices = instance.enumerate_physical_devices()?;
        if devices.is_empty() {
            anyhow::bail!("No Vulkan-capable GPU found");
        }

        // Prefer discrete GPU, fallback to first
        let device = devices
            .iter()
            .find(|&&d| {
                let props = instance.get_physical_device_properties(d);
                props.device_type == vk::PhysicalDeviceType::DISCRETE_GPU
            })
            .copied()
            .unwrap_or(devices[0]);

        let props = instance.get_physical_device_properties(device);
        let name = CStr::from_ptr(props.device_name.as_ptr())
            .to_string_lossy()
            .into_owned();

        println!("Selected Vulkan device: {}", name);
        if name.to_lowercase().contains("llvmpipe") || name.to_lowercase().contains("softpipe") {
            eprintln!(
                "WARNING: prad is running on a software Vulkan renderer ({}).\n\
                 This is ~100× slower than a real GPU and likely unintentional.\n\
                 On Linux/NVIDIA, ensure DISPLAY is set (e.g. via Xvfb) and\n\
                 VK_ICD_FILENAMES points to the NVIDIA ICD before running.",
                name
            );
        }

        Ok((device, name))
    }

    unsafe fn find_queue_families(
        instance: &ash::Instance,
        device: vk::PhysicalDevice,
        surface_loader: &khr::surface::Instance,
        surface: vk::SurfaceKHR,
    ) -> Result<(u32, u32)> {
        let families = instance.get_physical_device_queue_family_properties(device);

        let mut graphics = None;
        let mut compute = None;

        for (i, family) in families.iter().enumerate() {
            let i = i as u32;

            // Check for graphics + present support
            if family.queue_flags.contains(vk::QueueFlags::GRAPHICS) {
                if surface_loader.get_physical_device_surface_support(device, i, surface)? {
                    graphics = Some(i);
                }
            }

            // Check for compute support
            if family.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                compute = Some(i);
            }
        }

        Ok((
            graphics.context("No graphics queue family found")?,
            compute.context("No compute queue family found")?,
        ))
    }

    unsafe fn create_device(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        graphics_family: u32,
        compute_family: u32,
    ) -> Result<(ash::Device, vk::Queue, vk::Queue)> {
        let priorities = [1.0f32];

        let mut queue_infos = vec![vk::DeviceQueueCreateInfo {
            queue_family_index: graphics_family,
            queue_count: 1,
            p_queue_priorities: priorities.as_ptr(),
            ..Default::default()
        }];

        // Add compute queue if different family
        if compute_family != graphics_family {
            queue_infos.push(vk::DeviceQueueCreateInfo {
                queue_family_index: compute_family,
                queue_count: 1,
                p_queue_priorities: priorities.as_ptr(),
                ..Default::default()
            });
        }

        let device_extensions = [
            khr::swapchain::NAME.as_ptr(),
            #[cfg(target_os = "macos")]
            ash::khr::portability_subset::NAME.as_ptr(),
        ];

        let device_info = vk::DeviceCreateInfo {
            queue_create_info_count: queue_infos.len() as u32,
            p_queue_create_infos: queue_infos.as_ptr(),
            enabled_extension_count: device_extensions.len() as u32,
            pp_enabled_extension_names: device_extensions.as_ptr(),
            ..Default::default()
        };

        let device = instance.create_device(physical_device, &device_info, None)?;

        let graphics_queue = device.get_device_queue(graphics_family, 0);
        let compute_queue = device.get_device_queue(compute_family, 0);

        Ok((device, graphics_queue, compute_queue))
    }

    pub fn device_name(&self) -> &str {
        &self.device_name
    }

    pub fn vulkan_api_version(&self) -> String {
        unsafe {
            let props = self.instance.get_physical_device_properties(self.physical_device);
            let major = vk::api_version_major(props.api_version);
            let minor = vk::api_version_minor(props.api_version);
            let patch = vk::api_version_patch(props.api_version);
            format!("{}.{}.{}", major, minor, patch)
        }
    }

    pub fn device(&self) -> &ash::Device {
        &self.device
    }

    pub fn compute_queue(&self) -> vk::Queue {
        self.compute_queue
    }

    pub fn graphics_queue(&self) -> vk::Queue {
        self.graphics_queue
    }

    pub fn command_pool(&self) -> vk::CommandPool {
        self.command_pool
    }

    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }

    pub fn instance(&self) -> &ash::Instance {
        &self.instance
    }

    pub fn surface_loader(&self) -> &khr::surface::Instance {
        &self.surface_loader
    }

    pub fn surface(&self) -> vk::SurfaceKHR {
        self.surface
    }

    pub fn graphics_queue_family(&self) -> u32 {
        self.graphics_queue_family
    }

    pub fn compute_queue_family(&self) -> u32 {
        self.compute_queue_family
    }

    /// Get timestamp period in nanoseconds per tick
    pub fn timestamp_period(&self) -> f32 {
        unsafe {
            let props = self.instance.get_physical_device_properties(self.physical_device);
            props.limits.timestamp_period
        }
    }

    /// Check if timestamps are supported on compute queue
    pub fn timestamps_supported(&self) -> bool {
        unsafe {
            let families = self.instance.get_physical_device_queue_family_properties(self.physical_device);
            if let Some(family) = families.get(self.compute_queue_family as usize) {
                family.timestamp_valid_bits > 0
            } else {
                false
            }
        }
    }
}

impl Drop for VulkanContext {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().ok();
            self.device.destroy_command_pool(self.command_pool, None);
            self.device.destroy_device(None);
            // Only destroy the surface when it is non-null (windowed mode).
            // Headless mode stores vk::SurfaceKHR::null().
            if self.surface != vk::SurfaceKHR::null() {
                self.surface_loader.destroy_surface(self.surface, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
