//! GPU module - Vulkan context, buffers, and compute pipelines

mod context;
mod buffers;
mod swapchain;
mod compute;
mod renderer;
mod graphics;
mod timing;
mod egui_renderer;
pub mod diagnostic;

pub use context::VulkanContext;
pub use buffers::{GpuBuffer, FieldTexture, StagingBuffer, DetectorTexture};
pub use swapchain::Swapchain;
pub use compute::{ComputePipeline, SimParams};
pub use renderer::{Renderer, DetectorHit, render_hitmap_f32, apply_detector_response, DetectorRenderInfo};
pub use graphics::{DetectorPipeline, DisplayParams, Detector3DParams, VolumePipeline, VolumeParams, MarkerPipeline, MarkerParams};
pub use timing::{GpuTiming, BenchmarkResults};
pub use egui_renderer::EguiRenderer;
