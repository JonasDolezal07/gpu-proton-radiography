//! Renderer - ties together swapchain, compute, and presentation

use ash::vk;
use anyhow::{Result, Context};
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
use gpu_allocator::MemoryLocation;
use std::sync::{Arc, Mutex};

use super::{VulkanContext, Swapchain, GpuBuffer, FieldTexture, DetectorTexture, StagingBuffer, ComputePipeline, SimParams, DetectorPipeline, DisplayParams, Detector3DParams, VolumePipeline, VolumeParams, MarkerPipeline, MarkerParams, GpuTiming, BenchmarkResults, EguiRenderer};
use crate::loaders::{FieldData, ParticleData};
use crate::config::{PngExportConfig, ScaleMode, ColormapType, DetectorResponseConfig};
use crate::run_dir::{RunDir, RunDiagnostics};

const MAX_FRAMES_IN_FLIGHT: usize = 2;
const MAX_DETECTOR_HITS: usize = 1_000_000;
const DETECTOR_RESOLUTION: u32 = 1024;
const STEPS_PER_FRAME: u32 = 100;  // Total simulation steps per render frame
const STEPS_PER_DISPATCH: u32 = 100;  // Steps batched per GPU dispatch (keeps particles in registers)

/// Detector hit record (matches shader)
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct DetectorHit {
    pub position: [f32; 2],
    pub energy: f32,
    pub particle_id: u32,
}

/// Detector geometry needed by `apply_detector_response`.
///
/// Axis convention (matches GPU texture / buffer layout):
///   index = row * width + col, where col ↔ y_mm, row ↔ z_mm.
pub struct DetectorRenderInfo {
    pub width_px:          usize,
    pub height_px:         usize,
    /// µm per pixel along the y_mm axis (column direction).
    pub pixel_pitch_y_um:  f64,
    /// µm per pixel along the z_mm axis (row direction).
    pub pixel_pitch_z_um:  f64,
}

/// Apply physical detector response to hit counts (f32).
///
/// Pipeline: Gaussian blur → add background → Poisson noise.
/// Physically: expected = gaussian_blur(raw) + background;
///             if poisson_noise: observed ~ Poisson(expected).
///
/// The GPU path casts u32 raw counts to f32 before calling.
/// The standalone render path passes f32 binned counts directly.
pub fn apply_detector_response(
    raw_counts: &[f32],
    width: usize,
    height: usize,
    detector: &DetectorRenderInfo,
    response: &DetectorResponseConfig,
) -> Vec<f32> {
    let mut counts = raw_counts.to_vec();

    // 1. Gaussian blur on raw counts (before background)
    if response.blur_sigma_um > 1e-6 {
        let sigma_col = (response.blur_sigma_um / detector.pixel_pitch_y_um) as f32;
        let sigma_row = (response.blur_sigma_um / detector.pixel_pitch_z_um) as f32;
        gaussian_blur_2d(&mut counts, width, height, sigma_col, sigma_row);
    }

    // 2. Add uniform background (part of expected signal before Poisson draw)
    if response.background_counts > 0.0 {
        let bg = response.background_counts as f32;
        for c in counts.iter_mut() { *c += bg; }
    }

    // 3. Poisson noise: observed ~ Poisson(expected)
    if response.poisson_noise {
        use rand::SeedableRng;
        use rand_distr::Distribution;
        let mut rng: rand::rngs::StdRng = match response.noise_seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None    => rand::rngs::StdRng::from_entropy(),
        };
        for c in counts.iter_mut() {
            if *c > 0.0 {
                let lambda = (*c as f64).max(1e-9);
                *c = rand_distr::Poisson::new(lambda).unwrap().sample(&mut rng) as f32;
            }
        }
    }

    counts
}

/// Separable 2D Gaussian blur (clamp-to-edge boundary).
/// data: row-major, index = row * w + col.
/// sigma_col: σ in pixels along the column axis (y_mm direction).
/// sigma_row: σ in pixels along the row axis (z_mm direction).
fn gaussian_blur_2d(data: &mut [f32], w: usize, h: usize, sigma_col: f32, sigma_row: f32) {
    fn make_kernel(sigma: f32) -> Vec<f32> {
        let radius = (3.0 * sigma).ceil() as usize;
        let len = 2 * radius + 1;
        let mut k: Vec<f32> = (0..len)
            .map(|i| { let d = i as f32 - radius as f32; (-0.5 * (d / sigma).powi(2)).exp() })
            .collect();
        let sum: f32 = k.iter().sum();
        for v in k.iter_mut() { *v /= sum; }
        k
    }

    let kc = make_kernel(sigma_col);
    let kr = make_kernel(sigma_row);
    let rc = kc.len() / 2;
    let rr = kr.len() / 2;
    let mut tmp = vec![0.0f32; w * h];

    // Horizontal pass (along columns, each row)
    for row in 0..h {
        for col in 0..w {
            let mut acc = 0.0f32;
            for (ki, &kv) in kc.iter().enumerate() {
                let c = ((col as isize + ki as isize - rc as isize).clamp(0, w as isize - 1)) as usize;
                acc += data[row * w + c] * kv;
            }
            tmp[row * w + col] = acc;
        }
    }

    // Vertical pass (along rows, each column)
    for col in 0..w {
        for row in 0..h {
            let mut acc = 0.0f32;
            for (ki, &kv) in kr.iter().enumerate() {
                let r = ((row as isize + ki as isize - rr as isize).clamp(0, h as isize - 1)) as usize;
                acc += tmp[r * w + col] * kv;
            }
            data[row * w + col] = acc;
        }
    }
}

/// Write raw GPU hit counts as a binary file + JSON sidecar.
/// stem: shared timestamp string (no extension). Files: {stem}_raw_counts.bin / _meta.json.
fn write_raw_counts(
    output_dir: &std::path::Path,
    stem: &str,
    counts: &[u32],
    width: u32,
    height: u32,
    detector: &DetectorRenderInfo,
) -> Result<()> {
    let bin_path  = output_dir.join(format!("{}_raw_counts.bin",      stem));
    let meta_path = output_dir.join(format!("{}_raw_counts_meta.json", stem));

    std::fs::write(&bin_path, bytemuck::cast_slice::<u32, u8>(counts))
        .context("Failed to write raw_counts.bin")?;

    let total_hits: u64 = counts.iter().map(|&c| c as u64).sum();
    let max_count:  u32 = counts.iter().cloned().max().unwrap_or(0);

    let meta = serde_json::json!({
        "dtype": "uint32",
        "endian": "little",
        "shape": [height, width],
        "width_px": width,
        "height_px": height,
        "axes": { "row": "z_mm", "col": "y_mm" },
        "total_hits": total_hits,
        "max_count": max_count,
        "detector": {
            "pixel_pitch_y_um": detector.pixel_pitch_y_um,
            "pixel_pitch_z_um": detector.pixel_pitch_z_um,
        }
    });
    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap())
        .context("Failed to write raw_counts_meta.json")?;

    log::info!("Wrote raw counts → {:?}", bin_path);
    Ok(())
}

/// Write detector-response-processed counts as a binary file + JSON sidecar.
fn write_processed_counts(
    output_dir: &std::path::Path,
    stem: &str,
    counts: &[f32],
    width: u32,
    height: u32,
    detector: &DetectorRenderInfo,
    response: &DetectorResponseConfig,
    raw_total: u64,
) -> Result<()> {
    let bin_path  = output_dir.join(format!("{}_processed_counts.bin",      stem));
    let meta_path = output_dir.join(format!("{}_processed_counts_meta.json", stem));

    std::fs::write(&bin_path, bytemuck::cast_slice::<f32, u8>(counts))
        .context("Failed to write processed_counts.bin")?;

    let processed_total: f64 = counts.iter().map(|&c| c as f64).sum();
    let max_count: f32 = counts.iter().cloned().fold(0.0f32, f32::max);
    let poisson_info = if response.poisson_noise {
        serde_json::json!({ "enabled": true, "seed": response.noise_seed,
            "note": "Applied to processed detector counts after blur/background." })
    } else {
        serde_json::json!({ "enabled": false })
    };

    let meta = serde_json::json!({
        "dtype": "float32",
        "endian": "little",
        "shape": [height, width],
        "width_px": width,
        "height_px": height,
        "axes": { "row": "z_mm", "col": "y_mm" },
        "raw_total": raw_total,
        "processed_total": processed_total,
        "max_count": max_count,
        "detector": {
            "pixel_pitch_y_um": detector.pixel_pitch_y_um,
            "pixel_pitch_z_um": detector.pixel_pitch_z_um,
        },
        "detector_response": {
            "blur_sigma_um": response.blur_sigma_um,
            "blur_sigma_px_y": if response.blur_sigma_um > 1e-6 {
                response.blur_sigma_um / detector.pixel_pitch_y_um } else { 0.0 },
            "blur_sigma_px_z": if response.blur_sigma_um > 1e-6 {
                response.blur_sigma_um / detector.pixel_pitch_z_um } else { 0.0 },
            "background_counts": response.background_counts,
            "poisson_noise": poisson_info,
        }
    });
    std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap())
        .context("Failed to write processed_counts_meta.json")?;

    log::info!("Wrote processed counts → {:?}", bin_path);
    Ok(())
}

/// Shared pixel-rendering pipeline used by both export_detector_png and the standalone render subcommand.
///
/// `counts` is in row-major order: index = row * width + col.
/// Uses f32 throughout so weighted hit accumulation is a future drop-in.
pub fn render_hitmap_f32(
    counts: &[f32],
    width: u32,
    height: u32,
    cfg: &PngExportConfig,
) -> image::ImageBuffer<image::Rgb<u8>, Vec<u8>> {
    let max_count = counts.iter().cloned().fold(0.0f32, f32::max).max(1.0);
    let colorbar_width: u32 = if cfg.include_colorbar { 24 } else { 0 };
    let png_width = width + colorbar_width;

    image::ImageBuffer::from_fn(png_width, height, |x, y| {
        let value = if x < width {
            let count = counts[(y * width + x) as usize];
            let normalized = match cfg.scale {
                ScaleMode::Linear => count / max_count,
                ScaleMode::Log    => if count > 0.0 { (count + 1.0).ln() / (max_count + 1.0).ln() } else { 0.0 },
                ScaleMode::Sqrt   => (count / max_count).sqrt(),
            };
            (normalized * cfg.exposure).clamp(0.0, 1.0).powf(cfg.gamma)
        } else {
            // Colorbar strip: full range top→bottom (t=1 at top, t=0 at bottom)
            1.0 - (y as f32 / height.saturating_sub(1).max(1) as f32)
        };

        let rgb = apply_colormap(value, cfg.colormap);
        image::Rgb(rgb)
    })
}

fn apply_colormap(value: f32, colormap: ColormapType) -> [u8; 3] {
    match colormap {
        ColormapType::RcfFilm    => colormap_rcf(value),
        ColormapType::Scientific => colormap_scientific(value),
        ColormapType::Grayscale  => colormap_grayscale(value),
        ColormapType::Hot        => colormap_hot(value),
        ColormapType::Inverted   => colormap_inverted(value),
    }
}

fn colormap_rcf(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let c = if t < 0.1 {
        mix3([0.75, 0.88, 0.82], [0.6, 0.78, 0.8], t / 0.1)
    } else if t < 0.3 {
        mix3([0.6, 0.78, 0.8], [0.4, 0.55, 0.75], (t - 0.1) / 0.2)
    } else if t < 0.5 {
        mix3([0.4, 0.55, 0.75], [0.35, 0.35, 0.65], (t - 0.3) / 0.2)
    } else if t < 0.7 {
        mix3([0.35, 0.35, 0.65], [0.4, 0.25, 0.5], (t - 0.5) / 0.2)
    } else if t < 0.9 {
        mix3([0.4, 0.25, 0.5], [0.35, 0.18, 0.35], (t - 0.7) / 0.2)
    } else {
        mix3([0.35, 0.18, 0.35], [0.2, 0.1, 0.15], (t - 0.9) / 0.1)
    };
    [(c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8]
}

fn colormap_scientific(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let c = if t < 0.25 {
        mix3([0.02, 0.02, 0.05], [0.1, 0.2, 0.6], t / 0.25)
    } else if t < 0.5 {
        mix3([0.1, 0.2, 0.6], [0.2, 0.6, 0.8], (t - 0.25) / 0.25)
    } else if t < 0.75 {
        mix3([0.2, 0.6, 0.8], [0.9, 0.9, 0.95], (t - 0.5) / 0.25)
    } else {
        mix3([0.9, 0.9, 0.95], [1.0, 1.0, 0.7], (t - 0.75) / 0.25)
    };
    [(c[0] * 255.0) as u8, (c[1] * 255.0) as u8, (c[2] * 255.0) as u8]
}

fn colormap_grayscale(t: f32) -> [u8; 3] {
    let v = (t.clamp(0.0, 1.0) * 255.0) as u8;
    [v, v, v]
}

fn colormap_hot(t: f32) -> [u8; 3] {
    let t = t.clamp(0.0, 1.0);
    let (r, g, b) = if t < 1.0 / 3.0 {
        (t * 3.0, 0.0, 0.0)
    } else if t < 2.0 / 3.0 {
        (1.0, (t - 1.0 / 3.0) * 3.0, 0.0)
    } else {
        (1.0, 1.0, (t - 2.0 / 3.0) * 3.0)
    };
    [(r * 255.0) as u8, (g * 255.0) as u8, (b * 255.0) as u8]
}

/// Inverted grayscale: 0 counts = white, max counts = black.
fn colormap_inverted(t: f32) -> [u8; 3] {
    let v = ((1.0 - t.clamp(0.0, 1.0)) * 255.0) as u8;
    [v, v, v]
}

fn mix3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t, a[2] + (b[2] - a[2]) * t]
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
    swapchain: Option<Swapchain>,

    // Graphics resources
    detector_pipeline: Option<DetectorPipeline>,
    detector_texture: DetectorTexture,
    display_params: DisplayParams,
    detector_3d_params: Detector3DParams,
    volume_pipeline: Option<VolumePipeline>,
    volume_params: VolumeParams,
    marker_pipeline: Option<MarkerPipeline>,
    source_marker_params: MarkerParams,

    // GUI rendering
    egui_renderer: Option<EguiRenderer>,

    // Compute resources
    compute_pipeline: Option<ComputePipeline>,
    particle_buffer: Option<GpuBuffer>,
    field_texture: Option<FieldTexture>,
    e_field_texture: Option<FieldTexture>,
    detector_buffer: Option<GpuBuffer>,

    // Simulation state
    pub sim_params: SimParams,
    particle_count: u32,
    is_running: bool,
    total_hits: u32,
    frame_count: u32,
    is_shutting_down: bool,
    diagnostics_logged: bool,
    benchmark_reported: bool,
    // Source metadata for PNG sidecar (stored at load time, not in GPU sim_params)
    source_type: String,
    particle_energy_mev: f32,
    // PNG export config for auto-export and interactive export
    png_cfg: PngExportConfig,
    // Detector response config applied before rendering
    detector_response: DetectorResponseConfig,

    // Performance timing
    gpu_timing: GpuTiming,

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

        // Load shaders
        // Use 3D vertex shader for detector (positions quad in world space)
        const VERT_SHADER: &[u8] = include_bytes!("../../../shaders/detector3d.vert.spv");
        const FRAG_SHADER: &[u8] = include_bytes!("../../../shaders/detector.frag.spv");

        // Create graphics pipeline
        let detector_pipeline = DetectorPipeline::new(
            ctx.device(),
            &allocator,
            swapchain.format,
            swapchain.extent,
            swapchain.image_views(),
            VERT_SHADER,
            FRAG_SHADER,
        )?;

        // Create detector texture (1024x1024)
        let detector_texture = DetectorTexture::new(
            ctx.device(),
            &allocator,
            DETECTOR_RESOLUTION,
            DETECTOR_RESOLUTION,
        )?;

        // Initialize detector texture to zero and transition to shader read layout
        Self::init_detector_texture(&ctx, &detector_texture)?;

        // Update descriptor with detector texture
        detector_pipeline.update_descriptor(ctx.device(), detector_texture.view);

        // Create frame synchronization objects
        let frames = Self::create_frame_data(ctx.device(), ctx.command_pool())?;

        let sim_params = SimParams {
            dt: 1e-12,
            q_over_m: 9.58e7,  // Proton: e/m_p
            n_particles: 0,
            steps_per_dispatch: STEPS_PER_DISPATCH,
            max_steps: 25_000,
            _pad_a: 0,
            _pad_b: 0,
            _pad_c: 0,
            field_min: [0.0; 4],
            field_max: [1.0, 1.0, 1.0, 0.0],
            detector_pos:    [0.0, 0.0, 1.0, 0.0],
            detector_normal: [1.0, 0.0, 0.0, 0.0],
            detector_extent: [0.25, 0.25, 0.0, 0.0],
            detector_up:     [0.0, 1.0, 0.0, 0.0],
        };

        let display_params = DisplayParams {
            max_count: 100.0,
            gamma: 0.5,
            exposure: 1.0,
            use_log_scale: 1,  // Log scale by default for better dynamic range
            colormap_mode: 0,  // RCF film (realistic)
        };

        let detector_3d_params = Detector3DParams::default();
        let volume_params = VolumeParams::default();

        // Create GPU timing
        let gpu_timing = GpuTiming::new(
            ctx.device(),
            ctx.timestamp_period(),
            ctx.timestamps_supported(),
        )?;

        Ok(Self {
            allocator,
            swapchain: Some(swapchain),
            detector_pipeline: Some(detector_pipeline),
            detector_texture,
            display_params,
            detector_3d_params,
            volume_pipeline: None,  // Created when field is uploaded
            volume_params,
            marker_pipeline: None,  // Created when field is uploaded
            source_marker_params: MarkerParams::default(),
            egui_renderer: None,  // Created on demand
            compute_pipeline: None,
            particle_buffer: None,
            field_texture: None,
            e_field_texture: None,
            detector_buffer: None,
            sim_params,
            particle_count: 0,
            is_running: false,
            total_hits: 0,
            frame_count: 0,
            is_shutting_down: false,
            diagnostics_logged: false,
            benchmark_reported: false,
            source_type: String::new(),
            particle_energy_mev: 0.0,
            png_cfg: PngExportConfig::default(),
            detector_response: DetectorResponseConfig::default(),
            gpu_timing,
            frames,
            current_frame: 0,
            ctx,
        })
    }

    /// Create a renderer suitable for headless batch runs (no swapchain, no graphics pipeline).
    pub fn new_headless(ctx: Arc<VulkanContext>) -> Result<Self> {
        let allocator = Allocator::new(&AllocatorCreateDesc {
            instance: ctx.instance().clone(),
            device: ctx.device().clone(),
            physical_device: ctx.physical_device(),
            debug_settings: Default::default(),
            buffer_device_address: false,
            allocation_sizes: Default::default(),
        }).context("Failed to create GPU allocator (headless)")?;

        let allocator = Arc::new(Mutex::new(allocator));

        // Create detector texture — compute shader writes hit counts into it as a storage image.
        let detector_texture = DetectorTexture::new(
            ctx.device(),
            &allocator,
            DETECTOR_RESOLUTION,
            DETECTOR_RESOLUTION,
        )?;

        // Clear and transition to SHADER_READ_ONLY_OPTIMAL (the compute barrier expects this as
        // the initial layout before transitioning to GENERAL for writes).
        Self::init_detector_texture(&ctx, &detector_texture)?;

        // Frame synchronisation objects (needed by run_compute_step)
        let frames = Self::create_frame_data(ctx.device(), ctx.command_pool())?;

        let sim_params = SimParams {
            dt: 1e-12,
            q_over_m: 9.58e7,
            n_particles: 0,
            steps_per_dispatch: STEPS_PER_DISPATCH,
            max_steps: 25_000,
            _pad_a: 0,
            _pad_b: 0,
            _pad_c: 0,
            field_min: [0.0; 4],
            field_max: [1.0, 1.0, 1.0, 0.0],
            detector_pos:    [0.0, 0.0, 1.0, 0.0],
            detector_normal: [1.0, 0.0, 0.0, 0.0],
            detector_extent: [0.25, 0.25, 0.0, 0.0],
            detector_up:     [0.0, 1.0, 0.0, 0.0],
        };

        let display_params = DisplayParams {
            max_count: 100.0,
            gamma: 0.5,
            exposure: 1.0,
            use_log_scale: 1,
            colormap_mode: 0,
        };

        let gpu_timing = GpuTiming::new(
            ctx.device(),
            ctx.timestamp_period(),
            ctx.timestamps_supported(),
        )?;

        Ok(Self {
            allocator,
            swapchain: None,
            detector_pipeline: None,
            detector_texture,
            display_params,
            detector_3d_params: Detector3DParams::default(),
            volume_pipeline: None,
            volume_params: VolumeParams::default(),
            marker_pipeline: None,
            source_marker_params: MarkerParams::default(),
            egui_renderer: None,
            compute_pipeline: None,
            particle_buffer: None,
            field_texture: None,
            e_field_texture: None,
            detector_buffer: None,
            sim_params,
            particle_count: 0,
            is_running: false,
            total_hits: 0,
            frame_count: 0,
            is_shutting_down: false,
            diagnostics_logged: false,
            benchmark_reported: false,
            source_type: String::new(),
            particle_energy_mev: 0.0,
            png_cfg: PngExportConfig::default(),
            detector_response: DetectorResponseConfig::default(),
            gpu_timing,
            frames,
            current_frame: 0,
            ctx,
        })
    }

    fn init_detector_texture(ctx: &VulkanContext, texture: &DetectorTexture) -> Result<()> {
        let device = ctx.device();
        let queue = ctx.compute_queue();

        unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo {
                command_pool: ctx.command_pool(),
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

            // Transition to transfer dst
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

            // Clear to zero
            let clear_value = vk::ClearColorValue { uint32: [0, 0, 0, 0] };
            device.cmd_clear_color_image(
                cmd,
                texture.image,
                vk::ImageLayout::TRANSFER_DST_OPTIMAL,
                &clear_value,
                &[vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                }],
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
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[barrier],
            );

            device.end_command_buffer(cmd)?;

            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd,
                ..Default::default()
            };

            device.queue_submit(queue, &[submit_info], vk::Fence::null())?;
            device.queue_wait_idle(queue)?;
            device.free_command_buffers(ctx.command_pool(), &[cmd]);
        }

        Ok(())
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
        let num_voxels = (field.nx * field.ny * field.nz) as usize;

        // Upload B-field texture
        let mut b_texture = FieldTexture::new(device, &self.allocator, field.nx, field.ny, field.nz)?;
        let mut b_rgba = vec![0.0f32; num_voxels * 4];
        for i in 0..num_voxels {
            b_rgba[i * 4]     = field.data[i * 3];
            b_rgba[i * 4 + 1] = field.data[i * 3 + 1];
            b_rgba[i * 4 + 2] = field.data[i * 3 + 2];
        }
        let b_bytes = (b_rgba.len() * std::mem::size_of::<f32>()) as vk::DeviceSize;
        let mut b_staging = StagingBuffer::new(device, &self.allocator, b_bytes)?;
        b_staging.write(&b_rgba)?;
        self.upload_texture_data(&mut b_texture, &b_staging, field.nx, field.ny, field.nz)?;
        b_staging.cleanup(device, &self.allocator);

        // Upload E-field texture (zero-filled if no E data in file)
        let mut e_texture = FieldTexture::new(device, &self.allocator, field.nx, field.ny, field.nz)?;
        let mut e_rgba = vec![0.0f32; num_voxels * 4];
        for i in 0..num_voxels {
            e_rgba[i * 4]     = field.e_data[i * 3];
            e_rgba[i * 4 + 1] = field.e_data[i * 3 + 1];
            e_rgba[i * 4 + 2] = field.e_data[i * 3 + 2];
        }
        let e_bytes = (e_rgba.len() * std::mem::size_of::<f32>()) as vk::DeviceSize;
        let mut e_staging = StagingBuffer::new(device, &self.allocator, e_bytes)?;
        e_staging.write(&e_rgba)?;
        self.upload_texture_data(&mut e_texture, &e_staging, field.nx, field.ny, field.nz)?;
        e_staging.cleanup(device, &self.allocator);

        // Update sim params with field bounds
        self.sim_params.field_min = [field.bounds.x_min, field.bounds.y_min, field.bounds.z_min, 0.0];
        self.sim_params.field_max = [field.bounds.x_max, field.bounds.y_max, field.bounds.z_max, 0.0];

        if let Some(mut old) = self.field_texture.take() {
            old.cleanup(device, &self.allocator);
        }
        self.field_texture = Some(b_texture);

        if let Some(mut old) = self.e_field_texture.take() {
            old.cleanup(device, &self.allocator);
        }
        self.e_field_texture = Some(e_texture);

        // Update volume params with field bounds
        self.volume_params.volume_min = self.sim_params.field_min;
        self.volume_params.volume_max = self.sim_params.field_max;

        // Create volume pipeline if not already created (only in windowed mode)
        if self.volume_pipeline.is_none() {
            if let Some(dp) = &self.detector_pipeline {
                const VERT_SHADER: &[u8] = include_bytes!("../../../shaders/fullscreen.vert.spv");
                const VOLUME_FRAG: &[u8] = include_bytes!("../../../shaders/volume.frag.spv");

                let volume_pipeline = VolumePipeline::new(
                    device,
                    dp.render_pass(),
                    VERT_SHADER,
                    VOLUME_FRAG,
                )?;

                // Update descriptor with field texture view
                volume_pipeline.update_descriptor(device, self.field_texture.as_ref().unwrap().view);

                self.volume_pipeline = Some(volume_pipeline);
            }
        }

        // Create marker pipeline if not already created (only in windowed mode)
        if self.marker_pipeline.is_none() {
            if let Some(dp) = &self.detector_pipeline {
                const MARKER_VERT: &[u8] = include_bytes!("../../../shaders/marker.vert.spv");
                const MARKER_FRAG: &[u8] = include_bytes!("../../../shaders/marker.frag.spv");

                let marker_pipeline = MarkerPipeline::new(
                    device,
                    dp.render_pass(),
                    MARKER_VERT,
                    MARKER_FRAG,
                )?;

                self.marker_pipeline = Some(marker_pipeline);
            }
        }

        if self.volume_pipeline.is_some() {
            // Just update the field texture descriptor
            self.volume_pipeline.as_ref().unwrap().update_descriptor(
                device,
                self.field_texture.as_ref().unwrap().view,
            );
        }

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

        // Create detector buffer for hit recording
        let detector_size = (std::mem::size_of::<u32>() * 4
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

        // Reset total hits
        self.total_hits = 0;

        log::info!("Uploaded {} particles", particles.count);
        Ok(())
    }

    pub fn update_descriptors(&mut self) -> Result<()> {
        let pipeline = self.compute_pipeline.as_ref().context("No compute pipeline")?;
        let particles = self.particle_buffer.as_ref().context("No particle buffer")?;
        let b_field = self.field_texture.as_ref().context("No B-field texture")?;
        let e_field = self.e_field_texture.as_ref().context("No E-field texture")?;
        let detector = self.detector_buffer.as_ref().context("No detector buffer")?;

        pipeline.update_descriptors(
            self.ctx.device(),
            particles.buffer,
            particles.size,
            b_field.view,
            b_field.sampler,
            detector.buffer,
            detector.size,
            self.detector_texture.storage_view,
            e_field.view,
            e_field.sampler,
        );

        Ok(())
    }

    pub fn set_sim_params(
        &mut self,
        dt: f32,
        detector_pos: [f32; 3],
        detector_normal: [f32; 3],
        detector_up: [f32; 3],
        detector_extent: [f32; 2],
    ) {
        self.sim_params.dt = dt;
        self.sim_params.detector_pos    = [detector_pos[0],    detector_pos[1],    detector_pos[2],    0.0];
        self.sim_params.detector_normal = [detector_normal[0], detector_normal[1], detector_normal[2], 0.0];
        self.sim_params.detector_up     = [detector_up[0],     detector_up[1],     detector_up[2],     0.0];
        self.sim_params.detector_extent = [detector_extent[0], detector_extent[1], 0.0, 0.0];
    }

    /// Set the source position for the marker visualization
    pub fn set_source_position(&mut self, position: [f32; 3]) {
        // Size based on field dimensions (about 2% of field extent)
        let field_size = (self.sim_params.field_max[0] - self.sim_params.field_min[0]).abs();
        let marker_size = (field_size * 0.03).max(0.002);  // At least 2mm

        self.source_marker_params.position = [position[0], position[1], position[2], marker_size];
        self.source_marker_params.color = [1.0, 0.15, 0.1, 1.0];  // Bright red
    }

    pub fn set_source_metadata(&mut self, source_type: &str, particle_energy_mev: f32) {
        self.source_type = source_type.to_string();
        self.particle_energy_mev = particle_energy_mev;
    }

    pub fn set_png_config(&mut self, cfg: PngExportConfig) {
        self.png_cfg = cfg;
    }

    pub fn set_detector_response_config(&mut self, cfg: DetectorResponseConfig) {
        self.detector_response = cfg;
    }

    /// Initialize egui renderer (call once after renderer is created)
    pub fn init_egui(&mut self) -> Result<()> {
        if self.egui_renderer.is_some() {
            return Ok(());
        }

        let device = self.ctx.device();
        let render_pass = match &self.detector_pipeline {
            Some(dp) => dp.render_pass(),
            None => return Ok(()), // headless — no GUI
        };

        const EGUI_VERT: &[u8] = include_bytes!("../../../shaders/egui.vert.spv");
        const EGUI_FRAG: &[u8] = include_bytes!("../../../shaders/egui.frag.spv");

        let egui_renderer = EguiRenderer::new(
            device.clone(),
            self.allocator.clone(),
            render_pass,
            self.ctx.graphics_queue(),
            self.ctx.graphics_queue_family(),
            EGUI_VERT,
            EGUI_FRAG,
        )?;

        self.egui_renderer = Some(egui_renderer);
        log::info!("Egui renderer initialized");
        Ok(())
    }

    /// Update egui font texture if needed
    pub fn apply_egui_textures_delta(&mut self, delta: &egui::TexturesDelta) -> Result<()> {
        if let Some(egui_renderer) = &mut self.egui_renderer {
            egui_renderer.apply_textures_delta(delta)?;
        }
        Ok(())
    }

    /// Render egui to the current frame (call during render_frame)
    pub fn render_egui(
        &mut self,
        clipped_primitives: &[egui::ClippedPrimitive],
        screen_size: [f32; 2],
        pixels_per_point: f32,
    ) -> Result<()> {
        if let Some(egui_renderer) = &mut self.egui_renderer {
            let device = self.ctx.device();
            let frame = &self.frames[self.current_frame];
            egui_renderer.render(
                frame.command_buffer,
                clipped_primitives,
                screen_size,
                pixels_per_point,
            )?;
        }
        Ok(())
    }

    pub fn start_simulation(&mut self) {
        self.is_running = true;
        self.frame_count = 0;
        self.diagnostics_logged = false;
        self.benchmark_reported = false;

        log::info!("Simulation started");
        log::info!("  Particles: {}", self.sim_params.n_particles);
        log::info!("  Steps per frame: {}", STEPS_PER_FRAME);
        log::info!("  Field bounds: {:?} to {:?}",
            &self.sim_params.field_min[..3], &self.sim_params.field_max[..3]);
        log::info!("  Detector: pos={:?}, normal={:?}, extent={:?}",
            &self.sim_params.detector_pos[..3],
            &self.sim_params.detector_normal[..3],
            &self.sim_params.detector_extent[..2]);

        // Calculate dynamic max_count based on particle count and detector resolution
        // For uniform distribution: avg_hits = n_particles / (width * height)
        // Caustics can be 10-100x brighter, so we scale up for headroom
        // With log scale, this gives us good dynamic range without saturation
        let detector_pixels = (DETECTOR_RESOLUTION * DETECTOR_RESOLUTION) as f32;
        let avg_hits_per_pixel = self.sim_params.n_particles as f32 / detector_pixels;
        // Set max_count to ~50x average to capture caustic peaks without saturation
        // With log scale, this provides ~4 decades of dynamic range
        self.display_params.max_count = (avg_hits_per_pixel * 50.0).max(100.0);
        self.display_params.use_log_scale = 1;
        self.display_params.exposure = 1.0;  // Start at 1.0, let log scale do the work
        self.display_params.gamma = 0.7;  // Slightly less aggressive gamma for RCF look
        log::info!("  Display: max_count={:.0} (avg/pixel={:.1}), log_scale={}, exposure={}",
            self.display_params.max_count,
            avg_hits_per_pixel,
            self.display_params.use_log_scale,
            self.display_params.exposure);

        // Configure and start benchmark timing
        self.gpu_timing.configure(
            self.sim_params.n_particles as u64,
            STEPS_PER_FRAME as u64,
            STEPS_PER_DISPATCH as u64,
        );
        self.gpu_timing.start_benchmark();

        // Update volume rendering camera
        self.update_volume_camera();

        // Print controls help
        log::info!("");
        log::info!("=== CONTROLS ===");
        log::info!("  Space  - Pause/resume simulation");
        log::info!("  C      - Toggle colormap (RCF film / Scientific)");
        log::info!("  L      - Toggle log scale");
        log::info!("  +/-    - Adjust exposure");
        log::info!("  [/]    - Adjust gamma");
        log::info!("  S      - Export detector data to CSV");
        log::info!("  P      - Export radiograph as PNG");
        log::info!("  H      - Show help");
        log::info!("  Mouse  - Orbit (left), Pan (right), Zoom (scroll)");
        log::info!("  PNG auto-exports to output/png/ on completion");
        log::info!("");
    }

    /// Update volume rendering camera based on field bounds
    fn update_volume_camera(&mut self) {
        use glam::{Mat4, Vec3};

        // Calculate volume center and size
        let vol_min = Vec3::new(
            self.sim_params.field_min[0],
            self.sim_params.field_min[1],
            self.sim_params.field_min[2],
        );
        let vol_max = Vec3::new(
            self.sim_params.field_max[0],
            self.sim_params.field_max[1],
            self.sim_params.field_max[2],
        );
        let vol_center = (vol_min + vol_max) * 0.5;
        let vol_size = vol_max - vol_min;
        let vol_radius = vol_size.length() * 0.5;

        // Position camera to view the volume from an angle
        let camera_distance = vol_radius * 3.0;
        let camera_pos = vol_center + Vec3::new(
            camera_distance * 0.5,
            camera_distance * 0.3,
            -camera_distance * 0.8,
        );

        // Create view matrix (look at volume center)
        let view = Mat4::look_at_rh(camera_pos, vol_center, Vec3::Y);

        // Create projection matrix
        let (sw, sh) = self.swapchain.as_ref()
            .map(|sc| (sc.extent.width, sc.extent.height))
            .unwrap_or((1280, 720));
        let aspect = sw as f32 / sh as f32;
        let proj = Mat4::perspective_rh(
            45.0_f32.to_radians(),
            aspect,
            0.01,
            100.0,
        );

        // Compute inverse view-projection
        let view_proj = proj * view;
        let inv_view_proj = view_proj.inverse();

        // Update volume params
        self.volume_params.inv_view_proj = inv_view_proj.to_cols_array_2d();
        self.volume_params.camera_pos = [camera_pos.x, camera_pos.y, camera_pos.z, 1.0];
        self.volume_params.volume_min = self.sim_params.field_min;
        self.volume_params.volume_max = self.sim_params.field_max;
        // Higher quality volume rendering: more steps, smaller step size
        self.volume_params.step_size = vol_radius / 128.0;  // ~128 steps through radius
        self.volume_params.density_scale = 0.15;  // Lower density for less opaque volume
        self.volume_params.brightness = 8.0;  // Brighter colors
        self.volume_params.num_steps = 256;  // More steps for smoother rendering

        log::info!("  Volume camera: pos={:?}, looking at {:?}",
            &self.volume_params.camera_pos[..3], vol_center);
    }

    pub fn stop_simulation(&mut self) {
        self.is_running = false;
        // Log hit count when stopping
        if let Some(detector) = &self.detector_buffer {
            let mut header = [0u32; 4];
            if detector.read(&mut header).is_ok() {
                log::info!("Simulation stopped - total hits: {}", header[0]);
            } else {
                log::info!("Simulation stopped");
            }
        } else {
            log::info!("Simulation stopped");
        }
    }

    pub fn toggle_simulation(&mut self) {
        if self.is_running {
            self.stop_simulation();
        } else {
            self.start_simulation();
        }
    }

    /// Record compute dispatch commands for simulation steps
    fn record_compute_commands(&self, device: &ash::Device, cmd: vk::CommandBuffer) {
        if let Some(pipeline) = &self.compute_pipeline {
            unsafe {
                // Transition detector texture to GENERAL for compute writes
                let barrier = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    new_layout: vk::ImageLayout::GENERAL,
                    image: self.detector_texture.image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_access_mask: vk::AccessFlags::SHADER_READ,
                    dst_access_mask: vk::AccessFlags::SHADER_WRITE,
                    ..Default::default()
                };

                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );

                // Calculate number of dispatches needed
                // Each dispatch does STEPS_PER_DISPATCH steps with particles in registers
                let num_dispatches = (STEPS_PER_FRAME + STEPS_PER_DISPATCH - 1) / STEPS_PER_DISPATCH;

                for _ in 0..num_dispatches {
                    pipeline.record_dispatch(device, cmd, &self.sim_params, 256);

                    // Memory barrier between dispatches (only needed if multiple dispatches)
                    if num_dispatches > 1 {
                        let mem_barrier = vk::MemoryBarrier {
                            src_access_mask: vk::AccessFlags::SHADER_WRITE,
                            dst_access_mask: vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
                            ..Default::default()
                        };

                        device.cmd_pipeline_barrier(
                            cmd,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::PipelineStageFlags::COMPUTE_SHADER,
                            vk::DependencyFlags::empty(),
                            &[mem_barrier],
                            &[],
                            &[],
                        );
                    }
                }

                // Transition detector texture back to SHADER_READ_ONLY_OPTIMAL for fragment
                let barrier = vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::GENERAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    image: self.detector_texture.image,
                    subresource_range: vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    },
                    src_access_mask: vk::AccessFlags::SHADER_WRITE,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    ..Default::default()
                };

                device.cmd_pipeline_barrier(
                    cmd,
                    vk::PipelineStageFlags::COMPUTE_SHADER,
                    vk::PipelineStageFlags::FRAGMENT_SHADER,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[barrier],
                );
            }
        }
    }

    /// Render a frame (without egui)
    pub fn render_frame(&mut self) -> Result<bool> {
        self.render_frame_with_egui(None)
    }

    /// Render a frame with optional egui overlay
    pub fn render_frame_with_egui(
        &mut self,
        egui_data: Option<(&[egui::ClippedPrimitive], [f32; 2], f32)>,
    ) -> Result<bool> {
        // Don't render during shutdown
        if self.is_shutting_down {
            return Ok(false);
        }

        let device = self.ctx.device();
        let frame = &self.frames[self.current_frame];

        unsafe {
            // Acquire next swapchain image (panics if called headlessly — programmer error)
            let (image_index, suboptimal) = match self.swapchain.as_mut().unwrap().acquire_next_image(frame.image_available) {
                Ok(result) => result,
                Err(_) => return Ok(true), // Need resize
            };

            if suboptimal {
                return Ok(true);
            }

            // Wait for previous use of this frame
            device.wait_for_fences(&[frame.in_flight_fence], true, u64::MAX)?;
            device.reset_fences(&[frame.in_flight_fence])?;

            // Display params are set in start_simulation()

            let cmd = frame.command_buffer;
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(cmd, &begin_info)?;

            // Start frame timing
            self.gpu_timing.begin_frame(device, cmd);

            // Run compute (simulation step) if active
            let compute_ran = self.is_running && self.compute_pipeline.is_some() && self.particle_buffer.is_some();
            if compute_ran {
                self.gpu_timing.begin_compute(device, cmd);
                self.record_compute_commands(device, cmd);
                self.gpu_timing.end_compute(device, cmd);
            }

            let swapchain_extent = self.swapchain.as_ref().unwrap().extent;

            // Begin render pass
            self.gpu_timing.begin_render(device, cmd);
            self.detector_pipeline.as_ref().unwrap().begin_render_pass(
                device,
                cmd,
                image_index as usize,
                swapchain_extent,
            );

            // Render detector first (opaque, writes depth)
            self.detector_pipeline.as_ref().unwrap().draw(
                device,
                cmd,
                swapchain_extent,
                &self.detector_3d_params,
            );

            // Render volume second (transparent, tests against depth)
            // Volume fragments behind detector will be discarded
            // Skip during active benchmark to measure pure compute performance
            if !self.gpu_timing.is_benchmarking() {
                if let Some(ref volume_pipeline) = self.volume_pipeline {
                    volume_pipeline.record_commands(
                        device,
                        cmd,
                        swapchain_extent,
                        &self.volume_params,
                    );
                }

                // Render source marker (red sphere at source position)
                if let Some(ref marker_pipeline) = self.marker_pipeline {
                    marker_pipeline.draw(
                        device,
                        cmd,
                        swapchain_extent,
                        &self.source_marker_params,
                    );
                }

                // Render egui overlay
                if let Some((primitives, screen_size, ppp)) = egui_data {
                    if let Some(ref mut egui_renderer) = self.egui_renderer {
                        if let Err(e) = egui_renderer.render(cmd, primitives, screen_size, ppp) {
                            log::warn!("Egui render error: {}", e);
                        }
                    }
                }
            }

            // End render pass
            self.detector_pipeline.as_ref().unwrap().end_render_pass(device, cmd);
            self.gpu_timing.end_render(device, cmd);

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
            let suboptimal = self.swapchain.as_mut().unwrap().present(
                self.ctx.graphics_queue(),
                image_index,
                frame.render_finished,
            )?;

            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
            self.frame_count += 1;

            // For accurate timing during benchmarks, wait for GPU to complete
            // This serializes CPU/GPU but gives us accurate measurements
            if self.gpu_timing.is_benchmarking() {
                device.device_wait_idle()?;
            }

            // End frame timing AFTER GPU sync - this measures actual GPU execution time
            self.gpu_timing.end_frame(device, compute_ran);

            // Log progress every 30 frames during active simulation (silent after completion)
            if self.is_running && !self.benchmark_reported && self.frame_count % 30 == 0 {
                if let Some(detector) = &self.detector_buffer {
                    let mut header = [0u32; 4];
                    if detector.read(&mut header).is_ok() {
                        let hits = header[0];
                        let pct = 100.0 * hits as f32 / self.particle_count.max(1) as f32;

                        // Get current benchmark stats
                        let bench = self.gpu_timing.get_benchmark();
                        let avg_fps = bench.avg_fps();

                        log::info!("Frame {}: {} / {} hits ({:.1}%) | {:.1} fps",
                            self.frame_count, hits, self.particle_count, pct, avg_fps);

                        // Simulation is complete when all particles are accounted for
                        // (hit the detector or exited the domain).
                        let exits = header[1];
                        let accounted = hits.saturating_add(exits);
                        let all_accounted = accounted >= self.particle_count;
                        let mostly_done  = hits >= self.particle_count * 99 / 100;

                        if !self.benchmark_reported && (all_accounted || mostly_done) {
                            self.benchmark_reported = true;

                            // Stop benchmark and report final results
                            let results = self.gpu_timing.stop_benchmark();
                            results.log_summary();

                            // Log hit diagnostics
                            log::info!("Simulation complete ({:.1}% hit) - running diagnostics...",
                                100.0 * hits as f32 / self.particle_count.max(1) as f32);
                            self.log_hit_diagnostics();

                            // Auto-export PNG
                            let png_dir = std::path::PathBuf::from("output/png");
                            let cfg = self.png_cfg.clone();
                            match self.export_detector_png(&png_dir, &cfg) {
                                Ok(path) => log::info!("Auto-exported radiograph to {:?}", path),
                                Err(e) => log::warn!("Failed to auto-export PNG: {}", e),
                            }
                            log::info!("Simulation complete — window stays open for inspection. P to export, Esc to quit.");
                        }
                        self.total_hits = hits;
                    }
                }
            }

            Ok(suboptimal)
        }
    }

    /// Run compute shader only (no graphics rendering) - for batch/headless mode
    /// Returns true when all particles have finished (hit detector or left domain)
    pub fn run_compute_step(&mut self) -> Result<bool> {
        if !self.is_running || self.compute_pipeline.is_none() || self.particle_buffer.is_none() {
            return Ok(true); // Nothing to do
        }

        let device = self.ctx.device();
        let frame = &self.frames[self.current_frame];

        unsafe {
            // Wait for previous use of this frame
            device.wait_for_fences(&[frame.in_flight_fence], true, u64::MAX)?;
            device.reset_fences(&[frame.in_flight_fence])?;

            let cmd = frame.command_buffer;
            device.reset_command_buffer(cmd, vk::CommandBufferResetFlags::empty())?;

            let begin_info = vk::CommandBufferBeginInfo::default();
            device.begin_command_buffer(cmd, &begin_info)?;

            // Run compute only
            self.record_compute_commands(device, cmd);

            device.end_command_buffer(cmd)?;

            // Submit without semaphores (no swapchain sync needed)
            let submit_info = vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd,
                ..Default::default()
            };

            device.queue_submit(
                self.ctx.graphics_queue(),
                &[submit_info],
                frame.in_flight_fence,
            )?;

            self.current_frame = (self.current_frame + 1) % MAX_FRAMES_IN_FLIGHT;
            self.frame_count += 1;

            // Check if simulation is complete
            if let Some(detector) = &self.detector_buffer {
                let mut header = [0u32; 4];
                if detector.read(&mut header).is_ok() {
                    let hits  = header[0];
                    let exits = header[1];  // particles that left domain without hitting
                    let accounted = hits.saturating_add(exits);
                    let pct = 100.0 * accounted as f32 / self.particle_count.max(1) as f32;

                    // Log progress periodically
                    if self.frame_count % 10 == 0 {
                        log::info!("Batch frame {}: {:.1}% complete ({} hits, {} domain exits)",
                            self.frame_count, pct, hits, exits);
                    }

                    // Simulation is complete when every particle has either hit the
                    // detector or left the domain.  This is an exact count — no
                    // heuristic needed.
                    if accounted >= self.particle_count {
                        device.device_wait_idle()?;
                        return Ok(true);
                    }

                    self.total_hits = hits;
                }
            }

            Ok(false) // Not done yet
        }
    }

    /// Return (hits, exits) from the GPU detector buffer header.
    pub fn hit_exit_counts(&self) -> (u32, u32) {
        let Some(detector) = &self.detector_buffer else { return (0, 0) };
        let mut header = [0u32; 4];
        if detector.read(&mut header).is_ok() {
            (header[0], header[1])
        } else {
            (0, 0)
        }
    }

    /// True once every particle has either hit the detector or left the domain.
    pub fn is_simulation_complete(&self) -> bool {
        if self.particle_count == 0 { return false; }
        let (hits, exits) = self.hit_exit_counts();
        hits.saturating_add(exits) >= self.particle_count
    }

    /// True while the simulation is actively stepping particles.
    pub fn is_running(&self) -> bool { self.is_running }

    /// Resize the swapchain.  Returns `Ok(true)` when the swapchain extent
    /// still doesn't match the requested size (MoltenVK returned a stale
    /// `current_extent`); the caller should keep `needs_resize = true` and
    /// retry next frame.
    pub fn resize(&mut self, width: u32, height: u32) -> Result<bool> {
        if width == 0 || height == 0 {
            return Ok(false);
        }

        unsafe {
            self.ctx.device().device_wait_idle()?;
        }

        self.swapchain.as_mut().unwrap().recreate(
            self.ctx.instance(),
            self.ctx.device(),
            self.ctx.physical_device(),
            self.ctx.surface_loader(),
            self.ctx.surface(),
            self.ctx.graphics_queue_family(),
            width,
            height,
        )?;

        // Recreate framebuffers for new swapchain (includes depth buffer)
        self.detector_pipeline.as_mut().unwrap().recreate_framebuffers(
            self.ctx.device(),
            &self.allocator,
            self.swapchain.as_ref().unwrap().image_views(),
            self.swapchain.as_ref().unwrap().extent,
        )?;

        let actual = self.swapchain.as_ref().unwrap().extent;
        let extent_mismatch = actual.width != width || actual.height != height;
        log::info!("Swapchain resized to {}x{}", actual.width, actual.height);
        Ok(extent_mismatch)
    }

    /// Update volume and detector rendering from external camera
    pub fn update_camera(&mut self, camera: &crate::camera::Camera) {
        let pos = camera.position();
        let inv_view_proj = camera.inv_view_proj();
        let view_proj = camera.view_proj();

        // Update volume params
        self.volume_params.inv_view_proj = inv_view_proj.to_cols_array_2d();
        self.volume_params.view_proj = view_proj.to_cols_array_2d();
        self.volume_params.camera_pos = [pos.x, pos.y, pos.z, 1.0];

        // Update detector 3D params
        self.detector_3d_params.view_proj = view_proj.to_cols_array_2d();
        self.detector_3d_params.detector_pos = self.sim_params.detector_pos;
        self.detector_3d_params.detector_normal = self.sim_params.detector_normal;
        self.detector_3d_params.detector_extent = self.sim_params.detector_extent;
        // Copy display params
        self.detector_3d_params.max_count = self.display_params.max_count;
        self.detector_3d_params.gamma = self.display_params.gamma;
        self.detector_3d_params.exposure = self.display_params.exposure;
        self.detector_3d_params.use_log_scale = self.display_params.use_log_scale;
        self.detector_3d_params.colormap_mode = self.display_params.colormap_mode;

        // Update marker params with current view_proj
        self.source_marker_params.view_proj = view_proj.to_cols_array_2d();
    }

    /// Toggle between colormap modes (RCF film vs scientific)
    pub fn toggle_colormap(&mut self) {
        self.display_params.colormap_mode = (self.display_params.colormap_mode + 1) % 2;
        let mode_name = match self.display_params.colormap_mode {
            0 => "RCF Film (realistic)",
            1 => "Scientific (dark→light)",
            _ => "Unknown",
        };
        log::info!("Colormap: {}", mode_name);
    }

    /// Toggle log scale on/off
    pub fn toggle_log_scale(&mut self) {
        self.display_params.use_log_scale = if self.display_params.use_log_scale != 0 { 0 } else { 1 };
        log::info!("Log scale: {}", if self.display_params.use_log_scale != 0 { "ON" } else { "OFF" });
    }

    /// Adjust exposure by a factor
    pub fn adjust_exposure(&mut self, factor: f32) {
        self.display_params.exposure = (self.display_params.exposure * factor).clamp(0.1, 10.0);
        log::info!("Exposure: {:.2}", self.display_params.exposure);
    }

    /// Adjust gamma
    pub fn adjust_gamma(&mut self, delta: f32) {
        self.display_params.gamma = (self.display_params.gamma + delta).clamp(0.2, 2.0);
        log::info!("Gamma: {:.2}", self.display_params.gamma);
    }

    /// Get current display settings for UI display
    pub fn display_info(&self) -> String {
        format!(
            "Colormap: {} | Log: {} | Exp: {:.1} | Gamma: {:.2}",
            if self.display_params.colormap_mode == 0 { "RCF" } else { "Sci" },
            if self.display_params.use_log_scale != 0 { "ON" } else { "OFF" },
            self.display_params.exposure,
            self.display_params.gamma,
        )
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

        let allocation = detector.allocation.as_ref().context("No allocation")?;
        let ptr = allocation.mapped_ptr().context("Buffer not mapped")?;

        unsafe {
            let hits_ptr = (ptr.as_ptr() as *const u8).add(16) as *const DetectorHit;
            std::ptr::copy_nonoverlapping(hits_ptr, hits.as_mut_ptr(), hit_count);
        }

        Ok(hits)
    }

    /// Analyze and log hit distribution for sanity checking
    pub fn log_hit_diagnostics(&self) {
        if let Ok(hits) = self.read_detector_hits() {
            if hits.is_empty() {
                log::info!("Hit diagnostics: No hits recorded");
                return;
            }

            let n = hits.len();
            log::info!("Hit diagnostics ({} hits):", n);

            let mut sum_x = 0.0f64;
            let mut sum_y = 0.0f64;
            let mut sum_x2 = 0.0f64;
            let mut sum_y2 = 0.0f64;
            let mut min_x = f32::MAX;
            let mut max_x = f32::MIN;
            let mut min_y = f32::MAX;
            let mut max_y = f32::MIN;

            for hit in &hits {
                let x = hit.position[0] as f64;
                let y = hit.position[1] as f64;
                sum_x  += x;
                sum_y  += y;
                sum_x2 += x * x;
                sum_y2 += y * y;
                min_x = min_x.min(hit.position[0]);
                max_x = max_x.max(hit.position[0]);
                min_y = min_y.min(hit.position[1]);
                max_y = max_y.max(hit.position[1]);
            }

            let mean_x = sum_x / n as f64;
            let mean_y = sum_y / n as f64;
            let std_x  = ((sum_x2 / n as f64) - mean_x * mean_x).sqrt();
            let std_y  = ((sum_y2 / n as f64) - mean_y * mean_y).sqrt();

            // Report in mm (positions from shader are in metres)
            // position[0] → y_mm, position[1] → z_mm (detector-plane axes)
            log::info!("  Y: mean={:+.2} mm, std={:.2} mm, range=[{:.2}, {:.2}] mm",
                mean_x * 1e3, std_x * 1e3, min_x * 1e3, max_x * 1e3);
            log::info!("  Z: mean={:+.2} mm, std={:.2} mm, range=[{:.2}, {:.2}] mm",
                mean_y * 1e3, std_y * 1e3, min_y * 1e3, max_y * 1e3);
        }
    }

    /// Texture resolution used for the detector accumulation buffer.
    pub fn detector_texture_size(&self) -> (u32, u32) {
        (DETECTOR_RESOLUTION, DETECTOR_RESOLUTION)
    }

    /// Compute hit distribution statistics without logging, for metadata.json.
    pub fn compute_hit_diagnostics(&self) -> Option<RunDiagnostics> {
        let hits = self.read_detector_hits().ok()?;
        if hits.is_empty() {
            return None;
        }

        let n = hits.len() as u64;
        let mut sum_y = 0.0f64;
        let mut sum_z = 0.0f64;
        let mut sum_y2 = 0.0f64;
        let mut sum_z2 = 0.0f64;

        for hit in &hits {
            let y = hit.position[0] as f64;
            let z = hit.position[1] as f64;
            sum_y  += y;  sum_z  += z;
            sum_y2 += y * y; sum_z2 += z * z;
        }

        let mean_y = sum_y / n as f64;
        let mean_z = sum_z / n as f64;
        let std_y  = ((sum_y2 / n as f64) - mean_y * mean_y).max(0.0).sqrt();
        let std_z  = ((sum_z2 / n as f64) - mean_z * mean_z).max(0.0).sqrt();

        Some(RunDiagnostics {
            n_particles: self.sim_params.n_particles,
            n_hits: n,
            hit_fraction: n as f64 / self.sim_params.n_particles.max(1) as f64,
            mean_y_m: mean_y,
            std_y_m: std_y,
            mean_z_m: mean_z,
            std_z_m: std_z,
        })
    }

    /// Export counts + PNG into a structured run directory.
    ///
    /// Writes:
    ///   counts/raw_counts.bin        (u32 little-endian)
    ///   counts/processed_counts.bin  (f32 little-endian)
    ///   counts/hits.bin              (u32 n_hits header + f32 y_mm,z_mm,energy_MeV triples)
    ///   images/radiograph.png
    ///
    /// Returns `(raw_counts, processed_counts)` for use in metadata construction.
    pub fn export_to_run_dir(
        &self,
        run_dir: &RunDir,
        cfg: &PngExportConfig,
        save_hits: bool,
    ) -> Result<(Vec<u32>, Vec<f32>)> {
        let tex_w = DETECTOR_RESOLUTION as usize;
        let tex_h = DETECTOR_RESOLUTION as usize;

        let half_y_mm = self.sim_params.detector_extent[0] as f64 * 1e3;
        let half_z_mm = self.sim_params.detector_extent[1] as f64 * 1e3;
        let pixel_pitch_y_um = half_y_mm * 2.0 / tex_w as f64 * 1e3;
        let pixel_pitch_z_um = half_z_mm * 2.0 / tex_h as f64 * 1e3;
        let detector_info = DetectorRenderInfo {
            width_px: tex_w, height_px: tex_h,
            pixel_pitch_y_um, pixel_pitch_z_um,
        };

        let raw_counts = self.read_detector_counts_from_gpu()?;

        std::fs::write(run_dir.raw_counts_path(), bytemuck::cast_slice::<u32, u8>(&raw_counts))
            .context("Failed to write raw_counts.bin")?;
        log::info!("Wrote raw counts       → {:?}", run_dir.raw_counts_path());

        let raw_f32: Vec<f32> = raw_counts.iter().map(|&c| c as f32).collect();
        let processed = apply_detector_response(
            &raw_f32, tex_w, tex_h, &detector_info, &self.detector_response,
        );

        std::fs::write(run_dir.processed_counts_path(), bytemuck::cast_slice::<f32, u8>(&processed))
            .context("Failed to write processed_counts.bin")?;
        log::info!("Wrote processed counts → {:?}", run_dir.processed_counts_path());

        let img = render_hitmap_f32(&processed, tex_w as u32, tex_h as u32, cfg);
        img.save(run_dir.radiograph_png_path())
            .map_err(|e| anyhow::anyhow!("Failed to save radiograph.png: {}", e))?;
        log::info!("Wrote radiograph       → {:?}", run_dir.radiograph_png_path());

        if save_hits {
            let hits = self.read_detector_hits()?;
            Self::write_hits_bin(&hits, &run_dir.hits_bin_path())?;
            log::info!("Wrote hits             → {:?}", run_dir.hits_bin_path());
        }

        Ok((raw_counts, processed))
    }

    /// Write per-hit binary: 4-byte little-endian u32 count, then (y_mm, z_mm, energy_MeV)
    /// as f32 LE triples.  Readable in Python:
    ///   n = np.frombuffer(data[:4], dtype='<u4')[0]
    ///   hits = np.frombuffer(data[4:], dtype='<f4').reshape(n, 3)
    fn write_hits_bin(hits: &[DetectorHit], path: &std::path::Path) -> Result<()> {
        const PROTON_MASS_KG: f64 = 1.672_621_923_69e-27;
        const MEV_J: f64          = 1.602_176_634e-13;

        let n = hits.len() as u32;
        let mut buf = Vec::with_capacity(4 + hits.len() * 12);
        buf.extend_from_slice(&n.to_le_bytes());
        for hit in hits {
            let y_mm       = hit.position[0] * 1e3;
            let z_mm       = hit.position[1] * 1e3;
            // hit.energy is (γ-1)c² [J/kg]; multiply by proton mass to get KE in Joules
            let energy_mev = (hit.energy as f64 * PROTON_MASS_KG / MEV_J) as f32;
            buf.extend_from_slice(&y_mm.to_le_bytes());
            buf.extend_from_slice(&z_mm.to_le_bytes());
            buf.extend_from_slice(&energy_mev.to_le_bytes());
        }
        std::fs::write(path, &buf).context("Failed to write hits.bin")?;
        Ok(())
    }

    /// Export detector hits to CSV (positions in mm, energies in MeV) plus a
    /// JSON metadata sidecar.
    pub fn export_detector_data(&self, path: &std::path::Path) -> Result<()> {
        use std::io::Write;

        const PROTON_MASS_KG: f64 = 1.672_621_923_69e-27;
        const MEV_J: f64          = 1.602_176_634e-13;

        let hits = self.read_detector_hits()?;
        if hits.is_empty() {
            log::warn!("No hits to export");
            return Ok(());
        }

        // ── CSV ──────────────────────────────────────────────────────────────
        let mut file = std::fs::File::create(path)
            .context("Failed to create export file")?;

        // Coordinate convention: +x is beam axis; detector plane is y–z.
        // hit.position[0] = dot(local, y_axis) → y_mm
        // hit.position[1] = dot(local, z_axis) → z_mm
        writeln!(file, "y_mm,z_mm,ke_MeV,particle_id")?;

        for hit in &hits {
            let y_mm   = hit.position[0] as f64 * 1e3;
            let z_mm   = hit.position[1] as f64 * 1e3;
            let ke_mev = (hit.energy as f64) * PROTON_MASS_KG / MEV_J;
            writeln!(file, "{:.6},{:.6},{:.6},{}", y_mm, z_mm, ke_mev, hit.particle_id)?;
        }

        // ── metadata sidecar ─────────────────────────────────────────────────
        let stem = path.file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("output");
        let meta_path = path.with_file_name(format!("{}_meta.json", stem));

        let meta = serde_json::json!({
            "n_particles":      self.sim_params.n_particles,
            "n_hits":           hits.len(),
            "detector_pos_m":   &self.sim_params.detector_pos[..3],
            "detector_normal":  &self.sim_params.detector_normal[..3],
            "detector_extent_m": &self.sim_params.detector_extent[..2],
            "columns":          ["y_mm", "z_mm", "ke_MeV", "particle_id"],
            "position_units":   "mm  (detector-plane: y_mm along up-axis, z_mm along cross(normal,up))",
            "energy_units":     "MeV"
        });

        std::fs::write(&meta_path,
            serde_json::to_string_pretty(&meta).unwrap())
            .context("Failed to write metadata sidecar")?;

        log::info!("Exported {} hits to {:?}", hits.len(), path);
        Ok(())
    }

    /// Export detector texture as PNG image.
    ///
    /// Read R32_UINT detector texture back from GPU into a flat Vec<u32>.
    /// Row-major layout: index = row * width + col, where col ↔ y_mm, row ↔ z_mm.
    fn read_detector_counts_from_gpu(&self) -> Result<Vec<u32>> {
        let device = self.ctx.device();
        let queue  = self.ctx.compute_queue();
        let tex_w  = DETECTOR_RESOLUTION;
        let tex_h  = DETECTOR_RESOLUTION;

        let buffer_size = (tex_w * tex_h * std::mem::size_of::<u32>() as u32) as vk::DeviceSize;
        let staging = StagingBuffer::new_readback(device, &self.allocator, buffer_size)
            .context("Failed to create staging buffer for readback")?;

        unsafe {
            let alloc_info = vk::CommandBufferAllocateInfo {
                command_pool: self.ctx.command_pool(),
                level: vk::CommandBufferLevel::PRIMARY,
                command_buffer_count: 1,
                ..Default::default()
            };
            let cmd = device.allocate_command_buffers(&alloc_info)?[0];

            device.begin_command_buffer(cmd, &vk::CommandBufferBeginInfo {
                flags: vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT,
                ..Default::default()
            })?;

            let subresource_range = vk::ImageSubresourceRange {
                aspect_mask: vk::ImageAspectFlags::COLOR,
                base_mip_level: 0,
                level_count: 1,
                base_array_layer: 0,
                layer_count: 1,
            };

            device.cmd_pipeline_barrier(cmd,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::PipelineStageFlags::TRANSFER,
                vk::DependencyFlags::empty(), &[], &[],
                &[vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    new_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    image: self.detector_texture.image,
                    subresource_range,
                    src_access_mask: vk::AccessFlags::SHADER_READ,
                    dst_access_mask: vk::AccessFlags::TRANSFER_READ,
                    ..Default::default()
                }],
            );

            device.cmd_copy_image_to_buffer(cmd,
                self.detector_texture.image,
                vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                staging.buffer(),
                &[vk::BufferImageCopy {
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
                    image_extent: vk::Extent3D { width: tex_w, height: tex_h, depth: 1 },
                }],
            );

            device.cmd_pipeline_barrier(cmd,
                vk::PipelineStageFlags::TRANSFER,
                vk::PipelineStageFlags::FRAGMENT_SHADER,
                vk::DependencyFlags::empty(), &[], &[],
                &[vk::ImageMemoryBarrier {
                    old_layout: vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
                    new_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
                    image: self.detector_texture.image,
                    subresource_range,
                    src_access_mask: vk::AccessFlags::TRANSFER_READ,
                    dst_access_mask: vk::AccessFlags::SHADER_READ,
                    ..Default::default()
                }],
            );

            device.end_command_buffer(cmd)?;
            device.queue_submit(queue, &[vk::SubmitInfo {
                command_buffer_count: 1,
                p_command_buffers: &cmd,
                ..Default::default()
            }], vk::Fence::null())?;
            device.queue_wait_idle(queue)?;
            device.free_command_buffers(self.ctx.command_pool(), &[cmd]);
        }

        let mut raw_counts = vec![0u32; (tex_w * tex_h) as usize];
        let mut staging = staging;
        staging.read(&mut raw_counts)?;
        staging.cleanup(device, &self.allocator);

        Ok(raw_counts)
    }

    /// Reads the R32_UINT detector texture from GPU, applies detector response,
    /// exports raw + processed count grids, renders a PNG, and writes all sidecars.
    ///
    /// output_dir: base directory for this run (counts go here; PNG goes into output_dir/png/).
    pub fn export_detector_png(
        &self,
        output_dir: &std::path::Path,
        cfg: &PngExportConfig,
    ) -> Result<std::path::PathBuf> {
        use chrono::Local;

        std::fs::create_dir_all(output_dir)
            .context("Failed to create output directory")?;

        let timestamp = Local::now().format("%Y%m%d_%H%M%S");
        let stem = format!("radiograph_{}", timestamp);

        // ── Detector geometry ─────────────────────────────────────────────────
        let tex_w = DETECTOR_RESOLUTION as usize;
        let tex_h = DETECTOR_RESOLUTION as usize;
        // detector_extent[0] = half-width (y axis), detector_extent[1] = half-height (z axis), in metres.
        // GPU texture: col ↔ y_mm, row ↔ z_mm.
        let half_y_mm = self.sim_params.detector_extent[0] as f64 * 1e3;
        let half_z_mm = self.sim_params.detector_extent[1] as f64 * 1e3;
        let pixel_pitch_y_um = half_y_mm * 2.0 / tex_w as f64 * 1e3;
        let pixel_pitch_z_um = half_z_mm * 2.0 / tex_h as f64 * 1e3;
        let detector_info = DetectorRenderInfo {
            width_px: tex_w, height_px: tex_h,
            pixel_pitch_y_um, pixel_pitch_z_um,
        };

        // ── GPU readback ──────────────────────────────────────────────────────
        let raw_counts = self.read_detector_counts_from_gpu()?;

        // ── Write raw counts ──────────────────────────────────────────────────
        write_raw_counts(output_dir, &stem, &raw_counts,
            tex_w as u32, tex_h as u32, &detector_info)?;

        // ── Apply detector response ───────────────────────────────────────────
        let raw_total: u64 = raw_counts.iter().map(|&c| c as u64).sum();
        let raw_f32: Vec<f32> = raw_counts.iter().map(|&c| c as f32).collect();
        let processed = apply_detector_response(
            &raw_f32, tex_w, tex_h, &detector_info, &self.detector_response,
        );

        // ── Write processed counts ────────────────────────────────────────────
        write_processed_counts(output_dir, &stem, &processed,
            tex_w as u32, tex_h as u32, &detector_info, &self.detector_response, raw_total)?;

        // ── Resolve PNG output resolution ─────────────────────────────────────
        let out_pixels = cfg.output_pixels.unwrap_or([tex_w as u32, tex_h as u32]);
        let (out_w, out_h) = (out_pixels[0], out_pixels[1]);

        let display_counts: Vec<f32> = if out_w == tex_w as u32 && out_h == tex_h as u32 {
            processed.clone()
        } else {
            let mut scaled = vec![0.0f32; (out_w * out_h) as usize];
            for oy in 0..out_h {
                for ox in 0..out_w {
                    let sx = (ox * tex_w as u32 / out_w).min(tex_w as u32 - 1) as usize;
                    let sy = (oy * tex_h as u32 / out_h).min(tex_h as u32 - 1) as usize;
                    scaled[(oy * out_w + ox) as usize] = processed[sy * tex_w + sx];
                }
            }
            scaled
        };

        let total_hits: u64 = raw_counts.iter().map(|&c| c as u64).sum();
        let max_count = display_counts.iter().cloned().fold(0.0f32, f32::max).max(1.0);

        // ── Render PNG ───────────────────────────────────────────────────────
        let output_path = output_dir.join(format!("{}.png", stem));
        let img = render_hitmap_f32(&display_counts, out_w, out_h, cfg);
        img.save(&output_path).context("Failed to save PNG")?;

        // ── PNG Sidecar JSON ──────────────────────────────────────────────────
        if cfg.include_metadata {
            let colorbar_width: u32 = if cfg.include_colorbar { 24 } else { 0 };
            let scale_name = match cfg.scale {
                ScaleMode::Linear => "linear",
                ScaleMode::Log    => "log",
                ScaleMode::Sqrt   => "sqrt",
            };
            let colormap_name = match cfg.colormap {
                ColormapType::RcfFilm    => "rcf_film",
                ColormapType::Scientific => "scientific",
                ColormapType::Grayscale  => "grayscale",
                ColormapType::Hot        => "hot",
                ColormapType::Inverted   => "inverted",
            };

            let meta = serde_json::json!({
                "image": {
                    "data_width_px":     out_w,
                    "data_height_px":    out_h,
                    "png_width_px":      out_w + colorbar_width,
                    "png_height_px":     out_h,
                    "colorbar_width_px": colorbar_width,
                    "scale":             scale_name,
                    "colormap":          colormap_name,
                    "max_count":         max_count,
                    "total_hits":        total_hits,
                },
                "render": {
                    "render_source":    "gpu_detector_texture",
                    "gamma":            cfg.gamma,
                    "exposure":         cfg.exposure,
                    "normalization":    "max",
                    "zero_count_color": [0, 0, 0],
                    "colorbar":         cfg.include_colorbar,
                },
                "detector": {
                    "width_mm":         half_y_mm * 2.0,
                    "height_mm":        half_z_mm * 2.0,
                    "axes":             { "row": "z_mm", "col": "y_mm" },
                    "y_range_mm":       [-half_y_mm, half_y_mm],
                    "z_range_mm":       [-half_z_mm, half_z_mm],
                    "pixel_pitch_y_um": pixel_pitch_y_um,
                    "pixel_pitch_z_um": pixel_pitch_z_um,
                },
                "detector_response": {
                    "blur_sigma_um":    self.detector_response.blur_sigma_um,
                    "background_counts": self.detector_response.background_counts,
                    "poisson_noise":    self.detector_response.poisson_noise,
                },
                "source": {
                    "source_type":  self.source_type,
                    "energy_MeV":   self.particle_energy_mev,
                    "n_particles":  self.sim_params.n_particles,
                },
            });

            let meta_path = output_dir.join(format!("{}_meta.json", stem));
            std::fs::write(&meta_path, serde_json::to_string_pretty(&meta).unwrap())
                .context("Failed to write PNG metadata sidecar")?;
            log::info!("Wrote PNG sidecar → {:?}", meta_path);
        }

        log::info!("Exported radiograph → {:?}", output_path);
        Ok(output_path)
    }

    pub fn cleanup(&mut self) {
        self.is_shutting_down = true;

        let device = self.ctx.device();

        unsafe {
            device.device_wait_idle().ok();
        }

        // Clean up buffers and textures
        if let Some(mut buf) = self.particle_buffer.take() {
            buf.cleanup(device, &self.allocator);
        }
        if let Some(mut tex) = self.field_texture.take() {
            tex.cleanup(device, &self.allocator);
        }
        if let Some(mut tex) = self.e_field_texture.take() {
            tex.cleanup(device, &self.allocator);
        }
        if let Some(mut buf) = self.detector_buffer.take() {
            buf.cleanup(device, &self.allocator);
        }
        if let Some(mut pipe) = self.compute_pipeline.take() {
            pipe.cleanup(device);
        }

        // Clean up detector texture
        self.detector_texture.cleanup(device, &self.allocator);

        // Clean up volume pipeline
        if let Some(mut pipe) = self.volume_pipeline.take() {
            pipe.cleanup(device);
        }

        // Clean up marker pipeline
        if let Some(mut pipe) = self.marker_pipeline.take() {
            pipe.cleanup(device);
        }

        // Clean up graphics pipeline (includes depth buffer)
        if let Some(mut dp) = self.detector_pipeline.take() {
            dp.cleanup(device, &self.allocator);
        }

        // Clean up GPU timing
        self.gpu_timing.cleanup(device);

        // Clean up frames
        unsafe {
            for frame in &self.frames {
                device.destroy_semaphore(frame.image_available, None);
                device.destroy_semaphore(frame.render_finished, None);
                device.destroy_fence(frame.in_flight_fence, None);
            }
        }

        if let Some(mut sc) = self.swapchain.take() {
            sc.cleanup(device);
        }
    }
}
