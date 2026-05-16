//! Proton Radiography Tracer - GPU Accelerated
//!
//! Traces charged particles through electromagnetic fields using Vulkan compute.
//!
//! Usage:
//!   proton_tracer <config.json>              # Interactive mode with visualization
//!   proton_tracer <config.json> --batch      # Batch mode: run simulation and export CSV
//!   proton_tracer <config.json> --batch -o output_dir  # Specify output directory

#[cfg(target_os = "macos")]
#[macro_use]
extern crate objc;

mod gpu;
mod loaders;
mod units;
mod camera;
mod config;
mod gui;
mod run_dir;
mod overrides;
mod sweep;
mod inspect;

use anyhow::Result;
use winit::{
    application::ApplicationHandler,
    event::{WindowEvent, MouseButton, ElementState, MouseScrollDelta},
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};
use std::sync::Arc;
use std::path::PathBuf;

use gpu::{VulkanContext, Renderer, render_hitmap_f32};
use loaders::{FieldData, ParticleData, SimConfig, SimSourceGeometry};
use camera::Camera;
use config::{PngExportConfig, ScaleMode, ColormapType, DetectorResponseConfig};
use gui::{Gui, DeckDisplay, RunState};

/// Command-line arguments
struct CliArgs {
    config_path: Option<String>,
    batch_mode: bool,
    output_dir: PathBuf,
    run_opts: run_dir::RunOptions,
    overrides: Vec<overrides::ConfigOverride>,
}

impl CliArgs {
    /// Parse legacy-style args: `proton_tracer [config] [--batch] [-o dir]`
    fn parse_legacy(argv: &[String]) -> Self {
        let mut config_path = None;
        let mut batch_mode = false;
        let mut output_dir = PathBuf::from("output");

        let mut i = 0;
        while i < argv.len() {
            match argv[i].as_str() {
                "--batch" | "-b" => batch_mode = true,
                "--output" | "-o" => {
                    i += 1;
                    if i < argv.len() {
                        output_dir = PathBuf::from(&argv[i]);
                    }
                }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                arg if !arg.starts_with('-') && config_path.is_none() => {
                    config_path = Some(arg.to_string());
                }
                _ => {}
            }
            i += 1;
        }

        Self { config_path, batch_mode, output_dir, run_opts: run_dir::RunOptions::default(), overrides: vec![] }
    }
}

fn print_help() {
    println!("Proton Radiography Tracer");
    println!();
    println!("Usage:");
    println!("  proton-tracer run   <deck.toml> [-o dir]        Batch run");
    println!("  proton-tracer gui   [deck.toml]                 Interactive GUI");
    println!("  proton-tracer demo  [preset]    [--batch -o d]  Demo preset");
    println!("  proton-tracer init  [preset]    [-o deck.toml]  Emit preset template");
    println!("  proton-tracer explain <deck>                    Print resolved geometry");
    println!("  proton-tracer validate <deck>                   Schema check");
    println!("  proton-tracer render  --hits hits.csv ...       Re-render without GPU");
    println!("  proton-tracer sweep   <deck> --param k=v1,v2   Parameter sweep");
    println!("  proton-tracer inspect <run_dir|sweep_dir>       Print run/sweep summary");
    println!("  proton-tracer analyze <run_dir> [--raw]         Count statistics");
    println!();
    println!("Demo presets: zpinch, sausage-weak, sausage-strong, kink-weak, kink-strong, mixed");
    println!("Init presets: blank, zpinch, kink-strong");
}

fn parse_run_args(argv: &[String]) -> CliArgs {
    let mut config_path = None;
    let mut output_dir = PathBuf::from("output");
    let mut overwrite = false;
    let mut resume = false;
    let mut cli_overrides: Vec<overrides::ConfigOverride> = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "-o" | "--output" => { i += 1; if i < argv.len() { output_dir = PathBuf::from(&argv[i]); } }
            "--overwrite"     => { overwrite = true; }
            "--resume"        => { resume = true; }
            "--set" => {
                i += 1;
                if let Some(s) = argv.get(i) {
                    match overrides::parse_override(s) {
                        Ok(ov) => cli_overrides.push(ov),
                        Err(e) => { eprintln!("Error: {}", e); std::process::exit(1); }
                    }
                } else {
                    eprintln!("Error: --set requires a key=value argument");
                    std::process::exit(1);
                }
            }
            arg if !arg.starts_with('-') && config_path.is_none() => {
                config_path = Some(arg.to_string());
            }
            _ => {}
        }
        i += 1;
    }
    CliArgs {
        config_path,
        batch_mode: true,
        output_dir,
        run_opts: run_dir::RunOptions { overwrite, resume },
        overrides: cli_overrides,
    }
}

fn parse_gui_args(argv: &[String]) -> CliArgs {
    let mut config_path = None;
    let mut output_dir = PathBuf::from("output");
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "-o" | "--output" => { i += 1; if i < argv.len() { output_dir = PathBuf::from(&argv[i]); } }
            arg if !arg.starts_with('-') && config_path.is_none() => {
                config_path = Some(arg.to_string());
            }
            _ => {}
        }
        i += 1;
    }
    CliArgs { config_path, batch_mode: false, output_dir, run_opts: run_dir::RunOptions::default(), overrides: vec![] }
}

fn parse_demo_args(argv: &[String]) -> CliArgs {
    let mut preset: Option<String> = None;
    let mut batch_mode = false;
    let mut output_dir = PathBuf::from("output");
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--batch" | "-b" => batch_mode = true,
            "-o" | "--output" => { i += 1; if i < argv.len() { output_dir = PathBuf::from(&argv[i]); } }
            arg if !arg.starts_with('-') && preset.is_none() => {
                preset = Some(arg.to_string());
            }
            _ => {}
        }
        i += 1;
    }
    let config_path = preset.as_deref().and_then(demo_preset_path);
    if preset.is_some() && config_path.is_none() {
        log::warn!("Unknown demo preset: {}. Use: zpinch, kink-strong, etc.",
            preset.as_deref().unwrap_or(""));
    }
    CliArgs { config_path, batch_mode, output_dir, run_opts: run_dir::RunOptions::default(), overrides: vec![] }
}

fn demo_preset_path(preset: &str) -> Option<String> {
    match preset {
        "zpinch" | "z-pinch"     => Some("data/instabilities/zpinch.json".into()),
        "sausage-weak"            => Some("data/instabilities/sausage_weak.json".into()),
        "sausage-strong"          => Some("data/instabilities/sausage_strong.json".into()),
        "kink-weak"               => Some("data/instabilities/kink_weak.json".into()),
        "kink-strong"             => Some("data/instabilities/kink_strong.json".into()),
        "mixed"                   => Some("data/instabilities/mixed.json".into()),
        _ => None,
    }
}

/// Application state
struct App {
    window: Option<Arc<Window>>,
    vulkan: Option<Arc<VulkanContext>>,
    renderer: Option<Renderer>,
    camera: Option<Camera>,
    config_path: Option<String>,
    output_dir: PathBuf,
    batch_mode: bool,
    batch_frames_remaining: u32,
    needs_resize: bool,
    /// Set to true when Vulkan or renderer initialisation fails so main() can
    /// exit with a non-zero code and validate scripts can detect the failure.
    init_failed: bool,
    png_cfg: PngExportConfig,
    // GUI state
    egui_ctx: Option<egui::Context>,
    egui_winit: Option<egui_winit::State>,
    gui: Option<Gui>,
    gui_wants_input: bool,
    /// True once the current GUI-triggered run has been finalized (avoid double-export).
    gui_run_finalized: bool,
    // Phase 2: reproducible run system
    run_dir: Option<run_dir::RunDir>,
    run_metadata: Option<run_dir::RunMetadata>,
    resolved_config: Option<SimConfig>,
    run_start: Option<std::time::Instant>,
    /// Process argv — stored so GUI-triggered runs can populate RunMetadata.
    argv: Vec<String>,
    /// Phase 5A: --set overrides applied before config SI conversion.
    overrides: Vec<overrides::ConfigOverride>,
}

impl App {
    fn new(
        args: CliArgs,
        run_dir: Option<run_dir::RunDir>,
        run_metadata: Option<run_dir::RunMetadata>,
        argv: Vec<String>,
    ) -> Self {
        Self {
            window: None,
            vulkan: None,
            renderer: None,
            camera: None,
            config_path: args.config_path,
            output_dir: args.output_dir,
            batch_mode: args.batch_mode,
            batch_frames_remaining: if args.batch_mode { 100 } else { 0 },
            needs_resize: false,
            init_failed: false,
            png_cfg: PngExportConfig::default(),
            egui_ctx: None,
            egui_winit: None,
            gui: None,
            gui_wants_input: false,
            gui_run_finalized: false,
            run_dir,
            run_metadata,
            resolved_config: None,
            run_start: Some(std::time::Instant::now()),
            argv,
            overrides: args.overrides,
        }
    }

    /// Ensure output directory exists
    fn ensure_output_dir(&self) -> Result<()> {
        if !self.output_dir.exists() {
            std::fs::create_dir_all(&self.output_dir)?;
            log::info!("Created output directory: {:?}", self.output_dir);
        }
        Ok(())
    }

    /// Export detector data to output directory
    fn export_to_output_dir(&self, config_name: &str) -> Result<PathBuf> {
        self.ensure_output_dir()?;

        let renderer = self.renderer.as_ref().unwrap();
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);

        let filename = format!("{}_{}.csv", config_name, timestamp);
        let path = self.output_dir.join(&filename);

        renderer.export_detector_data(&path)?;
        Ok(path)
    }

    fn load_simulation(&mut self) -> Result<()> {
        let config_path = match &self.config_path {
            Some(p) => p.clone(),
            None => return Ok(()),
        };

        log::info!("Loading simulation from: {}", config_path);
        if !self.overrides.is_empty() {
            log::info!("Applying {} CLI override(s):", self.overrides.len());
        }

        // Load and convert config — applies --set overrides before SI conversion.
        let mut config = SimConfig::load_with_overrides(&config_path, &self.overrides)?;

        // Resolve field path relative to the config file
        let config_dir = std::path::Path::new(&config_path)
            .parent()
            .unwrap_or(std::path::Path::new("."));

        let field_path = config_dir.join(&config.field_path);
        log::info!("  Loading field from: {}", field_path.display());
        let mut field = FieldData::load(&field_path)?;
        log::info!("  Loaded field: {}x{}x{}", field.nx, field.ny, field.nz);

        // Optional separate E-field file
        if let Some(ref e_path_str) = config.e_field_path {
            let e_path = config_dir.join(e_path_str);
            log::info!("  Loading separate E-field from: {}", e_path.display());
            let e_field = FieldData::load(&e_path)?;
            field.set_e_from_separate_file(e_field)?;
            log::info!("  E-field merged");
        }

        // Both B and E are fully resolved — log diagnostics
        field.log_diagnostics();

        // ── Field SHA-256 + metadata update ───────────────────────────────────
        if let Some(meta) = &mut self.run_metadata {
            meta.input_files.field_path = Some(field_path.display().to_string());
            meta.input_files.field_sha256 = run_dir::sha256_file(&field_path).ok();
            if let Some(ref e_path_str) = config.e_field_path {
                let e_abs = config_dir.join(e_path_str);
                meta.input_files.e_field_path = Some(e_abs.display().to_string());
                meta.input_files.e_field_sha256 = run_dir::sha256_file(&e_abs).ok();
            }
            if let Some(rd) = &self.run_dir {
                let _ = rd.write_metadata(meta);
            }
        }

        // ── Resolve beam center ───────────────────────────────────────────────
        let beam_center: [f32; 3] = match &config.source.geometry {
            SimSourceGeometry::Pencil { position_m, .. } |
            SimSourceGeometry::Point  { position_m, .. } |
            SimSourceGeometry::Disk   { center_m: position_m, .. } => *position_m,
            SimSourceGeometry::ParallelBeam { center_m, .. } => {
                center_m.unwrap_or_else(|| {
                    if let Some(src_m) = config.source.source_distance_m {
                        let b = &field.bounds;
                        let cx = (b.x_min + b.x_max) / 2.0;
                        let cy = (b.y_min + b.y_max) / 2.0;
                        let cz = (b.z_min + b.z_max) / 2.0;
                        let d = config.source.beam_direction();
                        [cx - d[0] * src_m as f32,
                         cy - d[1] * src_m as f32,
                         cz - d[2] * src_m as f32]
                    } else {
                        log::warn!("No beam_center or source_distance_mm — defaulting to [-0.1, 0, 0]");
                        [-0.1, 0.0, 0.0]
                    }
                })
            }
        };
        if let SimSourceGeometry::ParallelBeam { ref mut center_m, .. } = config.source.geometry {
            *center_m = Some(beam_center);
        }

        // ── Resolve detector center ───────────────────────────────────────────
        let detector_center: [f32; 3] = if let Some(c) = config.detector.center_m {
            [c[0] as f32, c[1] as f32, c[2] as f32]
        } else if let Some(dist_m) = config.detector.distance_m {
            let beam_dir = config.source.beam_direction();
            let b = &field.bounds;
            let field_exit = [
                if beam_dir[0] > 0.0 { b.x_max } else if beam_dir[0] < 0.0 { b.x_min } else { 0.0 },
                if beam_dir[1] > 0.0 { b.y_max } else if beam_dir[1] < 0.0 { b.y_min } else { 0.0 },
                if beam_dir[2] > 0.0 { b.z_max } else if beam_dir[2] < 0.0 { b.z_min } else { 0.0 },
            ];
            let d = dist_m as f32;
            [field_exit[0] + d * beam_dir[0],
             field_exit[1] + d * beam_dir[1],
             field_exit[2] + d * beam_dir[2]]
        } else {
            anyhow::bail!("Detector center not resolved — set center_mm in the detector block");
        };
        config.detector.center_m = Some([
            detector_center[0] as f64,
            detector_center[1] as f64,
            detector_center[2] as f64,
        ]);

        // Auto-compute dt if not supplied
        if !config.dt_was_supplied {
            config.dt_s = compute_recommended_dt(&config, &field);
            log::info!("  dt auto-computed: {}", units::fmt_time(config.dt_s));
        }

        // ── RunDir: copy deck + write resolved_config.json ────────────────────
        if let Some(rd) = &self.run_dir {
            // Copy input deck snapshot
            let ext = std::path::Path::new(&config_path)
                .extension()
                .and_then(|e| e.to_str())
                .unwrap_or("toml");
            let deck_dest = rd.input_deck_path(ext);
            match std::fs::copy(&config_path, &deck_dest) {
                Ok(_) => {
                    log::info!("Saved input deck       → {:?}", deck_dest);
                    if let Some(meta) = &mut self.run_metadata {
                        meta.outputs.input_deck = Some(format!("input_deck.{}", ext));
                    }
                }
                Err(e) => log::warn!("Failed to copy input deck: {}", e),
            }

            // Write fully-resolved config (all geometry and dt now set)
            match serde_json::to_string_pretty(&config) {
                Ok(json) => match std::fs::write(rd.resolved_config_path(), json) {
                    Ok(_) => log::info!("Saved resolved config  → {:?}", rd.resolved_config_path()),
                    Err(e) => log::warn!("Failed to write resolved_config.json: {}", e),
                },
                Err(e) => log::warn!("Failed to serialise resolved config: {}", e),
            }

            // Collect dt warning into metadata if applicable
            if let Some(meta) = &mut self.run_metadata {
                if config.dt_was_supplied {
                    // dt_rec needs field — approximate from field data
                    let dt_rec = compute_recommended_dt(&config, &field);
                    if config.dt_s > dt_rec * 5.0 {
                        meta.warnings.push(format!(
                            "Chosen dt is {:.1}× above recommended minimum — accuracy may suffer",
                            config.dt_s / dt_rec
                        ));
                    }
                }
                let _ = rd.write_metadata(meta);
            }
        }

        // Experiment summary + dt recommendation
        log_experiment_summary(&config, &field);

        // Generate particles
        log::info!("  Generating {} particles...", config.source.n_particles);
        let particles = ParticleData::generate(&config.source)?;
        log::info!("  Generated {} particles", particles.count);

        // Upload field and particles to GPU
        let renderer = self.renderer.as_mut().unwrap();
        renderer.upload_field(&field)?;
        renderer.upload_particles(&particles)?;

        renderer.set_sim_params(
            config.dt_s as f32,
            detector_center,
            config.detector.normal,
            config.detector.up,
            [(config.detector.width_m  / 2.0) as f32,
             (config.detector.height_m / 2.0) as f32],
        );

        let source_pos = match &config.source.geometry {
            SimSourceGeometry::Pencil { position_m, .. } |
            SimSourceGeometry::Point  { position_m, .. } |
            SimSourceGeometry::Disk   { center_m: position_m, .. } => *position_m,
            SimSourceGeometry::ParallelBeam { center_m, .. } => {
                center_m.unwrap_or([0.0, 0.0, 0.0])
            }
        };
        renderer.set_source_position(source_pos);

        let source_type_name = match &config.source.geometry {
            SimSourceGeometry::ParallelBeam { .. } => "parallel",
            SimSourceGeometry::Pencil { .. }       => "pencil",
            SimSourceGeometry::Point  { .. }       => "point",
            SimSourceGeometry::Disk   { .. }       => "disk",
        };
        renderer.set_source_metadata(source_type_name, config.source.particle_energy_mev as f32);
        renderer.set_png_config(self.png_cfg.clone());
        renderer.set_detector_response_config(config.detector_response.clone());

        const BORIS_SHADER: &[u8] = include_bytes!("../../shaders/boris.spv");
        renderer.load_compute_shader(BORIS_SHADER)?;
        renderer.update_descriptors()?;

        if let Some(camera) = &mut self.camera {
            use glam::Vec3;
            let b = &field.bounds;
            camera.look_at_bounds(
                Vec3::new(b.x_min, b.y_min, b.z_min),
                Vec3::new(b.x_max, b.y_max, b.z_max),
            );
            renderer.update_camera(camera);
        }

        self.resolved_config = Some(config);

        log::info!("Simulation ready — press Space to start");
        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        // Create window - minimized and hidden in batch mode
        let window_attrs = if self.batch_mode {
            log::info!("Creating hidden window for compute...");
            Window::default_attributes()
                .with_title("Proton Tracer (Batch)")
                .with_inner_size(winit::dpi::LogicalSize::new(640, 480))
                .with_visible(false)  // Hidden window
        } else {
            log::info!("Creating window...");
            Window::default_attributes()
                .with_title("Proton Tracer")
                .with_inner_size(winit::dpi::LogicalSize::new(1280, 720))
        };

        match event_loop.create_window(window_attrs) {
            Ok(window) => {
                let window = Arc::new(window);
                self.window = Some(window.clone());

                // Initialize Vulkan
                log::info!("Initializing Vulkan...");
                match VulkanContext::new(&window) {
                    Ok(vulkan) => {
                        log::info!("Vulkan initialized: {}", vulkan.device_name());
                        let vulkan = Arc::new(vulkan);

                        // Update metadata with GPU hardware info
                        if let Some(meta) = &mut self.run_metadata {
                            meta.hardware.gpu = Some(vulkan.device_name().to_string());
                            meta.hardware.vulkan_api_version = Some(vulkan.vulkan_api_version());
                            if let Some(rd) = &self.run_dir {
                                let _ = rd.write_metadata(meta);
                            }
                        }

                        self.vulkan = Some(vulkan.clone());

                        // Create renderer
                        let size = window.inner_size();
                        match Renderer::new(vulkan, size.width, size.height) {
                            Ok(renderer) => {
                                log::info!("Renderer created");
                                self.renderer = Some(renderer);

                                // Create camera
                                let camera = Camera::new(size.width, size.height);
                                self.camera = Some(camera);

                                // Initialize egui (only in interactive mode)
                                if !self.batch_mode {
                                    // Initialize egui renderer in Vulkan
                                    if let Err(e) = self.renderer.as_mut().unwrap().init_egui() {
                                        log::warn!("Failed to init egui renderer: {}", e);
                                    }

                                    let egui_ctx = egui::Context::default();
                                    let egui_winit = egui_winit::State::new(
                                        egui_ctx.clone(),
                                        egui::ViewportId::ROOT,
                                        &window,
                                        Some(window.scale_factor() as f32),
                                        None,
                                        None,
                                    );
                                    let gui = Gui::new();
                                    self.egui_ctx = Some(egui_ctx);
                                    self.egui_winit = Some(egui_winit);
                                    self.gui = Some(gui);
                                    log::info!("GUI initialized");
                                }

                                // Load simulation if config provided via CLI (batch mode or direct run)
                                if self.config_path.is_some() {
                                    if let Err(e) = self.load_simulation() {
                                        log::error!("Failed to load simulation: {}", e);
                                    } else {
                                        // Only auto-start in batch mode
                                        if self.batch_mode {
                                            self.renderer.as_mut().unwrap().start_simulation();
                                        }
                                    }
                                } else {
                                    log::info!("No config provided - select a preset from the GUI and press Run");
                                }
                            }
                            Err(e) => {
                                log::error!("Failed to create renderer: {}", e);
                                self.init_failed = true;
                                event_loop.exit();
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to initialize Vulkan: {}", e);
                        self.init_failed = true;
                        event_loop.exit();
                    }
                }
            }
            Err(e) => {
                log::error!("Failed to create window: {}", e);
                self.init_failed = true;
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        // Let egui handle events first
        if let (Some(egui_winit), Some(window)) = (&mut self.egui_winit, &self.window) {
            let response = egui_winit.on_window_event(window, &event);
            if response.consumed {
                return;
            }
        }

        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested, shutting down...");
                // Export PNG before cleanup
                if let Some(renderer) = &self.renderer {
                    
                    if let Err(e) = renderer.export_detector_png(&self.output_dir, &self.png_cfg) {
                        log::warn!("Failed to export PNG on exit: {}", e);
                    }
                }
                if let Some(renderer) = &mut self.renderer {
                    renderer.cleanup();
                }
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                log::debug!("Window resized to {}x{}", size.width, size.height);
                self.needs_resize = true;
            }
            WindowEvent::ScaleFactorChanged { scale_factor, .. } => {
                // egui_winit already updates pixels_per_point in on_window_event above.
                // We just need to trigger a resize so the viewport matches the new
                // physical size (scale_factor * logical_size).
                log::debug!("Scale factor changed to {:.2}", scale_factor);
                self.needs_resize = true;
            }
            WindowEvent::RedrawRequested => {
                // BATCH MODE: Run compute only, no graphics
                if self.batch_mode {
                    if let Some(renderer) = &mut self.renderer {
                        // Run compute-only step
                        match renderer.run_compute_step() {
                            Ok(complete) => {
                                if complete {
                                    log::info!("Batch mode: simulation complete, exporting...");
                                    drop(renderer);

                                    if self.run_dir.is_some() {
                                        let run_time_s = self.run_start
                                            .map(|t| t.elapsed().as_secs_f64())
                                            .unwrap_or(0.0);
                                        if let (Some(rd), Some(meta), Some(renderer)) =
                                            (&self.run_dir, &mut self.run_metadata, &self.renderer)
                                        {
                                            finalize_run_dir_export(renderer, rd, meta, &self.png_cfg, run_time_s);
                                        }
                                    } else {
                                        // ── Legacy: timestamp-based flat export ───────────────
                                        let config_name = self.config_path.as_ref()
                                            .and_then(|p| std::path::Path::new(p).file_stem())
                                            .and_then(|s| s.to_str())
                                            .unwrap_or("output");

                                        match self.export_to_output_dir(config_name) {
                                            Ok(path) => log::info!("Exported to: {:?}", path),
                                            Err(e) => log::error!("Export failed: {}", e),
                                        }
                                        if let Some(renderer) = &self.renderer {
                                            match renderer.export_detector_png(
                                                &self.output_dir, &self.png_cfg,
                                            ) {
                                                Ok(p) => log::info!("PNG exported to: {:?}", p),
                                                Err(e) => log::error!("Failed to export PNG: {}", e),
                                            }
                                        }
                                    }

                                    // Cleanup and exit
                                    if let Some(renderer) = &mut self.renderer {
                                        renderer.cleanup();
                                    }
                                    event_loop.exit();
                                    return;
                                }
                            }
                            Err(e) => {
                                log::error!("Compute error: {}", e);
                                event_loop.exit();
                                return;
                            }
                        }
                    }
                } else {
                    // INTERACTIVE MODE: Full rendering with graphics

                    // Run egui frame and collect actions
                    let mut egui_output = None;
                    let mut deck_load_request: Option<String> = None;
                    let mut run_start_request: Option<String> = None;
                    let mut export_requested = false;

                    if let (Some(egui_ctx), Some(egui_winit), Some(gui), Some(window)) =
                        (&self.egui_ctx, &mut self.egui_winit, &mut self.gui, &self.window)
                    {
                        let raw_input = egui_winit.take_egui_input(window);
                        let output = egui_ctx.run(raw_input, |ctx| {
                            self.gui_wants_input = gui.render(ctx);
                        });
                        egui_winit.handle_platform_output(window, output.platform_output.clone());
                        let shapes = output.shapes.clone();
                        let ppp = output.pixels_per_point;
                        let primitives = egui_ctx.tessellate(shapes, ppp);
                        egui_output = Some((output, primitives));

                        deck_load_request = gui.take_deck_load_request();
                        run_start_request = gui.take_run_start_request();
                        export_requested  = gui.take_export_request();
                    }

                    // ── Deck preview (no GPU work — just parse config for display) ──
                    if let Some(deck_path) = deck_load_request {
                        match build_deck_display(&deck_path) {
                            Ok(display) => {
                                if let Some(gui) = &mut self.gui {
                                    gui.set_deck_display(display);
                                }
                            }
                            Err(e) => {
                                log::warn!("Failed to preview deck {}: {}", deck_path, e);
                                if let Some(gui) = &mut self.gui {
                                    gui.set_run_state(RunState::Failed(e.to_string()));
                                }
                            }
                        }
                    }

                    // ── Run start ─────────────────────────────────────────────────
                    if let Some(output_dir_str) = run_start_request {
                        let deck_path = self.gui.as_ref()
                            .and_then(|g| g.selected_config_path())
                            .unwrap_or_default();
                        log::info!("GUI run: deck={} out={}", deck_path, output_dir_str);

                        self.config_path = Some(deck_path.clone());
                        let output_dir = PathBuf::from(&output_dir_str);

                        // Create run directory (overwrite if already exists — timestamp in name keeps them unique)
                        match run_dir::RunDir::open(output_dir.clone(), &run_dir::RunOptions { overwrite: true, resume: false }) {
                            Ok(rd) => {
                                if let Err(e) = run_dir::attach_log_tee(&rd.log_path()) {
                                    log::warn!("Failed to attach log tee: {}", e);
                                }
                                let run_name = output_dir.file_name()
                                    .and_then(|n| n.to_str())
                                    .unwrap_or("gui_run")
                                    .to_string();
                                let mut meta = run_dir::RunMetadata::new_running(
                                    run_name, deck_path, self.argv.clone()
                                );
                                let _ = rd.write_metadata(&meta);

                                // GPU info (if already initialized)
                                if let Some(vulkan) = &self.vulkan {
                                    meta.hardware.gpu = Some(vulkan.device_name().to_string());
                                    meta.hardware.vulkan_api_version = Some(vulkan.vulkan_api_version());
                                    let _ = rd.write_metadata(&meta);
                                }

                                self.run_dir = Some(rd);
                                self.run_metadata = Some(meta);
                                self.run_start = Some(std::time::Instant::now());
                                self.gui_run_finalized = false;

                                match self.load_simulation() {
                                    Ok(()) => {
                                        if let Some(renderer) = &mut self.renderer {
                                            renderer.start_simulation();
                                        }
                                        if let Some(gui) = &mut self.gui {
                                            let n = self.resolved_config.as_ref()
                                                .map(|c| c.source.n_particles)
                                                .unwrap_or(0);
                                            gui.update_progress(0, 0, n);
                                            gui.set_run_state(RunState::Running);
                                        }
                                    }
                                    Err(e) => {
                                        log::error!("Failed to load simulation: {}", e);
                                        if let Some(gui) = &mut self.gui {
                                            gui.set_run_state(RunState::Failed(e.to_string()));
                                        }
                                    }
                                }
                            }
                            Err(e) => {
                                log::error!("Failed to create run directory: {}", e);
                                if let Some(gui) = &mut self.gui {
                                    gui.set_run_state(RunState::Failed(e.to_string()));
                                }
                            }
                        }
                    }

                    // ── Progress update + completion detection ─────────────────────
                    if !self.gui_run_finalized {
                        if let (Some(renderer), Some(rc)) = (&self.renderer, &self.resolved_config) {
                            if renderer.is_running() {
                                let (hits, exits) = renderer.hit_exit_counts();
                                let n = rc.source.n_particles;
                                if let Some(gui) = &mut self.gui {
                                    gui.update_progress(hits, exits, n);
                                }

                                if hits.saturating_add(exits) >= n {
                                    log::info!("GUI run complete: {} hits, {} exits", hits, exits);
                                    let wall_s = self.run_start
                                        .map(|t| t.elapsed().as_secs_f64())
                                        .unwrap_or(0.0);
                                    let run_dir_path = self.run_dir.as_ref()
                                        .map(|rd| rd.root().to_path_buf());

                                    if let (Some(rd), Some(meta)) =
                                        (&self.run_dir, &mut self.run_metadata)
                                    {
                                        finalize_run_dir_export(renderer, rd, meta, &self.png_cfg, wall_s);
                                    }
                                    self.gui_run_finalized = true;
                                    run_dir::detach_log_tee();

                                    if let Some(gui) = &mut self.gui {
                                        let path = run_dir_path
                                            .unwrap_or_else(|| PathBuf::from(&self.output_dir));
                                        gui.set_run_state(RunState::Complete(path));
                                    }
                                }
                            }
                        }
                    }

                    // ── Export request (from post-run panel) ──────────────────────
                    if export_requested {
                        if let Some(renderer) = &self.renderer {
                            if let Err(e) = renderer.export_detector_png(&self.output_dir, &self.png_cfg) {
                                log::error!("Failed to export PNG: {}", e);
                            }
                        }
                    }

                    if let Some(renderer) = &mut self.renderer {
                        // Handle resize if needed
                        let mut did_resize = false;
                        if self.needs_resize {
                            if let Some(window) = &self.window {
                                let size = window.inner_size();
                                if let Err(e) = renderer.resize(size.width, size.height) {
                                    log::error!("Failed to resize: {}", e);
                                }
                                self.needs_resize = false;
                                // Resize camera regardless of swapchain outcome
                                if let Some(camera) = &mut self.camera {
                                    camera.resize(size.width, size.height);
                                    renderer.update_camera(camera);
                                }
                                did_resize = true;
                            }
                        }

                        // Apply texture deltas (font atlas updates) from this frame's egui output.
                        // Must happen before rendering. In egui 0.29, output.textures_delta
                        // is the only source — ctx.fonts().font_image_delta() returns None
                        // because it was already consumed inside ctx.run().
                        if let Some((output, _)) = &egui_output {
                            if let Err(e) = renderer.apply_egui_textures_delta(&output.textures_delta) {
                                log::warn!("Failed to apply egui texture delta: {}", e);
                            }
                        }

                        // Skip GPU submission for the frame where we just recreated the
                        // swapchain — gives the CAMetalLayer one cycle to settle before
                        // we submit new work, reducing the AppKit race that causes SIGBUS.
                        if !did_resize {
                            // Prepare egui data for rendering
                            let egui_data = egui_output.as_ref().map(|(output, primitives)| {
                                let size = self.window.as_ref()
                                    .map(|w| w.inner_size())
                                    .unwrap_or(winit::dpi::PhysicalSize::new(1280, 720));
                                (
                                    primitives.as_slice(),
                                    [size.width as f32, size.height as f32],
                                    output.pixels_per_point,
                                )
                            });

                            // Render frame with graphics and egui
                            match renderer.render_frame_with_egui(egui_data) {
                                Ok(needs_resize) => {
                                    if needs_resize {
                                        self.needs_resize = true;
                                    }
                                }
                                Err(e) => {
                                    log::error!("Render error: {}", e);
                                }
                            }
                        }
                    }
                }

                if let Some(window) = &self.window {
                    window.request_redraw();
                }
            }
            WindowEvent::KeyboardInput { event, .. } => {
                use winit::keyboard::{Key, NamedKey};
                if event.state.is_pressed() {
                    match event.logical_key {
                        Key::Named(NamedKey::Escape) => {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.cleanup();
                            }
                            event_loop.exit();
                        }
                        Key::Named(NamedKey::Space) => {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.toggle_simulation();
                            }
                        }
                        Key::Named(NamedKey::Tab) => {
                            // Toggle GUI visibility
                            if let Some(gui) = &mut self.gui {
                                gui.visible = !gui.visible;
                                log::info!("GUI {}", if gui.visible { "shown" } else { "hidden" });
                            }
                        }
                        Key::Character(ref c) if c == "r" => {
                            log::info!("R pressed - reset particles");
                            // TODO: Reset particle positions
                        }
                        // Export detector data to CSV
                        Key::Character(ref c) if c == "s" || c == "S" => {
                            // Extract config name for filename
                            let config_name = self.config_path.as_ref()
                                .and_then(|p| std::path::Path::new(p).file_stem())
                                .and_then(|s| s.to_str())
                                .unwrap_or("output");

                            match self.export_to_output_dir(config_name) {
                                Ok(path) => log::info!("Saved detector data to {:?}", path),
                                Err(e) => log::error!("Failed to export: {}", e),
                            }
                        }
                        // Export detector as PNG image (P key)
                        Key::Character(ref c) if c == "p" || c == "P" => {
                            if let Some(renderer) = &self.renderer {
                                
                                match renderer.export_detector_png(&self.output_dir, &self.png_cfg) {
                                    Ok(path) => log::info!("Saved radiograph to {:?}", path),
                                    Err(e) => log::error!("Failed to export PNG: {}", e),
                                }
                            }
                        }
                        // Toggle colormap (C = cycle colormap)
                        Key::Character(ref c) if c == "c" || c == "C" => {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.toggle_colormap();
                            }
                        }
                        // Toggle log scale (L)
                        Key::Character(ref c) if c == "l" || c == "L" => {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.toggle_log_scale();
                            }
                        }
                        // Adjust exposure (+ / -)
                        Key::Character(ref c) if c == "=" || c == "+" => {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.adjust_exposure(1.2);
                            }
                        }
                        Key::Character(ref c) if c == "-" || c == "_" => {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.adjust_exposure(0.8);
                            }
                        }
                        // Adjust gamma ([ / ])
                        Key::Character(ref c) if c == "[" || c == "{" => {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.adjust_gamma(-0.1);
                            }
                        }
                        Key::Character(ref c) if c == "]" || c == "}" => {
                            if let Some(renderer) = &mut self.renderer {
                                renderer.adjust_gamma(0.1);
                            }
                        }
                        // Show help
                        Key::Character(ref c) if c == "h" || c == "H" || c == "?" => {
                            log::info!("=== KEYBOARD CONTROLS ===");
                            log::info!("  Space   - Toggle simulation");
                            log::info!("  C       - Cycle colormap (RCF film / Scientific / Grayscale / Hot / Inverted)");
                            log::info!("  L       - Toggle log scale");
                            log::info!("  +/-     - Adjust exposure");
                            log::info!("  [/]     - Adjust gamma");
                            log::info!("  S       - Save detector data (CSV)");
                            log::info!("  P       - Export radiograph package (raw counts, processed counts, PNG, metadata)");
                            log::info!("  H/?     - Show this help");
                            log::info!("  Esc     - Exit");
                            log::info!("");
                            if let Some(renderer) = &self.renderer {
                                log::info!("  {}", renderer.display_info());
                            }
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Some(camera) = &mut self.camera {
                    match (button, state) {
                        (MouseButton::Left, ElementState::Pressed) => {
                            camera.start_orbit();
                        }
                        (MouseButton::Right, ElementState::Pressed) => {
                            camera.start_pan();
                        }
                        (MouseButton::Left | MouseButton::Right, ElementState::Released) => {
                            camera.stop_drag();
                        }
                        _ => {}
                    }
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Some(camera) = &mut self.camera {
                    camera.handle_mouse_move(position.x as f32, position.y as f32);

                    // Update renderer with new camera
                    if let Some(renderer) = &mut self.renderer {
                        renderer.update_camera(camera);
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                if let Some(camera) = &mut self.camera {
                    let scroll = match delta {
                        MouseScrollDelta::LineDelta(_, y) => y,
                        MouseScrollDelta::PixelDelta(pos) => pos.y as f32 / 100.0,
                    };
                    camera.handle_scroll(scroll);

                    // Update renderer with new camera
                    if let Some(renderer) = &mut self.renderer {
                        renderer.update_camera(camera);
                    }
                }
            }
            _ => {}
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }
}

/// Export counts + PNG into a run directory and finalise metadata.
///
/// Called from both batch and interactive paths after simulation completes.
/// Returns true if the export succeeded.
fn finalize_run_dir_export(
    renderer: &Renderer,
    run_dir: &run_dir::RunDir,
    meta: &mut run_dir::RunMetadata,
    png_cfg: &PngExportConfig,
    wall_s: f64,
) -> bool {
    let export_ok = match renderer.export_to_run_dir(run_dir, png_cfg) {
        Ok((raw, _)) => {
            let hits: u64 = raw.iter().map(|&c| c as u64).sum();
            log::info!("Total detector hits: {}", hits);
            meta.outputs.raw_counts = Some("counts/raw_counts.bin".to_string());
            meta.outputs.processed_counts = Some("counts/processed_counts.bin".to_string());
            meta.outputs.radiograph_png = Some("images/radiograph.png".to_string());
            true
        }
        Err(e) => {
            log::error!("Export failed: {}", e);
            false
        }
    };

    meta.diagnostics = renderer.compute_hit_diagnostics();
    meta.performance = Some(run_dir::PerfInfo { total_runtime_s: wall_s });
    meta.run.status = "complete".to_string();
    meta.run.completed_at = Some(chrono::Utc::now().to_rfc3339());

    let scale_str = match png_cfg.scale {
        ScaleMode::Linear => "linear",
        ScaleMode::Log    => "log",
        ScaleMode::Sqrt   => "sqrt",
    };
    let cmap_str = match png_cfg.colormap {
        ColormapType::RcfFilm    => "rcf_film",
        ColormapType::Scientific => "scientific",
        ColormapType::Grayscale  => "grayscale",
        ColormapType::Hot        => "hot",
        ColormapType::Inverted   => "inverted",
    };
    meta.render = Some(run_dir::RenderProvenance {
        source: "gpu_detector_texture".to_string(),
        scale: scale_str.to_string(),
        colormap: cmap_str.to_string(),
        gamma: png_cfg.gamma,
        exposure: png_cfg.exposure,
    });

    let (tex_w, tex_h) = renderer.detector_texture_size();
    meta.counts_format = Some(run_dir::CountsFormat {
        raw: run_dir::CountsBinarySpec {
            dtype: "uint32".to_string(),
            endianness: "little".to_string(),
            shape: [tex_h, tex_w],
            layout: "row-major".to_string(),
            row_axis: "detector_z".to_string(),
            col_axis: "detector_y".to_string(),
            units: "proton counts".to_string(),
        },
        processed: run_dir::CountsBinarySpec {
            dtype: "float32".to_string(),
            endianness: "little".to_string(),
            shape: [tex_h, tex_w],
            layout: "row-major".to_string(),
            row_axis: "detector_z".to_string(),
            col_axis: "detector_y".to_string(),
            units: "detector-response-adjusted counts".to_string(),
        },
    });

    match run_dir.write_metadata(meta) {
        Ok(_)  => log::info!("Wrote metadata.json    → {:?}", run_dir.metadata_path()),
        Err(e) => log::error!("Failed to write final metadata: {}", e),
    }

    export_ok
}

/// Quick deck preview: parse config file and return display-ready params.
/// Does NOT load the field or do any GPU work.
fn build_deck_display(deck_path: &str) -> anyhow::Result<DeckDisplay> {
    let config = SimConfig::load(deck_path)?;
    let deck_name = std::path::Path::new(deck_path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("run")
        .to_string();

    let geometry = match &config.source.geometry {
        SimSourceGeometry::ParallelBeam { .. } => "parallel beam",
        SimSourceGeometry::Pencil       { .. } => "pencil beam",
        SimSourceGeometry::Point        { .. } => "point source",
        SimSourceGeometry::Disk         { .. } => "disk source",
    }.to_string();

    Ok(DeckDisplay {
        deck_name,
        n_particles: config.source.n_particles,
        energy_mev: config.source.particle_energy_mev,
        dt_ps: if config.dt_was_supplied { config.dt_s * 1e12 } else { 0.0 },
        dt_auto: !config.dt_was_supplied,
        max_steps: config.max_steps,
        field_file: config.field_path.clone(),
        geometry,
        detector_center_mm: config.detector.center_m
            .map(|c| [c[0] * 1e3, c[1] * 1e3, c[2] * 1e3]),
        detector_size_mm: [
            config.detector.width_m  * 1e3,
            config.detector.height_m * 1e3,
        ],
    })
}

/// Compute the minimum of three physics-motivated dt candidates.
fn compute_recommended_dt(config: &SimConfig, field: &FieldData) -> f64 {
    let v = config.source.particle_speed_m_s;
    let (_, b_max) = field.b_magnitude_range();
    let (dx, dy, dz) = field.spacing();
    let min_cell = (dx as f64).min(dy as f64).min(dz as f64);

    let d = config.source.beam_direction();
    let b = &field.bounds;
    let transit = (
        (b.x_max - b.x_min) as f64 * (d[0] as f64).abs()
      + (b.y_max - b.y_min) as f64 * (d[1] as f64).abs()
      + (b.z_max - b.z_min) as f64 * (d[2] as f64).abs()
    ).max(1e-4);

    let dt_larmor = if b_max > 1e-10 {
        std::f64::consts::TAU / (units::PROTON_QM * b_max as f64) / 20.0
    } else {
        f64::INFINITY
    };
    let dt_grid    = 0.25 * min_cell / v;
    let dt_transit = transit / (v * config.max_steps as f64);

    dt_larmor.min(dt_grid).min(dt_transit)
}

/// Build a human-readable experiment summary as a list of lines.
fn build_experiment_summary(config: &SimConfig, field: &FieldData) -> Vec<String> {
    use units::*;
    let mut out: Vec<String> = Vec::new();

    let src  = &config.source;
    let v    = src.particle_speed_m_s;
    let beta = v / C_M_S;

    let (_, b_max) = field.b_magnitude_range();
    let (_, e_max) = field.e_magnitude_range();
    let (dx, dy, dz) = field.spacing();
    let min_cell = (dx as f64).min(dy as f64).min(dz as f64);

    let d = src.beam_direction();
    let b = &field.bounds;
    let transit_m = (
        (b.x_max - b.x_min) as f64 * (d[0] as f64).abs()
      + (b.y_max - b.y_min) as f64 * (d[1] as f64).abs()
      + (b.z_max - b.z_min) as f64 * (d[2] as f64).abs()
    ).max(1e-4);
    let transit_t = transit_m / v;

    let dt_larmor = if b_max > 1e-10 {
        std::f64::consts::TAU / (PROTON_QM * b_max as f64) / 20.0
    } else {
        f64::INFINITY
    };
    let dt_grid    = 0.25 * min_cell / v;
    let dt_transit = transit_m / (v * config.max_steps as f64);
    let dt_rec     = dt_larmor.min(dt_grid).min(dt_transit);

    let (beam_radius_opt, source_distance_m) = match &src.geometry {
        SimSourceGeometry::ParallelBeam { radius_m, .. } => (Some(*radius_m), src.source_distance_m),
        SimSourceGeometry::Disk { radius_m, .. } => (Some(*radius_m as f64), None),
        SimSourceGeometry::Pencil { .. } | SimSourceGeometry::Point { .. } => (None, None),
    };
    let detector_distance_m = config.detector.distance_m
        .or_else(|| config.detector.center_m.map(|c| {
            let dir = src.beam_direction();
            let exit_x = if dir[0] > 0.0 { b.x_max } else { b.x_min } as f64;
            (c[0] - exit_x).abs()
        }))
        .unwrap_or(0.0);

    out.push("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".into());
    out.push("Experimental setup:".into());
    out.push(format!("  Proton KE             : {:.2} MeV  (β = {:.2}% c)",
        src.particle_energy_mev, beta * 100.0));
    if let Some(r) = beam_radius_opt {
        out.push(format!("  Source radius         : {}", fmt_dist(r)));
    }
    if let SimSourceGeometry::Pencil { position_m, .. }
       | SimSourceGeometry::Point   { position_m, .. }
       | SimSourceGeometry::Disk    { center_m: position_m, .. } = &src.geometry {
        out.push(format!("  Source position       : ({:.4}, {:.4}, {:.4}) m",
            position_m[0], position_m[1], position_m[2]));
    }
    if let SimSourceGeometry::Point { half_angle_rad, .. }
       | SimSourceGeometry::Disk   { half_angle_rad, .. } = &src.geometry {
        if *half_angle_rad > 0.0 {
            out.push(format!("  Cone half-angle       : {:.2}°", half_angle_rad.to_degrees()));
        }
    }
    if let Some(sd) = source_distance_m {
        let mag = (sd + detector_distance_m) / sd;
        out.push(format!("  Source → field centre : {}", fmt_dist(sd)));
        out.push(format!("  Field exit → detector : {}", fmt_dist(detector_distance_m)));
        out.push(format!("  Magnification         : {:.2}×", mag));
    } else {
        out.push(format!("  Field exit → detector : {}", fmt_dist(detector_distance_m)));
    }
    out.push(format!("  Field transit         : {}  ({} along beam)",
        fmt_time(transit_t), fmt_dist(transit_m)));
    if b_max > 1e-10 {
        let r_larmor = v / (PROTON_QM * b_max as f64);
        out.push(format!("  Larmor radius (Bmax)  : {}  ({})",
            fmt_dist(r_larmor), fmt_b(b_max as f64)));
    }
    if e_max > 1.0 {
        out.push(format!("  E-field (max)         : {}", fmt_e(e_max as f64)));
    }

    out.push("Simulation timing:".into());
    if config.dt_was_supplied {
        out.push(format!("  dt                    : {}  [user supplied]", fmt_time(config.dt_s)));
    } else {
        out.push(format!("  dt                    : {}  [auto]", fmt_time(config.dt_s)));
    }
    out.push("  Recommended candidates:".into());
    if b_max > 1e-10 {
        out.push(format!("    Larmor / 20         : {}  (Bmax = {})",
            fmt_time(dt_larmor), fmt_b(b_max as f64)));
    } else {
        out.push("    Larmor / 20         : n/a  (B ≈ 0)".into());
    }
    out.push(format!("    Grid crossing / 4   : {}  (min cell {})",
        fmt_time(dt_grid), fmt_dist(min_cell)));
    out.push(format!("    Transit / max_steps : {}  ({} steps)",
        fmt_time(dt_transit), config.max_steps));
    out.push(format!("  → minimum recommended : {}", fmt_time(dt_rec)));

    if config.dt_was_supplied && config.dt_s > dt_rec * 5.0 {
        out.push(format!("  WARNING: dt is {:.1}× above recommended minimum — accuracy may suffer",
            config.dt_s / dt_rec));
    }

    // Step-budget check: warn when max_steps × dt × v < source→detector distance.
    // Uses straight-line distance (no field deflection) — a lower bound on path length.
    let source_pos: Option<[f64; 3]> = match &src.geometry {
        SimSourceGeometry::ParallelBeam { center_m, .. } =>
            center_m.map(|c| [c[0] as f64, c[1] as f64, c[2] as f64]),
        SimSourceGeometry::Pencil { position_m, .. } |
        SimSourceGeometry::Point  { position_m, .. } =>
            Some([position_m[0] as f64, position_m[1] as f64, position_m[2] as f64]),
        SimSourceGeometry::Disk { center_m, .. } =>
            Some([center_m[0] as f64, center_m[1] as f64, center_m[2] as f64]),
    };
    if let (Some(src_pos), Some(det_ctr)) = (source_pos, config.detector.center_m) {
        let dist = ((det_ctr[0] - src_pos[0]).powi(2)
                  + (det_ctr[1] - src_pos[1]).powi(2)
                  + (det_ctr[2] - src_pos[2]).powi(2)).sqrt();
        let dist_avail = v * config.dt_s * config.max_steps as f64;
        let steps_needed = (dist / (v * config.dt_s)).ceil() as u64;
        if steps_needed > config.max_steps as u64 {
            out.push(format!(
                "  WARNING: step budget too small for source → detector path:"));
            out.push(format!(
                "    Source → detector distance : {}",
                fmt_dist(dist)));
            out.push(format!(
                "    Max simulated distance     : {}  ({} steps × {})",
                fmt_dist(dist_avail), config.max_steps, fmt_time(config.dt_s)));
            out.push(format!(
                "    Steps needed (straight-line, zero field) : {}",
                steps_needed));
            out.push(format!(
                "    max_steps configured       : {}",
                config.max_steps));
            out.push(
                "    Particles will not reach the detector — increase max_steps.".into());
        }
    }

    out.push("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━".into());
    out
}

/// Log experiment summary via the logger.
fn log_experiment_summary(config: &SimConfig, field: &FieldData) {
    for line in build_experiment_summary(config, field) {
        log::info!("{}", line);
    }
}

// ── Render helpers shared by both render modes ────────────────────────────────

fn parse_colormap(s: Option<&str>) -> Result<ColormapType> {
    match s {
        Some("rcf" | "rcf_film") => Ok(ColormapType::RcfFilm),
        Some("scientific")       => Ok(ColormapType::Scientific),
        Some("grayscale")        => Ok(ColormapType::Grayscale),
        Some("hot")              => Ok(ColormapType::Hot),
        Some("inverted")         => Ok(ColormapType::Inverted),
        other => anyhow::bail!("Unknown colormap: {:?}. Use: rcf, scientific, grayscale, hot, inverted", other),
    }
}

fn parse_scale(s: Option<&str>) -> Result<ScaleMode> {
    match s {
        Some("linear") => Ok(ScaleMode::Linear),
        Some("log")    => Ok(ScaleMode::Log),
        Some("sqrt")   => Ok(ScaleMode::Sqrt),
        other => anyhow::bail!("Unknown scale: {:?}. Use: linear, log, sqrt", other),
    }
}

fn colormap_to_str(c: ColormapType) -> &'static str {
    match c {
        ColormapType::RcfFilm    => "rcf_film",
        ColormapType::Scientific => "scientific",
        ColormapType::Grayscale  => "grayscale",
        ColormapType::Hot        => "hot",
        ColormapType::Inverted   => "inverted",
    }
}

fn scale_to_str(s: ScaleMode) -> &'static str {
    match s { ScaleMode::Linear => "linear", ScaleMode::Log => "log", ScaleMode::Sqrt => "sqrt" }
}

// ── render <run_dir> ──────────────────────────────────────────────────────────

/// Re-render a radiograph from stored counts in a structured run directory.
/// No GPU required — loads `counts/processed_counts.bin` (or raw) and calls
/// `render_hitmap_f32`. Updates `metadata.json` render provenance in-place.
fn render_from_run_dir(dir: &std::path::Path, argv: &[String]) -> Result<()> {
    use anyhow::Context as _;

    // ── Parse CLI flags ───────────────────────────────────────────────────────
    let mut colormap_arg: Option<ColormapType> = None;
    let mut scale_arg:    Option<ScaleMode>    = None;
    let mut gamma_arg:    Option<f32>          = None;
    let mut exposure_arg: Option<f32>          = None;
    let mut out_path:     Option<String>       = None;
    let mut use_raw = false;

    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--colormap" | "-c" => { i += 1; colormap_arg = Some(parse_colormap(argv.get(i).map(String::as_str))?); }
            "--scale"    | "-s" => { i += 1; scale_arg    = Some(parse_scale(argv.get(i).map(String::as_str))?); }
            "--gamma"           => { i += 1; gamma_arg    = argv.get(i).and_then(|s| s.parse().ok()); }
            "--exposure"        => { i += 1; exposure_arg = argv.get(i).and_then(|s| s.parse().ok()); }
            "-o" | "--out"      => { i += 1; out_path = argv.get(i).cloned(); }
            "--counts" => {
                i += 1;
                match argv.get(i).map(String::as_str) {
                    Some("raw")       => use_raw = true,
                    Some("processed") => use_raw = false,
                    other => anyhow::bail!("--counts expects raw|processed, got {:?}", other),
                }
            }
            _ => {}
        }
        i += 1;
    }

    // ── Read + validate metadata.json ─────────────────────────────────────────
    let meta_path = dir.join("metadata.json");
    let meta_str  = std::fs::read_to_string(&meta_path)
        .with_context(|| format!("Cannot read {:?}", meta_path))?;
    let mut meta: run_dir::RunMetadata = serde_json::from_str(&meta_str)
        .context("Failed to parse metadata.json")?;

    // Extract shape from counts_format (Copy types — borrow ends immediately)
    let (shape, bin_path, source_label) = {
        let cf = meta.counts_format.as_ref()
            .ok_or_else(|| anyhow::anyhow!(
                "metadata.json has no counts_format — was this run completed?"
            ))?;
        if use_raw {
            (cf.raw.shape, dir.join("counts/raw_counts.bin"), "counts/raw_counts.bin")
        } else {
            (cf.processed.shape, dir.join("counts/processed_counts.bin"), "counts/processed_counts.bin")
        }
    };
    let [height, width] = shape;

    // ── File size validation ──────────────────────────────────────────────────
    let expected_bytes = (height as usize) * (width as usize) * 4;
    let actual_bytes   = std::fs::metadata(&bin_path)
        .with_context(|| format!("Cannot stat {:?}", bin_path))?.len() as usize;

    if actual_bytes != expected_bytes {
        anyhow::bail!(
            "{} size does not match metadata shape [{}, {}]:\n  \
             expected {} bytes ({}×{} × 4), found {} bytes",
            source_label, height, width,
            expected_bytes, height, width, actual_bytes
        );
    }

    // ── Load counts as f32 ────────────────────────────────────────────────────
    let raw_bytes = std::fs::read(&bin_path)
        .with_context(|| format!("Cannot read {:?}", bin_path))?;

    let counts: Vec<f32> = if use_raw {
        // u32 LE → f32
        raw_bytes.chunks_exact(4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32)
            .collect()
    } else {
        // f32 LE
        raw_bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()
    };

    log::info!("Loaded {} ({} × {} pixels, {} values)", source_label, height, width, counts.len());

    // ── Build render config: CLI flag → stored metadata → default ─────────────
    let stored = meta.render.as_ref();
    let colormap = colormap_arg.unwrap_or_else(||
        stored.and_then(|r| parse_colormap(Some(&r.colormap)).ok())
              .unwrap_or(ColormapType::RcfFilm));
    let scale = scale_arg.unwrap_or_else(||
        stored.and_then(|r| parse_scale(Some(&r.scale)).ok())
              .unwrap_or(ScaleMode::Log));
    let gamma    = gamma_arg   .unwrap_or_else(|| stored.map(|r| r.gamma)   .unwrap_or(0.5));
    let exposure = exposure_arg.unwrap_or_else(|| stored.map(|r| r.exposure).unwrap_or(1.0));

    let cfg = PngExportConfig {
        output_pixels: Some([width, height]),
        colormap,
        scale,
        gamma,
        exposure,
        include_colorbar: false,
        include_metadata: false,
    };

    // ── Render ────────────────────────────────────────────────────────────────
    let img = render_hitmap_f32(&counts, width, height, &cfg);

    let out_png = out_path.as_ref()
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| dir.join("images/radiograph.png"));

    if let Some(parent) = out_png.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    img.save(&out_png).map_err(|e| anyhow::anyhow!("Failed to save PNG: {}", e))?;
    log::info!("Wrote {}", out_png.display());

    // ── Update metadata render provenance ─────────────────────────────────────
    // Only write back when PNG goes into the run dir (not when -o redirects elsewhere).
    if out_path.is_none() {
        meta.render = Some(run_dir::RenderProvenance {
            source:   source_label.to_string(),
            scale:    scale_to_str(scale).to_string(),
            colormap: colormap_to_str(colormap).to_string(),
            gamma,
            exposure,
        });
        let updated = serde_json::to_string_pretty(&meta)
            .context("Failed to serialise updated metadata")?;
        std::fs::write(&meta_path, updated)
            .context("Failed to write updated metadata.json")?;
        log::info!("Updated metadata.json render provenance  (source: {})", source_label);
    }

    Ok(())
}

/// Standalone re-render subcommand: read a hit CSV and produce a PNG without GPU.
///
/// Usage:
///   proton_tracer render --hits hits.csv [--meta hits_meta.json | --detector-width-mm W --detector-height-mm H]
///                        [--out radiograph.png] [--colormap rcf_film|scientific|grayscale|hot|inverted]
///                        [--scale linear|log|sqrt] [--gamma 0.5] [--exposure 1.0]
///                        [--width 1024] [--height 1024]
fn run_render_subcommand(argv: &[String]) -> Result<()> {
    use anyhow::Context as _;
    use std::io::{BufRead, BufReader};

    // ── Dispatch: run directory vs legacy CSV ─────────────────────────────────
    // If the first positional arg is a directory, use the counts-based path.
    let run_dir_arg = argv.iter()
        .find(|a| !a.starts_with('-'))
        .filter(|a| std::path::Path::new(a).is_dir());
    if let Some(dir) = run_dir_arg {
        let dir = std::path::Path::new(dir).to_path_buf();
        // Strip the directory arg so render_from_run_dir only sees option flags
        let flags: Vec<String> = argv.iter()
            .filter(|a| a.as_str() != dir.to_str().unwrap_or(""))
            .cloned()
            .collect();
        return render_from_run_dir(&dir, &flags);
    }

    let mut hits_path: Option<String> = None;
    let mut meta_path: Option<String> = None;
    let mut out_path:  Option<String> = None;
    let mut det_width_mm:  Option<f64> = None;
    let mut det_height_mm: Option<f64> = None;
    let mut colormap  = ColormapType::RcfFilm;
    let mut scale     = ScaleMode::Log;
    let mut gamma     = 0.5f32;
    let mut exposure  = 1.0f32;
    let mut out_w: Option<u32> = None;
    let mut out_h: Option<u32> = None;
    let mut blur_sigma_um    = 0.0f64;
    let mut background_counts = 0.0f64;
    let mut poisson_noise    = false;
    let mut noise_seed: Option<u64> = None;

    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--hits" => { i += 1; hits_path = argv.get(i).cloned(); }
            "--meta" => { i += 1; meta_path = argv.get(i).cloned(); }
            "--out"  => { i += 1; out_path  = argv.get(i).cloned(); }
            "--detector-width-mm"  => { i += 1; det_width_mm  = argv.get(i).and_then(|s| s.parse().ok()); }
            "--detector-height-mm" => { i += 1; det_height_mm = argv.get(i).and_then(|s| s.parse().ok()); }
            "--colormap" => {
                i += 1;
                colormap = match argv.get(i).map(|s| s.as_str()) {
                    Some("rcf_film")    => ColormapType::RcfFilm,
                    Some("scientific")  => ColormapType::Scientific,
                    Some("grayscale")   => ColormapType::Grayscale,
                    Some("hot")         => ColormapType::Hot,
                    Some("inverted")    => ColormapType::Inverted,
                    other => anyhow::bail!("Unknown colormap: {:?}. Use: rcf_film, scientific, grayscale, hot, inverted", other),
                };
            }
            "--scale" => {
                i += 1;
                scale = match argv.get(i).map(|s| s.as_str()) {
                    Some("linear") => ScaleMode::Linear,
                    Some("log")    => ScaleMode::Log,
                    Some("sqrt")   => ScaleMode::Sqrt,
                    other => anyhow::bail!("Unknown scale: {:?}. Use: linear, log, sqrt", other),
                };
            }
            "--gamma"           => { i += 1; if let Some(v) = argv.get(i).and_then(|s| s.parse().ok()) { gamma = v; } }
            "--exposure"        => { i += 1; if let Some(v) = argv.get(i).and_then(|s| s.parse().ok()) { exposure = v; } }
            "--width"           => { i += 1; out_w = argv.get(i).and_then(|s| s.parse().ok()); }
            "--height"          => { i += 1; out_h = argv.get(i).and_then(|s| s.parse().ok()); }
            "--blur-sigma-um"   => { i += 1; if let Some(v) = argv.get(i).and_then(|s| s.parse().ok()) { blur_sigma_um = v; } }
            "--background"      => { i += 1; if let Some(v) = argv.get(i).and_then(|s| s.parse().ok()) { background_counts = v; } }
            "--poisson-noise"   => { poisson_noise = true; }
            "--noise-seed"      => { i += 1; noise_seed = argv.get(i).and_then(|s| s.parse().ok()); }
            "--help" | "-h" => {
                println!("proton-tracer render — re-render a radiograph without GPU");
                println!();
                println!("Modes:");
                println!("  proton-tracer render <run_dir> [options]");
                println!("      Load counts/processed_counts.bin from a structured run directory.");
                println!("      Writes images/radiograph.png and updates metadata.json.");
                println!();
                println!("  proton-tracer render --hits hits.csv [options]");
                println!("      Re-bin a hit CSV and render to PNG (legacy path).");
                println!();
                println!("Run-directory options:");
                println!("  --counts raw|processed     Which counts file to load (default: processed)");
                println!("  --colormap <name>          rcf | scientific | grayscale | hot | inverted");
                println!("  --scale <name>             linear | log | sqrt");
                println!("  --gamma <f>                Gamma correction (default: from metadata or 0.5)");
                println!("  --exposure <f>             Exposure multiplier (default: from metadata or 1.0)");
                println!("  -o <png>                   Custom output path (metadata not updated)");
                println!();
                println!("CSV options:");
                println!("  --hits <csv>               Hit CSV (y_mm, z_mm columns required)");
                println!("  --meta <json>              Sidecar JSON from CSV export");
                println!("  --detector-width-mm <f>    Detector width  (if --meta absent)");
                println!("  --detector-height-mm <f>   Detector height (if --meta absent)");
                println!("  --out <png>                Output PNG path");
                println!("  --colormap / --scale / --gamma / --exposure   (same as above)");
                println!("  --width <px>               Output PNG width in pixels");
                println!("  --height <px>              Output PNG height in pixels");
                return Ok(());
            }
            _ => {}
        }
        i += 1;
    }

    let hits_path = hits_path.ok_or_else(|| anyhow::anyhow!("--hits is required"))?;

    // ── Detector geometry ─────────────────────────────────────────────────────
    let (det_w_mm, det_h_mm, n_particles): (f64, f64, u64) = if let Some(ref mp) = meta_path {
        let json: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(mp).context("Failed to read meta JSON")?
        ).context("Failed to parse meta JSON")?;

        // Support two sidecar formats:
        //   PNG sidecar: {"detector": {"width_mm": ..., "height_mm": ...}, "source": {"n_particles": ...}}
        //   CSV sidecar: {"detector_extent_m": [half_w, half_h], "n_particles": ...}
        let (w, h) = if let (Some(w), Some(h)) = (
            json["detector"]["width_mm"].as_f64(),
            json["detector"]["height_mm"].as_f64(),
        ) {
            (w, h)
        } else if let Some(ext) = json["detector_extent_m"].as_array() {
            let half_w = ext.get(0).and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("detector_extent_m[0] missing"))?;
            let half_h = ext.get(1).and_then(|v| v.as_f64())
                .ok_or_else(|| anyhow::anyhow!("detector_extent_m[1] missing"))?;
            (half_w * 2.0 * 1e3, half_h * 2.0 * 1e3)
        } else {
            anyhow::bail!("meta JSON has no detector geometry (expected detector.width_mm or detector_extent_m)");
        };
        let n = json["source"]["n_particles"].as_u64()
            .or_else(|| json["n_particles"].as_u64())
            .unwrap_or(0);
        (w, h, n)
    } else {
        let w = det_width_mm.ok_or_else(||
            anyhow::anyhow!("Provide --meta or both --detector-width-mm and --detector-height-mm"))?;
        let h = det_height_mm.ok_or_else(||
            anyhow::anyhow!("Provide --meta or both --detector-width-mm and --detector-height-mm"))?;
        (w, h, 0)
    };

    // ── Output path ───────────────────────────────────────────────────────────
    let out_png = out_path.map(std::path::PathBuf::from).unwrap_or_else(|| {
        let base = std::path::Path::new(&hits_path);
        base.with_file_name("radiograph.png")
    });

    // ── Output resolution ─────────────────────────────────────────────────────
    // If neither --width nor --height given, default to 1024×<aspect>
    let (grid_w, grid_h) = match (out_w, out_h) {
        (Some(w), Some(h)) => (w, h),
        (Some(w), None)    => {
            let h = (w as f64 * det_h_mm / det_w_mm.max(1e-9)).round() as u32;
            (w, h.max(1))
        }
        (None, Some(h))    => {
            let w = (h as f64 * det_w_mm / det_h_mm.max(1e-9)).round() as u32;
            (w.max(1), h)
        }
        (None, None)       => {
            let w = 1024u32;
            let h = (w as f64 * det_h_mm / det_w_mm.max(1e-9)).round() as u32;
            (w, h.max(1))
        }
    };

    // ── Read CSV (column-name-based) ──────────────────────────────────────────
    let file = std::fs::File::open(&hits_path)
        .with_context(|| format!("Cannot open hits CSV: {}", hits_path))?;
    let reader = BufReader::new(file);
    let mut lines = reader.lines();

    // Find header; skip comment lines
    let mut header_line: Option<String> = None;
    for line in lines.by_ref() {
        let line: String = line?;
        if !line.starts_with('#') {
            header_line = Some(line);
            break;
        }
    }
    let header = header_line.ok_or_else(|| anyhow::anyhow!("CSV has no header row"))?;
    let cols: Vec<&str> = header.split(',').map(str::trim).collect();
    let y_col = cols.iter().position(|&c| c == "y_mm")
        .ok_or_else(|| anyhow::anyhow!("CSV missing 'y_mm' column"))?;
    let z_col = cols.iter().position(|&c| c == "z_mm")
        .ok_or_else(|| anyhow::anyhow!("CSV missing 'z_mm' column"))?;
    let w_col = cols.iter().position(|&c| c == "weight");

    // ── Bin hits ──────────────────────────────────────────────────────────────
    let half_y = det_w_mm / 2.0;
    let half_z = det_h_mm / 2.0;
    let mut counts = vec![0.0f32; (grid_w * grid_h) as usize];
    let mut n_hits: u64 = 0;

    for line in lines {
        let line: String = line?;
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') { continue; }
        let fields: Vec<&str> = line.split(',').collect();
        let y_mm: f64 = fields.get(y_col)
            .and_then(|s: &&str| s.trim().parse::<f64>().ok())
            .unwrap_or(f64::NAN);
        let z_mm: f64 = fields.get(z_col)
            .and_then(|s: &&str| s.trim().parse::<f64>().ok())
            .unwrap_or(f64::NAN);
        if y_mm.is_nan() || z_mm.is_nan() { continue; }
        let weight: f32 = w_col
            .and_then(|wc| fields.get(wc))
            .and_then(|s: &&str| s.trim().parse::<f32>().ok())
            .unwrap_or(1.0);

        // Map mm coords to pixel indices — clamp to grid bounds
        let px = ((y_mm + half_y) / (det_w_mm) * grid_w as f64) as i64;
        let py = ((z_mm + half_z) / (det_h_mm) * grid_h as f64) as i64;
        if px >= 0 && px < grid_w as i64 && py >= 0 && py < grid_h as i64 {
            counts[(py as u32 * grid_w + px as u32) as usize] += weight;
            n_hits += 1;
        }
    }

    log::info!("Binned {} hits into {}×{} grid", n_hits, grid_w, grid_h);

    // ── Apply detector response ───────────────────────────────────────────────
    let pixel_pitch_y_um = det_w_mm / grid_w as f64 * 1e3;
    let pixel_pitch_z_um = det_h_mm / grid_h as f64 * 1e3;
    let detector_info = gpu::DetectorRenderInfo {
        width_px: grid_w as usize, height_px: grid_h as usize,
        pixel_pitch_y_um, pixel_pitch_z_um,
    };
    let response = DetectorResponseConfig {
        blur_sigma_um, background_counts, poisson_noise, noise_seed,
    };
    let processed = gpu::apply_detector_response(
        &counts, grid_w as usize, grid_h as usize, &detector_info, &response,
    );

    // ── Build PngExportConfig from CLI flags ──────────────────────────────────
    let cfg = PngExportConfig {
        output_pixels: Some([grid_w, grid_h]),
        colormap,
        scale,
        gamma,
        exposure,
        include_colorbar: false,
        include_metadata: meta_path.is_some(),
    };

    // ── Render ────────────────────────────────────────────────────────────────
    let img = render_hitmap_f32(&processed, grid_w, grid_h, &cfg);
    if let Some(parent) = out_png.parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)?;
        }
    }
    img.save(&out_png).map_err(|e| anyhow::anyhow!("Failed to save PNG: {}", e))?;
    log::info!("Saved radiograph to {:?}", out_png);

    // ── Write sidecar if we have enough metadata ──────────────────────────────
    if cfg.include_metadata {
        let scale_name = match scale { ScaleMode::Linear => "linear", ScaleMode::Log => "log", ScaleMode::Sqrt => "sqrt" };
        let colormap_name = match colormap {
            ColormapType::RcfFilm    => "rcf_film",
            ColormapType::Scientific => "scientific",
            ColormapType::Grayscale  => "grayscale",
            ColormapType::Hot        => "hot",
            ColormapType::Inverted   => "inverted",
        };
        let max_count = processed.iter().cloned().fold(0.0f32, f32::max);
        let meta = serde_json::json!({
            "image": {
                "data_width_px":  grid_w,
                "data_height_px": grid_h,
                "png_width_px":   grid_w,
                "png_height_px":  grid_h,
                "colorbar_width_px": 0,
                "scale":          scale_name,
                "colormap":       colormap_name,
                "max_count":      max_count,
                "total_hits":     n_hits,
            },
            "render": {
                "render_source":  "csv_hits",
                "gamma":          gamma,
                "exposure":       exposure,
                "normalization":  "max",
                "zero_count_color": [0, 0, 0],
                "colorbar":       false,
            },
            "detector": {
                "width_mm":       det_w_mm,
                "height_mm":      det_h_mm,
                "axes":           { "row": "z_mm", "col": "y_mm" },
                "y_range_mm":     [-half_y, half_y],
                "z_range_mm":     [-half_z, half_z],
                "pixel_pitch_y_um": pixel_pitch_y_um,
                "pixel_pitch_z_um": pixel_pitch_z_um,
            },
            "detector_response": {
                "blur_sigma_um":   blur_sigma_um,
                "background_counts": background_counts,
                "poisson_noise":   poisson_noise,
            },
            "source": {
                "n_particles": n_particles,
            },
        });
        let meta_path_out = out_png.with_extension("").with_extension("_meta.json");
        // Simpler: replace .png with _meta.json
        let stem = out_png.file_stem().unwrap_or_default().to_string_lossy();
        let meta_file = out_png.with_file_name(format!("{}_meta.json", stem));
        std::fs::write(&meta_file, serde_json::to_string_pretty(&meta).unwrap())?;
        log::info!("Wrote sidecar to {:?}", meta_file);
        let _ = meta_path_out; // suppress unused warning
    }

    Ok(())
}

// ── Non-GPU subcommands ───────────────────────────────────────────────────────

fn run_init_subcommand(argv: &[String]) -> Result<()> {
    let mut preset: Option<String> = None;
    let mut out_path: Option<String> = None;
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "-o" | "--output" => { i += 1; out_path = argv.get(i).cloned(); }
            "--help" | "-h" => {
                println!("proton-tracer init [preset] [-o output.toml]");
                println!("  Presets: blank (default), zpinch, kink-strong");
                return Ok(());
            }
            arg if !arg.starts_with('-') && preset.is_none() => {
                preset = Some(arg.to_string());
            }
            _ => {}
        }
        i += 1;
    }
    let preset = preset.as_deref().unwrap_or("blank");
    let template = match preset {
        "blank"        => DECK_TEMPLATE_BLANK,
        "zpinch"       => DECK_TEMPLATE_ZPINCH,
        "kink-strong"  => DECK_TEMPLATE_KINK_STRONG,
        other => anyhow::bail!("Unknown preset: {}. Use: blank, zpinch, kink-strong", other),
    };
    match out_path {
        Some(ref p) => {
            std::fs::write(p, template)?;
            println!("Wrote preset '{}' to {}", preset, p);
        }
        None => print!("{}", template),
    }
    Ok(())
}

fn run_validate_subcommand(argv: &[String]) -> Result<()> {
    let mut deck_path: Option<String> = None;
    let mut cli_overrides: Vec<overrides::ConfigOverride> = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--set" => {
                i += 1;
                if let Some(s) = argv.get(i) {
                    match overrides::parse_override(s) {
                        Ok(ov) => cli_overrides.push(ov),
                        Err(e) => { eprintln!("ERR {}", e); std::process::exit(1); }
                    }
                }
            }
            arg if !arg.starts_with('-') && deck_path.is_none() => {
                deck_path = Some(arg.to_string());
            }
            _ => {}
        }
        i += 1;
    }
    let deck_path = deck_path
        .ok_or_else(|| anyhow::anyhow!("Usage: proton-tracer validate <deck.toml> [--set key=value ...]"))?;

    match SimConfig::load_with_overrides(&deck_path, &cli_overrides) {
        Ok(_)  => { println!("OK  {}", deck_path); Ok(()) }
        Err(e) => {
            eprintln!("ERR {}: {}", deck_path, e);
            std::process::exit(1);
        }
    }
}

fn run_inspect_subcommand(argv: &[String]) -> Result<()> {
    let path = argv.iter().find(|a| !a.starts_with('-'))
        .ok_or_else(|| anyhow::anyhow!(
            "Usage: proton-tracer inspect <run_dir|sweep_dir>"
        ))?;
    inspect::inspect(std::path::Path::new(path))
}

fn run_analyze_subcommand(argv: &[String]) -> Result<()> {
    let mut path: Option<PathBuf> = None;
    let mut use_raw = false;
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "--raw"       => use_raw = true,
            "--help"|"-h" => {
                println!("proton-tracer analyze <run_dir> [--raw]");
                println!("  Compute count statistics from processed_counts.bin (default) or raw_counts.bin.");
                return Ok(());
            }
            arg if !arg.starts_with('-') && path.is_none() => {
                path = Some(PathBuf::from(arg));
            }
            _ => {}
        }
        i += 1;
    }
    let path = path.ok_or_else(|| anyhow::anyhow!(
        "Usage: proton-tracer analyze <run_dir> [--raw]"
    ))?;
    inspect::analyze(&path, use_raw)
}

fn run_explain_subcommand(argv: &[String]) -> Result<()> {
    let mut deck_path: Option<String> = None;
    let mut out_dir: Option<String> = None;
    let mut cli_overrides: Vec<overrides::ConfigOverride> = Vec::new();
    let mut i = 0;
    while i < argv.len() {
        match argv[i].as_str() {
            "-o" | "--output" => { i += 1; out_dir = argv.get(i).cloned(); }
            "--set" => {
                i += 1;
                if let Some(s) = argv.get(i) {
                    cli_overrides.push(overrides::parse_override(s)?);
                }
            }
            arg if !arg.starts_with('-') && deck_path.is_none() => {
                deck_path = Some(arg.to_string());
            }
            _ => {}
        }
        i += 1;
    }
    let deck_path = deck_path
        .ok_or_else(|| anyhow::anyhow!("Usage: proton-tracer explain <deck.toml> [-o run_dir] [--set key=value ...]"))?;

    if !cli_overrides.is_empty() {
        log::info!("Applying {} CLI override(s):", cli_overrides.len());
    }
    let mut config = SimConfig::load_with_overrides(&deck_path, &cli_overrides)?;
    let config_dir = std::path::Path::new(&deck_path)
        .parent().unwrap_or(std::path::Path::new("."));
    let field_path = config_dir.join(&config.field_path);

    println!("Config: {}", deck_path);
    println!("Field:  {}", field_path.display());

    let mut field = FieldData::load(&field_path)?;

    if let Some(ref e_path_str) = config.e_field_path.clone() {
        let e_path = config_dir.join(e_path_str);
        let e_field = FieldData::load(&e_path)?;
        field.set_e_from_separate_file(e_field)?;
    }

    // Resolve beam center (mirrors load_simulation)
    let beam_center: [f32; 3] = match &config.source.geometry {
        SimSourceGeometry::Pencil { position_m, .. } |
        SimSourceGeometry::Point  { position_m, .. } |
        SimSourceGeometry::Disk   { center_m: position_m, .. } => *position_m,
        SimSourceGeometry::ParallelBeam { center_m, .. } => {
            center_m.unwrap_or_else(|| {
                if let Some(src_m) = config.source.source_distance_m {
                    let b = &field.bounds;
                    let cx = (b.x_min + b.x_max) / 2.0;
                    let cy = (b.y_min + b.y_max) / 2.0;
                    let cz = (b.z_min + b.z_max) / 2.0;
                    let d = config.source.beam_direction();
                    [cx - d[0] * src_m as f32,
                     cy - d[1] * src_m as f32,
                     cz - d[2] * src_m as f32]
                } else {
                    [-0.1, 0.0, 0.0]
                }
            })
        }
    };
    if let SimSourceGeometry::ParallelBeam { ref mut center_m, .. } = config.source.geometry {
        *center_m = Some(beam_center);
    }

    // Resolve detector center
    if let Some(dist_m) = config.detector.distance_m {
        let beam_dir = config.source.beam_direction();
        let b = &field.bounds;
        let field_exit = [
            if beam_dir[0] > 0.0 { b.x_max } else if beam_dir[0] < 0.0 { b.x_min } else { 0.0 },
            if beam_dir[1] > 0.0 { b.y_max } else if beam_dir[1] < 0.0 { b.y_min } else { 0.0 },
            if beam_dir[2] > 0.0 { b.z_max } else if beam_dir[2] < 0.0 { b.z_min } else { 0.0 },
        ];
        let d = dist_m as f32;
        config.detector.center_m = Some([
            (field_exit[0] + d * beam_dir[0]) as f64,
            (field_exit[1] + d * beam_dir[1]) as f64,
            (field_exit[2] + d * beam_dir[2]) as f64,
        ]);
    }

    if !config.dt_was_supplied {
        config.dt_s = compute_recommended_dt(&config, &field);
    }

    for line in build_experiment_summary(&config, &field) {
        println!("{}", line);
    }

    // Output plan when -o is given
    if let Some(ref out) = out_dir {
        const TEX: u32 = 1024; // DETECTOR_RESOLUTION
        let size_mib = (TEX * TEX * 4) as f64 / (1024.0 * 1024.0);
        println!();
        println!("Output plan (dry-run — files not created):");
        println!("  {}/", out);
        println!("    input_deck.toml");
        println!("    resolved_config.json");
        println!("    metadata.json");
        println!("    log.txt");
        println!("    counts/");
        println!("      raw_counts.bin         [uint32, {}×{}, {:.0} MiB]", TEX, TEX, size_mib);
        println!("      processed_counts.bin   [float32, {}×{}, {:.0} MiB]", TEX, TEX, size_mib);
        println!("    images/");
        println!("      radiograph.png");
        println!("    tables/");
        println!("      (hits.csv not written by default)");
    }

    Ok(())
}

// ── Deck templates for `init` ─────────────────────────────────────────────────

const DECK_TEMPLATE_BLANK: &str = r#"name = "my_experiment"

[field]
path = "data/my_field.bfld"
scale_B = 1.0
scale_E = 0.0

[source]
type = "disk"
center_mm = [-80.0, 0.0, 0.0]
direction = [1.0, 0.0, 0.0]
radius_um = 40.0
energy_MeV = 14.7
energy_spread_percent = 5.0
cone_half_angle_deg = 0.0
n_particles = 1000000

[detector]
center_mm = [100.0, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
width_mm = 250.0
height_mm = 250.0
pixels = [1024, 1024]

[numerics]
integrator = "boris"
dt_ps = 1.0
max_steps = 10000

[render]
scale = "log"
colormap = "rcf"
exposure = 1.0

[output]
write_raw_counts = true
write_processed_counts = true
write_png = true
write_metadata = true
"#;

const DECK_TEMPLATE_ZPINCH: &str = r#"name = "zpinch"

[field]
path = "data/instabilities/zpinch.bfld"
scale_B = 1.0
scale_E = 0.0

[source]
type = "parallel"
direction = [1.0, 0.0, 0.0]
beam_radius_mm = 30.0
source_distance_mm = 100.0
energy_MeV = 14.7
energy_spread_percent = 0.0
n_particles = 100000

[detector]
center_mm = [110.0, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]
up = [0.0, 1.0, 0.0]
width_mm = 500.0
height_mm = 500.0
pixels = [512, 512]

[numerics]
integrator = "boris"
dt_ps = 1.0
max_steps = 20000

[render]
scale = "log"
colormap = "rcf"
exposure = 1.0

[output]
write_raw_counts = true
write_processed_counts = true
write_png = true
write_metadata = true
"#;

const DECK_TEMPLATE_KINK_STRONG: &str = r#"name = "kink_strong"

[field]
path = "data/instabilities/kink_strong.bfld"
scale_B = 1.0
scale_E = 0.0

[source]
type = "disk"
center_mm = [-80.0, 0.0, 0.0]
direction = [1.0, 0.0, 0.0]
radius_um = 40.0
energy_MeV = 14.7
energy_spread_percent = 5.0
cone_half_angle_deg = 0.0
n_particles = 1000000

[detector]
center_mm = [100.0, 0.0, 0.0]
normal = [1.0, 0.0, 0.0]
up = [0.0, 0.0, 1.0]
width_mm = 250.0
height_mm = 250.0
pixels = [1024, 1024]

[numerics]
integrator = "boris"
dt_ps = 1.0
max_steps = 10000

[render]
scale = "log"
colormap = "rcf"
exposure = 1.0

[output]
write_raw_counts = true
write_processed_counts = true
write_png = true
write_metadata = true
"#;

fn main() -> Result<()> {
    let argv: Vec<String> = std::env::args().collect();
    let subcommand = argv.get(1).map(|s| s.as_str());

    // Non-GPU subcommands: init logger now and dispatch immediately.
    let is_nongpu = matches!(
        subcommand,
        Some("render" | "init" | "validate" | "explain" | "inspect" | "sweep" | "analyze")
    );
    if is_nongpu {
        env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
            .init();
        return match subcommand {
            Some("render")   => run_render_subcommand(&argv[2..]),
            Some("init")     => run_init_subcommand(&argv[2..]),
            Some("validate") => run_validate_subcommand(&argv[2..]),
            Some("explain")  => run_explain_subcommand(&argv[2..]),
            Some("inspect")  => run_inspect_subcommand(&argv[2..]),
            Some("analyze")  => run_analyze_subcommand(&argv[2..]),
            Some("sweep")    => {
                let spec = sweep::parse_sweep_args(&argv[2..])?;
                sweep::run_sweep(&spec)
            }
            _ => unreachable!(),
        };
    }

    // GPU-based modes: parse args first.
    let args = match subcommand {
        Some("run")  => parse_run_args(&argv[2..]),
        Some("gui")  => parse_gui_args(&argv[2..]),
        Some("demo") => parse_demo_args(&argv[2..]),
        _            => CliArgs::parse_legacy(&argv[1..]),
    };

    // For the `run` subcommand: create the run directory and install TeeLogger
    // before Vulkan so all GPU init logs land in log.txt too.
    // For GUI / demo / legacy: install global logger (stderr only; file sink added per run).
    let (run_dir, run_meta) = if matches!(subcommand, Some("run")) {
        let rd = run_dir::RunDir::open(args.output_dir.clone(), &args.run_opts)?;
        run_dir::init_global_logger()?;
        run_dir::attach_log_tee(&rd.log_path())?;
        let run_name = args.output_dir
            .file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("run")
            .to_string();
        let deck_path = args.config_path.clone().unwrap_or_default();
        let mut meta = run_dir::RunMetadata::new_running(run_name, deck_path, argv.clone());
        if !args.overrides.is_empty() {
            meta.cli_overrides = Some(args.overrides.iter().map(|ov| run_dir::CliOverrideRecord {
                key: ov.canonical_key.clone(),
                value: ov.raw_value.clone(),
            }).collect());
        }
        rd.write_metadata(&meta)?;
        log::info!("Run directory: {:?}", rd.root());
        (Some(rd), Some(meta))
    } else {
        run_dir::init_global_logger()?;
        (None, None)
    };

    match subcommand {
        Some("run")  => log::info!("Proton Tracer — batch run"),
        Some("gui")  => log::info!("Proton Tracer — GUI mode"),
        Some("demo") => log::info!("Proton Tracer — demo mode"),
        _            => log::info!("Proton Tracer starting..."),
    }
    if args.batch_mode && run_dir.is_none() {
        log::info!("Batch mode — output: {:?}", args.output_dir);
    }

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(args, run_dir, run_meta, argv);
    event_loop.run_app(&mut app)?;

    if app.init_failed {
        std::process::exit(1);
    }

    log::info!("Proton Tracer shut down cleanly");
    Ok(())
}
