//! Proton Radiography Tracer - GPU Accelerated
//!
//! Traces charged particles through electromagnetic fields using Vulkan compute.

#[cfg(target_os = "macos")]
#[macro_use]
extern crate objc;

mod gpu;
mod loaders;

use anyhow::Result;
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};
use std::sync::Arc;

use gpu::{VulkanContext, Renderer};
use loaders::{FieldData, ParticleData, SimConfig};

/// Application state
struct App {
    window: Option<Arc<Window>>,
    vulkan: Option<Arc<VulkanContext>>,
    renderer: Option<Renderer>,
    config_path: Option<String>,
    needs_resize: bool,
}

impl App {
    fn new(config_path: Option<String>) -> Self {
        Self {
            window: None,
            vulkan: None,
            renderer: None,
            config_path,
            needs_resize: false,
        }
    }

    fn load_simulation(&mut self) -> Result<()> {
        let config_path = match &self.config_path {
            Some(p) => p.clone(),
            None => return Ok(()),
        };

        log::info!("Loading simulation from: {}", config_path);

        // Load config
        let config = SimConfig::load(&config_path)?;

        // Load field data
        let field_path = std::path::Path::new(&config_path)
            .parent()
            .unwrap_or(std::path::Path::new("."))
            .join(&config.field_path);

        let field = FieldData::load(&field_path)?;
        log::info!("Loaded field: {}x{}x{}", field.nx, field.ny, field.nz);

        // Generate particles
        let particles = ParticleData::generate(&config.source)?;
        log::info!("Generated {} particles", particles.count);

        // Upload to GPU
        let renderer = self.renderer.as_mut().unwrap();

        renderer.upload_field(&field)?;
        renderer.upload_particles(&particles)?;

        // Set simulation parameters
        renderer.set_sim_params(
            config.dt as f32,
            [
                config.source.detector_distance * config.source.detector_normal[0],
                config.source.detector_distance * config.source.detector_normal[1],
                config.source.detector_distance * config.source.detector_normal[2],
            ],
            config.source.detector_normal,
        );

        // Load compute shader
        const BORIS_SHADER: &[u8] = include_bytes!("../../shaders/boris.spv");
        renderer.load_compute_shader(BORIS_SHADER)?;
        renderer.update_descriptors()?;

        log::info!("Simulation ready - press Space to start");

        Ok(())
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.window.is_some() {
            return;
        }

        log::info!("Creating window...");
        let window_attrs = Window::default_attributes()
            .with_title("Proton Tracer")
            .with_inner_size(winit::dpi::LogicalSize::new(1280, 720));

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
                        self.vulkan = Some(vulkan.clone());

                        // Create renderer
                        let size = window.inner_size();
                        match Renderer::new(vulkan, size.width, size.height) {
                            Ok(renderer) => {
                                log::info!("Renderer created");
                                self.renderer = Some(renderer);

                                // Load simulation if config provided
                                if let Err(e) = self.load_simulation() {
                                    log::error!("Failed to load simulation: {}", e);
                                }
                            }
                            Err(e) => {
                                log::error!("Failed to create renderer: {}", e);
                                event_loop.exit();
                            }
                        }
                    }
                    Err(e) => {
                        log::error!("Failed to initialize Vulkan: {}", e);
                        event_loop.exit();
                    }
                }
            }
            Err(e) => {
                log::error!("Failed to create window: {}", e);
                event_loop.exit();
            }
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => {
                log::info!("Close requested, shutting down...");
                if let Some(renderer) = &mut self.renderer {
                    renderer.cleanup();
                }
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                log::debug!("Window resized to {}x{}", size.width, size.height);
                self.needs_resize = true;
            }
            WindowEvent::RedrawRequested => {
                if let Some(renderer) = &mut self.renderer {
                    // Handle resize if needed
                    if self.needs_resize {
                        if let Some(window) = &self.window {
                            let size = window.inner_size();
                            if let Err(e) = renderer.resize(size.width, size.height) {
                                log::error!("Failed to resize: {}", e);
                            }
                            self.needs_resize = false;
                        }
                    }

                    // Render frame
                    match renderer.render_frame() {
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
                        Key::Character(ref c) if c == "r" => {
                            log::info!("R pressed - reset particles");
                            // TODO: Reset particle positions
                        }
                        Key::Character(ref c) if c == "s" => {
                            if let Some(renderer) = &self.renderer {
                                // Save detector hits
                                match renderer.read_detector_hits() {
                                    Ok(hits) => {
                                        log::info!("Read {} detector hits", hits.len());
                                        // TODO: Save to file
                                    }
                                    Err(e) => log::error!("Failed to read hits: {}", e),
                                }
                            }
                        }
                        _ => {}
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

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info"))
        .init();

    log::info!("Proton Tracer starting...");

    let args: Vec<String> = std::env::args().collect();
    let config_path = args.get(1).cloned();

    let event_loop = EventLoop::new()?;
    event_loop.set_control_flow(ControlFlow::Poll);

    let mut app = App::new(config_path);
    event_loop.run_app(&mut app)?;

    log::info!("Proton Tracer shut down cleanly");
    Ok(())
}
