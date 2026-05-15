//! GPU timing infrastructure for reliable benchmarking
//!
//! Since MoltenVK doesn't support GPU timestamp queries, we measure total
//! frame time with GPU synchronization. Compute vs render breakdown requires
//! separate command buffer submissions with fences.

use ash::vk;
use anyhow::{Result, Context};
use std::time::Instant;

/// Number of timestamp query slots (kept for future use on other platforms)
const NUM_TIMESTAMPS: u32 = 8;

/// Benchmark results in milliseconds
#[derive(Debug, Clone, Default)]
pub struct BenchmarkResults {
    pub total_frames: u64,
    pub total_particles: u64,
    pub total_steps: u64,           // Steps per frame
    pub steps_per_dispatch: u64,    // Steps batched per dispatch (for bandwidth calc)
    pub total_gpu_ms: f64,          // Total GPU time (compute + render combined)
    pub total_wall_ms: f64,         // Total wall clock time
}

impl BenchmarkResults {
    pub fn avg_gpu_ms(&self) -> f64 {
        if self.total_frames > 0 {
            self.total_gpu_ms / self.total_frames as f64
        } else {
            0.0
        }
    }

    pub fn avg_fps(&self) -> f64 {
        if self.total_wall_ms > 0.0 {
            self.total_frames as f64 / (self.total_wall_ms / 1000.0)
        } else {
            0.0
        }
    }

    pub fn particles_per_sec(&self) -> f64 {
        // Total particle-steps = particles × steps × frames
        if self.total_gpu_ms > 0.0 {
            let total_particle_steps = self.total_particles as f64
                * self.total_steps as f64
                * self.total_frames as f64;
            (total_particle_steps * 1000.0) / self.total_gpu_ms
        } else {
            0.0
        }
    }

    pub fn particle_bandwidth_gbps(&self) -> f64 {
        // With batching: particles are read/written once per DISPATCH, not per step
        // Each particle: 32 bytes read + 32 bytes write = 64 bytes per dispatch
        // Dispatches per frame = ceil(steps / steps_per_dispatch)
        if self.total_gpu_ms > 0.0 && self.steps_per_dispatch > 0 {
            let dispatches_per_frame = (self.total_steps + self.steps_per_dispatch - 1) / self.steps_per_dispatch;
            let total_bytes = self.total_particles as f64
                * 64.0  // read + write
                * dispatches_per_frame as f64
                * self.total_frames as f64;
            (total_bytes * 1000.0) / (self.total_gpu_ms * 1e9)
        } else {
            0.0
        }
    }

    pub fn texture_bandwidth_gbps(&self) -> f64 {
        // Estimate texture bandwidth from B-field sampling
        // Each step samples the field (trilinear = up to 8 texels × 16 bytes = 128 bytes)
        // But texture cache is very effective, estimate 10% cache miss rate
        if self.total_gpu_ms > 0.0 {
            let samples_per_particle = self.total_steps as f64;
            let bytes_per_sample = 128.0 * 0.1;  // 10% cache miss estimate
            let total_bytes = self.total_particles as f64
                * samples_per_particle
                * bytes_per_sample
                * self.total_frames as f64;
            (total_bytes * 1000.0) / (self.total_gpu_ms * 1e9)
        } else {
            0.0
        }
    }

    pub fn log_summary(&self) {
        let m4_max_bandwidth = 120.0; // GB/s theoretical max for M4
        let particle_bw = self.particle_bandwidth_gbps();
        let texture_bw = self.texture_bandwidth_gbps();
        let total_bw = particle_bw + texture_bw;
        let efficiency = (total_bw / m4_max_bandwidth * 100.0).min(100.0);

        let dispatches_per_frame = if self.steps_per_dispatch > 0 {
            (self.total_steps + self.steps_per_dispatch - 1) / self.steps_per_dispatch
        } else {
            self.total_steps
        };

        log::info!("═══════════════════════════════════════════════════════════");
        log::info!("                    BENCHMARK RESULTS                       ");
        log::info!("═══════════════════════════════════════════════════════════");
        log::info!("Configuration:");
        log::info!("  Particles:        {:>10}", format_number(self.total_particles));
        log::info!("  Steps/frame:      {:>10}", self.total_steps);
        log::info!("  Steps/dispatch:   {:>10} (batched)", self.steps_per_dispatch);
        log::info!("  Dispatches/frame: {:>10}", dispatches_per_frame);
        log::info!("  Total frames:     {:>10}", self.total_frames);
        log::info!("───────────────────────────────────────────────────────────");
        log::info!("Timing:");
        log::info!("  GPU time:      {:>10.2} ms/frame", self.avg_gpu_ms());
        log::info!("  Total GPU:     {:>10.1} ms", self.total_gpu_ms);
        log::info!("  Avg FPS:       {:>10.1} fps", self.avg_fps());
        log::info!("───────────────────────────────────────────────────────────");
        log::info!("Throughput:");
        log::info!("  Particle-steps/sec: {:.2} billion", self.particles_per_sec() / 1e9);
        log::info!("───────────────────────────────────────────────────────────");
        log::info!("Memory:");
        log::info!("  Particle buffer:  {:>6.1} GB/s (actual)", particle_bw);
        log::info!("  M4 theoretical:   {:>6.0} GB/s", m4_max_bandwidth);
        log::info!("═══════════════════════════════════════════════════════════");
    }
}

fn format_number(n: u64) -> String {
    if n >= 1_000_000 {
        format!("{:.1}M", n as f64 / 1_000_000.0)
    } else if n >= 1_000 {
        format!("{:.1}K", n as f64 / 1_000.0)
    } else {
        format!("{}", n)
    }
}

/// GPU timing using CPU-side measurement with GPU synchronization
pub struct GpuTiming {
    // Vulkan query pool (kept for platforms that support timestamps)
    query_pool: vk::QueryPool,
    #[allow(dead_code)]
    timestamp_period_ns: f64,
    #[allow(dead_code)]
    timestamps_valid: bool,

    // Per-frame timing - measured AFTER GPU sync
    frame_start: Instant,
    last_frame_end: Instant,

    // Benchmark accumulation (only during active simulation)
    benchmark: BenchmarkResults,
    benchmark_active: bool,

    // Configuration
    particles_per_frame: u64,
    steps_per_frame: u64,
    steps_per_dispatch: u64,

    // Frame counter for logging
    frame_count: u64,
}

impl GpuTiming {
    pub fn new(device: &ash::Device, timestamp_period: f32, timestamps_supported: bool) -> Result<Self> {
        let query_pool = if timestamps_supported {
            let create_info = vk::QueryPoolCreateInfo {
                query_type: vk::QueryType::TIMESTAMP,
                query_count: NUM_TIMESTAMPS,
                ..Default::default()
            };

            unsafe {
                device.create_query_pool(&create_info, None)
                    .context("Failed to create timestamp query pool")?
            }
        } else {
            log::info!("GPU timestamps not supported - using CPU timing with GPU sync");
            vk::QueryPool::null()
        };

        let now = Instant::now();
        Ok(Self {
            query_pool,
            timestamp_period_ns: timestamp_period as f64,
            timestamps_valid: timestamps_supported,
            frame_start: now,
            last_frame_end: now,
            benchmark: BenchmarkResults::default(),
            benchmark_active: false,
            particles_per_frame: 0,
            steps_per_frame: 0,
            steps_per_dispatch: 0,
            frame_count: 0,
        })
    }

    /// Configure benchmark parameters
    pub fn configure(&mut self, particles: u64, steps_per_frame: u64, steps_per_dispatch: u64) {
        self.particles_per_frame = particles;
        self.steps_per_frame = steps_per_frame;
        self.steps_per_dispatch = steps_per_dispatch;
    }

    /// Start a new benchmark (call when simulation starts)
    pub fn start_benchmark(&mut self) {
        self.benchmark = BenchmarkResults {
            total_particles: self.particles_per_frame,
            total_steps: self.steps_per_frame,
            steps_per_dispatch: self.steps_per_dispatch,
            ..Default::default()
        };
        self.benchmark_active = true;
        self.last_frame_end = Instant::now();
        let dispatches = (self.steps_per_frame + self.steps_per_dispatch - 1) / self.steps_per_dispatch;
        log::info!("Benchmark started: {} particles × {} steps/frame ({} steps/dispatch = {} dispatches)",
            self.particles_per_frame, self.steps_per_frame, self.steps_per_dispatch, dispatches);
    }

    /// Stop benchmark and return results
    pub fn stop_benchmark(&mut self) -> BenchmarkResults {
        self.benchmark_active = false;
        self.benchmark.clone()
    }

    /// Check if benchmark is active
    pub fn is_benchmarking(&self) -> bool {
        self.benchmark_active
    }

    /// Call at the start of frame - BEFORE any GPU work
    pub fn begin_frame(&mut self, _device: &ash::Device, _cmd: vk::CommandBuffer) {
        self.frame_start = Instant::now();
    }

    /// These are no-ops - we measure total frame time instead of trying to
    /// separate compute/render (which doesn't work without GPU timestamps)
    pub fn begin_compute(&mut self, _device: &ash::Device, _cmd: vk::CommandBuffer) {}
    pub fn end_compute(&mut self, _device: &ash::Device, _cmd: vk::CommandBuffer) {}
    pub fn begin_render(&mut self, _device: &ash::Device, _cmd: vk::CommandBuffer) {}
    pub fn end_render(&mut self, _device: &ash::Device, _cmd: vk::CommandBuffer) {}

    /// Call AFTER device_wait_idle() - this is when we know GPU has finished
    /// The time from frame_start to now includes all GPU work
    pub fn end_frame_after_sync(&mut self, compute_ran: bool) {
        let now = Instant::now();

        // GPU time = time from frame start to now (after GPU sync)
        let gpu_ms = self.frame_start.elapsed().as_secs_f64() * 1000.0;

        // Wall time = time since last frame ended (includes any CPU overhead)
        let wall_ms = self.last_frame_end.elapsed().as_secs_f64() * 1000.0;
        self.last_frame_end = now;

        self.frame_count += 1;

        // Only accumulate if benchmark is active AND compute actually ran
        if self.benchmark_active && compute_ran {
            self.benchmark.total_frames += 1;
            self.benchmark.total_gpu_ms += gpu_ms;
            self.benchmark.total_wall_ms += wall_ms;
        }
    }

    /// Legacy interface - calls end_frame_after_sync
    pub fn end_frame(&mut self, _device: &ash::Device, compute_ran: bool) {
        self.end_frame_after_sync(compute_ran);
    }

    /// Get current benchmark (snapshot)
    pub fn get_benchmark(&self) -> BenchmarkResults {
        self.benchmark.clone()
    }

    pub fn cleanup(&mut self, device: &ash::Device) {
        if self.query_pool != vk::QueryPool::null() {
            unsafe {
                device.destroy_query_pool(self.query_pool, None);
            }
            self.query_pool = vk::QueryPool::null();
        }
    }
}
