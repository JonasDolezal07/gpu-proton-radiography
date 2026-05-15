//! Particle data generation and loading

use anyhow::{Result, bail};
use glam::Vec3;

use super::SourceConfig;

/// Particle state for GPU
#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct Particle {
    pub position: [f32; 3],
    pub _pad0: f32,
    pub velocity: [f32; 3],
    pub is_active: u32,  // 1 = is_active, 0 = hit detector
}

/// Particle data on CPU
pub struct ParticleData {
    pub particles: Vec<Particle>,
    pub count: u32,
}

impl ParticleData {
    /// Generate particles from source configuration
    pub fn generate(config: &SourceConfig) -> Result<Self> {
        let n = config.n_protons as usize;
        let speed = config.proton_speed();

        let mut particles = Vec::with_capacity(n);

        match config.source_type.as_str() {
            "point" => {
                let source = Vec3::new(
                    config.point_position.as_ref().map(|p| p[0]).unwrap_or(0.0),
                    config.point_position.as_ref().map(|p| p[1]).unwrap_or(0.0),
                    config.point_position.as_ref().map(|p| p[2]).unwrap_or(-0.1),
                );
                let target = Vec3::new(
                    config.point_target.as_ref().map(|p| p[0]).unwrap_or(0.0),
                    config.point_target.as_ref().map(|p| p[1]).unwrap_or(0.0),
                    config.point_target.as_ref().map(|p| p[2]).unwrap_or(0.0),
                );

                let central_dir = (target - source).normalize();
                let spread = config.angular_spread;

                // Build perpendicular basis
                let perp1 = if central_dir.x.abs() < 0.9 {
                    central_dir.cross(Vec3::X).normalize()
                } else {
                    central_dir.cross(Vec3::Y).normalize()
                };
                let perp2 = central_dir.cross(perp1);

                for _ in 0..n {
                    let theta = rand_f32() * std::f32::consts::TAU;
                    let phi = rand_f32() * spread;

                    let dir = central_dir * phi.cos()
                        + perp1 * phi.sin() * theta.cos()
                        + perp2 * phi.sin() * theta.sin();

                    particles.push(Particle {
                        position: source.to_array(),
                        _pad0: 0.0,
                        velocity: (dir * speed).to_array(),
                        is_active: 1,
                    });
                }
            }
            "parallel" => {
                let center = Vec3::new(
                    config.beam_center.as_ref().map(|p| p[0]).unwrap_or(0.0),
                    config.beam_center.as_ref().map(|p| p[1]).unwrap_or(0.0),
                    config.beam_center.as_ref().map(|p| p[2]).unwrap_or(-0.1),
                );
                let dir = Vec3::new(
                    config.beam_direction.as_ref().map(|p| p[0]).unwrap_or(0.0),
                    config.beam_direction.as_ref().map(|p| p[1]).unwrap_or(0.0),
                    config.beam_direction.as_ref().map(|p| p[2]).unwrap_or(1.0),
                ).normalize();
                let radius = config.beam_radius;

                // Build perpendicular basis
                let perp1 = if dir.z.abs() < 0.9 {
                    dir.cross(Vec3::Z).normalize()
                } else {
                    dir.cross(Vec3::X).normalize()
                };
                let perp2 = dir.cross(perp1);

                let vel = dir * speed;

                for _ in 0..n {
                    let theta = rand_f32() * std::f32::consts::TAU;
                    let r = radius * rand_f32().sqrt();

                    let pos = center + perp1 * r * theta.cos() + perp2 * r * theta.sin();

                    particles.push(Particle {
                        position: pos.to_array(),
                        _pad0: 0.0,
                        velocity: vel.to_array(),
                        is_active: 1,
                    });
                }
            }
            other => bail!("Unknown source type: {}", other),
        }

        Ok(Self {
            count: particles.len() as u32,
            particles,
        })
    }

    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        self.particles.len() * std::mem::size_of::<Particle>()
    }
}

/// Simple random float [0, 1)
fn rand_f32() -> f32 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    let state = RandomState::new();
    let mut hasher = state.build_hasher();
    hasher.write_u64(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
    (hasher.finish() as f32) / (u64::MAX as f32)
}
