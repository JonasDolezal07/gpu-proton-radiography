//! Particle data generation and loading

use anyhow::{Context, Result};
use glam::Vec3;
use rand::SeedableRng;
use rand_distr::Distribution;

use super::{SimSourceConfig, SimSourceGeometry};
use crate::units::proton_speed_from_mev;

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
    /// Generate particles from source configuration.
    /// The geometry's `center_m` must be resolved before calling (see main.rs).
    pub fn generate(config: &SimSourceConfig) -> Result<Self> {
        let n = config.n_particles as usize;

        // Seeded RNG for energy sampling. Spatial/angular still uses rand_f32() (non-seeded).
        // TODO: unify all particle sampling under the same seeded RNG in a future pass.
        let mut energy_rng: rand::rngs::StdRng = match config.seed {
            Some(s) => rand::rngs::StdRng::seed_from_u64(s),
            None    => rand::rngs::StdRng::from_entropy(),
        };
        let energy_dist: Option<rand_distr::Normal<f64>> = if config.energy_spread_percent > 0.0 {
            let sigma = config.particle_energy_mev * config.energy_spread_percent / 100.0;
            Some(rand_distr::Normal::new(config.particle_energy_mev, sigma)
                .map_err(|e| anyhow::anyhow!("Invalid energy distribution: {}", e))?)
        } else {
            None
        };

        let mono_speed = config.particle_speed_m_s as f32;

        let mut particles = Vec::with_capacity(n);

        match &config.geometry {
            SimSourceGeometry::ParallelBeam { center_m, direction, radius_m, angular_spread_rad } => {
                let center = Vec3::from(
                    center_m.context("Beam center not resolved — ensure load_simulation() runs before generate()")?
                );
                let dir    = Vec3::from(*direction).normalize();
                let radius = *radius_m as f32;
                let spread = *angular_spread_rad;

                let perp1 = if dir.z.abs() < 0.9 {
                    dir.cross(Vec3::Z).normalize()
                } else {
                    dir.cross(Vec3::X).normalize()
                };
                let perp2 = dir.cross(perp1);

                let cos_spread = spread.cos();

                for _ in 0..n {
                    let spd = sample_speed(&energy_dist, &mut energy_rng, mono_speed);
                    // Position: uniform disk
                    let phi = rand_f32() * std::f32::consts::TAU;
                    let r   = radius * rand_f32().sqrt();
                    let pos = center + perp1 * r * phi.cos() + perp2 * r * phi.sin();

                    // Direction: uniform solid-angle cone of half-angle spread.
                    // When spread == 0, cos_psi == 1 and sin_psi == 0, reducing to dir.
                    let vel = if spread > 0.0 {
                        let az      = rand_f32() * std::f32::consts::TAU;
                        let cos_psi = 1.0 - rand_f32() * (1.0 - cos_spread);
                        let sin_psi = (1.0 - cos_psi * cos_psi).max(0.0).sqrt();
                        let v_dir   = dir * cos_psi
                            + perp1 * sin_psi * az.cos()
                            + perp2 * sin_psi * az.sin();
                        v_dir * spd
                    } else {
                        dir * spd
                    };

                    particles.push(Particle {
                        position:  pos.to_array(),
                        _pad0:     0.0,
                        velocity:  vel.to_array(),
                        is_active: 1,
                    });
                }
            }

            SimSourceGeometry::Pencil { position_m, direction } => {
                let pos = Vec3::from(*position_m);
                let dir = Vec3::from(*direction).normalize();
                for _ in 0..n {
                    let spd = sample_speed(&energy_dist, &mut energy_rng, mono_speed);
                    particles.push(Particle {
                        position:  pos.to_array(),
                        _pad0:     0.0,
                        velocity:  (dir * spd).to_array(),
                        is_active: 1,
                    });
                }
            }

            SimSourceGeometry::Disk { center_m, direction, radius_m, half_angle_rad } => {
                let center   = Vec3::from(*center_m);
                let dir      = Vec3::from(*direction).normalize();
                let radius   = *radius_m;
                let half     = *half_angle_rad;
                let cos_half = half.cos();

                let perp1 = if dir.z.abs() < 0.9 {
                    dir.cross(Vec3::Z).normalize()
                } else {
                    dir.cross(Vec3::X).normalize()
                };
                let perp2 = dir.cross(perp1);

                for _ in 0..n {
                    let spd = sample_speed(&energy_dist, &mut energy_rng, mono_speed);
                    // Position: uniform disk in the plane perpendicular to dir.
                    let phi = rand_f32() * std::f32::consts::TAU;
                    let r   = radius * rand_f32().sqrt();
                    let pos = center + perp1 * r * phi.cos() + perp2 * r * phi.sin();

                    // Direction: uniform solid-angle cone.
                    let az      = rand_f32() * std::f32::consts::TAU;
                    let cos_psi = if half > 0.0 {
                        1.0 - rand_f32() * (1.0 - cos_half)
                    } else {
                        1.0
                    };
                    let sin_psi = (1.0 - cos_psi * cos_psi).max(0.0).sqrt();
                    let v_dir   = dir * cos_psi
                        + perp1 * sin_psi * az.cos()
                        + perp2 * sin_psi * az.sin();

                    particles.push(Particle {
                        position:  pos.to_array(),
                        _pad0:     0.0,
                        velocity:  (v_dir * spd).to_array(),
                        is_active: 1,
                    });
                }
            }

            SimSourceGeometry::Point { position_m, direction, half_angle_rad } => {
                let pos  = Vec3::from(*position_m);
                let dir  = Vec3::from(*direction).normalize();
                let half = *half_angle_rad;
                let cos_half = half.cos();

                let perp1 = if dir.z.abs() < 0.9 {
                    dir.cross(Vec3::Z).normalize()
                } else {
                    dir.cross(Vec3::X).normalize()
                };
                let perp2 = dir.cross(perp1);

                for _ in 0..n {
                    let spd = sample_speed(&energy_dist, &mut energy_rng, mono_speed);
                    let az      = rand_f32() * std::f32::consts::TAU;
                    let cos_psi = if half > 0.0 {
                        1.0 - rand_f32() * (1.0 - cos_half)
                    } else {
                        1.0
                    };
                    let sin_psi = (1.0 - cos_psi * cos_psi).max(0.0).sqrt();
                    let v_dir   = dir * cos_psi
                        + perp1 * sin_psi * az.cos()
                        + perp2 * sin_psi * az.sin();
                    particles.push(Particle {
                        position:  pos.to_array(),
                        _pad0:     0.0,
                        velocity:  (v_dir * spd).to_array(),
                        is_active: 1,
                    });
                }
            }
        }

        if !particles.is_empty() {
            sort_particles_by_morton(&mut particles);
            log::info!("Sorted {} particles by Morton code for cache locality", particles.len());
        }

        if !particles.is_empty() {
            log::info!("Particle sanity check:");
            for i in [0, 1, 2, particles.len() / 2, particles.len() - 1] {
                if i < particles.len() {
                    let p = &particles[i];
                    let spd = (p.velocity[0].powi(2) + p.velocity[1].powi(2) + p.velocity[2].powi(2)).sqrt();
                    log::info!(
                        "  [{}] pos=({:.4}, {:.4}, {:.4}) |v|={:.3e} m/s",
                        i, p.position[0], p.position[1], p.position[2], spd
                    );
                }
            }
        }

        Ok(Self { count: particles.len() as u32, particles })
    }

    pub fn size_bytes(&self) -> usize {
        self.particles.len() * std::mem::size_of::<Particle>()
    }
}

/// Sample per-particle speed: if energy_dist is Some, draw a positive energy
/// (MeV) via rejection sampling and convert to SI speed; otherwise return mono_speed.
fn sample_speed(
    energy_dist: &Option<rand_distr::Normal<f64>>,
    rng: &mut rand::rngs::StdRng,
    mono_speed: f32,
) -> f32 {
    match energy_dist {
        None => mono_speed,
        Some(dist) => {
            let e_mev = loop {
                let e = dist.sample(rng);
                if e > 0.0 { break e; }
            };
            proton_speed_from_mev(e_mev) as f32
        }
    }
}

fn rand_f32() -> f32 {
    use std::collections::hash_map::RandomState;
    use std::hash::{BuildHasher, Hasher};

    static COUNTER: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

    let state = RandomState::new();
    let mut hasher = state.build_hasher();
    hasher.write_u64(COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed));
    (hasher.finish() as f32) / (u64::MAX as f32)
}

fn sort_particles_by_morton(particles: &mut [Particle]) {
    if particles.is_empty() { return; }

    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];
    for p in particles.iter() {
        for i in 0..3 {
            min[i] = min[i].min(p.position[i]);
            max[i] = max[i].max(p.position[i]);
        }
    }

    let range = [
        (max[0] - min[0]).max(1e-6),
        (max[1] - min[1]).max(1e-6),
        (max[2] - min[2]).max(1e-6),
    ];

    let mut indexed: Vec<(u32, usize)> = particles
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let x = (((p.position[0] - min[0]) / range[0]) * 1023.0) as u32;
            let y = (((p.position[1] - min[1]) / range[1]) * 1023.0) as u32;
            let z = (((p.position[2] - min[2]) / range[2]) * 1023.0) as u32;
            (morton_encode(x, y, z), i)
        })
        .collect();

    indexed.sort_unstable_by_key(|(morton, _)| *morton);

    let original: Vec<Particle> = particles.to_vec();
    for (new_idx, (_, old_idx)) in indexed.iter().enumerate() {
        particles[new_idx] = original[*old_idx];
    }
}

fn morton_encode(x: u32, y: u32, z: u32) -> u32 {
    spread_bits(x) | (spread_bits(y) << 1) | (spread_bits(z) << 2)
}

fn spread_bits(mut x: u32) -> u32 {
    x = x & 0x3FF;
    x = (x | (x << 16)) & 0x030000FF;
    x = (x | (x << 8))  & 0x0300F00F;
    x = (x | (x << 4))  & 0x030C30C3;
    x = (x | (x << 2))  & 0x09249249;
    x
}
