//! Bethe-Bloch stopping power tables for protons in various materials.
//!
//! Computes mass stopping power dE/dρx [MeV cm²/g] on a log-spaced kinetic
//! energy grid and packs it into `StoppingGpuData` for upload to binding 6.
//!
//! When no density grid is configured the table is all-zeros and the
//! density texture is 1×1×1 with ρ = 0 — no energy loss occurs.

use bytemuck::{Pod, Zeroable};

const K: f64 = 0.307075;      // 4πNₐrₑ²mₑc² [MeV cm²/g]
const ME_C2: f64 = 0.51099895;  // electron rest energy [MeV]
const MP_C2: f64 = 938.272046;  // proton rest energy [MeV]

pub const TABLE_N: usize = 256;
const E_MIN_MEV: f64 = 0.1;    // minimum table KE [MeV]
const E_MAX_MEV: f64 = 1000.0; // maximum table KE [MeV]

/// Material parameters for Bethe-Bloch.
#[derive(Debug, Clone, Copy)]
pub struct Material {
    pub z_over_a: f64, // effective Z/A (dimensionless)
    pub i_ev: f64,     // mean excitation energy [eV]
}

impl Material {
    pub const WATER:     Self = Self { z_over_a: 0.5551, i_ev: 75.0  };
    pub const PLASTIC:   Self = Self { z_over_a: 0.5702, i_ev: 57.4  }; // polypropylene
    pub const BERYLLIUM: Self = Self { z_over_a: 0.4439, i_ev: 63.7  };
    pub const ALUMINUM:  Self = Self { z_over_a: 0.4818, i_ev: 166.0 };
    pub const HYDROGEN:  Self = Self { z_over_a: 0.9922, i_ev: 19.2  };

    pub fn from_name(name: &str) -> Option<Self> {
        match name.to_lowercase().as_str() {
            "water"                => Some(Self::WATER),
            "plastic" | "ch2"      => Some(Self::PLASTIC),
            "beryllium" | "be"     => Some(Self::BERYLLIUM),
            "aluminum" | "aluminium" | "al" => Some(Self::ALUMINUM),
            "hydrogen" | "h"       => Some(Self::HYDROGEN),
            _ => None,
        }
    }

    pub fn custom(z_over_a: f64, i_ev: f64) -> Self {
        Self { z_over_a, i_ev }
    }
}

/// Bethe-Bloch mass stopping power [MeV cm²/g] for a proton at KE `ke_mev`.
fn bethe_bloch(ke_mev: f64, mat: &Material) -> f64 {
    let gamma = 1.0 + ke_mev / MP_C2;
    let beta2 = (1.0 - 1.0 / (gamma * gamma)).max(1e-12);

    // Maximum kinetic energy transfer to a free electron
    let tmax = 2.0 * ME_C2 * beta2 * gamma * gamma
        / (1.0 + 2.0 * gamma * ME_C2 / MP_C2 + (ME_C2 / MP_C2).powi(2));

    let i_mev = mat.i_ev * 1e-6;
    let arg = 2.0 * ME_C2 * beta2 * gamma * gamma * tmax / (i_mev * i_mev);

    let bracket = 0.5 * arg.ln() - beta2;
    if bracket <= 0.0 { return 0.0; }

    K * mat.z_over_a / beta2 * bracket
}

/// Data packed into the GPU storage buffer at binding 6 (std430 layout).
///
/// The dedx array holds mass stopping power [MeV cm²/g] at 256 log-spaced
/// kinetic energy samples from `E_MIN_MEV` to `E_MAX_MEV`.
///
/// Density grid world bounds are included so the shader can convert world
/// position → texture UV without touching push constants (which are full).
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
pub struct StoppingGpuData {
    pub dedx: [f32; TABLE_N],      // mass stopping power [MeV cm²/g]
    pub log_e_min: f32,            // log(E_min_MeV)
    pub log_e_inv_range: f32,      // 1 / (log(E_max_MeV) - log(E_min_MeV))
    pub dens_xmin: f32,
    pub dens_ymin: f32,
    pub dens_zmin: f32,
    pub dens_xmax: f32,
    pub dens_ymax: f32,
    pub dens_zmax: f32,
    pub _pad: [f32; 2],
}

impl StoppingGpuData {
    /// Build table from a material and density grid bounds [m].
    pub fn for_material(
        mat: &Material,
        dens_min: [f32; 3],
        dens_max: [f32; 3],
    ) -> Self {
        let log_e_min = E_MIN_MEV.ln();
        let log_e_max = E_MAX_MEV.ln();
        let log_e_range = log_e_max - log_e_min;

        let mut dedx = [0.0f32; TABLE_N];
        for i in 0..TABLE_N {
            let t = i as f64 / (TABLE_N - 1) as f64;
            let ke_mev = (log_e_min + t * log_e_range).exp();
            dedx[i] = bethe_bloch(ke_mev, mat) as f32;
        }

        Self {
            dedx,
            log_e_min: log_e_min as f32,
            log_e_inv_range: (1.0 / log_e_range) as f32,
            dens_xmin: dens_min[0], dens_ymin: dens_min[1], dens_zmin: dens_min[2],
            dens_xmax: dens_max[0], dens_ymax: dens_max[1], dens_zmax: dens_max[2],
            _pad: [0.0; 2],
        }
    }

    /// Vacuum fallback: zero table, bounds don't matter (density is 0 everywhere).
    pub fn vacuum() -> Self {
        Self {
            dedx: [0.0; TABLE_N],
            log_e_min: E_MIN_MEV.ln() as f32,
            log_e_inv_range: (1.0 / (E_MAX_MEV.ln() - E_MIN_MEV.ln())) as f32,
            dens_xmin: -1.0, dens_ymin: -1.0, dens_zmin: -1.0,
            dens_xmax:  1.0, dens_ymax:  1.0, dens_zmax:  1.0,
            _pad: [0.0; 2],
        }
    }
}
