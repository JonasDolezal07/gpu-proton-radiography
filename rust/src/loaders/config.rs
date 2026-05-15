//! Simulation configuration loading

use anyhow::{Context, Result};
use serde::Deserialize;
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

/// Complete simulation configuration
#[derive(Debug, Deserialize)]
pub struct SimConfig {
    pub field_path: String,
    pub source: SourceConfig,
    pub dt: f64,
    pub max_steps: u32,
    pub detector_bins: [u32; 2],
}

/// Proton source configuration
#[derive(Debug, Deserialize)]
pub struct SourceConfig {
    pub source_type: String,
    pub n_protons: u32,
    pub energy_MeV: f64,

    // Point source
    pub point_position: Option<[f32; 3]>,
    pub point_target: Option<[f32; 3]>,
    pub angular_spread: f32,

    // Parallel beam
    pub beam_center: Option<[f32; 3]>,
    pub beam_direction: Option<[f32; 3]>,
    pub beam_radius: f32,

    // Detector
    pub detector_distance: f32,
    pub detector_normal: [f32; 3],
}

impl SimConfig {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("Failed to open config: {}", path.display()))?;
        let reader = BufReader::new(file);
        let config: SimConfig = serde_json::from_reader(reader)
            .with_context(|| format!("Failed to parse config: {}", path.display()))?;
        Ok(config)
    }
}

impl SourceConfig {
    /// Calculate proton speed from energy (relativistic)
    pub fn proton_speed(&self) -> f32 {
        const C: f64 = 3.0e8;
        const PROTON_MASS_MEV: f64 = 938.3;

        let gamma = 1.0 + self.energy_MeV / PROTON_MASS_MEV;
        let beta = (1.0 - 1.0 / (gamma * gamma)).sqrt();
        (beta * C) as f32
    }
}
