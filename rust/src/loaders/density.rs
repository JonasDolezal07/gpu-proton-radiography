//! Loader for `.dens` density grid files.
//!
//! Binary format:
//!   Offset  Size  Field
//!   0       4     magic "DENS"
//!   4       4     version u32 (currently 1)
//!   8       4     nx u32
//!   12      4     ny u32
//!   16      4     nz u32
//!   20      4     x_min f32 [m]
//!   24      4     x_max f32 [m]
//!   28      4     y_min f32 [m]
//!   32      4     y_max f32 [m]
//!   36      4     z_min f32 [m]
//!   40      4     z_max f32 [m]
//!   44      20    padding (zeros)
//!   64      n×4   density data f32 [g/cm³], C-contiguous (x outermost, z innermost)

use std::path::Path;
use anyhow::{bail, Context, Result};

#[derive(Debug, Clone)]
pub struct DensityBounds {
    pub x_min: f32, pub x_max: f32,
    pub y_min: f32, pub y_max: f32,
    pub z_min: f32, pub z_max: f32,
}

pub struct DensityData {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    /// Density values [g/cm³], C-contiguous (x outermost, z innermost).
    pub data: Vec<f32>,
    pub bounds: DensityBounds,
}

impl DensityData {
    /// 1×1×1 vacuum grid — no energy loss anywhere.
    pub fn vacuum() -> Self {
        Self {
            nx: 1, ny: 1, nz: 1,
            data: vec![0.0f32],
            bounds: DensityBounds {
                x_min: -1.0, x_max: 1.0,
                y_min: -1.0, y_max: 1.0,
                z_min: -1.0, z_max: 1.0,
            },
        }
    }

    pub fn load(path: &Path) -> Result<Self> {
        let raw = std::fs::read(path)
            .with_context(|| format!("Cannot read density file {:?}", path))?;

        if raw.len() < 64 {
            bail!("Density file too short ({} bytes)", raw.len());
        }
        if &raw[0..4] != b"DENS" {
            bail!("Invalid density file magic (expected 'DENS')");
        }

        let version = u32::from_le_bytes(raw[4..8].try_into().unwrap());
        if version != 1 {
            bail!("Unsupported density file version {}", version);
        }

        let nx = u32::from_le_bytes(raw[8..12].try_into().unwrap());
        let ny = u32::from_le_bytes(raw[12..16].try_into().unwrap());
        let nz = u32::from_le_bytes(raw[16..20].try_into().unwrap());

        let read_f32 = |off: usize| f32::from_le_bytes(raw[off..off+4].try_into().unwrap());
        let bounds = DensityBounds {
            x_min: read_f32(20), x_max: read_f32(24),
            y_min: read_f32(28), y_max: read_f32(32),
            z_min: read_f32(36), z_max: read_f32(40),
        };

        let n = (nx * ny * nz) as usize;
        let needed = 64 + n * 4;
        if raw.len() < needed {
            bail!("Density file data too short: need {} bytes, got {}", needed, raw.len());
        }

        let data: Vec<f32> = (0..n)
            .map(|i| f32::from_le_bytes(raw[64 + i*4 .. 64 + i*4+4].try_into().unwrap()))
            .collect();

        Ok(Self { nx, ny, nz, data, bounds })
    }

    /// Write a density grid to a `.dens` file.
    pub fn write(&self, path: &Path) -> Result<()> {
        let n = (self.nx * self.ny * self.nz) as usize;
        let mut buf = vec![0u8; 64 + n * 4];

        buf[0..4].copy_from_slice(b"DENS");
        buf[4..8].copy_from_slice(&1u32.to_le_bytes());
        buf[8..12].copy_from_slice(&self.nx.to_le_bytes());
        buf[12..16].copy_from_slice(&self.ny.to_le_bytes());
        buf[16..20].copy_from_slice(&self.nz.to_le_bytes());

        let mut write_f32 = |off: usize, v: f32| {
            buf[off..off+4].copy_from_slice(&v.to_le_bytes());
        };
        write_f32(20, self.bounds.x_min); write_f32(24, self.bounds.x_max);
        write_f32(28, self.bounds.y_min); write_f32(32, self.bounds.y_max);
        write_f32(36, self.bounds.z_min); write_f32(40, self.bounds.z_max);

        for (i, &v) in self.data.iter().enumerate() {
            buf[64 + i*4 .. 64 + i*4+4].copy_from_slice(&v.to_le_bytes());
        }

        std::fs::write(path, &buf)
            .with_context(|| format!("Cannot write density file {:?}", path))
    }
}
