//! Field data loader
//!
//! Binary format (from Python):
//! - Header (64 bytes):
//!   - magic: "BFLD" (4 bytes)
//!   - version: u32
//!   - nx, ny, nz: 3x u32
//!   - bounds: 6x f32 (x_min, x_max, y_min, y_max, z_min, z_max)
//!   - padding to 64 bytes
//! - Data: nx * ny * nz * 3 * f32 (B-field vectors)

use anyhow::{Context, Result, bail};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

#[derive(Debug)]
pub struct FieldData {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub bounds: FieldBounds,
    pub data: Vec<f32>,  // Flattened: nx * ny * nz * 3
}

#[derive(Debug, Clone, Copy)]
pub struct FieldBounds {
    pub x_min: f32,
    pub x_max: f32,
    pub y_min: f32,
    pub y_max: f32,
    pub z_min: f32,
    pub z_max: f32,
}

impl FieldData {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path = path.as_ref();
        let file = File::open(path)
            .with_context(|| format!("Failed to open field file: {}", path.display()))?;
        let mut reader = BufReader::new(file);

        // Read header
        let mut header = [0u8; 64];
        reader.read_exact(&mut header)?;

        // Check magic
        if &header[0..4] != b"BFLD" {
            bail!("Invalid field file magic (expected BFLD)");
        }

        // Parse header
        let version = u32::from_le_bytes(header[4..8].try_into()?);
        if version != 1 {
            bail!("Unsupported field format version: {}", version);
        }

        let nx = u32::from_le_bytes(header[8..12].try_into()?);
        let ny = u32::from_le_bytes(header[12..16].try_into()?);
        let nz = u32::from_le_bytes(header[16..20].try_into()?);

        let bounds = FieldBounds {
            x_min: f32::from_le_bytes(header[20..24].try_into()?),
            x_max: f32::from_le_bytes(header[24..28].try_into()?),
            y_min: f32::from_le_bytes(header[28..32].try_into()?),
            y_max: f32::from_le_bytes(header[32..36].try_into()?),
            z_min: f32::from_le_bytes(header[36..40].try_into()?),
            z_max: f32::from_le_bytes(header[40..44].try_into()?),
        };

        // Read data
        let num_values = (nx * ny * nz * 3) as usize;
        let mut data = vec![0f32; num_values];
        let data_bytes = bytemuck::cast_slice_mut(&mut data);
        reader.read_exact(data_bytes)?;

        log::debug!(
            "Loaded field: {}x{}x{}, bounds: ({:.3}, {:.3}) x ({:.3}, {:.3}) x ({:.3}, {:.3})",
            nx, ny, nz,
            bounds.x_min, bounds.x_max,
            bounds.y_min, bounds.y_max,
            bounds.z_min, bounds.z_max
        );

        Ok(Self { nx, ny, nz, bounds, data })
    }

    /// Get grid spacing
    pub fn spacing(&self) -> (f32, f32, f32) {
        (
            (self.bounds.x_max - self.bounds.x_min) / (self.nx - 1) as f32,
            (self.bounds.y_max - self.bounds.y_min) / (self.ny - 1) as f32,
            (self.bounds.z_max - self.bounds.z_min) / (self.nz - 1) as f32,
        )
    }

    /// Size in bytes
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }
}
