//! Field data loader
//!
//! ## Binary format (.bfld)
//!
//! All integers and floats are little-endian.
//!
//! ### Header (64 bytes, fixed)
//! ```text
//! bytes  0..4   magic      = b"BFLD"
//! bytes  4..8   version    = 1 (B only) or 2 (B + E)
//! bytes  8..12  nx         u32
//! bytes 12..16  ny         u32
//! bytes 16..20  nz         u32
//! bytes 20..24  x_min      f32  [metres]
//! bytes 24..28  x_max      f32  [metres]
//! bytes 28..32  y_min      f32  [metres]
//! bytes 32..36  y_max      f32  [metres]
//! bytes 36..40  z_min      f32  [metres]
//! bytes 40..44  z_max      f32  [metres]
//! bytes 44..64  reserved   (zeros)
//! ```
//!
//! ### Data layout (both versions)
//! Each field block is `nx * ny * nz` voxels, each voxel is 3 × f32 (x, y, z components
//! interleaved: Fx, Fy, Fz).  Voxel order is C-contiguous with x outermost:
//! `[ix=0,iy=0,iz=0], [ix=0,iy=0,iz=1], ..., [ix=0,iy=1,iz=0], ..., [ix=nx-1,iy=ny-1,iz=nz-1]`
//!
//! Version 1: `B block` (nx·ny·nz·3 f32, units T)
//! Version 2: `B block` then `E block` (nx·ny·nz·3 f32, units V/m)

use anyhow::{Context, Result, bail};
use std::fs::File;
use std::io::{BufReader, Read};
use std::path::Path;

/// How the E-field was populated — tracks presence, not data values.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EFieldSource {
    /// Not supplied; e_data is all zeros.
    ZeroFilled,
    /// Read from the combined version-2 file (E block follows B block).
    EmbeddedV2,
    /// Loaded from a separate version-1 vector file via `e_field_path`.
    SeparateFile,
}

impl std::fmt::Display for EFieldSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EFieldSource::ZeroFilled    => write!(f, "not supplied; zero-filled"),
            EFieldSource::EmbeddedV2    => write!(f, "embedded in combined v2 file"),
            EFieldSource::SeparateFile  => write!(f, "loaded from separate file"),
        }
    }
}

#[derive(Debug)]
pub struct FieldData {
    pub nx: u32,
    pub ny: u32,
    pub nz: u32,
    pub bounds: FieldBounds,
    pub data: Vec<f32>,       // B-field, flattened: nx * ny * nz * 3  [T]
    pub e_data: Vec<f32>,     // E-field, flattened: nx * ny * nz * 3  [V/m]
    pub e_source: EFieldSource,
    pub version: u32,
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

        let mut header = [0u8; 64];
        reader.read_exact(&mut header)?;

        if &header[0..4] != b"BFLD" {
            bail!("Invalid field file magic (expected BFLD, got {:?})", &header[0..4]);
        }

        let version = u32::from_le_bytes(header[4..8].try_into()?);
        if version != 1 && version != 2 {
            bail!("Unsupported field format version: {} (expected 1 or 2)", version);
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

        let num_values = (nx * ny * nz * 3) as usize;
        if num_values == 0 {
            bail!("Field has zero voxels (nx={}, ny={}, nz={})", nx, ny, nz);
        }

        let mut data = vec![0f32; num_values];
        reader.read_exact(bytemuck::cast_slice_mut(&mut data))?;
        assert_eq!(data.len(), num_values, "B-field buffer length mismatch");

        let (e_data, e_source) = if version == 2 {
            let mut e = vec![0f32; num_values];
            reader.read_exact(bytemuck::cast_slice_mut(&mut e))?;
            assert_eq!(e.len(), num_values, "E-field buffer length mismatch (v2)");
            (e, EFieldSource::EmbeddedV2)
        } else {
            (vec![0.0f32; num_values], EFieldSource::ZeroFilled)
        };

        log::debug!(
            "Loaded field v{}: {}x{}x{}, bounds x=[{:.4},{:.4}] y=[{:.4},{:.4}] z=[{:.4},{:.4}]",
            version, nx, ny, nz,
            bounds.x_min, bounds.x_max,
            bounds.y_min, bounds.y_max,
            bounds.z_min, bounds.z_max,
        );

        Ok(Self { nx, ny, nz, bounds, data, e_data, e_source, version })
    }

    /// Merge a separately-loaded vector field as the E field.
    /// The source must be a version-1 file (its primary vector block is interpreted as E [V/m]).
    /// Dimensions and bounds must match this field exactly.
    pub fn set_e_from_separate_file(&mut self, e_field: FieldData) -> Result<()> {
        // Enforce version 1 — a v2 file's "data" block is its B component, not E.
        if e_field.version != 1 {
            bail!(
                "Separate E-field file must be version 1 (got version {}). \
                 A version-2 file stores B then E; pass a version-1 file \
                 whose single vector block is the E-field [V/m].",
                e_field.version
            );
        }

        // Grid dimensions
        if e_field.nx != self.nx || e_field.ny != self.ny || e_field.nz != self.nz {
            bail!(
                "E-field grid {}x{}x{} != B-field grid {}x{}x{}",
                e_field.nx, e_field.ny, e_field.nz,
                self.nx, self.ny, self.nz
            );
        }

        // Spatial bounds: allow only tiny floating-point serialisation differences (1e-5 relative)
        let b = &self.bounds;
        let e = &e_field.bounds;
        let tol = 1e-5_f32;
        let span_x = (b.x_max - b.x_min).abs().max(1e-12);
        let span_y = (b.y_max - b.y_min).abs().max(1e-12);
        let span_z = (b.z_max - b.z_min).abs().max(1e-12);
        let ok = ((e.x_min - b.x_min) / span_x).abs() <= tol
            && ((e.x_max - b.x_max) / span_x).abs() <= tol
            && ((e.y_min - b.y_min) / span_y).abs() <= tol
            && ((e.y_max - b.y_max) / span_y).abs() <= tol
            && ((e.z_min - b.z_min) / span_z).abs() <= tol
            && ((e.z_max - b.z_max) / span_z).abs() <= tol;
        if !ok {
            bail!(
                "E-field bounds ({:.6},{:.6},{:.6})-({:.6},{:.6},{:.6}) don't match \
                 B-field bounds ({:.6},{:.6},{:.6})-({:.6},{:.6},{:.6}) within {:.0e} relative tolerance",
                e.x_min, e.y_min, e.z_min, e.x_max, e.y_max, e.z_max,
                b.x_min, b.y_min, b.z_min, b.x_max, b.y_max, b.z_max,
                tol
            );
        }

        // Length invariant
        let expected = (self.nx * self.ny * self.nz * 3) as usize;
        if e_field.data.len() != expected {
            bail!(
                "E-field data length {} != expected {} (nx*ny*nz*3)",
                e_field.data.len(), expected
            );
        }

        self.e_data = e_field.data;
        self.e_source = EFieldSource::SeparateFile;
        Ok(())
    }

    /// True if E was actually supplied (regardless of whether all values happen to be zero).
    pub fn has_e_field(&self) -> bool {
        self.e_source != EFieldSource::ZeroFilled
    }

    /// Get grid spacing (dx, dy, dz)
    pub fn spacing(&self) -> (f32, f32, f32) {
        (
            (self.bounds.x_max - self.bounds.x_min) / (self.nx - 1) as f32,
            (self.bounds.y_max - self.bounds.y_min) / (self.ny - 1) as f32,
            (self.bounds.z_max - self.bounds.z_min) / (self.nz - 1) as f32,
        )
    }

    /// Size in bytes (B-field only)
    pub fn size_bytes(&self) -> usize {
        self.data.len() * std::mem::size_of::<f32>()
    }

    /// CPU-side trilinear sample (used only for diagnostics, not the GPU path).
    pub fn sample(&self, x: f32, y: f32, z: f32) -> [f32; 3] {
        let b = &self.bounds;
        let u = ((x - b.x_min) / (b.x_max - b.x_min)).clamp(0.0, 0.999);
        let v = ((y - b.y_min) / (b.y_max - b.y_min)).clamp(0.0, 0.999);
        let w = ((z - b.z_min) / (b.z_max - b.z_min)).clamp(0.0, 0.999);

        let ix = (u * (self.nx - 1) as f32) as usize;
        let iy = (v * (self.ny - 1) as f32) as usize;
        let iz = (w * (self.nz - 1) as f32) as usize;

        // Python write order: x outermost (ix outer), z innermost (iz inner)
        let idx = (ix * self.ny as usize * self.nz as usize
            + iy * self.nz as usize
            + iz) * 3;
        if idx + 2 < self.data.len() {
            [self.data[idx], self.data[idx + 1], self.data[idx + 2]]
        } else {
            [0.0, 0.0, 0.0]
        }
    }

    /// Public accessor: (min, max) magnitude of the B-field.
    pub fn b_magnitude_range(&self) -> (f32, f32) { Self::field_mag_range(&self.data) }

    /// Public accessor: (min, max) magnitude of the E-field.
    pub fn e_magnitude_range(&self) -> (f32, f32) { Self::field_mag_range(&self.e_data) }

    fn field_mag_range(data: &[f32]) -> (f32, f32) {
        let n = data.len() / 3;
        if n == 0 { return (0.0, 0.0); }
        let mut min_mag = f32::MAX;
        let mut max_mag = 0.0f32;
        for i in 0..n {
            let fx = data[i * 3];
            let fy = data[i * 3 + 1];
            let fz = data[i * 3 + 2];
            let mag = (fx*fx + fy*fy + fz*fz).sqrt();
            if mag < min_mag { min_mag = mag; }
            if mag > max_mag { max_mag = mag; }
        }
        (min_mag, max_mag)
    }

    /// Log electromagnetic field status.
    /// Call after both B and E are fully resolved (post-merge) so the log is accurate.
    pub fn log_diagnostics(&self) {
        let (b_min, b_max) = Self::field_mag_range(&self.data);
        let (e_min, e_max) = Self::field_mag_range(&self.e_data);

        log::info!("Electromagnetic field:");
        log::info!("  Grid       : {}x{}x{}", self.nx, self.ny, self.nz);
        log::info!("  Bounds (m) : x=[{:.4e}, {:.4e}]  y=[{:.4e}, {:.4e}]  z=[{:.4e}, {:.4e}]",
            self.bounds.x_min, self.bounds.x_max,
            self.bounds.y_min, self.bounds.y_max,
            self.bounds.z_min, self.bounds.z_max);
        log::info!("  B [T]      : |B| ∈ [{:.4e}, {:.4e}]", b_min, b_max);
        log::info!("  E source   : {}", self.e_source);
        if self.has_e_field() {
            log::info!("  E [V/m]    : |E| ∈ [{:.4e}, {:.4e}]", e_min, e_max);
            log::info!("  Mode       : electromagnetic (E + B)");
        } else {
            log::info!("  Mode       : magnetic-only  (E = 0)");
        }
    }

    /// Log simulation unit sanity numbers before launch.
    pub fn log_unit_sanity(&self, dt: f32, q_over_m: f32, particle_speed: f32) {
        const PROTON_MASS: f32 = 1.67262192369e-27;  // kg
        const MEV: f32 = 1.602176634e-13;            // J per MeV
        const C: f32 = 299_792_458.0;                // m/s

        let (_, b_max) = Self::field_mag_range(&self.data);
        let (_, e_max) = Self::field_mag_range(&self.e_data);

        let ke_mev = 0.5 * PROTON_MASS * particle_speed * particle_speed / MEV;

        log::info!("Simulation unit sanity:");
        log::info!("  dt                  = {:.3e} s", dt);
        log::info!("  q/m                 = {:.3e} C/kg  (proton: 9.58e7)", q_over_m);
        log::info!("  particle speed      = {:.3e} m/s   ({:.2}% of c)",
            particle_speed, 100.0 * particle_speed / C);
        log::info!("  kinetic energy      ≈ {:.3} MeV", ke_mev);
        log::info!("  max |B|             = {:.3e} T   → Larmor acc = {:.3e} m/s²",
            b_max, q_over_m * b_max * particle_speed);
        if self.has_e_field() {
            log::info!("  max |E|             = {:.3e} V/m → E acc      = {:.3e} m/s²",
                e_max, q_over_m * e_max);
            log::info!("  Δv/step from E      ≈ {:.3e} m/s  (should be << {:.3e} m/s)",
                q_over_m * e_max * dt, particle_speed);
        }
    }
}
