//! Simulation configuration: raw JSON layer → SI layer.
//!
//! ## Two-layer design
//!
//! `RawConfig` mirrors the JSON exactly, with explicit unit suffixes.  Legacy
//! bare names (metres, seconds) are accepted with deprecation warnings.
//!
//! `SimConfig` is the internal representation.  Every value is SI.
//! No other part of the engine sees `RawConfig`.
//!
//! ## Coordinate convention
//!
//! +x is the beam axis.  Source is upstream (x < 0); detector is downstream
//! (x > 0).  The detector plane is y–z.  CSV hit positions are y_mm, z_mm.
//!
//! ## Detector y/z axes
//!
//! detector_y_axis = normalize(up projected onto detector plane)
//! detector_z_axis = cross(normal, detector_y_axis)
//!
//! Defaults: normal=[1,0,0], up=[0,1,0] → y=[0,1,0], z=[0,0,1].
//!
//! ## Preferred config format (v2)
//!
//! ```json
//! {
//!   "field_path": "plasma.bfld",
//!   "detector": {
//!     "center_mm": [110, 0, 0],
//!     "normal": [1, 0, 0],
//!     "up": [0, 1, 0],
//!     "width_mm": 500,
//!     "height_mm": 500,
//!     "pixels": [512, 512]
//!   },
//!   "source": {
//!     "source_type": "parallel",
//!     "n_particles": 50000,
//!     "energy_MeV": 14.7,
//!     "beam_direction": [1, 0, 0],
//!     "source_distance_mm": 100,
//!     "beam_radius_mm": 30,
//!     "angular_spread_deg": 0.0
//!   },
//!   "dt_ps": 1.0,
//!   "max_steps": 20000
//! }
//! ```

use anyhow::{Context, Result, bail};
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use glam::Vec3;
use crate::units::{mm_to_m, ps_to_s, mev_to_j, proton_speed_from_mev};
use crate::config::DetectorResponseConfig;

// ── raw layer (mirrors JSON) ──────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RawConfig {
    pub field_path: String,
    #[serde(default)]
    pub e_field_path: Option<String>,
    pub source: RawSourceConfig,
    /// Top-level detector block (v2 format).
    pub detector: Option<RawDetectorConfig>,
    pub dt_ps: Option<f64>,
    /// Legacy: time step in seconds.
    #[serde(rename = "dt")]
    pub dt_s_legacy: Option<f64>,
    pub max_steps: u32,
    /// Legacy: pixel grid when no top-level detector block.
    pub detector_bins: Option<[u32; 2]>,
    /// Physical detector response (blur, background, noise). Default: identity.
    #[serde(default)]
    pub detector_response: DetectorResponseConfig,
}

#[derive(Debug, Deserialize)]
pub struct RawDetectorConfig {
    /// Detector center in world space [mm].  Required in v2.
    pub center_mm: Option<[f64; 3]>,
    /// Detector normal (default [1,0,0]).
    pub normal: Option<[f64; 3]>,
    /// Detector y-axis / up direction (default [0,1,0]).
    pub up: Option<[f64; 3]>,
    /// Full width along the y-axis [mm].
    pub width_mm: f64,
    /// Full height along the z-axis [mm].
    pub height_mm: f64,
    /// Pixel grid [nx, ny] (default [512, 512]).
    pub pixels: Option<[u32; 2]>,
}

#[derive(Debug, Deserialize)]
pub struct RawCommonSource {
    #[serde(alias = "n_protons")]
    pub n_particles: u64,
    #[allow(non_snake_case)]
    pub energy_MeV: f64,
    pub energy_spread_percent: Option<f64>,
    /// Exponential/TNSA spectrum: temperature parameter [MeV].  If set, overrides energy_spread_percent.
    #[serde(alias = "temperature_MeV")]
    pub temperature_mev: Option<f64>,
    /// Exponential spectrum: maximum cutoff energy [MeV].  Default: 100 × temperature_MeV.
    pub cutoff_mev: Option<f64>,
    pub seed: Option<u64>,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "source_type")]
pub enum RawSourceConfig {
    /// Finite-radius disk, all particles parallel (legacy name: "parallel").
    #[serde(rename = "parallel", alias = "parallel_beam")]
    ParallelBeam(RawParallelBeamSource),
    /// True single-ray source (Phase 1B).
    #[serde(rename = "pencil")]
    Pencil(RawPencilSource),
    /// Diverging cone from a point (Phase 2).
    #[serde(rename = "point")]
    Point(RawPointSource),
    /// Sampled positions over disk + cone directions (Phase 3).
    #[serde(rename = "disk")]
    Disk(RawDiskSource),
}

#[derive(Debug, Deserialize)]
pub struct RawParallelBeamSource {
    #[serde(flatten)]
    pub common: RawCommonSource,
    pub beam_center: Option<[f64; 3]>,
    pub source_distance_mm: Option<f64>,
    pub beam_direction: Option<[f64; 3]>,
    pub beam_radius_mm: Option<f64>,
    /// Legacy: beam radius in metres.
    pub beam_radius: Option<f64>,
    // Legacy alias: angular_spread was historically undocumented radians.
    // All existing configs use 0.0; nonzero legacy values must be migrated
    // manually (multiply by 180/π to convert to degrees).
    #[serde(alias = "angular_spread", default)]
    pub angular_spread_deg: f64,
    // Legacy detector fields (fallback when no top-level detector block).
    pub detector_distance_mm: Option<f64>,
    /// Legacy: detector distance in metres.
    pub detector_distance: Option<f64>,
    pub detector_normal: Option<[f64; 3]>,
    // Ignored legacy fields (kept for parse compatibility only).
    #[serde(default)]
    pub point_position: Option<[f64; 3]>,
    #[serde(default)]
    pub point_target: Option<[f64; 3]>,
}

// ── Phase 1B stub ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RawPencilSource {
    #[serde(flatten)]
    pub common: RawCommonSource,
    pub position_mm: Option<[f64; 3]>,
    pub direction: Option<[f64; 3]>,
    pub aim_at_mm: Option<[f64; 3]>,
}

// ── Phase 2 stub ──────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct RawPointSource {
    #[serde(flatten)]
    pub common: RawCommonSource,
    pub position_mm: Option<[f64; 3]>,
    pub direction: Option<[f64; 3]>,
    pub aim_at_mm: Option<[f64; 3]>,
    pub cone_half_angle_deg: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct RawDiskSource {
    #[serde(flatten)]
    pub common: RawCommonSource,
    pub center_mm: Option<[f64; 3]>,
    /// Accepted as an alias for `direction` (v1 limitation: disk normal and
    /// cone axis are always the same — the disk lies perpendicular to the
    /// emission direction).  If a future version allows them to differ,
    /// `normal` will become the disk-face orientation and `direction` the
    /// separate cone axis.
    pub normal: Option<[f64; 3]>,
    /// Physical source spot radius [µm].
    pub radius_um: Option<f64>,
    pub direction: Option<[f64; 3]>,
    pub aim_at_mm: Option<[f64; 3]>,
    pub cone_half_angle_deg: Option<f64>,
}

// ── SI layer ──────────────────────────────────────────────────────────────────

/// Resolved source geometry.  Variants are added as phases are implemented.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type")]
pub enum SimSourceGeometry {
    /// Finite-radius disk, all particles travel in one direction ± spread.
    #[serde(rename = "parallel")]
    ParallelBeam {
        /// Beam disk center [m].  None until resolved against field bounds in main.rs.
        center_m: Option<[f32; 3]>,
        direction: [f32; 3],
        radius_m: f64,
        angular_spread_rad: f32,
    },
    /// Single-ray source: all N particles share one position and one velocity.
    #[serde(rename = "pencil")]
    Pencil {
        position_m: [f32; 3],
        direction: [f32; 3],
    },
    /// Diverging cone from a point: all particles start at the same position,
    /// directions sampled uniformly within a cone of half-angle `half_angle_rad`.
    #[serde(rename = "point")]
    Point {
        position_m: [f32; 3],
        /// Cone axis (normalised).
        direction: [f32; 3],
        /// Cone half-angle [rad].
        half_angle_rad: f32,
    },
    /// Extended source: positions sampled over a disk, directions over a cone.
    ///
    /// **v1 constraint**: the disk normal and the cone axis are the same vector
    /// (`direction`).  The source disk always lies in the plane perpendicular to
    /// the emission direction.  The JSON field `normal` is accepted only as an
    /// alias for `direction`; it is *not* an independent disk-face orientation.
    /// If that distinction ever matters, add a separate `disk_normal` field and
    /// keep `direction` as the cone axis.
    #[serde(rename = "disk")]
    Disk {
        center_m: [f32; 3],
        /// Emission direction and disk normal (normalised, always identical in v1).
        direction: [f32; 3],
        /// Physical source spot radius [m].
        radius_m: f32,
        /// Cone half-angle [rad].
        half_angle_rad: f32,
    },
}

/// Fully resolved source configuration (SI units throughout).
#[derive(Debug, Clone, Serialize)]
pub struct SimSourceConfig {
    pub n_particles: u32,
    pub particle_energy_mev: f64,
    pub energy_j: f64,
    pub particle_speed_m_s: f64,
    /// Gaussian energy spread: σ = mean_MeV × energy_spread_percent / 100.
    /// 0.0 = monoenergetic.  Ignored when temperature_mev is Some.
    pub energy_spread_percent: f64,
    /// Exponential/TNSA spectrum temperature [MeV].  When Some, the spectrum is
    /// dN/dE ∝ exp(−E/T) with a hard cutoff at cutoff_mev.
    pub temperature_mev: Option<f64>,
    /// Hard cutoff energy [MeV] for the exponential spectrum.  Default: 100 × temperature_mev.
    pub cutoff_mev: Option<f64>,
    /// RNG seed for energy sampling. None = non-deterministic (from_entropy).
    pub seed: Option<u64>,
    pub geometry: SimSourceGeometry,
    /// Source–field-centre distance [m].  Used to resolve center_m in main.rs.
    pub source_distance_m: Option<f64>,
}

impl SimSourceConfig {
    /// Beam direction from geometry.
    pub fn beam_direction(&self) -> [f32; 3] {
        match &self.geometry {
            SimSourceGeometry::ParallelBeam { direction, .. } => *direction,
            SimSourceGeometry::Pencil       { direction, .. } => *direction,
            SimSourceGeometry::Point        { direction, .. } => *direction,
            SimSourceGeometry::Disk         { direction, .. } => *direction,
        }
    }
}

/// Fully resolved detector configuration (SI units throughout).
#[derive(Debug, Clone, Serialize)]
pub struct SimDetectorConfig {
    /// Detector center [m].  None until resolved against field bounds in main.rs.
    pub center_m: Option<[f64; 3]>,
    /// Distance from field exit face [m].  Used when center_m is None.
    pub distance_m: Option<f64>,
    /// Detector face normal (unit vector).
    pub normal: [f32; 3],
    /// Detector y-axis / up direction (unit vector).
    pub up: [f32; 3],
    /// Full width along the y-axis [m].
    pub width_m: f64,
    /// Full height along the z-axis [m].
    pub height_m: f64,
    pub pixels: [u32; 2],
    /// Write counts/hits.bin (y_mm, z_mm, energy_MeV per hit).
    pub save_hits: bool,
}

/// Top-level simulation config (SI).
#[derive(Debug, Clone, Serialize)]
pub struct SimConfig {
    pub field_path: String,
    pub e_field_path: Option<String>,
    #[serde(default = "one_f64")]
    pub scale_b: f64,
    #[serde(default)]
    pub scale_e: f64,
    pub source: SimSourceConfig,
    pub detector: SimDetectorConfig,
    pub dt_s: f64,
    pub dt_was_supplied: bool,
    pub max_steps: u32,
    pub detector_response: DetectorResponseConfig,
}

// ── conversion ────────────────────────────────────────────────────────────────

impl SimConfig {
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Self> {
        Self::load_with_overrides(path, &[])
    }

    pub fn load_with_overrides<P: AsRef<Path>>(
        path: P,
        overrides: &[crate::overrides::ConfigOverride],
    ) -> Result<Self> {
        let path = path.as_ref();
        if path.extension().and_then(|e| e.to_str()) == Some("toml") {
            let text = std::fs::read_to_string(path)
                .with_context(|| format!("Failed to read deck: {}", path.display()))?;
            let mut deck: DeckConfig = toml::from_str(&text)
                .with_context(|| format!("Failed to parse deck: {}", path.display()))?;
            if !overrides.is_empty() {
                apply_overrides_to_deck(&mut deck, overrides)?;
            }
            Self::try_from(deck)
        } else {
            if !overrides.is_empty() {
                anyhow::bail!(
                    "--set overrides are only supported for TOML input decks.\n\
                     Tip: convert with `proton_tracer init ... -o deck.toml`."
                );
            }
            let file = File::open(path)
                .with_context(|| format!("Failed to open config: {}", path.display()))?;
            let reader = BufReader::new(file);
            let raw: RawConfig = serde_json::from_reader(reader)
                .with_context(|| format!("Failed to parse config: {}", path.display()))?;
            Self::try_from(raw)
        }
    }
}

fn apply_overrides_to_deck(
    deck: &mut DeckConfig,
    overrides: &[crate::overrides::ConfigOverride],
) -> Result<()> {
    use crate::overrides::{parse_f64, parse_u32, parse_u64};
    for ov in overrides {
        let k = ov.canonical_key.as_str();
        let v = ov.raw_value.as_str();
        match k {
            "source.energy_MeV"           => deck.source.energy_mev          = parse_f64(k, v)?,
            "source.n_particles"           => deck.source.n_particles          = parse_u64(k, v)?,
            "source.beam_radius_mm"        => deck.source.beam_radius_mm       = Some(parse_f64(k, v)?),
            "source.angular_spread_deg"    => deck.source.angular_spread_deg   = parse_f64(k, v)?,
            "source.energy_spread_percent" => deck.source.energy_spread_percent = parse_f64(k, v)?,
            "source.temperature_MeV"       => deck.source.temperature_mev = Some(parse_f64(k, v)?),
            "source.cutoff_mev"            => deck.source.cutoff_mev = Some(parse_f64(k, v)?),
            "numerics.dt_ps"               => deck.numerics.dt_ps              = Some(parse_f64(k, v)?),
            "numerics.max_steps"           => deck.numerics.max_steps           = Some(parse_u32(k, v)?),
            "detector.width_mm"            => deck.detector.width_mm            = parse_f64(k, v)?,
            "detector.height_mm"           => deck.detector.height_mm           = parse_f64(k, v)?,
            "field.scale_B"                => deck.field.scale_b                = parse_f64(k, v)?,
            "field.scale_E"                => deck.field.scale_e                = parse_f64(k, v)?,
            _ => anyhow::bail!("Internal: unhandled canonical override key {:?}", k),
        }
        log::info!("  override applied: {}={}", k, v);
    }
    Ok(())
}

impl TryFrom<RawConfig> for SimConfig {
    type Error = anyhow::Error;

    fn try_from(raw: RawConfig) -> Result<Self> {
        // ── dt ────────────────────────────────────────────────────────────────
        let (dt_s, dt_was_supplied) = match (raw.dt_ps, raw.dt_s_legacy) {
            (Some(ps), _) => (ps_to_s(ps), true),
            (None, Some(s)) => {
                log::warn!("Config: 'dt' (seconds) is deprecated — use 'dt_ps' (picoseconds)");
                (s, true)
            }
            (None, None) => (0.0, false),
        };

        // ── source ────────────────────────────────────────────────────────────
        // Returns (SimSourceConfig, legacy_detector_distance_m, legacy_detector_normal)
        let (source, leg_det_dist_m, leg_det_normal) = match raw.source {
            RawSourceConfig::ParallelBeam(pb) => {
                let n_particles = u32::try_from(pb.common.n_particles)
                    .context("n_particles exceeds u32 max")?;
                let particle_energy_mev = pb.common.energy_MeV;
                let energy_j           = mev_to_j(particle_energy_mev);
                let particle_speed_m_s = proton_speed_from_mev(particle_energy_mev);

                let radius_m = match (pb.beam_radius_mm, pb.beam_radius) {
                    (Some(mm), _) => mm_to_m(mm),
                    (None, Some(m)) => {
                        log::warn!("Config: 'beam_radius' (metres) is deprecated — use 'beam_radius_mm'");
                        m
                    }
                    (None, None) => {
                        log::warn!("Config: 'beam_radius_mm' not set, defaulting to 30 mm");
                        0.03
                    }
                };

                let direction = pb.beam_direction
                    .map(|d| [d[0] as f32, d[1] as f32, d[2] as f32])
                    .unwrap_or([1.0, 0.0, 0.0]);

                let center_m = pb.beam_center
                    .map(|c| [c[0] as f32, c[1] as f32, c[2] as f32]);

                let source_distance_m = pb.source_distance_mm.map(mm_to_m);

                if pb.angular_spread_deg != 0.0 {
                    log::warn!(
                        "Config: angular_spread_deg = {:.4}°. \
                         If this was set via the legacy 'angular_spread' key (radians), \
                         migrate by multiplying the original value by 180/π.",
                        pb.angular_spread_deg
                    );
                }
                let angular_spread_rad = (pb.angular_spread_deg as f32).to_radians();

                let leg_dist = match (pb.detector_distance_mm, pb.detector_distance) {
                    (Some(mm), _) => Some(mm_to_m(mm)),
                    (None, Some(m)) => {
                        log::warn!(
                            "Config: 'detector_distance' (metres) is deprecated — \
                             use a top-level 'detector' block with 'center_mm'"
                        );
                        Some(m)
                    }
                    (None, None) => None,
                };
                let leg_normal = pb.detector_normal
                    .map(|n| [n[0] as f32, n[1] as f32, n[2] as f32]);

                let src = SimSourceConfig {
                    n_particles,
                    particle_energy_mev,
                    energy_j,
                    particle_speed_m_s,
                    energy_spread_percent: pb.common.energy_spread_percent.unwrap_or(0.0),
                    temperature_mev: pb.common.temperature_mev,
                    cutoff_mev: pb.common.cutoff_mev,
                    seed: pb.common.seed,
                    geometry: SimSourceGeometry::ParallelBeam {
                        center_m,
                        direction,
                        radius_m,
                        angular_spread_rad,
                    },
                    source_distance_m,
                };
                (src, leg_dist, leg_normal)
            }

            RawSourceConfig::Pencil(p) => {
                let n_particles = u32::try_from(p.common.n_particles)
                    .context("n_particles exceeds u32 max")?;
                let particle_energy_mev = p.common.energy_MeV;
                let energy_j           = mev_to_j(particle_energy_mev);
                let particle_speed_m_s = proton_speed_from_mev(particle_energy_mev);

                let position_m = p.position_mm
                    .map(|pos| [
                        mm_to_m(pos[0]) as f32,
                        mm_to_m(pos[1]) as f32,
                        mm_to_m(pos[2]) as f32,
                    ])
                    .ok_or_else(|| anyhow::anyhow!(
                        "source_type=\"pencil\" requires 'position_mm'"
                    ))?;

                let direction = match (p.direction, p.aim_at_mm) {
                    (Some(d), None) => {
                        let v = glam::Vec3::new(d[0] as f32, d[1] as f32, d[2] as f32).normalize();
                        v.to_array()
                    }
                    (None, Some(t)) => {
                        let target = glam::Vec3::new(
                            mm_to_m(t[0]) as f32,
                            mm_to_m(t[1]) as f32,
                            mm_to_m(t[2]) as f32,
                        );
                        let src = glam::Vec3::from(position_m);
                        (target - src).normalize().to_array()
                    }
                    (Some(_), Some(_)) => bail!(
                        "source_type=\"pencil\": specify either 'direction' or 'aim_at_mm', not both"
                    ),
                    (None, None) => bail!(
                        "source_type=\"pencil\": requires 'direction' or 'aim_at_mm'"
                    ),
                };

                let src = SimSourceConfig {
                    n_particles,
                    particle_energy_mev,
                    energy_j,
                    particle_speed_m_s,
                    energy_spread_percent: p.common.energy_spread_percent.unwrap_or(0.0),
                    temperature_mev: p.common.temperature_mev,
                    cutoff_mev: p.common.cutoff_mev,
                    seed: p.common.seed,
                    geometry: SimSourceGeometry::Pencil { position_m, direction },
                    source_distance_m: None,
                };
                (src, None, None)
            }
            RawSourceConfig::Point(p) => {
                let n_particles = u32::try_from(p.common.n_particles)
                    .context("n_particles exceeds u32 max")?;
                let particle_energy_mev = p.common.energy_MeV;
                let energy_j           = mev_to_j(particle_energy_mev);
                let particle_speed_m_s = proton_speed_from_mev(particle_energy_mev);

                let position_m = p.position_mm
                    .map(|pos| [
                        mm_to_m(pos[0]) as f32,
                        mm_to_m(pos[1]) as f32,
                        mm_to_m(pos[2]) as f32,
                    ])
                    .ok_or_else(|| anyhow::anyhow!(
                        "source_type=\"point\" requires 'position_mm'"
                    ))?;

                let direction = match (p.direction, p.aim_at_mm) {
                    (Some(d), None) => {
                        Vec3::new(d[0] as f32, d[1] as f32, d[2] as f32)
                            .normalize().to_array()
                    }
                    (None, Some(t)) => {
                        let target = Vec3::new(
                            mm_to_m(t[0]) as f32,
                            mm_to_m(t[1]) as f32,
                            mm_to_m(t[2]) as f32,
                        );
                        (target - Vec3::from(position_m)).normalize().to_array()
                    }
                    (Some(_), Some(_)) => bail!(
                        "source_type=\"point\": specify either 'direction' or 'aim_at_mm', not both"
                    ),
                    (None, None) => [1.0, 0.0, 0.0],
                };

                let half_angle_rad = p.cone_half_angle_deg
                    .unwrap_or(0.0) as f32 * std::f32::consts::PI / 180.0;

                let src = SimSourceConfig {
                    n_particles,
                    particle_energy_mev,
                    energy_j,
                    particle_speed_m_s,
                    energy_spread_percent: p.common.energy_spread_percent.unwrap_or(0.0),
                    temperature_mev: p.common.temperature_mev,
                    cutoff_mev: p.common.cutoff_mev,
                    seed: p.common.seed,
                    geometry: SimSourceGeometry::Point { position_m, direction, half_angle_rad },
                    source_distance_m: None,
                };
                (src, None, None)
            }
            RawSourceConfig::Disk(d) => {
                let n_particles = u32::try_from(d.common.n_particles)
                    .context("n_particles exceeds u32 max")?;
                let particle_energy_mev = d.common.energy_MeV;
                let energy_j           = mev_to_j(particle_energy_mev);
                let particle_speed_m_s = proton_speed_from_mev(particle_energy_mev);

                let center_m = d.center_mm
                    .map(|c| [
                        mm_to_m(c[0]) as f32,
                        mm_to_m(c[1]) as f32,
                        mm_to_m(c[2]) as f32,
                    ])
                    .ok_or_else(|| anyhow::anyhow!(
                        "source_type=\"disk\" requires 'center_mm'"
                    ))?;

                // direction / normal / aim_at_mm all describe the cone axis + disk normal.
                // Prefer `direction`, accept `normal` as alias, compute from `aim_at_mm`.
                let direction = match (d.direction.or(d.normal), d.aim_at_mm) {
                    (Some(d_vec), None) => {
                        Vec3::new(d_vec[0] as f32, d_vec[1] as f32, d_vec[2] as f32)
                            .normalize().to_array()
                    }
                    (None, Some(t)) => {
                        let target = Vec3::new(
                            mm_to_m(t[0]) as f32,
                            mm_to_m(t[1]) as f32,
                            mm_to_m(t[2]) as f32,
                        );
                        (target - Vec3::from(center_m)).normalize().to_array()
                    }
                    (Some(_), Some(_)) => bail!(
                        "source_type=\"disk\": specify at most one of \
                         'direction'/'normal' and 'aim_at_mm'"
                    ),
                    (None, None) => [1.0, 0.0, 0.0],
                };

                let radius_m = d.radius_um.unwrap_or(0.0) as f32 * 1e-6;
                let half_angle_rad = d.cone_half_angle_deg
                    .unwrap_or(0.0) as f32 * std::f32::consts::PI / 180.0;

                let src = SimSourceConfig {
                    n_particles,
                    particle_energy_mev,
                    energy_j,
                    particle_speed_m_s,
                    energy_spread_percent: d.common.energy_spread_percent.unwrap_or(0.0),
                    temperature_mev: d.common.temperature_mev,
                    cutoff_mev: d.common.cutoff_mev,
                    seed: d.common.seed,
                    geometry: SimSourceGeometry::Disk {
                        center_m, direction, radius_m, half_angle_rad,
                    },
                    source_distance_m: None,
                };
                (src, None, None)
            }
        };

        // ── detector ─────────────────────────────────────────────────────────
        let detector = if let Some(rd) = raw.detector {
            let normal = rd.normal
                .map(|n| [n[0] as f32, n[1] as f32, n[2] as f32])
                .unwrap_or([1.0, 0.0, 0.0]);
            let up = rd.up
                .map(|u| [u[0] as f32, u[1] as f32, u[2] as f32])
                .unwrap_or([0.0, 1.0, 0.0]);
            let pixels = rd.pixels.unwrap_or([512, 512]);
            SimDetectorConfig {
                center_m: rd.center_mm.map(|c| [
                    mm_to_m(c[0]), mm_to_m(c[1]), mm_to_m(c[2]),
                ]),
                distance_m: None,
                normal,
                up,
                width_m:  mm_to_m(rd.width_mm),
                height_m: mm_to_m(rd.height_mm),
                pixels,
                save_hits: true,
            }
        } else {
            // Legacy: detector described inside the source block.
            let distance_m = leg_det_dist_m.ok_or_else(|| anyhow::anyhow!(
                "Config: detector geometry missing.  Add a top-level 'detector' block \
                 with 'center_mm', 'width_mm', 'height_mm' — or set \
                 'detector_distance_mm' inside the source block (legacy)."
            ))?;
            let normal = leg_det_normal.unwrap_or([1.0, 0.0, 0.0]);
            let pixels = raw.detector_bins.unwrap_or([512, 512]);
            SimDetectorConfig {
                center_m: None,
                distance_m: Some(distance_m),
                normal,
                up: [0.0, 1.0, 0.0],
                width_m:  0.50,
                height_m: 0.50,
                pixels,
                save_hits: true,
            }
        };

        Ok(SimConfig {
            field_path: raw.field_path,
            e_field_path: raw.e_field_path,
            scale_b: 1.0,
            scale_e: 1.0,
            source,
            detector,
            dt_s,
            dt_was_supplied,
            max_steps: raw.max_steps,
            detector_response: raw.detector_response,
        })
    }
}

// ── TOML deck format (canonical new schema) ───────────────────────────────────

/// Top-level TOML input deck.
#[derive(Debug, Deserialize)]
pub struct DeckConfig {
    pub name: Option<String>,
    pub field: DeckFieldBlock,
    pub source: DeckSourceBlock,
    pub detector: DeckDetectorBlock,
    #[serde(default)]
    pub numerics: DeckNumerics,
    #[serde(default)]
    pub render: DeckRender,
    #[serde(default)]
    pub output: DeckOutput,
    #[serde(default)]
    pub detector_response: DetectorResponseConfig,
}

#[derive(Debug, Deserialize)]
pub struct DeckFieldBlock {
    pub path: String,
    pub e_path: Option<String>,
    #[serde(rename = "scale_B", default = "deck_one")]
    pub scale_b: f64,
    #[serde(rename = "scale_E", default)]
    pub scale_e: f64,
}

/// Flat source block — all source types share one TOML table, discriminated by `type`.
#[derive(Debug, Deserialize)]
pub struct DeckSourceBlock {
    #[serde(rename = "type")]
    pub source_type: String,
    pub n_particles: u64,
    #[serde(rename = "energy_MeV")]
    pub energy_mev: f64,
    #[serde(default)]
    pub energy_spread_percent: f64,
    /// Exponential/TNSA spectrum temperature [MeV].  Overrides energy_spread_percent when set.
    #[serde(rename = "temperature_MeV")]
    pub temperature_mev: Option<f64>,
    /// Hard cutoff energy [MeV] for the exponential spectrum.
    pub cutoff_mev: Option<f64>,
    pub seed: Option<u64>,
    // parallel
    pub direction: Option<[f64; 3]>,
    pub beam_radius_mm: Option<f64>,
    pub source_distance_mm: Option<f64>,
    #[serde(default)]
    pub angular_spread_deg: f64,
    // pencil / point  — explicit world position
    pub position_mm: Option<[f64; 3]>,
    pub aim_at_mm: Option<[f64; 3]>,
    // disk (and optional explicit center for parallel)
    pub center_mm: Option<[f64; 3]>,
    // disk / point
    pub radius_um: Option<f64>,
    pub cone_half_angle_deg: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct DeckDetectorBlock {
    pub center_mm: [f64; 3],
    #[serde(default = "deck_default_normal")]
    pub normal: [f64; 3],
    #[serde(default = "deck_default_up")]
    pub up: [f64; 3],
    pub width_mm: f64,
    pub height_mm: f64,
    #[serde(default = "deck_default_pixels")]
    pub pixels: [u32; 2],
    #[serde(default = "deck_default_true")]
    pub save_hits: bool,
}

#[derive(Debug, Deserialize, Default)]
pub struct DeckNumerics {
    pub integrator: Option<String>,
    pub dt_ps: Option<f64>,
    pub max_steps: Option<u32>,
}

#[derive(Debug, Deserialize, Default)]
pub struct DeckRender {
    pub scale: Option<String>,
    pub colormap: Option<String>,
    pub exposure: Option<f32>,
}

#[derive(Debug, Deserialize, Default)]
pub struct DeckOutput {
    pub write_raw_counts: Option<bool>,
    pub write_processed_counts: Option<bool>,
    pub write_png: Option<bool>,
    pub write_metadata: Option<bool>,
}

fn deck_one() -> f64 { 1.0 }
fn one_f64()  -> f64 { 1.0 }
fn deck_default_normal() -> [f64; 3] { [1.0, 0.0, 0.0] }
fn deck_default_up()     -> [f64; 3] { [0.0, 1.0, 0.0] }
fn deck_default_pixels() -> [u32; 2] { [512, 512] }
fn deck_default_true()   -> bool     { true }

impl TryFrom<DeckConfig> for SimConfig {
    type Error = anyhow::Error;

    fn try_from(deck: DeckConfig) -> Result<Self> {
        let (dt_s, dt_was_supplied) = match deck.numerics.dt_ps {
            Some(ps) => (ps_to_s(ps), true),
            None => (0.0, false),
        };
        let max_steps = deck.numerics.max_steps.unwrap_or(10000);

        let s = &deck.source;
        let n_particles = u32::try_from(s.n_particles).context("n_particles exceeds u32 max")?;
        let particle_energy_mev = s.energy_mev;
        let energy_j           = mev_to_j(particle_energy_mev);
        let particle_speed_m_s = proton_speed_from_mev(particle_energy_mev);

        let (geometry, source_distance_m) = match s.source_type.as_str() {
            "parallel" => {
                let direction = s.direction
                    .map(|d| [d[0] as f32, d[1] as f32, d[2] as f32])
                    .unwrap_or([1.0, 0.0, 0.0]);
                let radius_m = s.beam_radius_mm.map(mm_to_m).unwrap_or_else(|| {
                    log::warn!("Deck: beam_radius_mm not set, defaulting to 30 mm");
                    0.03
                });
                let angular_spread_rad = (s.angular_spread_deg as f32).to_radians();
                let source_distance_m  = s.source_distance_mm.map(mm_to_m);
                let center_m = s.center_mm
                    .map(|c| [mm_to_m(c[0]) as f32, mm_to_m(c[1]) as f32, mm_to_m(c[2]) as f32]);
                (SimSourceGeometry::ParallelBeam { center_m, direction, radius_m, angular_spread_rad },
                 source_distance_m)
            }
            "pencil" => {
                let position_m = deck_require_position(s, "pencil")?;
                let direction = deck_resolve_direction(s.direction, s.aim_at_mm,
                    Some(position_m), "pencil", true)?;
                (SimSourceGeometry::Pencil { position_m, direction }, None)
            }
            "point" => {
                let position_m = deck_require_position(s, "point")?;
                let direction = deck_resolve_direction(s.direction, s.aim_at_mm,
                    Some(position_m), "point", false)?;
                let half_angle_rad = s.cone_half_angle_deg.unwrap_or(0.0) as f32
                    * std::f32::consts::PI / 180.0;
                (SimSourceGeometry::Point { position_m, direction, half_angle_rad }, None)
            }
            "disk" => {
                let center_m = s.center_mm
                    .map(|c| [mm_to_m(c[0]) as f32, mm_to_m(c[1]) as f32, mm_to_m(c[2]) as f32])
                    .ok_or_else(|| anyhow::anyhow!("source type \"disk\" requires center_mm"))?;
                let direction = deck_resolve_direction(s.direction, s.aim_at_mm,
                    Some(center_m), "disk", false)?;
                let radius_m       = s.radius_um.unwrap_or(0.0) as f32 * 1e-6;
                let half_angle_rad = s.cone_half_angle_deg.unwrap_or(0.0) as f32
                    * std::f32::consts::PI / 180.0;
                (SimSourceGeometry::Disk { center_m, direction, radius_m, half_angle_rad }, None)
            }
            other => bail!("Unknown source type: {:?}. Use: parallel, pencil, point, disk", other),
        };

        let source = SimSourceConfig {
            n_particles,
            particle_energy_mev,
            energy_j,
            particle_speed_m_s,
            energy_spread_percent: s.energy_spread_percent,
            temperature_mev: s.temperature_mev,
            cutoff_mev: s.cutoff_mev,
            seed: s.seed,
            geometry,
            source_distance_m,
        };

        let d = &deck.detector;
        let detector = SimDetectorConfig {
            center_m: Some([
                mm_to_m(d.center_mm[0]),
                mm_to_m(d.center_mm[1]),
                mm_to_m(d.center_mm[2]),
            ]),
            distance_m: None,
            normal: [d.normal[0] as f32, d.normal[1] as f32, d.normal[2] as f32],
            up:     [d.up[0]     as f32, d.up[1]     as f32, d.up[2]     as f32],
            width_m:  mm_to_m(d.width_mm),
            height_m: mm_to_m(d.height_mm),
            pixels: d.pixels,
            save_hits: d.save_hits,
        };

        Ok(SimConfig {
            field_path: deck.field.path,
            e_field_path: deck.field.e_path,
            scale_b: deck.field.scale_b,
            scale_e: deck.field.scale_e,
            source,
            detector,
            dt_s,
            dt_was_supplied,
            max_steps,
            detector_response: deck.detector_response,
        })
    }
}

fn deck_require_position(s: &DeckSourceBlock, src_type: &str) -> Result<[f32; 3]> {
    s.position_mm
        .map(|p| [mm_to_m(p[0]) as f32, mm_to_m(p[1]) as f32, mm_to_m(p[2]) as f32])
        .ok_or_else(|| anyhow::anyhow!("source type \"{}\" requires position_mm", src_type))
}

/// Resolve direction from explicit vector or aim_at_mm target.
/// `require`: if true, errors on (None, None); if false, returns [1,0,0].
fn deck_resolve_direction(
    direction: Option<[f64; 3]>,
    aim_at_mm: Option<[f64; 3]>,
    from_m: Option<[f32; 3]>,
    src_type: &str,
    require: bool,
) -> Result<[f32; 3]> {
    match (direction, aim_at_mm) {
        (Some(d), None) => {
            Ok(Vec3::new(d[0] as f32, d[1] as f32, d[2] as f32).normalize().to_array())
        }
        (None, Some(t)) => {
            let target = Vec3::new(
                mm_to_m(t[0]) as f32,
                mm_to_m(t[1]) as f32,
                mm_to_m(t[2]) as f32,
            );
            let from = Vec3::from(from_m.unwrap_or([0.0, 0.0, 0.0]));
            Ok((target - from).normalize().to_array())
        }
        (Some(_), Some(_)) => bail!(
            "source type \"{}\": specify either direction or aim_at_mm, not both", src_type
        ),
        (None, None) => {
            if require {
                bail!("source type \"{}\": requires direction or aim_at_mm", src_type)
            } else {
                Ok([1.0, 0.0, 0.0])
            }
        }
    }
}
