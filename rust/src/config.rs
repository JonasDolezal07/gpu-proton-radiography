//! Unified simulation configuration
//!
//! This module defines ALL configurable parameters for the proton radiography simulation.
//! Every aspect of the physics, rendering, and output is controlled through this config.
//! No magic numbers should exist outside this configuration.

use serde::{Deserialize, Serialize};

/// Master configuration for the entire simulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SimulationConfig {
    /// Proton source configuration
    pub source: SourceConfig,

    /// Magnetic field configuration
    pub field: FieldConfig,

    /// Detector/screen configuration
    pub detector: DetectorConfig,

    /// Fiducial grid (D-grid) for calibration
    pub grid: GridConfig,

    /// Simulation physics parameters
    pub physics: PhysicsConfig,

    /// Output and export settings
    pub output: OutputConfig,

    /// Display and rendering settings
    pub display: DisplayConfig,
}

// ============================================================================
// SOURCE CONFIGURATION
// ============================================================================

/// Proton source configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceConfig {
    /// Source type
    pub source_type: SourceType,

    /// Number of protons to simulate
    pub n_protons: u32,

    /// Source position relative to field center [m]
    /// For point source: exact position
    /// For parallel beam: beam center
    pub position: [f32; 3],

    /// Beam direction (normalized) or target point
    pub direction: DirectionMode,

    /// Energy configuration
    pub energy: EnergyConfig,

    /// Spatial extent of the source
    pub spatial: SpatialConfig,

    /// Angular spread of emitted protons
    pub angular: AngularConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SourceType {
    /// Point source (diverging cone)
    Point,
    /// Parallel beam (collimated)
    Parallel,
    /// Isotropic (4π emission) - for specific experiments
    Isotropic,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DirectionMode {
    /// Explicit direction vector (will be normalized)
    Vector([f32; 3]),
    /// Aim at a target point
    Target([f32; 3]),
    /// Auto: aim at field center (0,0,0)
    AutoCenter,
}

/// Energy distribution configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyConfig {
    /// Mean/central energy [MeV]
    pub mean_mev: f32,

    /// Energy spectrum type
    pub spectrum: EnergySpectrum,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnergySpectrum {
    /// Single energy (delta function)
    Monoenergetic,

    /// Gaussian distribution
    Gaussian {
        /// Standard deviation [MeV]
        sigma_mev: f32,
    },

    /// Maxwell-Boltzmann distribution
    MaxwellBoltzmann {
        /// Temperature [MeV] (kT equivalent)
        temperature_mev: f32,
    },

    /// TNSA-like exponential decay: N(E) ~ exp(-E/kT)
    /// Common for laser-driven sources
    TnsaExponential {
        /// Characteristic temperature [MeV]
        temperature_mev: f32,
        /// Maximum energy cutoff [MeV]
        max_energy_mev: f32,
    },

    /// Uniform distribution between min and max
    Uniform {
        min_mev: f32,
        max_mev: f32,
    },
}

/// Spatial extent of the source
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialConfig {
    /// Source size mode
    pub mode: SpatialMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SpatialMode {
    /// Mathematical point source (zero extent)
    Point,

    /// Gaussian disk (2D, perpendicular to beam)
    GaussianDisk {
        /// 1-sigma radius [m] (typical: 20-50 μm for TNSA)
        sigma_m: f32,
    },

    /// Uniform disk
    UniformDisk {
        /// Radius [m]
        radius_m: f32,
    },

    /// Gaussian sphere (3D)
    GaussianSphere {
        /// 1-sigma radius [m]
        sigma_m: f32,
    },
}

/// Angular spread configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AngularConfig {
    /// Angular spread mode
    pub mode: AngularMode,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AngularMode {
    /// Perfect collimation (parallel beam only)
    Collimated,

    /// Uniform cone
    UniformCone {
        /// Half-angle [radians]
        half_angle_rad: f32,
    },

    /// Gaussian angular distribution
    GaussianCone {
        /// 1-sigma angle [radians]
        sigma_rad: f32,
    },
}

// ============================================================================
// FIELD CONFIGURATION
// ============================================================================

/// Magnetic field configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldConfig {
    /// Field source type
    pub source: FieldSource,

    /// Global scale factor (multiply all B values)
    pub scale_factor: f32,

    /// Field center offset [m] (shift the entire field)
    pub center_offset: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FieldSource {
    /// Load from file
    File(FileFieldConfig),

    /// Generate analytically
    Analytical(AnalyticalFieldConfig),
}

/// File-based field configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileFieldConfig {
    /// Path to field file (relative to config or absolute)
    pub path: String,

    /// File format
    pub format: FieldFileFormat,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum FieldFileFormat {
    /// Native .bfld format (64-byte header + float32 data)
    Bfld,
    /// HDF5 format (common for GORGON output)
    Hdf5,
    /// Raw binary (requires dimension specification)
    RawBinary,
    /// ASCII/CSV grid
    Csv,
    /// VTK structured grid
    Vtk,
}

/// Analytical field configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticalFieldConfig {
    /// Field type
    pub field_type: AnalyticalFieldType,

    /// Grid resolution [nx, ny, nz]
    pub resolution: [u32; 3],

    /// Domain bounds: [x_min, x_max, y_min, y_max, z_min, z_max] [m]
    pub bounds: [f32; 6],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AnalyticalFieldType {
    /// No field (straight-line propagation for testing)
    None,

    /// Single current-carrying wire along Z axis
    SingleWire {
        /// Total current [A]
        current_a: f32,
        /// Wire radius [m] (for internal field calculation)
        wire_radius_m: f32,
    },

    /// Z-pinch (cylindrical current distribution)
    ZPinch {
        /// Total current [A]
        current_a: f32,
        /// Pinch radius [m]
        pinch_radius_m: f32,
        /// Axial length [m]
        length_m: f32,
    },

    /// Z-pinch with sausage (m=0) instability
    Sausage {
        /// Total current [A]
        current_a: f32,
        /// Mean pinch radius [m]
        pinch_radius_m: f32,
        /// Perturbation amplitude (fraction of radius)
        amplitude: f32,
        /// Perturbation wavelength [m]
        wavelength_m: f32,
        /// Number of wavelengths
        n_periods: f32,
    },

    /// Z-pinch with kink (m=1) instability
    Kink {
        /// Total current [A]
        current_a: f32,
        /// Pinch radius [m]
        pinch_radius_m: f32,
        /// Displacement amplitude [m]
        amplitude_m: f32,
        /// Wavelength [m]
        wavelength_m: f32,
    },

    /// Harris current sheet (magnetic reconnection)
    HarrisSheet {
        /// Asymptotic field strength [T]
        b0_tesla: f32,
        /// Sheet half-thickness [m]
        thickness_m: f32,
        /// Guide field strength [T] (Bz component)
        guide_field_tesla: f32,
        /// Sheet orientation axis
        orientation: SheetOrientation,
    },

    /// Double Harris sheet (X-point reconnection)
    DoubleHarrisSheet {
        /// Asymptotic field strength [T]
        b0_tesla: f32,
        /// Sheet half-thickness [m]
        thickness_m: f32,
        /// Separation between sheets [m]
        separation_m: f32,
    },

    /// Wire array (multiple parallel wires)
    WireArray {
        /// Number of wires
        n_wires: u32,
        /// Array radius [m]
        array_radius_m: f32,
        /// Total current [A] (divided among wires)
        total_current_a: f32,
        /// Individual wire radius [m]
        wire_radius_m: f32,
    },

    /// Custom: user-provided formula (future extension)
    Custom {
        /// Formula string (parsed at runtime)
        formula: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SheetOrientation {
    /// Field reversal in X direction (By changes sign across x=0)
    AlongX,
    /// Field reversal in Y direction
    AlongY,
    /// Field reversal in Z direction
    AlongZ,
}

// ============================================================================
// DETECTOR CONFIGURATION
// ============================================================================

/// Detector/screen configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorConfig {
    /// Detector positioning mode
    pub position_mode: DetectorPositionMode,

    /// Detector size [width, height] in meters
    pub extent_m: [f32; 2],

    /// Resolution [bins_x, bins_y]
    pub resolution: [u32; 2],

    /// Detector normal direction (which way it faces)
    pub normal: [f32; 3],
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DetectorPositionMode {
    /// Automatic: place at field exit + distance along beam direction
    Auto {
        /// Distance from field boundary [m]
        distance_from_field_m: f32,
    },

    /// Manual: explicit world position
    Manual {
        /// Detector center position [m]
        position: [f32; 3],
    },
}

// ============================================================================
// FIDUCIAL GRID CONFIGURATION
// ============================================================================

/// Fiducial grid (D-grid) configuration for calibration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GridConfig {
    /// Enable grid
    pub enabled: bool,

    /// Grid position along beam axis [m] (relative to field center)
    pub position_m: f32,

    /// Grid pattern
    pub pattern: GridPattern,

    /// Wire/line thickness [m]
    pub wire_thickness_m: f32,

    /// What happens when a proton hits the grid
    pub interaction: GridInteraction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GridPattern {
    /// Square grid
    Square {
        /// Spacing between wires [m]
        spacing_m: f32,
    },

    /// Hexagonal grid
    Hexagonal {
        /// Cell size [m]
        cell_size_m: f32,
    },

    /// Parallel lines only
    Lines {
        /// Spacing [m]
        spacing_m: f32,
        /// Orientation angle [radians]
        angle_rad: f32,
    },

    /// Crosshair (single cross at center)
    Crosshair {
        /// Arm length [m]
        arm_length_m: f32,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum GridInteraction {
    /// Absorb (particle stops)
    Absorb,
    /// Scatter (randomize direction, reduce energy)
    Scatter {
        /// Energy loss fraction
        energy_loss: f32,
    },
}

// ============================================================================
// PHYSICS CONFIGURATION
// ============================================================================

/// Physics simulation parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhysicsConfig {
    /// Time step [seconds]
    pub dt_s: f64,

    /// Maximum number of integration steps per particle
    pub max_steps: u32,

    /// Domain boundary margin factor
    /// Domain extends beyond field bounds by this factor
    pub domain_margin: f32,

    /// Integration method
    pub integrator: IntegratorType,

    /// Include electric field effects (future)
    pub include_electric_field: bool,

    /// Relativistic corrections
    pub relativistic: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum IntegratorType {
    /// Boris pusher (standard for particle-in-cell)
    Boris,
    /// Vay pusher (better for relativistic particles)
    Vay,
    /// Simple leapfrog (for testing)
    Leapfrog,
}

// ============================================================================
// OUTPUT CONFIGURATION
// ============================================================================

/// Output and export configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OutputConfig {
    /// Output directory
    pub directory: String,

    /// Filename prefix for exports
    pub prefix: String,

    /// Auto-export settings
    pub auto_export: AutoExportConfig,

    /// What to export
    pub export_png: bool,
    pub export_csv: bool,
    pub export_config: bool,  // Save config snapshot with output

    /// PNG export settings
    pub png: PngExportConfig,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AutoExportConfig {
    /// Enable auto-export when simulation stabilizes
    pub enabled: bool,

    /// Stability threshold: export when hit count changes by less than this %
    pub stability_threshold_percent: f32,

    /// Number of frames to check stability over
    pub stability_frames: u32,
}

/// Physical detector response applied to raw hit counts before rendering.
/// These parameters model what the detector hardware does to the signal.
///
/// Apply order: blur → add background → Poisson noise.
/// Physically: expected = gaussian_blur(raw) + background;
///             if poisson_noise: observed ~ Poisson(expected).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetectorResponseConfig {
    /// Gaussian PSF sigma in microns. 0.0 = no blur.
    #[serde(default)]
    pub blur_sigma_um: f64,
    /// Uniform background added to every pixel (counts). 0.0 = no background.
    #[serde(default)]
    pub background_counts: f64,
    /// Apply Poisson noise after blur + background. Default false.
    /// Note: Monte Carlo particle sampling already creates shot noise;
    /// this adds detector-counting noise on top.
    #[serde(default)]
    pub poisson_noise: bool,
    /// RNG seed for Poisson sampling. None = non-deterministic.
    #[serde(default)]
    pub noise_seed: Option<u64>,
}

impl Default for DetectorResponseConfig {
    fn default() -> Self {
        Self {
            blur_sigma_um:     0.0,
            background_counts: 0.0,
            poisson_noise:     false,
            noise_seed:        None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ScaleMode {
    Linear,
    Log,
    Sqrt,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PngExportConfig {
    /// Output resolution [width, height] in pixels.
    /// None → use the detector's own pixel grid ([nx, ny] from config).
    /// If only one dimension is desired, set both and preserve the physical
    /// aspect ratio manually: height = width * height_mm / width_mm.
    pub output_pixels: Option<[u32; 2]>,

    pub colormap: ColormapType,
    pub scale: ScaleMode,
    pub gamma: f32,
    pub exposure: f32,

    /// Append a 24-px wide colorbar strip on the right edge of the PNG.
    /// When true, png_width = data_width + 24; record both in the sidecar.
    pub include_colorbar: bool,

    /// Write a _meta.json sidecar alongside the PNG.
    pub include_metadata: bool,
}

// ============================================================================
// DISPLAY CONFIGURATION
// ============================================================================

/// Display and rendering configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DisplayConfig {
    /// Colormap for radiograph
    pub colormap: ColormapType,

    /// Gamma correction
    pub gamma: f32,

    /// Exposure multiplier
    pub exposure: f32,

    /// Use logarithmic scale
    pub log_scale: bool,

    /// Normalization mode
    pub normalization: NormalizationMode,

    /// 3D visualization options
    pub show_volume: bool,
    pub show_source_marker: bool,
    pub show_detector_bounds: bool,
    pub show_field_bounds: bool,
    pub show_grid: bool,

    /// Performance display
    pub show_fps: bool,
    pub show_stats: bool,

    /// Window settings
    pub window: WindowConfig,

    /// Rendering quality
    pub render: RenderConfig,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum ColormapType {
    /// Realistic RCF film response
    RcfFilm,
    /// Scientific: dark-to-light (viridis-like)
    Scientific,
    /// Grayscale
    Grayscale,
    /// Hot (black-red-yellow-white)
    Hot,
    /// Inverted (white background, dark features)
    Inverted,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NormalizationMode {
    /// Auto-normalize to max count
    Auto,
    /// Fixed maximum count
    Fixed { max_count: f32 },
    /// Percentile-based (ignore outliers)
    Percentile { percentile: f32 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WindowConfig {
    /// Initial window width
    pub width: u32,
    /// Initial window height
    pub height: u32,
    /// Start fullscreen
    pub fullscreen: bool,
    /// VSync
    pub vsync: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RenderConfig {
    /// Steps per frame (higher = faster simulation, lower fps)
    pub steps_per_frame: u32,

    /// Volume rendering step size
    pub volume_step_size: f32,

    /// Volume rendering density scale
    pub volume_density: f32,
}

// ============================================================================
// DEFAULT IMPLEMENTATIONS
// ============================================================================

impl Default for SimulationConfig {
    fn default() -> Self {
        Self {
            source: SourceConfig::default(),
            field: FieldConfig::default(),
            detector: DetectorConfig::default(),
            grid: GridConfig::default(),
            physics: PhysicsConfig::default(),
            output: OutputConfig::default(),
            display: DisplayConfig::default(),
        }
    }
}

impl Default for SourceConfig {
    fn default() -> Self {
        Self {
            source_type: SourceType::Point,
            n_protons: 1_000_000,
            position: [-0.05, 0.0, 0.0],
            direction: DirectionMode::AutoCenter,
            energy: EnergyConfig::default(),
            spatial: SpatialConfig::default(),
            angular: AngularConfig::default(),
        }
    }
}

impl Default for EnergyConfig {
    fn default() -> Self {
        Self {
            mean_mev: 14.7,  // D-3He fusion protons
            spectrum: EnergySpectrum::Monoenergetic,
        }
    }
}

impl Default for SpatialConfig {
    fn default() -> Self {
        Self {
            mode: SpatialMode::Point,
        }
    }
}

impl Default for AngularConfig {
    fn default() -> Self {
        Self {
            mode: AngularMode::UniformCone { half_angle_rad: 0.05 },
        }
    }
}

impl Default for FieldConfig {
    fn default() -> Self {
        Self {
            source: FieldSource::Analytical(AnalyticalFieldConfig::default()),
            scale_factor: 1.0,
            center_offset: [0.0, 0.0, 0.0],
        }
    }
}

impl Default for AnalyticalFieldConfig {
    fn default() -> Self {
        Self {
            field_type: AnalyticalFieldType::ZPinch {
                current_a: 1_000_000.0,  // 1 MA
                pinch_radius_m: 0.002,    // 2 mm
                length_m: 0.02,           // 2 cm
            },
            resolution: [128, 128, 256],
            bounds: [-0.03, 0.03, -0.03, 0.03, -0.02, 0.02],
        }
    }
}

impl Default for DetectorConfig {
    fn default() -> Self {
        Self {
            position_mode: DetectorPositionMode::Auto {
                distance_from_field_m: 0.20,
            },
            extent_m: [0.25, 0.25],
            resolution: [1024, 1024],
            normal: [1.0, 0.0, 0.0],
        }
    }
}

impl Default for GridConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            position_m: -0.02,  // Before field
            pattern: GridPattern::Square { spacing_m: 0.0005 },  // 500 μm
            wire_thickness_m: 0.00005,  // 50 μm
            interaction: GridInteraction::Absorb,
        }
    }
}

impl Default for PhysicsConfig {
    fn default() -> Self {
        Self {
            dt_s: 5e-13,
            max_steps: 20000,
            domain_margin: 1.5,
            integrator: IntegratorType::Boris,
            include_electric_field: false,
            relativistic: false,  // 14.7 MeV protons are ~17% c, borderline
        }
    }
}

impl Default for OutputConfig {
    fn default() -> Self {
        Self {
            directory: "output".to_string(),
            prefix: "radiograph".to_string(),
            auto_export: AutoExportConfig::default(),
            export_png: true,
            export_csv: true,
            export_config: true,
            png: PngExportConfig::default(),
        }
    }
}

impl Default for AutoExportConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            stability_threshold_percent: 0.1,
            stability_frames: 30,
        }
    }
}

impl Default for PngExportConfig {
    fn default() -> Self {
        Self {
            output_pixels: None,
            colormap: ColormapType::RcfFilm,
            scale: ScaleMode::Log,
            gamma: 0.5,
            exposure: 1.0,
            include_colorbar: false,
            include_metadata: true,
        }
    }
}

impl Default for DisplayConfig {
    fn default() -> Self {
        Self {
            colormap: ColormapType::RcfFilm,
            gamma: 0.5,
            exposure: 1.0,
            log_scale: false,
            normalization: NormalizationMode::Auto,
            show_volume: true,
            show_source_marker: true,
            show_detector_bounds: true,
            show_field_bounds: true,
            show_grid: true,
            show_fps: true,
            show_stats: true,
            window: WindowConfig::default(),
            render: RenderConfig::default(),
        }
    }
}

impl Default for WindowConfig {
    fn default() -> Self {
        Self {
            width: 1280,
            height: 720,
            fullscreen: false,
            vsync: true,
        }
    }
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self {
            steps_per_frame: 100,
            volume_step_size: 0.005,
            volume_density: 0.5,
        }
    }
}

// ============================================================================
// UTILITY METHODS
// ============================================================================

impl SimulationConfig {
    /// Load configuration from a JSON file
    pub fn load(path: &str) -> anyhow::Result<Self> {
        let content = std::fs::read_to_string(path)?;
        let config: Self = serde_json::from_str(&content)?;
        Ok(config)
    }

    /// Save configuration to a JSON file
    pub fn save(&self, path: &str) -> anyhow::Result<()> {
        let content = serde_json::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Create from legacy SimConfig format (backwards compatibility)
    pub fn from_legacy(legacy: &crate::loaders::SimConfig) -> Self {
        use crate::loaders::SimSourceGeometry;

        let mut config = Self::default();

        // Source
        config.source.n_protons = legacy.source.n_particles;
        config.source.energy.mean_mev = legacy.source.particle_energy_mev as f32;

        match &legacy.source.geometry {
            SimSourceGeometry::ParallelBeam { center_m, direction, angular_spread_rad, .. } => {
                config.source.source_type = SourceType::Parallel;
                if let Some(c) = center_m {
                    config.source.position = *c;
                }
                config.source.direction = DirectionMode::Vector(*direction);
                config.source.angular.mode = AngularMode::UniformCone {
                    half_angle_rad: *angular_spread_rad,
                };
            }
            SimSourceGeometry::Pencil { position_m, direction } => {
                config.source.source_type = SourceType::Parallel;
                config.source.position = *position_m;
                config.source.direction = DirectionMode::Vector(*direction);
                config.source.angular.mode = AngularMode::UniformCone { half_angle_rad: 0.0 };
            }
            SimSourceGeometry::Point { position_m, direction, half_angle_rad } => {
                config.source.source_type = SourceType::Parallel;
                config.source.position = *position_m;
                config.source.direction = DirectionMode::Vector(*direction);
                config.source.angular.mode = AngularMode::UniformCone { half_angle_rad: *half_angle_rad };
            }
            SimSourceGeometry::Disk { center_m, direction, half_angle_rad, .. } => {
                config.source.source_type = SourceType::Parallel;
                config.source.position = *center_m;
                config.source.direction = DirectionMode::Vector(*direction);
                config.source.angular.mode = AngularMode::UniformCone { half_angle_rad: *half_angle_rad };
            }
        }

        // Field (from file)
        config.field.source = FieldSource::File(FileFieldConfig {
            path: legacy.primary_field_path().to_string(),
            format: FieldFileFormat::Bfld,
        });

        // Detector
        let det_dist = legacy.detector.distance_m
            .unwrap_or_else(|| legacy.detector.center_m.map(|c| c[0].abs()).unwrap_or(0.05));
        config.detector.position_mode = DetectorPositionMode::Auto {
            distance_from_field_m: det_dist as f32,
        };
        config.detector.normal = legacy.detector.normal;
        config.detector.resolution = [legacy.detector.pixels[0], legacy.detector.pixels[1]];

        // Physics
        config.physics.dt_s = legacy.dt_s;
        config.physics.max_steps = legacy.max_steps;

        config
    }

    /// Get the beam direction as a normalized vector
    pub fn beam_direction(&self) -> [f32; 3] {
        match &self.source.direction {
            DirectionMode::Vector(v) => {
                let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
                if len > 0.0 {
                    [v[0] / len, v[1] / len, v[2] / len]
                } else {
                    [1.0, 0.0, 0.0]  // Default to +X
                }
            }
            DirectionMode::Target(t) => {
                let dx = t[0] - self.source.position[0];
                let dy = t[1] - self.source.position[1];
                let dz = t[2] - self.source.position[2];
                let len = (dx * dx + dy * dy + dz * dz).sqrt();
                if len > 0.0 {
                    [dx / len, dy / len, dz / len]
                } else {
                    [1.0, 0.0, 0.0]
                }
            }
            DirectionMode::AutoCenter => {
                // Aim at origin
                let dx = -self.source.position[0];
                let dy = -self.source.position[1];
                let dz = -self.source.position[2];
                let len = (dx * dx + dy * dy + dz * dz).sqrt();
                if len > 0.0 {
                    [dx / len, dy / len, dz / len]
                } else {
                    [1.0, 0.0, 0.0]
                }
            }
        }
    }
}

impl SourceConfig {
    /// Get proton speed from energy [m/s]
    pub fn mean_speed(&self) -> f64 {
        // E = 0.5 * m * v^2
        // v = sqrt(2 * E / m)
        let energy_j = self.energy.mean_mev as f64 * 1.602e-13;  // MeV to Joules
        let proton_mass = 1.673e-27;  // kg
        (2.0 * energy_j / proton_mass).sqrt()
    }
}

impl EnergySpectrum {
    /// Sample an energy from this distribution [MeV]
    pub fn sample(&self, mean_mev: f32, rng: &mut impl rand::Rng) -> f32 {
        use rand::distributions::Distribution;

        match self {
            EnergySpectrum::Monoenergetic => mean_mev,

            EnergySpectrum::Gaussian { sigma_mev } => {
                let normal = rand_distr::Normal::new(mean_mev as f64, *sigma_mev as f64)
                    .unwrap_or_else(|_| rand_distr::Normal::new(mean_mev as f64, 0.1).unwrap());
                normal.sample(rng).max(0.1) as f32  // Minimum 0.1 MeV
            }

            EnergySpectrum::MaxwellBoltzmann { temperature_mev } => {
                // Maxwell-Boltzmann: f(E) ~ sqrt(E) * exp(-E/kT)
                // Use rejection sampling or gamma distribution
                let gamma = rand_distr::Gamma::new(1.5, *temperature_mev as f64)
                    .unwrap_or_else(|_| rand_distr::Gamma::new(1.5, 1.0).unwrap());
                gamma.sample(rng).max(0.1) as f32
            }

            EnergySpectrum::TnsaExponential { temperature_mev, max_energy_mev } => {
                // Exponential: f(E) ~ exp(-E/kT) with cutoff
                let exp = rand_distr::Exp::new(1.0 / (*temperature_mev as f64))
                    .unwrap_or_else(|_| rand_distr::Exp::new(1.0).unwrap());
                let e = exp.sample(rng) as f32;
                e.min(*max_energy_mev).max(0.1)
            }

            EnergySpectrum::Uniform { min_mev, max_mev } => {
                let uniform = rand::distributions::Uniform::new(*min_mev, *max_mev);
                uniform.sample(rng)
            }
        }
    }
}
