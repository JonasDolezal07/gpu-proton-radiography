//! Global adaptive timestep schedule.
//!
//! The simulation has three phases along the beam path:
//!   1. Pre-field   (source → field entry face)    — vacuum, large dt
//!   2. In-field    (across the field volume)       — plasma, small dt (Larmor/20)
//!   3. Post-field  (field exit → detector)         — vacuum, large dt
//!
//! `dt_large` is used when `t_sim < t_entry` or `t_sim > t_exit`.
//! `dt_small` is used in between.
//!
//! Only active when the user has NOT supplied an explicit `dt_ps`. When the user
//! supplies a fixed dt, the schedule is None and `dt_small` is used throughout.

use serde::{Deserialize, Serialize};
use crate::loaders::{SimConfig, SimSourceGeometry, FieldData, FieldBounds};

const PROTON_QM: f64 = 9.57883392e7;   // q/m for proton [C/kg]

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DtSchedule {
    /// Timestep used outside the field (vacuum) [s].
    pub dt_large_s: f64,
    /// Timestep used inside the field (Larmor-constrained) [s].
    pub dt_small_s: f64,
    /// Simulated time when particles reach the field entry face [s].
    pub t_entry_s: f64,
    /// Simulated time when particles leave the field (+ safety margin) [s].
    pub t_exit_s: f64,
    /// Ratio dt_large / dt_small — approximate speedup over a fixed-dt run.
    pub speedup_estimate: f64,
}

impl DtSchedule {
    /// Choose dt for the given simulated time.
    #[inline]
    pub fn pick(&self, t_sim_s: f64) -> f32 {
        if t_sim_s >= self.t_entry_s && t_sim_s <= self.t_exit_s {
            self.dt_small_s as f32
        } else {
            self.dt_large_s as f32
        }
    }

    /// Compute the schedule from the loaded config and field.
    pub fn compute(config: &SimConfig, field: &FieldData) -> Self {
        let v = config.source.particle_speed_m_s;

        // ── dt_small: Larmor/20 ∧ grid_crossing/4 ──────────────────────────
        let (_, b_max) = field.b_magnitude_range();
        let dt_larmor = if b_max > 1e-10_f32 {
            std::f64::consts::TAU / (PROTON_QM * b_max as f64) / 20.0
        } else {
            f64::MAX
        };
        let (dx, dy, dz) = field.spacing();
        let min_cell = (dx as f64).min(dy as f64).min(dz as f64);
        let dt_grid = 0.25 * min_cell / v;
        let dt_small = dt_larmor.min(dt_grid);

        // ── Phase boundary times ────────────────────────────────────────────
        let b = &field.bounds;
        let dir = config.source.beam_direction();
        let [ddx, ddy, ddz] = [dir[0] as f64, dir[1] as f64, dir[2] as f64];

        let src = source_position_m(config);

        // Distance from source to the first field face hit along the beam.
        let t_entry = time_to_face_entry(src, [ddx, ddy, ddz], b, v);

        // Distance from source to the last field face exit along the beam.
        let field_transit = time_to_face_exit(src, [ddx, ddy, ddz], b, v) - t_entry;
        let t_exit = t_entry + field_transit.max(0.0) * 1.30; // 30% safety margin

        // ── dt_large: unconstrained except "don't skip over the field" ──────
        // Cap at t_entry/10 (at least 10 large steps before the field) and at
        // 20× dt_small (avoids absurd ratios when the source is very close).
        let dt_large = dt_small * 20.0;
        let dt_large = if t_entry > 0.0 {
            dt_large.min(t_entry / 10.0)
        } else {
            dt_small   // source already inside or touching field
        };

        let speedup_estimate = dt_large / dt_small;

        Self { dt_large_s: dt_large, dt_small_s: dt_small, t_entry_s: t_entry, t_exit_s: t_exit, speedup_estimate }
    }
}

// ── helpers ──────────────────────────────────────────────────────────────────

fn source_position_m(config: &SimConfig) -> [f64; 3] {
    match &config.source.geometry {
        SimSourceGeometry::Pencil { position_m, .. } |
        SimSourceGeometry::Point  { position_m, .. } =>
            [position_m[0] as f64, position_m[1] as f64, position_m[2] as f64],
        SimSourceGeometry::Disk { center_m, .. } =>
            [center_m[0] as f64, center_m[1] as f64, center_m[2] as f64],
        SimSourceGeometry::ParallelBeam { center_m, .. } =>
            center_m.map(|c| [c[0] as f64, c[1] as f64, c[2] as f64])
                    .unwrap_or([0.0, 0.0, 0.0]),
    }
}

/// Time [s] for a particle starting at `src` traveling in direction `dir`
/// (unit vector) at speed `v` [m/s] to reach the nearest face of the field box.
/// Returns 0.0 if the source is already inside or behind the field.
fn time_to_face_entry(
    src: [f64; 3],
    dir: [f64; 3],
    b: &FieldBounds,
    v: f64,
) -> f64 {
    let faces = [
        (b.x_min as f64, b.x_max as f64, src[0], dir[0]),
        (b.y_min as f64, b.y_max as f64, src[1], dir[1]),
        (b.z_min as f64, b.z_max as f64, src[2], dir[2]),
    ];

    let mut t_entry = 0.0_f64;
    for (fmin, fmax, p, d) in faces {
        if d.abs() < 1e-12 { continue; }
        let t_min = (fmin - p) / d;
        let t_max = (fmax - p) / d;
        let (ta, tb) = if d > 0.0 { (t_min, t_max) } else { (t_max, t_min) };
        // ta is when this axis enters the slab; must be > 0 and be the last enter
        if ta > 0.0 { t_entry = t_entry.max(ta); }
        let _ = tb; // t_exit handled separately
    }

    (t_entry / v).max(0.0)
}

/// Time [s] for the particle to exit the far face of the field box.
fn time_to_face_exit(
    src: [f64; 3],
    dir: [f64; 3],
    b: &FieldBounds,
    v: f64,
) -> f64 {
    let faces = [
        (b.x_min as f64, b.x_max as f64, src[0], dir[0]),
        (b.y_min as f64, b.y_max as f64, src[1], dir[1]),
        (b.z_min as f64, b.z_max as f64, src[2], dir[2]),
    ];

    let mut t_exit = f64::MAX;
    for (fmin, fmax, p, d) in faces {
        if d.abs() < 1e-12 { continue; }
        let t_min = (fmin - p) / d;
        let t_max = (fmax - p) / d;
        let (_ta, tb) = if d > 0.0 { (t_min, t_max) } else { (t_max, t_min) };
        if tb > 0.0 { t_exit = t_exit.min(tb); }
    }

    if t_exit == f64::MAX { t_exit = 0.0; }
    (t_exit / v).max(0.0)
}
