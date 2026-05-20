//! Physics constants and unit-conversion helpers.
//!
//! Internal representation is always SI (m, s, J, T, V/m).
//! These helpers convert between SI and human-friendly experiment units and
//! format values with appropriate SI prefixes for log output.

pub const PROTON_MASS_KG: f64 = 1.672_621_923_69e-27;
pub const MEV_J: f64 = 1.602_176_634e-13;   // joules per MeV
pub const C_M_S: f64 = 299_792_458.0;        // speed of light [m/s]
pub const PROTON_QM: f64 = 9.578_833_149e7;  // proton q/m [C/kg]

const PROTON_MASS_MEV: f64 = 938.272_046;    // proton rest mass [MeV/c²]

// ── unit conversions ──────────────────────────────────────────────────────────

pub fn mm_to_m(mm: f64) -> f64 { mm * 1e-3 }
pub fn m_to_mm(m: f64)  -> f64 { m  * 1e3  }
pub fn ps_to_s(ps: f64) -> f64 { ps * 1e-12 }
pub fn s_to_ps(s: f64)  -> f64 { s  * 1e12  }
pub fn mev_to_j(mev: f64) -> f64 { mev * MEV_J }
pub fn j_to_mev(j: f64)   -> f64 { j   / MEV_J }

/// Relativistic proton speed from kinetic energy [m/s].
pub fn proton_speed_from_mev(ke_mev: f64) -> f64 {
    let gamma = 1.0 + ke_mev / PROTON_MASS_MEV;
    (1.0 - 1.0 / (gamma * gamma)).sqrt() * C_M_S
}

/// Specific relativistic momentum |u| = γv [m/s] from kinetic energy.
/// This is what particles store in their velocity field (u = γv, not v).
pub fn proton_momentum_per_mass_from_mev(ke_mev: f64) -> f64 {
    let gamma = 1.0 + ke_mev / PROTON_MASS_MEV;
    C_M_S * (gamma * gamma - 1.0).sqrt()
}

// ── SI-prefix formatters ──────────────────────────────────────────────────────

pub fn fmt_b(val_t: f64) -> String {
    let a = val_t.abs();
    if a >= 1.0       { format!("{:.3} T",  val_t) }
    else if a >= 1e-3 { format!("{:.3} mT", val_t * 1e3) }
    else              { format!("{:.3} µT", val_t * 1e6) }
}

pub fn fmt_e(val_vm: f64) -> String {
    let a = val_vm.abs();
    if a >= 1e6       { format!("{:.3} MV/m", val_vm * 1e-6) }
    else if a >= 1e3  { format!("{:.3} kV/m", val_vm * 1e-3) }
    else              { format!("{:.3} V/m",   val_vm) }
}

pub fn fmt_dist(val_m: f64) -> String {
    let a = val_m.abs();
    if a >= 1.0       { format!("{:.2} m",  val_m) }
    else if a >= 1e-3 { format!("{:.2} mm", val_m * 1e3) }
    else              { format!("{:.2} µm", val_m * 1e6) }
}

pub fn fmt_time(val_s: f64) -> String {
    let a = val_s.abs();
    if a >= 1e-6      { format!("{:.3} µs", val_s * 1e6) }
    else if a >= 1e-9 { format!("{:.3} ns", val_s * 1e9) }
    else              { format!("{:.3} ps", val_s * 1e12) }
}
