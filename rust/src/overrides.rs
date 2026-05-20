use anyhow::{bail, Result};

/// A single `--set key=value` override parsed from the CLI.
#[derive(Debug, Clone)]
pub struct ConfigOverride {
    /// Canonical key matching TOML deck names (e.g. `source.energy_MeV`).
    pub canonical_key: String,
    /// Raw value string exactly as typed (e.g. `"14.7"`).
    pub raw_value: String,
}

const SUPPORTED_KEYS: &str = "\
  source.energy_MeV           source.n_particles\n\
  source.beam_radius_mm       source.angular_spread_deg\n\
  source.energy_spread_percent\n\
  source.temperature_MeV      source.cutoff_mev\n\
  numerics.dt_ps              numerics.max_steps\n\
  detector.width_mm           detector.height_mm\n\
  field.scale_B               field.scale_E";

/// Parse a `key=value` string into a [`ConfigOverride`].
///
/// Accepts lowercase aliases (e.g. `source.energy_mev`) but stores the
/// canonical TOML form (e.g. `source.energy_MeV`) so metadata is consistent.
pub fn parse_override(s: &str) -> Result<ConfigOverride> {
    let eq = s.find('=').ok_or_else(|| {
        anyhow::anyhow!(
            "--set requires key=value format (e.g. source.energy_MeV=14.7), got: {:?}",
            s
        )
    })?;
    let key   = s[..eq].trim();
    let value = s[eq + 1..].trim();
    let canonical_key = canonicalize_key(key)?.to_string();
    Ok(ConfigOverride { canonical_key, raw_value: value.to_string() })
}

pub(crate) fn canonicalize_key(key: &str) -> Result<&'static str> {
    match key {
        "source.energy_MeV" | "source.energy_mev"  => Ok("source.energy_MeV"),
        "source.n_particles"                         => Ok("source.n_particles"),
        "source.beam_radius_mm"                      => Ok("source.beam_radius_mm"),
        "source.angular_spread_deg"                  => Ok("source.angular_spread_deg"),
        "source.energy_spread_percent"               => Ok("source.energy_spread_percent"),
        "source.temperature_MeV" | "source.temperature_mev" => Ok("source.temperature_MeV"),
        "source.cutoff_mev" | "source.cutoff_MeV"   => Ok("source.cutoff_mev"),
        "numerics.dt_ps" | "dt_ps"                  => Ok("numerics.dt_ps"),
        "numerics.max_steps" | "max_steps"           => Ok("numerics.max_steps"),
        "detector.width_mm"                          => Ok("detector.width_mm"),
        "detector.height_mm"                         => Ok("detector.height_mm"),
        "field.scale_B" | "field.scale_b"            => Ok("field.scale_B"),
        "field.scale_E" | "field.scale_e"            => Ok("field.scale_E"),
        other => bail!(
            "Unknown --set key: {:?}\nSupported keys:\n{}",
            other, SUPPORTED_KEYS
        ),
    }
}

/// Parse f64 for an override, naming the key in the error.
pub(crate) fn parse_f64(key: &str, value: &str) -> Result<f64> {
    value.parse::<f64>()
        .map_err(|_| anyhow::anyhow!("expected a number for {:?}, got {:?}", key, value))
}

/// Parse u64 for an override, naming the key in the error.
pub(crate) fn parse_u64(key: &str, value: &str) -> Result<u64> {
    value.parse::<u64>()
        .map_err(|_| anyhow::anyhow!("expected a non-negative integer for {:?}, got {:?}", key, value))
}

/// Parse u32 for an override, naming the key in the error.
pub(crate) fn parse_u32(key: &str, value: &str) -> Result<u32> {
    value.parse::<u32>()
        .map_err(|_| anyhow::anyhow!("expected a non-negative integer for {:?}, got {:?}", key, value))
}
