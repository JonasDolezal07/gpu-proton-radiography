use anyhow::{bail, Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

// ── Data structures ───────────────────────────────────────────────────────────

pub struct SweepParam {
    pub key: String,         // canonical key, e.g. "source.energy_MeV"
    pub values: Vec<String>, // e.g. ["5", "10", "15", "20"]
}

pub struct SweepSpec {
    pub deck_path: String,
    pub params: Vec<SweepParam>,
    pub output_dir: PathBuf,
    pub overwrite: bool,
}

// ── Manifest ──────────────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct SweepManifest {
    pub sweep_schema_version: u32,
    pub deck: String,
    pub params: Vec<ManifestParam>,
    pub runs: Vec<ManifestRun>,
    pub started_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct ManifestParam {
    pub key: String,
    pub values: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestRun {
    pub label: String,
    pub run_dir: String,
    pub overrides: Vec<ManifestOverride>,
    pub status: String, // "pending" | "running" | "complete" | "failed"
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ManifestOverride {
    pub key: String,
    pub value: String,
}

// ── CLI parsing ───────────────────────────────────────────────────────────────

pub fn parse_sweep_args(argv: &[String]) -> Result<SweepSpec> {
    let mut deck_path: Option<String> = None;
    let mut output_dir: Option<PathBuf> = None;
    let mut params: Vec<SweepParam> = Vec::new();
    let mut overwrite = false;
    let mut i = 0;

    while i < argv.len() {
        match argv[i].as_str() {
            "--help" | "-h" => { print_sweep_help(); std::process::exit(0); }
            "-o" | "--output" => {
                i += 1;
                output_dir = argv.get(i).map(PathBuf::from);
            }
            "--overwrite" => overwrite = true,
            "--param" => {
                i += 1;
                let spec = argv.get(i)
                    .ok_or_else(|| anyhow::anyhow!("--param requires a value"))?;
                params.push(parse_param_spec(spec)?);
            }
            arg if !arg.starts_with('-') && deck_path.is_none() => {
                deck_path = Some(arg.to_string());
            }
            other => bail!("Unknown sweep option: {:?}", other),
        }
        i += 1;
    }

    let deck_path = deck_path.ok_or_else(|| anyhow::anyhow!(
        "Usage: proton_tracer sweep <deck.toml> --param key=val1,val2,...\n\
         Run `proton_tracer sweep --help` for details."
    ))?;

    if params.is_empty() {
        bail!("sweep requires at least one --param key=val1,val2,...");
    }

    // Zip mode: all params must have the same number of values.
    let n = params[0].values.len();
    for p in &params[1..] {
        if p.values.len() != n {
            bail!(
                "All --param value lists must have the same length for zip mode.\n\
                 {:?} has {} values but {:?} has {}.\n\
                 (Cartesian product via --product is planned for a future release.)",
                params[0].key, n, p.key, p.values.len()
            );
        }
    }

    let output_dir = output_dir
        .unwrap_or_else(|| find_next_sweep_dir(&PathBuf::from("runs")));

    Ok(SweepSpec { deck_path, params, output_dir, overwrite })
}

fn parse_param_spec(spec: &str) -> Result<SweepParam> {
    let eq = spec.find('=').ok_or_else(|| {
        anyhow::anyhow!(
            "--param requires key=values format (e.g. source.energy_MeV=5,10,15), got: {:?}",
            spec
        )
    })?;
    let key_raw = spec[..eq].trim();
    let values_str = spec[eq + 1..].trim();

    let key = crate::overrides::canonicalize_key(key_raw)?.to_string();

    let values: Vec<String> = if values_str.contains(':') {
        expand_range(values_str)?
    } else {
        values_str.split(',').map(|s| s.trim().to_string()).collect()
    };

    if values.is_empty() {
        bail!("--param {:?}: value list is empty", key);
    }
    Ok(SweepParam { key, values })
}

fn expand_range(spec: &str) -> Result<Vec<String>> {
    let parts: Vec<&str> = spec.split(':').collect();
    if parts.len() != 3 {
        bail!("Range syntax is start:stop:step (e.g. 5:20:5), got {:?}", spec);
    }
    let start: f64 = parts[0].trim().parse()
        .with_context(|| format!("Range start is not a number: {:?}", parts[0]))?;
    let stop: f64 = parts[1].trim().parse()
        .with_context(|| format!("Range stop is not a number: {:?}", parts[1]))?;
    let step: f64 = parts[2].trim().parse()
        .with_context(|| format!("Range step is not a number: {:?}", parts[2]))?;

    if step == 0.0 {
        bail!("Range step must not be zero");
    }
    if step > 0.0 && start > stop {
        bail!("Range start ({}) > stop ({}) with positive step", start, stop);
    }
    if step < 0.0 && start < stop {
        bail!("Range start ({}) < stop ({}) with negative step", start, stop);
    }

    let mut values = Vec::new();
    let mut v = start;
    let eps = step.abs() * 1e-9;
    while (step > 0.0 && v <= stop + eps) || (step < 0.0 && v >= stop - eps) {
        if (v - v.round()).abs() < 1e-9 {
            values.push(format!("{}", v.round() as i64));
        } else {
            // Up to 6 significant figures, strip trailing zeros.
            let s = format!("{:.6}", v);
            values.push(s.trim_end_matches('0').trim_end_matches('.').to_string());
        }
        v += step;
    }

    if values.is_empty() {
        bail!("Range {}:{}:{} produced no values", start, stop, step);
    }
    Ok(values)
}

fn find_next_sweep_dir(base: &Path) -> PathBuf {
    for n in 1..=999 {
        let candidate = base.join(format!("sweep_{:03}", n));
        if !candidate.exists() {
            return candidate;
        }
    }
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    base.join(format!("sweep_{}", ts))
}

fn make_run_label(overrides: &[ManifestOverride]) -> String {
    overrides.iter().map(|o| {
        let short = o.key.rsplit('.').next().unwrap_or(&o.key);
        format!("{}_{}", short, o.value)
    }).collect::<Vec<_>>().join("_")
}

// ── Manifest I/O ──────────────────────────────────────────────────────────────

fn write_manifest(sweep_dir: &Path, manifest: &SweepManifest) -> Result<()> {
    let json = serde_json::to_string_pretty(manifest)?;
    std::fs::write(sweep_dir.join("sweep_manifest.json"), json)
        .context("Failed to write sweep_manifest.json")?;
    Ok(())
}

// ── Execution ─────────────────────────────────────────────────────────────────

pub fn run_sweep(spec: &SweepSpec) -> Result<()> {
    if spec.output_dir.exists() {
        if spec.overwrite {
            std::fs::remove_dir_all(&spec.output_dir)
                .with_context(|| format!("Cannot remove {:?}", spec.output_dir))?;
        } else {
            bail!(
                "Sweep directory {:?} already exists.\n\
                 Use --overwrite to replace it, or omit -o to auto-number.",
                spec.output_dir
            );
        }
    }
    std::fs::create_dir_all(&spec.output_dir)
        .with_context(|| format!("Cannot create sweep directory: {:?}", spec.output_dir))?;

    let n_runs = spec.params[0].values.len();

    // Build the zipped run list.
    let run_overrides: Vec<Vec<ManifestOverride>> = (0..n_runs).map(|i| {
        spec.params.iter().map(|p| ManifestOverride {
            key: p.key.clone(),
            value: p.values[i].clone(),
        }).collect()
    }).collect();

    // Initialise manifest with all runs as "pending".
    let mut manifest = SweepManifest {
        sweep_schema_version: 1,
        deck: spec.deck_path.clone(),
        params: spec.params.iter().map(|p| ManifestParam {
            key: p.key.clone(),
            values: p.values.clone(),
        }).collect(),
        runs: run_overrides.iter().map(|ovs| {
            let label = make_run_label(ovs);
            ManifestRun {
                label: label.clone(),
                run_dir: label,
                overrides: ovs.clone(),
                status: "pending".to_string(),
            }
        }).collect(),
        started_at: chrono::Utc::now().to_rfc3339(),
        completed_at: None,
    };
    write_manifest(&spec.output_dir, &manifest)?;

    let binary = std::env::current_exe()
        .context("Cannot determine path to current executable")?;

    let mut n_complete = 0usize;
    let mut n_failed   = 0usize;

    for (idx, overrides) in run_overrides.iter().enumerate() {
        let label     = manifest.runs[idx].label.clone();
        let run_out   = spec.output_dir.join(&label);

        manifest.runs[idx].status = "running".to_string();
        write_manifest(&spec.output_dir, &manifest)?;

        log::info!("── sweep {}/{}: {} ──", idx + 1, n_runs, label);

        let mut cmd = std::process::Command::new(&binary);
        cmd.arg("run").arg(&spec.deck_path).arg("-o").arg(&run_out);
        for ov in overrides {
            cmd.arg("--set").arg(format!("{}={}", ov.key, ov.value));
        }

        let status = cmd.status()
            .with_context(|| format!("Failed to launch sweep run {:?}", label))?;

        if status.success() {
            manifest.runs[idx].status = "complete".to_string();
            n_complete += 1;
            log::info!("  → complete");
        } else {
            manifest.runs[idx].status = "failed".to_string();
            n_failed += 1;
            log::warn!("  → FAILED (exit {:?}) — continuing sweep", status.code());
        }
        write_manifest(&spec.output_dir, &manifest)?;
    }

    manifest.completed_at = Some(chrono::Utc::now().to_rfc3339());
    write_manifest(&spec.output_dir, &manifest)?;

    log::info!(
        "Sweep complete: {}/{} succeeded, {} failed",
        n_complete, n_runs, n_failed
    );
    log::info!("Sweep directory: {:?}", spec.output_dir);

    if n_failed > 0 {
        bail!("{} sweep run(s) failed — see sweep_manifest.json for details", n_failed);
    }
    Ok(())
}

// ── Help ──────────────────────────────────────────────────────────────────────

fn print_sweep_help() {
    println!("proton-tracer sweep — run a parameter sweep without editing deck files");
    println!();
    println!("Usage:");
    println!("  proton-tracer sweep <deck.toml> --param key=val1,val2,...");
    println!("  proton-tracer sweep <deck.toml> --param key=start:stop:step");
    println!();
    println!("Options:");
    println!("  --param key=values    Parameter to sweep (repeatable — zip mode)");
    println!("  -o <dir>              Output directory (default: auto runs/sweep_NNN)");
    println!("  --overwrite           Remove existing sweep directory before starting");
    println!();
    println!("Examples:");
    println!("  proton-tracer sweep kink.toml --param source.energy_MeV=5,10,15,20");
    println!("  proton-tracer sweep kink.toml --param source.energy_MeV=5:20:5");
    println!("  proton-tracer sweep kink.toml \\");
    println!("    --param source.energy_MeV=5,10,15 \\");
    println!("    --param numerics.max_steps=10000,20000,30000");
    println!();
    println!("Supported --param keys:");
    println!("  source.energy_MeV         source.n_particles");
    println!("  source.beam_radius_mm      source.angular_spread_deg");
    println!("  source.energy_spread_percent");
    println!("  numerics.dt_ps             numerics.max_steps");
    println!("  detector.width_mm          detector.height_mm");
    println!("  field.scale_B              field.scale_E");
    println!();
    println!("Output layout:");
    println!("  runs/sweep_001/");
    println!("    sweep_manifest.json      ← live status, updated per run");
    println!("    energy_MeV_5/            ← one full run directory per point");
    println!("    energy_MeV_10/");
    println!("    ...");
    println!();
    println!("All runs continue even if one fails (see manifest for status).");
}
