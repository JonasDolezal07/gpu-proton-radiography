use anyhow::{Context, Result};
use std::path::Path;

// ── Public entry points ───────────────────────────────────────────────────────

/// Inspect a run directory or sweep directory — auto-detected.
pub fn inspect(path: &Path) -> Result<()> {
    if path.join("sweep_manifest.json").exists() {
        print_sweep_summary(path)
    } else if path.join("metadata.json").exists() {
        print_run_summary(path)
    } else {
        anyhow::bail!(
            "{:?} is not a run or sweep directory\n\
             (no metadata.json or sweep_manifest.json found)",
            path
        )
    }
}

/// Analyze count statistics for a single run directory.
pub fn analyze(run_dir: &Path, use_raw: bool) -> Result<()> {
    let meta = read_metadata(run_dir)?;
    let mode = if use_raw { "raw counts" } else { "processed counts" };
    println!("Analysis: {}  ({})", meta.run.run_name, mode);
    println!();

    if use_raw {
        let path = run_dir.join("counts").join("raw_counts.bin");
        let bytes = std::fs::read(&path)
            .with_context(|| format!("Cannot read {:?}", path))?;
        if bytes.len() % 4 != 0 {
            anyhow::bail!("raw_counts.bin size is not a multiple of 4 bytes");
        }
        let counts: Vec<u32> = bytes.chunks_exact(4)
            .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let shape = meta.counts_format.as_ref().map(|cf| cf.raw.shape);
        validate_pixel_count(counts.len(), shape, "raw_counts.bin")?;
        print_raw_stats(&counts, shape);
    } else {
        let path = run_dir.join("counts").join("processed_counts.bin");
        let bytes = std::fs::read(&path)
            .with_context(|| format!("Cannot read {:?}", path))?;
        if bytes.len() % 4 != 0 {
            anyhow::bail!("processed_counts.bin size is not a multiple of 4 bytes");
        }
        let counts: Vec<f32> = bytes.chunks_exact(4)
            .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
            .collect();
        let shape = meta.counts_format.as_ref().map(|cf| cf.processed.shape);
        validate_pixel_count(counts.len(), shape, "processed_counts.bin")?;
        print_processed_stats(&counts, shape);
    }
    Ok(())
}

// ── Single-run summary ────────────────────────────────────────────────────────

fn print_run_summary(run_dir: &Path) -> Result<()> {
    let meta = read_metadata(run_dir)?;

    field("Run",    &meta.run.run_name);
    field("Status", &meta.run.status);

    // Deck — prefer the copied file inside the run dir
    let local_deck = run_dir.join("input_deck.toml");
    if local_deck.exists() {
        field("Deck", "input_deck.toml");
    } else {
        field("Deck", &meta.input_files.deck_path);
    }

    // Git
    let git = match (meta.code.git_commit.as_deref(), meta.code.git_dirty) {
        (Some(c), dirty) => {
            let short = &c[..c.len().min(7)];
            if dirty { format!("{} (dirty)", short) } else { format!("{} (clean)", short) }
        }
        (None, _) => "—".to_string(),
    };
    field("Git", &git);

    field("GPU", meta.hardware.gpu.as_deref().unwrap_or("—"));

    // Particles / hits
    if let Some(ref d) = meta.diagnostics {
        field("Particles", &fmt_int(d.n_particles as u64));
        let hits_str = format!("{}  ({:.1}%)", fmt_int(d.n_hits), d.hit_fraction * 100.0);
        field("Hits", &hits_str);
    } else {
        field("Particles", "—");
        field("Hits", "—");
    }

    // Runtime
    match meta.performance.as_ref() {
        Some(p) => field("Runtime", &format!("{:.2} s", p.total_runtime_s)),
        None    => field("Runtime", "—"),
    }

    // Counts shape
    if let Some(ref cf) = meta.counts_format {
        let s = cf.processed.shape;
        field("Counts", &format!("{}×{}  {}", s[0], s[1], cf.processed.dtype));
    } else {
        field("Counts", "—");
    }

    // Image
    match meta.outputs.radiograph_png.as_deref() {
        Some(p) => field("Image", p),
        None    => field("Image", "—"),
    }

    // CLI overrides
    match meta.cli_overrides.as_deref() {
        Some(ovs) if !ovs.is_empty() => {
            println!("Overrides:");
            for ov in ovs { println!("  {} = {}", ov.key, ov.value); }
        }
        _ => field("Overrides", "(none)"),
    }

    // Warnings
    if meta.warnings.is_empty() {
        field("Warnings", "(none)");
    } else {
        println!("Warnings:");
        for w in &meta.warnings { println!("  {}", w); }
    }

    Ok(())
}

// ── Sweep summary ─────────────────────────────────────────────────────────────

fn print_sweep_summary(sweep_dir: &Path) -> Result<()> {
    let manifest_path = sweep_dir.join("sweep_manifest.json");
    let manifest_str = std::fs::read_to_string(&manifest_path)
        .with_context(|| format!("Cannot read {:?}", manifest_path))?;
    let manifest: crate::sweep::SweepManifest = serde_json::from_str(&manifest_str)
        .context("Failed to parse sweep_manifest.json")?;

    let n         = manifest.runs.len();
    let n_done    = manifest.runs.iter().filter(|r| r.status == "complete").count();
    let n_failed  = manifest.runs.iter().filter(|r| r.status == "failed").count();
    let n_pending = n - n_done - n_failed;

    let dir_name = sweep_dir.file_name()
        .and_then(|n| n.to_str()).unwrap_or("?");

    field("Sweep",  &format!("{}  ({} runs)", dir_name, n));
    field("Deck",   &manifest.deck);
    field("Done",   manifest.completed_at.as_deref().unwrap_or("(running)"));
    field("Runs",   &format!("{} complete  {} failed  {} pending",
        n_done, n_failed, n_pending));
    println!();

    // Read child metadata for each run and build table rows.
    struct RowData<'a> {
        label:   &'a str,
        status:  &'a str,
        hits:    String,
        hit_pct: String,
        runtime: String,
    }

    let rows: Vec<RowData<'_>> = manifest.runs.iter().map(|run| {
        let child_meta = sweep_dir.join(&run.run_dir).join("metadata.json");
        let m: Option<crate::run_dir::RunMetadata> = std::fs::read_to_string(&child_meta)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok());

        let hits = m.as_ref()
            .and_then(|m| m.diagnostics.as_ref())
            .map(|d| fmt_int(d.n_hits))
            .unwrap_or_else(|| "—".to_string());

        let hit_pct = m.as_ref()
            .and_then(|m| m.diagnostics.as_ref())
            .map(|d| format!("{:.1}%", d.hit_fraction * 100.0))
            .unwrap_or_else(|| "—".to_string());

        let runtime = m.as_ref()
            .and_then(|m| m.performance.as_ref())
            .map(|p| format!("{:.2} s", p.total_runtime_s))
            .unwrap_or_else(|| "—".to_string());

        RowData { label: &run.label, status: &run.status, hits, hit_pct, runtime }
    }).collect();

    // Column widths
    let lw = rows.iter().map(|r| r.label.len()).max().unwrap_or(5).max(5);
    let sw = rows.iter().map(|r| r.status.len()).max().unwrap_or(6).max(6);
    let hw = rows.iter().map(|r| r.hits.len()).max().unwrap_or(4).max(4);
    let pw = rows.iter().map(|r| r.hit_pct.len()).max().unwrap_or(5).max(5);
    let rw = rows.iter().map(|r| r.runtime.len()).max().unwrap_or(7).max(7);

    // Header
    println!("  {:<3}  {:<lw$}  {:<sw$}  {:>hw$}  {:>pw$}  {:>rw$}",
        "#", "Label", "Status", "Hits", "Hit %", "Runtime",
        lw = lw, sw = sw, hw = hw, pw = pw, rw = rw);
    println!("  {}  {}  {}  {}  {}  {}",
        "─".repeat(3), "─".repeat(lw), "─".repeat(sw),
        "─".repeat(hw), "─".repeat(pw), "─".repeat(rw));

    for (i, row) in rows.iter().enumerate() {
        println!("  {:<3}  {:<lw$}  {:<sw$}  {:>hw$}  {:>pw$}  {:>rw$}",
            i + 1, row.label, row.status, row.hits, row.hit_pct, row.runtime,
            lw = lw, sw = sw, hw = hw, pw = pw, rw = rw);
    }

    Ok(())
}

// ── Stat printers ─────────────────────────────────────────────────────────────

fn print_raw_stats(counts: &[u32], shape: Option<[u32; 2]>) {
    let n       = counts.len() as u64;
    let total: u64 = counts.iter().map(|&x| x as u64).sum();
    let max_val = counts.iter().copied().max().unwrap_or(0);
    let nonzero = counts.iter().filter(|&&x| x > 0).count() as u64;

    let mean_all = total as f64 / n as f64;
    let mean_nz  = if nonzero > 0 { total as f64 / nonzero as f64 } else { 0.0 };
    let dyn_range = if mean_nz > 0.0 { max_val as f64 / mean_nz } else { 0.0 };
    let variance = counts.iter()
        .map(|&x| { let d = x as f64 - mean_all; d * d }).sum::<f64>() / n as f64;
    let std = variance.sqrt();

    if let Some(s) = shape {
        field("Grid", &format!("{}×{}  ({} pixels)", s[0], s[1], fmt_int(s[0] as u64 * s[1] as u64)));
    }
    field("Total counts",    &fmt_int(total));
    field("Non-zero",        &format!("{}  ({:.1}%)", fmt_int(nonzero), nonzero as f64 / n as f64 * 100.0));
    field("Max count",       &max_val.to_string());
    field("Mean (all)",      &format!("{:.3}", mean_all));
    field("Mean (non-zero)", &format!("{:.3}", mean_nz));
    field("Std",             &format!("{:.3}", std));
    if dyn_range > 0.0 {
        field("Dynamic range", &format!("{:.0}×", dyn_range));
    }
}

fn print_processed_stats(counts: &[f32], shape: Option<[u32; 2]>) {
    let n = counts.len() as u64;
    let total: f64 = counts.iter().map(|&x| x as f64).sum();
    let max_val = counts.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let nonzero = counts.iter().filter(|&&x| x > 0.0).count() as u64;

    let mean_all = total / n as f64;
    let nz_sum: f64 = counts.iter().filter(|&&x| x > 0.0).map(|&x| x as f64).sum();
    let mean_nz = if nonzero > 0 { nz_sum / nonzero as f64 } else { 0.0 };
    let dyn_range = if mean_nz > 0.0 { max_val as f64 / mean_nz } else { 0.0 };
    let variance = counts.iter()
        .map(|&x| { let d = x as f64 - mean_all; d * d }).sum::<f64>() / n as f64;
    let std = variance.sqrt();

    if let Some(s) = shape {
        field("Grid", &format!("{}×{}  ({} pixels)", s[0], s[1], fmt_int(s[0] as u64 * s[1] as u64)));
    }
    field("Total signal",    &format!("{:.1}", total));
    field("Non-zero",        &format!("{}  ({:.1}%)", fmt_int(nonzero), nonzero as f64 / n as f64 * 100.0));
    field("Max",             &format!("{:.2}", max_val));
    field("Mean (all)",      &format!("{:.4}", mean_all));
    field("Mean (non-zero)", &format!("{:.4}", mean_nz));
    field("Std",             &format!("{:.4}", std));
    if dyn_range > 0.0 {
        field("Dynamic range", &format!("{:.0}×", dyn_range));
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn read_metadata(run_dir: &Path) -> Result<crate::run_dir::RunMetadata> {
    let path = run_dir.join("metadata.json");
    let s = std::fs::read_to_string(&path)
        .with_context(|| format!("Cannot read {:?}", path))?;
    serde_json::from_str(&s).context("Failed to parse metadata.json")
}

fn validate_pixel_count(actual: usize, shape: Option<[u32; 2]>, file: &str) -> Result<()> {
    if let Some(s) = shape {
        let expected = (s[0] as usize) * (s[1] as usize);
        if actual != expected {
            anyhow::bail!(
                "{}: {} values but metadata shape {}×{} expects {}",
                file, actual, s[0], s[1], expected
            );
        }
    }
    Ok(())
}

/// Print a label: value line with consistent column alignment.
fn field(label: &str, value: &str) {
    println!("{:<16} {}", format!("{}:", label), value);
}

/// Format an integer with thousands separators.
fn fmt_int(n: u64) -> String {
    let s = n.to_string();
    let mut out = Vec::with_capacity(s.len() + s.len() / 3);
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 { out.push(','); }
        out.push(c);
    }
    out.iter().rev().collect()
}
