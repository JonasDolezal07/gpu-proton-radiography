//! Reproducible run directory management.
//!
//! A run directory is a self-contained record of one simulation run:
//! inputs, outputs, provenance, and logs all in one place.
//!
//! Layout:
//!   <run_dir>/
//!     input_deck.toml        ← copy of the deck actually used
//!     resolved_config.json   ← fully-resolved SI config
//!     metadata.json          ← provenance, hardware, diagnostics
//!     log.txt                ← mirror of terminal output
//!     counts/
//!       raw_counts.bin       ← u32 little-endian [H×W] row-major
//!       processed_counts.bin ← f32 little-endian [H×W] row-major
//!     images/
//!       radiograph.png
//!     tables/
//!       hits.csv             ← optional

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::{Mutex, OnceLock};

// ── Directory management ──────────────────────────────────────────────────────

pub struct RunOptions {
    pub overwrite: bool,
    pub resume: bool,
}

impl Default for RunOptions {
    fn default() -> Self {
        Self { overwrite: false, resume: false }
    }
}

pub struct RunDir {
    root: PathBuf,
}

impl RunDir {
    /// Open or create a run directory.
    ///
    /// - Non-existent: create it.
    /// - Exists and empty: use it.
    /// - Exists and non-empty + `overwrite`: remove and recreate.
    /// - Exists and non-empty + `resume`: reuse, create missing subdirs.
    /// - Exists and non-empty + neither: return an error.
    pub fn open(root: PathBuf, opts: &RunOptions) -> Result<Self> {
        if root.exists() {
            let non_empty = root
                .read_dir()
                .with_context(|| format!("Cannot list {:?}", root))?
                .next()
                .is_some();

            if non_empty {
                if opts.overwrite {
                    std::fs::remove_dir_all(&root)
                        .with_context(|| format!("Cannot remove {:?}", root))?;
                    log::info!("Overwriting existing run directory: {:?}", root);
                } else if opts.resume {
                    log::warn!("Resuming into non-empty run directory: {:?}", root);
                    let rd = Self { root };
                    rd.create_subdirs()?;
                    return Ok(rd);
                } else {
                    anyhow::bail!(
                        "Output directory {:?} already exists and is not empty.\n\
                         Use --overwrite to replace it, or choose a new directory.",
                        root
                    );
                }
            }
        }

        std::fs::create_dir_all(&root)
            .with_context(|| format!("Cannot create run directory: {:?}", root))?;

        let rd = Self { root };
        rd.create_subdirs()?;
        Ok(rd)
    }

    fn create_subdirs(&self) -> Result<()> {
        for sub in ["counts", "images", "tables"] {
            std::fs::create_dir_all(self.root.join(sub))?;
        }
        Ok(())
    }

    // ── Path accessors ────────────────────────────────────────────────────────

    pub fn root(&self) -> &Path { &self.root }
    pub fn counts_dir(&self)   -> PathBuf { self.root.join("counts") }
    pub fn images_dir(&self)   -> PathBuf { self.root.join("images") }
    pub fn tables_dir(&self)   -> PathBuf { self.root.join("tables") }

    pub fn raw_counts_path(&self)       -> PathBuf { self.counts_dir().join("raw_counts.bin") }
    pub fn processed_counts_path(&self) -> PathBuf { self.counts_dir().join("processed_counts.bin") }
    pub fn hits_bin_path(&self)         -> PathBuf { self.counts_dir().join("hits.bin") }
    pub fn radiograph_png_path(&self)   -> PathBuf { self.images_dir().join("radiograph.png") }
    pub fn hits_csv_path(&self)         -> PathBuf { self.tables_dir().join("hits.csv") }
    pub fn resolved_config_path(&self)  -> PathBuf { self.root.join("resolved_config.json") }
    pub fn metadata_path(&self)         -> PathBuf { self.root.join("metadata.json") }
    pub fn log_path(&self)              -> PathBuf { self.root.join("log.txt") }

    pub fn input_deck_path(&self, original_ext: &str) -> PathBuf {
        self.root.join(format!("input_deck.{}", original_ext))
    }

    // ── Metadata I/O ──────────────────────────────────────────────────────────

    pub fn write_metadata(&self, meta: &RunMetadata) -> Result<()> {
        let json = serde_json::to_string_pretty(meta)
            .context("Failed to serialise metadata")?;
        std::fs::write(self.metadata_path(), json)
            .context("Failed to write metadata.json")?;
        Ok(())
    }

    // ── Logging ───────────────────────────────────────────────────────────────

    /// Install global TeeLogger (stderr only), then attach file sink to log.txt.
    pub fn init_logging(&self) -> Result<()> {
        init_global_logger()?;
        attach_log_tee(&self.log_path())
    }
}

// ── Tee logger ────────────────────────────────────────────────────────────────

static LOGGER: OnceLock<TeeLogger> = OnceLock::new();

struct TeeLogger {
    inner: env_logger::Logger,
    /// Swappable file sink. None = stderr only.
    file: Mutex<Option<BufWriter<File>>>,
}

impl log::Log for TeeLogger {
    fn enabled(&self, metadata: &log::Metadata) -> bool {
        self.inner.enabled(metadata)
    }

    fn log(&self, record: &log::Record) {
        self.inner.log(record);
        if self.enabled(record.metadata()) {
            if let Ok(mut guard) = self.file.lock() {
                if let Some(ref mut w) = *guard {
                    let _ = writeln!(w, "[{}] {} — {}", record.level(), record.target(), record.args());
                    let _ = w.flush();
                }
            }
        }
    }

    fn flush(&self) {
        self.inner.flush();
        if let Ok(mut guard) = self.file.lock() {
            if let Some(ref mut w) = *guard {
                let _ = w.flush();
            }
        }
    }
}

/// Install the global TeeLogger (stderr only, no file sink).
///
/// Safe to call multiple times; only acts on the first call.
/// Must be called before any `env_logger::init()` or other logger install.
pub fn init_global_logger() -> Result<()> {
    if LOGGER.get().is_some() {
        return Ok(());
    }
    let env_logger = env_logger::Builder::from_env(
        env_logger::Env::default().default_filter_or("info"),
    )
    .build();
    let level = env_logger.filter();
    let _ = LOGGER.set(TeeLogger { inner: env_logger, file: Mutex::new(None) });
    log::set_logger(LOGGER.get().unwrap())
        .map_err(|e| anyhow::anyhow!("Logger install failed: {}", e))?;
    log::set_max_level(level);
    Ok(())
}

/// Attach (or replace) the log file sink. Subsequent log lines are also written
/// to `path` (append mode). Safe to call per-run to redirect to a new file.
pub fn attach_log_tee(path: &Path) -> Result<()> {
    let tee = LOGGER.get()
        .ok_or_else(|| anyhow::anyhow!("Call init_global_logger() first"))?;
    let f = OpenOptions::new()
        .create(true).append(true).open(path)
        .with_context(|| format!("Failed to open log file: {:?}", path))?;
    *tee.file.lock().unwrap() = Some(BufWriter::new(f));
    Ok(())
}

/// Remove the current log file sink (revert to stderr only).
pub fn detach_log_tee() {
    if let Some(tee) = LOGGER.get() {
        *tee.file.lock().unwrap() = None;
    }
}

// ── Metadata structs ──────────────────────────────────────────────────────────

#[derive(Debug, Serialize, Deserialize)]
pub struct RunMetadata {
    pub metadata_schema_version: u32,
    pub run: RunRecord,
    pub code: CodeInfo,
    pub hardware: HardwareInfo,
    pub input_files: InputFiles,
    pub outputs: OutputFiles,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub counts_format: Option<CountsFormat>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub render: Option<RenderProvenance>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub diagnostics: Option<RunDiagnostics>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub performance: Option<PerfInfo>,
    pub warnings: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub cli_overrides: Option<Vec<CliOverrideRecord>>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RunRecord {
    pub run_name: String,
    pub status: String,
    pub argv: Vec<String>,
    pub started_at: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub completed_at: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CodeInfo {
    pub name: String,
    pub version: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub git_commit: Option<String>,
    pub git_dirty: bool,
    pub build_profile: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HardwareInfo {
    pub hostname: String,
    pub os: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vulkan_api_version: Option<String>,
    pub backend: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct InputFiles {
    pub deck_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub field_sha256: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub e_field_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub e_field_sha256: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OutputFiles {
    pub resolved_config: String,
    pub log: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_deck: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub raw_counts: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub processed_counts: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hits_bin: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub radiograph_png: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hits_csv: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CountsFormat {
    pub raw: CountsBinarySpec,
    pub processed: CountsBinarySpec,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CountsBinarySpec {
    pub dtype: String,
    pub endianness: String,
    pub shape: [u32; 2],
    pub layout: String,
    pub row_axis: String,
    pub col_axis: String,
    pub units: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RenderProvenance {
    pub source: String,
    pub scale: String,
    pub colormap: String,
    pub gamma: f32,
    pub exposure: f32,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct RunDiagnostics {
    pub n_particles: u32,
    pub n_hits: u64,
    pub hit_fraction: f64,
    pub mean_y_m: f64,
    pub std_y_m: f64,
    pub mean_z_m: f64,
    pub std_z_m: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct PerfInfo {
    pub total_runtime_s: f64,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct CliOverrideRecord {
    pub key: String,
    pub value: String,
}

// ── Constructor ───────────────────────────────────────────────────────────────

impl RunMetadata {
    pub fn new_running(run_name: String, deck_path: String, argv: Vec<String>) -> Self {
        let started_at = chrono::Utc::now().to_rfc3339();

        let hostname = std::process::Command::new("hostname")
            .output()
            .ok()
            .and_then(|o| String::from_utf8(o.stdout).ok())
            .map(|s| s.trim().to_string())
            .unwrap_or_else(|| "unknown".to_string());

        Self {
            metadata_schema_version: 1,
            run: RunRecord {
                run_name,
                status: "running".to_string(),
                argv,
                started_at,
                completed_at: None,
            },
            code: CodeInfo {
                name: env!("CARGO_PKG_NAME").to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
                git_commit: option_env!("GIT_COMMIT").map(str::to_string),
                git_dirty: option_env!("GIT_DIRTY").map(|s| s == "true").unwrap_or(false),
                build_profile: if cfg!(debug_assertions) { "debug" } else { "release" }.to_string(),
            },
            hardware: HardwareInfo {
                hostname,
                os: std::env::consts::OS.to_string(),
                gpu: None,
                vulkan_api_version: None,
                backend: "Vulkan".to_string(),
            },
            input_files: InputFiles {
                deck_path,
                field_path: None,
                field_sha256: None,
                e_field_path: None,
                e_field_sha256: None,
            },
            outputs: OutputFiles {
                resolved_config: "resolved_config.json".to_string(),
                log: "log.txt".to_string(),
                input_deck: None,
                raw_counts: None,
                processed_counts: None,
                hits_bin: None,
                radiograph_png: None,
                hits_csv: None,
            },
            counts_format: None,
            render: None,
            diagnostics: None,
            performance: None,
            warnings: Vec::new(),
            cli_overrides: None,
        }
    }
}

// ── SHA-256 helper ────────────────────────────────────────────────────────────

pub fn sha256_file(path: &Path) -> Result<String> {
    use sha2::{Digest, Sha256};
    let bytes = std::fs::read(path)
        .with_context(|| format!("Cannot read {:?} for SHA-256", path))?;
    let hash = Sha256::digest(&bytes);
    Ok(format!("{:x}", hash))
}
