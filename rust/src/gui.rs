//! GUI module — deck launcher and run viewer.
//!
//! The GUI is a read-only launcher. It does not edit simulation parameters.
//! Parameter editing happens in deck files (.toml / .json).

use std::path::PathBuf;
use egui::{Context, Ui, Window, ComboBox};

// ── Preset catalogue ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum Preset {
    None,
    // Instabilities
    ZPinch,
    SausageWeak,
    SausageStrong,
    KinkWeak,
    KinkStrong,
    Mixed,
    // MAGPIE
    MagpieWires,
    MagpieStagnationMild,
    MagpieStagnationStrong,
    MagpieStagnation3MeV,
    MagpieParallel,
    // Custom file
    CustomFile(String),
}

impl Preset {
    pub fn config_path(&self) -> Option<String> {
        match self {
            Preset::None => None,
            Preset::ZPinch => Some("data/instabilities/zpinch.json".to_string()),
            Preset::SausageWeak => Some("data/instabilities/sausage_weak.json".to_string()),
            Preset::SausageStrong => Some("data/instabilities/sausage_strong.json".to_string()),
            Preset::KinkWeak => Some("data/instabilities/kink_weak.json".to_string()),
            Preset::KinkStrong => Some("data/instabilities/kink_strong.json".to_string()),
            Preset::Mixed => Some("data/instabilities/mixed.json".to_string()),
            Preset::MagpieWires => Some("data/magpie/magpie_wires.json".to_string()),
            Preset::MagpieStagnationMild => Some("data/magpie/magpie_stagnation_mild.json".to_string()),
            Preset::MagpieStagnationStrong => Some("data/magpie/magpie_stagnation_strong.json".to_string()),
            Preset::MagpieStagnation3MeV => Some("data/magpie/magpie_stagnation_3MeV.json".to_string()),
            Preset::MagpieParallel => Some("data/magpie/magpie_parallel.json".to_string()),
            Preset::CustomFile(path) => Some(path.clone()),
        }
    }

    fn display_name(&self) -> &str {
        match self {
            Preset::None => "-- Select preset --",
            Preset::ZPinch => "Z-Pinch",
            Preset::SausageWeak => "Sausage (weak)",
            Preset::SausageStrong => "Sausage (strong)",
            Preset::KinkWeak => "Kink (weak)",
            Preset::KinkStrong => "Kink (strong)",
            Preset::Mixed => "Mixed instabilities",
            Preset::MagpieWires => "MAGPIE Wires",
            Preset::MagpieStagnationMild => "MAGPIE Stagnation (mild)",
            Preset::MagpieStagnationStrong => "MAGPIE Stagnation (strong)",
            Preset::MagpieStagnation3MeV => "MAGPIE Stagnation (3 MeV)",
            Preset::MagpieParallel => "MAGPIE Parallel Beam",
            Preset::CustomFile(_) => "Custom file",
        }
    }
}

// ── Run state ─────────────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq)]
pub enum RunState {
    /// No deck loaded.
    Idle,
    /// Deck is being parsed / GPU resources allocated.
    Preparing,
    /// Simulation is actively stepping particles.
    Running,
    /// All particles accounted for; outputs written to disk.
    Complete(PathBuf),
    /// Something went wrong.
    Failed(String),
}

impl RunState {
    pub fn is_active(&self) -> bool {
        matches!(self, RunState::Preparing | RunState::Running)
    }
}

// ── Deck display ──────────────────────────────────────────────────────────────

/// Key params from a loaded deck, shown read-only before and during a run.
#[derive(Debug, Clone)]
pub struct DeckDisplay {
    pub deck_name: String,
    pub n_particles: u32,
    pub energy_mev: f64,
    pub dt_ps: f64,
    pub dt_auto: bool,
    pub max_steps: u32,
    pub field_file: String,
    pub geometry: String,
    pub detector_center_mm: Option<[f64; 3]>,
    pub detector_size_mm: [f64; 2],  // [width, height]
}

// ── Gui ───────────────────────────────────────────────────────────────────────

pub struct Gui {
    pub visible: bool,

    // Deck selection
    pub selected_preset: Preset,
    custom_file_path: String,

    // Current run state
    pub run_state: RunState,

    // Resolved params from the last loaded deck
    deck_display: Option<DeckDisplay>,

    // Output directory the user chose
    pub output_dir: String,

    // Live progress counters (updated each frame while Running)
    progress_hits: u32,
    progress_exits: u32,
    progress_n_particles: u32,

    // Pending actions consumed by App
    deck_load_request: Option<String>,   // path to preview-load
    run_start_request: Option<String>,   // output_dir → triggers full run

    pub export_requested: bool,
}

impl Gui {
    pub fn new() -> Self {
        Self {
            visible: true,
            selected_preset: Preset::None,
            custom_file_path: String::new(),
            run_state: RunState::Idle,
            deck_display: None,
            output_dir: "output/run".to_string(),
            progress_hits: 0,
            progress_exits: 0,
            progress_n_particles: 0,
            deck_load_request: None,
            run_start_request: None,
            export_requested: false,
        }
    }

    // ── App → GUI ─────────────────────────────────────────────────────────────

    pub fn set_deck_display(&mut self, display: DeckDisplay) {
        // Auto-generate output dir from deck name + timestamp
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        self.output_dir = format!("output/{}_{}", display.deck_name, ts);
        self.deck_display = Some(display);
        // If we were Preparing just for a preview, go back to Idle
        if self.run_state == RunState::Preparing {
            self.run_state = RunState::Idle;
        }
    }

    pub fn set_run_state(&mut self, state: RunState) {
        self.run_state = state;
    }

    pub fn update_progress(&mut self, hits: u32, exits: u32, n_particles: u32) {
        self.progress_hits = hits;
        self.progress_exits = exits;
        self.progress_n_particles = n_particles;
    }

    // ── GUI → App ─────────────────────────────────────────────────────────────

    /// Take pending deck-preview request (App should call SimConfig::load and set_deck_display).
    pub fn take_deck_load_request(&mut self) -> Option<String> {
        self.deck_load_request.take()
    }

    /// Take pending run-start request. Returns the chosen output directory.
    pub fn take_run_start_request(&mut self) -> Option<String> {
        self.run_start_request.take()
    }

    pub fn take_export_request(&mut self) -> bool {
        std::mem::take(&mut self.export_requested)
    }

    /// Deck path for the currently selected preset (for use by App).
    pub fn selected_config_path(&self) -> Option<String> {
        self.selected_preset.config_path()
    }

    // ── Render ────────────────────────────────────────────────────────────────

    pub fn render(&mut self, ctx: &Context) -> bool {
        if !self.visible { return false; }

        let mut wants_input = false;
        Window::new("Proton Radiography")
            .default_width(300.0)
            .max_width(340.0)
            .resizable(true)
            .show(ctx, |ui| {
                wants_input = true;
                self.render_ui(ui);
            });
        wants_input
    }

    fn render_ui(&mut self, ui: &mut Ui) {
        self.render_deck_selector(ui);

        if let Some(display) = &self.deck_display.clone() {
            ui.separator();
            self.render_deck_info(ui, display);
        }

        ui.separator();
        self.render_run_section(ui);

        match &self.run_state.clone() {
            RunState::Preparing => {
                ui.separator();
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Preparing...");
                });
            }
            RunState::Running => {
                ui.separator();
                self.render_progress(ui);
            }
            RunState::Complete(path) => {
                let path = path.clone();
                ui.separator();
                self.render_complete(ui, &path);
            }
            RunState::Failed(err) => {
                let err = err.clone();
                ui.separator();
                ui.colored_label(egui::Color32::RED, format!("Error: {}", err));
            }
            RunState::Idle => {}
        }
    }

    fn render_deck_selector(&mut self, ui: &mut Ui) {
        let old_preset = self.selected_preset.clone();

        ui.horizontal(|ui| {
            ui.label("Deck:");
            ComboBox::from_id_salt("preset_selector")
                .selected_text(self.selected_preset.display_name())
                .show_ui(ui, |ui| {
                    ui.selectable_value(&mut self.selected_preset, Preset::None, "-- Select preset --");
                    ui.separator();
                    ui.label("Instabilities");
                    ui.selectable_value(&mut self.selected_preset, Preset::ZPinch, "Z-Pinch");
                    ui.selectable_value(&mut self.selected_preset, Preset::SausageWeak, "Sausage (weak)");
                    ui.selectable_value(&mut self.selected_preset, Preset::SausageStrong, "Sausage (strong)");
                    ui.selectable_value(&mut self.selected_preset, Preset::KinkWeak, "Kink (weak)");
                    ui.selectable_value(&mut self.selected_preset, Preset::KinkStrong, "Kink (strong)");
                    ui.selectable_value(&mut self.selected_preset, Preset::Mixed, "Mixed instabilities");
                    ui.separator();
                    ui.label("MAGPIE");
                    ui.selectable_value(&mut self.selected_preset, Preset::MagpieWires, "Wires");
                    ui.selectable_value(&mut self.selected_preset, Preset::MagpieStagnationMild, "Stagnation (mild)");
                    ui.selectable_value(&mut self.selected_preset, Preset::MagpieStagnationStrong, "Stagnation (strong)");
                    ui.selectable_value(&mut self.selected_preset, Preset::MagpieStagnation3MeV, "Stagnation (3 MeV)");
                    ui.selectable_value(&mut self.selected_preset, Preset::MagpieParallel, "Parallel beam");
                    ui.separator();
                    if ui.selectable_label(
                        matches!(self.selected_preset, Preset::CustomFile(_)),
                        "Custom file..."
                    ).clicked() {
                        self.selected_preset = Preset::CustomFile(self.custom_file_path.clone());
                    }
                });
        });

        if matches!(self.selected_preset, Preset::CustomFile(_)) {
            ui.horizontal(|ui| {
                ui.label("Path:");
                if ui.text_edit_singleline(&mut self.custom_file_path).changed() {
                    self.selected_preset = Preset::CustomFile(self.custom_file_path.clone());
                }
            });
        }

        // Trigger preview load when preset changes
        if self.selected_preset != old_preset || matches!(self.selected_preset, Preset::CustomFile(_)) {
            if let Some(path) = self.selected_preset.config_path() {
                if !path.is_empty() && self.selected_preset != old_preset {
                    self.deck_display = None;
                    self.run_state = RunState::Preparing;
                    self.deck_load_request = Some(path);
                }
            }
        }
    }

    fn render_deck_info(&self, ui: &mut Ui, d: &DeckDisplay) {
        ui.label(egui::RichText::new("Deck").strong());
        egui::Grid::new("deck_info")
            .num_columns(2)
            .spacing([8.0, 2.0])
            .show(ui, |ui| {
                ui.label("Particles:");
                ui.label(format!("{}", d.n_particles));
                ui.end_row();

                ui.label("Energy:");
                ui.label(format!("{:.1} MeV", d.energy_mev));
                ui.end_row();

                ui.label("dt:");
                if d.dt_auto {
                    ui.label("auto");
                } else {
                    ui.label(format!("{:.3} ps", d.dt_ps));
                }
                ui.end_row();

                ui.label("Max steps:");
                ui.label(format!("{}", d.max_steps));
                ui.end_row();

                ui.label("Field:");
                let short = std::path::Path::new(&d.field_file)
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or(&d.field_file);
                ui.label(short);
                ui.end_row();

                ui.label("Source:");
                ui.label(&d.geometry);
                ui.end_row();

                if let Some(c) = d.detector_center_mm {
                    ui.label("Detector:");
                    ui.label(format!(
                        "({:.0}, {:.0}, {:.0}) mm  {:.0}×{:.0} mm",
                        c[0], c[1], c[2], d.detector_size_mm[0], d.detector_size_mm[1]
                    ));
                    ui.end_row();
                } else {
                    ui.label("Detector:");
                    ui.label(format!("{:.0}×{:.0} mm (auto-pos)", d.detector_size_mm[0], d.detector_size_mm[1]));
                    ui.end_row();
                }
            });
    }

    fn render_run_section(&mut self, ui: &mut Ui) {
        ui.horizontal(|ui| {
            ui.label("Output:");
            ui.text_edit_singleline(&mut self.output_dir);
        });

        ui.add_space(4.0);

        let can_run = self.deck_display.is_some() && !self.run_state.is_active();
        ui.horizontal(|ui| {
            if ui.add_enabled(can_run, egui::Button::new("Run")).clicked() {
                self.run_start_request = Some(self.output_dir.clone());
                self.run_state = RunState::Preparing;
                self.progress_hits = 0;
                self.progress_exits = 0;
            }
            if matches!(self.run_state, RunState::Complete(_)) {
                if ui.button("Export PNG").clicked() {
                    self.export_requested = true;
                }
            }
        });
    }

    fn render_progress(&self, ui: &mut Ui) {
        let n = self.progress_n_particles.max(1);
        let accounted = self.progress_hits.saturating_add(self.progress_exits);
        let pct = 100.0 * accounted as f32 / n as f32;

        ui.label(egui::RichText::new("Running").strong());
        let bar_rect = ui.available_rect_before_wrap();
        let bar_height = 6.0;
        ui.add(egui::ProgressBar::new(pct / 100.0).desired_width(bar_rect.width()));
        let _ = bar_height;

        egui::Grid::new("progress_grid")
            .num_columns(2)
            .spacing([8.0, 2.0])
            .show(ui, |ui| {
                ui.label("Hits:");
                ui.label(fmt_count(self.progress_hits));
                ui.end_row();
                ui.label("Exits:");
                ui.label(fmt_count(self.progress_exits));
                ui.end_row();
                ui.label("Progress:");
                ui.label(format!("{:.1}%  ({} / {})", pct, fmt_count(accounted), fmt_count(n)));
                ui.end_row();
            });
    }

    fn render_complete(&mut self, ui: &mut Ui, path: &std::path::Path) {
        ui.colored_label(egui::Color32::from_rgb(80, 200, 100), "Run complete");

        let path_str = path.display().to_string();
        ui.label(egui::RichText::new(&path_str).small().monospace());

        ui.horizontal(|ui| {
            if ui.button("Open in Finder").clicked() {
                #[cfg(target_os = "macos")]
                { let _ = std::process::Command::new("open").arg(path).spawn(); }
                #[cfg(target_os = "linux")]
                { let _ = std::process::Command::new("xdg-open").arg(path).spawn(); }
                #[cfg(target_os = "windows")]
                { let _ = std::process::Command::new("explorer").arg(path).spawn(); }
            }
            if ui.button("Run again").clicked() {
                self.run_state = RunState::Idle;
            }
        });
    }
}

fn fmt_count(n: impl Into<u64>) -> String {
    let n = n.into();
    if n >= 1_000_000 { format!("{:.2}M", n as f64 / 1e6) }
    else if n >= 1_000 { format!("{:.1}k", n as f64 / 1e3) }
    else { n.to_string() }
}
