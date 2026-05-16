//! Data loaders for field and particle binary formats

mod field;
mod particles;
mod config;

pub use field::{FieldData, FieldBounds, EFieldSource};
pub use particles::{ParticleData, Particle};
pub use config::{SimConfig, SimSourceConfig, SimDetectorConfig, SimSourceGeometry};
