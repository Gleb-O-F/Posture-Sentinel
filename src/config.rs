use serde::{Deserialize, Serialize};
use std::fs;
use anyhow::Result;

#[derive(Serialize, Deserialize)]
pub struct Config {
    pub camera_id: i32,
    pub model_path: String,
    pub threshold_slouch: f32,
    pub threshold_lean: f32,
    pub violation_timeout_secs: u64,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            camera_id: 0,
            model_path: "models/blazepose.onnx".to_string(),
            threshold_slouch: 0.05,
            threshold_lean: 0.15,
            violation_timeout_secs: 3,
        }
    }
}

impl Config {
    pub fn load(path: &str) -> Result<Self> {
        let content = fs::read_to_string(path)?;
        Ok(toml::from_str(&content)?)
    }

    pub fn save(&self, path: &str) -> Result<()> {
        let content = toml::to_string_pretty(self)?;
        fs::write(path, content)?;
        Ok(())
    }
}
