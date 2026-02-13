use winit::{
    event_loop::EventLoop,
    window::{Window, WindowBuilder},
};
use anyhow::Result;

pub struct BlurOverlay {
    window: Window,
}

impl BlurOverlay {
    pub fn new(event_loop: &EventLoop<()>) -> Result<Self> {
        let window = WindowBuilder::new()
            .with_transparent(true)
            .with_decorations(false)
            .build(event_loop)?;
        
        Ok(Self { window })
    }

    pub fn set_blur_intensity(&self, intensity: f32) {
        // TODO: Implement gradual blur effect
        // Use window-vibrancy or custom shader
    }
}
