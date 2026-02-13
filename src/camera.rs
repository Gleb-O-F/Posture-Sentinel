use opencv::{
    prelude::*,
    videoio::{self, VideoCapture, VideoCaptureAPIs},
};
use anyhow::Result;

pub struct Camera {
    capture: VideoCapture,
}

impl Camera {
    pub fn new(device_id: i32) -> Result<Self> {
        let mut capture = VideoCapture::new(device_id, VideoCaptureAPIs::CAP_ANY as i32)?;
        
        if !capture.is_opened()? {
            anyhow::bail!("Failed to open camera");
        }
        
        Ok(Self { capture })
    }

    pub fn read_frame(&mut self) -> Result<Mat> {
        let mut frame = Mat::default();
        self.capture.read(&mut frame)?;
        
        if frame.empty() {
            anyhow::bail!("Empty frame");
        }
        
        Ok(frame)
    }
}
