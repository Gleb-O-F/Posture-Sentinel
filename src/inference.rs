use ort::{Session, SessionBuilder, Value, Tensor, ArrayExt};
use anyhow::Result;
use opencv::core::Mat;

pub struct PoseDetector {
    session: Session,
    input_width: i32,
    input_height: i32,
}

impl PoseDetector {
    pub fn new(model_path: &str) -> Result<Self> {
        let session = SessionBuilder::new()?
            .with_execution_providers([ort::CUDAExecutionProvider::default().build()])?
            .with_optimization_level(ort::GraphOptimizationLevel::All)?
            .commit_from_file(model_path)?;

        let (input_width, input_height) = Self::get_input_dimensions(&session)?;

        Ok(Self {
            session,
            input_width,
            input_height,
        })
    }

    fn get_input_dimensions(session: &Session) -> Result<(i32, i32)> {
        let input_names = session.input_names()?;
        let input_name = input_names.get(0).ok_or_else(|| {
            anyhow::anyhow!("No input names found")
        })?;

        let input_shape = session.input_shape(input_name.as_str())?;
        
        if input_shape.len() >= 4 {
            let height = input_shape[2] as i32;
            let width = input_shape[3] as i32;
            Ok((width, height))
        } else {
            Ok((256, 256))
        }
    }

    pub fn detect(&self, frame: &Mat) -> Result<Vec<Landmark>> {
        let input_tensor = self.preprocess(frame)?;
        
        let outputs = self.session.run(vec![input_tensor])?;
        
        let landmarks = self.postprocess(&outputs)?;
        
        Ok(landmarks)
    }

    fn preprocess(&self, frame: &Mat) -> Result<Tensor<f32>> {
        use opencv::imgproc;
        use opencv::core;

        let mut rgb = Mat::default();
        opencv::imgproc::cvt_color(frame, &mut rgb, opencv::imgproc::COLOR_BGR2RGB, 0)?;

        let mut resized = Mat::default();
        opencv::imgproc::resize(
            &rgb, 
            &mut resized, 
            opencv::core::Size::new(self.input_width, self.input_height),
            0.0, 
            0.0, 
            opencv::imgproc::INTER_LINEAR
        )?;

        let mut float_img = Mat::default();
        resized.convert_to(&mut float_img, core::CV_32FC3, 1.0 / 255.0, 0.0)?;

        let h = self.input_height as usize;
        let w = self.input_width as usize;
        
        let mut data = vec![0.0f32; 1 * 3 * h * w];
        
        for y in 0..h {
            for x in 0..w {
                let pixel = resized.at_2d::<opencv::core::Vec3f>(y as i32, x as i32)?;
                let idx = y * w + x;
                data[idx] = pixel[0] / 255.0;
                data[h * w + idx] = pixel[1] / 255.0;
                data[2 * h * w + idx] = pixel[2] / 255.0;
            }
        }

        let tensor = Tensor::from_array(data, [1, 3, h as i64, w as i64])?;
        
        Ok(tensor)
    }

    fn postprocess(&self, outputs: &[Value]) -> Result<Vec<Landmark>> {
        let output_name = self.session.output_names()?.get(0).ok_or_else(|| {
            anyhow::anyhow!("No output names found")
        })?;

        let output = outputs.get(0).ok_or_else(|| {
            anyhow::anyhow!("No output found")
        })?;

        let tensor: Tensor<f32> = output.try_extract()?;
        let shape = tensor.shape();
        
        let landmarks_count = if shape.len() >= 2 { shape[1] as usize } else { 33 };
        
        let mut landmarks = Vec::with_capacity(landmarks_count);
        
        for i in 0..landmarks_count {
            let x = if shape.len() >= 3 {
                tensor[[0, i, 0]]
            } else {
                0.0
            };
            let y = if shape.len() >= 3 {
                tensor[[0, i, 1]]
            } else {
                0.0
            };
            let z = if shape.len() >= 3 && shape[2] > 2 {
                tensor[[0, i, 2]]
            } else {
                0.0
            };
            
            let visibility = if shape.len() >= 3 && shape[2] > 3 {
                tensor[[0, i, 3]]
            } else {
                1.0
            };
            
            landmarks.push(Landmark {
                x, y, z, visibility,
            });
        }

        Ok(landmarks)
    }
}

#[derive(Debug, Clone)]
pub struct Landmark {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub visibility: f32,
}
