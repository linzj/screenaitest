use anyhow::Result;
use std::path::Path;
use tflitec::interpreter::{Interpreter, Options};
use tflitec::model::Model;

use crate::utils::BBox;

/// ClusterSort model for determining if two boxes belong to the same line
/// Based on IDA analysis of chrome_screen_ai.dll cluster_sort model
pub struct ClusterSort {
    model: Model<'static>,
}

impl ClusterSort {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let model_path = model_dir
            .join("gocr")
            .join("layout")
            .join("cluster_sort")
            .join("model_v2.tflite");

        if !model_path.exists() {
            return Err(anyhow::anyhow!(
                "ClusterSort model not found: {}",
                model_path.display()
            ));
        }

        let model = Model::new(model_path.to_str().unwrap())?;

        Ok(Self { model })
    }

    /// Check if two boxes belong to the same line
    /// Returns true if same line, false if different lines
    pub fn is_same_line(&self, box1: &BBox, box2: &BBox, image_size: (f32, f32)) -> Result<bool> {
        let mut options = Options::default();
        options.is_xnnpack_enabled = true;
        let interpreter = Interpreter::new(&self.model, Some(options))?;
        interpreter.allocate_tensors()?;

        // Prepare interleaved features (30 dimensions)
        let features = self.prepare_features(box1, box2, image_size);

        // Set inputs based on actual model structure:
        // Input 0: [1] int64 = 0
        // Input 1: [1] int64 = 0
        // Input 2: [1, 30] float32 features
        let zeros_i64: [i64; 1] = [0];

        let input0 = interpreter.input(0)?;
        input0.set_data(&zeros_i64)?;

        let input1 = interpreter.input(1)?;
        input1.set_data(&zeros_i64)?;

        let input2 = interpreter.input(2)?;
        input2.set_data(&features)?;

        // Run inference
        interpreter.invoke()?;

        // Get output
        let output = interpreter.output(0)?;
        let output_data: &[f32] = output.data();

        // output[0] < 0.5 means same line
        // output[0] > 1.0 means different line
        Ok(output_data[0] < 0.5)
    }

    /// Prepare interleaved features for two boxes
    /// Format: [box1_feat[0], box2_feat[0], box1_feat[1], box2_feat[1], ...]
    /// Each box has 15 features: [x1, y1, x2, y2, cx, cy, w, h, area, aspect, conf, 0, 0, 0, 0]
    fn prepare_features(&self, box1: &BBox, box2: &BBox, image_size: (f32, f32)) -> Vec<f32> {
        let (img_w, img_h) = image_size;

        let feat1 = self.box_to_features(box1, img_w, img_h);
        let feat2 = self.box_to_features(box2, img_w, img_h);

        // Interleave features
        let mut result = vec![0.0f32; 30];
        for i in 0..15 {
            result[i * 2] = feat1[i];
            result[i * 2 + 1] = feat2[i];
        }

        result
    }

    /// Convert a box to 15-dimensional feature vector
    /// Based on IDA analysis, coordinates should be normalized to [0, 1] range
    fn box_to_features(&self, b: &BBox, img_w: f32, img_h: f32) -> [f32; 15] {
        let (cx, cy) = b.center();
        let w = b.width();
        let h = b.height();
        let area = w * h;
        let aspect = if h > 0.0 { w / h } else { 1.0 };

        // Normalize coordinates to [0, 1]
        // For image_size = 4096, this means dividing by 4096
        [
            b.x1 / img_w,           // x1 normalized
            b.y1 / img_h,           // y1 normalized
            b.x2 / img_w,           // x2 normalized
            b.y2 / img_h,           // y2 normalized
            cx / img_w,             // center x normalized
            cy / img_h,             // center y normalized
            w / img_w,              // width normalized
            h / img_h,              // height normalized
            area / (img_w * img_h), // area normalized
            aspect,                 // aspect ratio (not normalized)
            b.conf,                 // confidence (not normalized)
            0.0,
            0.0,
            0.0,
            0.0, // padding
        ]
    }
}
