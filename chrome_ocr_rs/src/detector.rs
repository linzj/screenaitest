use anyhow::{anyhow, Result};
use image::{GrayImage, ImageBuffer, Luma};
use std::path::Path;
use tflitec::interpreter::{Interpreter, Options};
use tflitec::model::Model;

use crate::utils::BBox;

const TARGET_SIZE: u32 = 4096;

pub struct TextDetector {
    model: Model<'static>,
    #[allow(dead_code)]
    input_sizes: Vec<usize>,
    pub scale: f32,
    pub offset_x: f32,
    pub offset_y: f32,
}

impl TextDetector {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let model_path = model_dir
            .join("gocr")
            .join("gocr_models")
            .join("detection")
            .join("gocr_group_rpn_text_detection_model_2024_q4.tflite");

        if !model_path.exists() {
            return Err(anyhow!(
                "Detection model not found: {}",
                model_path.display()
            ));
        }

        let model = Model::new(model_path.to_str().unwrap())?;

        // Create a temporary interpreter to get input sizes
        let input_sizes = {
            let options = Options::default();
            let interpreter = Interpreter::new(&model, Some(options))?;
            interpreter.allocate_tensors()?;

            let mut sizes: Vec<usize> = Vec::new();
            let input_count = interpreter.input_tensor_count();
            for idx in 0..input_count {
                let tensor = interpreter.input(idx)?;
                let shape = tensor.shape();
                if shape.rank() >= 2 {
                    sizes.push(shape.dimensions()[1]);
                }
            }
            sizes.sort();
            sizes
        };

        println!("  TextDetector: scales {:?}", input_sizes);

        Ok(Self {
            model,
            input_sizes,
            scale: 1.0,
            offset_x: 0.0,
            offset_y: 0.0,
        })
    }

    /// Preprocess image to 4096x4096 grayscale
    fn preprocess(&mut self, image: &GrayImage) -> GrayImage {
        let (w, h) = (image.width(), image.height());

        // Calculate scale to fit in 4096x4096
        let max_dim = w.max(h);
        self.scale = TARGET_SIZE as f32 / max_dim as f32;
        let new_w = (w as f32 * self.scale) as u32;
        let new_h = (h as f32 * self.scale) as u32;

        // Center the image
        self.offset_x = (TARGET_SIZE - new_w) as f32 / 2.0;
        self.offset_y = (TARGET_SIZE - new_h) as f32 / 2.0;

        // Create 4096x4096 canvas with white background
        let mut canvas: GrayImage =
            ImageBuffer::from_pixel(TARGET_SIZE, TARGET_SIZE, Luma([255u8]));

        // Resize and paste
        let resized =
            image::imageops::resize(image, new_w, new_h, image::imageops::FilterType::Lanczos3);

        image::imageops::overlay(
            &mut canvas,
            &resized,
            self.offset_x as i64,
            self.offset_y as i64,
        );

        canvas
    }

    /// Detect text regions in image
    pub fn detect(&mut self, image: &GrayImage, threshold: f32) -> Result<Vec<BBox>> {
        let preprocessed = self.preprocess(image);

        // Create interpreter for this detection with multi-threading
        let mut options = Options::default();
        options.thread_count = 4; // Use 4 threads
        let interpreter = Interpreter::new(&self.model, Some(options))?;
        interpreter.allocate_tensors()?;

        // Set multi-scale inputs - match by tensor size
        let input_count = interpreter.input_tensor_count();
        for idx in 0..input_count {
            let tensor = interpreter.input(idx)?;
            let shape = tensor.shape();
            let dims = shape.dimensions();

            // Get the size from tensor shape (should be [1, size, size, 1])
            if dims.len() >= 2 {
                let size = dims[1];

                let resized = image::imageops::resize(
                    &preprocessed,
                    size as u32,
                    size as u32,
                    image::imageops::FilterType::Triangle,
                );

                let input_data: Vec<u8> = resized.as_raw().to_vec();
                tensor.set_data(&input_data)?;
            }
        }

        // Run inference
        interpreter.invoke()?;

        // Parse outputs - look for feature maps with shape [1, H, W, 7]
        let mut boxes = Vec::new();
        let output_count = interpreter.output_tensor_count();

        for output_idx in 0..output_count {
            let tensor = interpreter.output(output_idx)?;
            let shape = tensor.shape();
            let dims = shape.dimensions();

            if dims.len() == 4 && dims[3] == 7 {
                let feat_h = dims[1];
                let feat_w = dims[2];

                // Get output data as f32
                let data: &[f32] = tensor.data();

                for y in 0..feat_h {
                    for x in 0..feat_w {
                        let idx = (y * feat_w + x) * 7;
                        let conf = data[idx];

                        if conf > threshold {
                            // Calculate center position in 4096 space
                            let cx = (x as f32 + 0.5) * TARGET_SIZE as f32 / feat_w as f32;
                            let cy = (y as f32 + 0.5) * TARGET_SIZE as f32 / feat_h as f32;
                            let bw = TARGET_SIZE as f32 / feat_w as f32 * 1.2;
                            let bh = TARGET_SIZE as f32 / feat_h as f32 * 1.2;

                            // Check if within content area
                            let scaled_w = image.width() as f32 * self.scale;
                            let scaled_h = image.height() as f32 * self.scale;

                            if cx > self.offset_x
                                && cx < self.offset_x + scaled_w
                                && cy > self.offset_y
                                && cy < self.offset_y + scaled_h
                            {
                                boxes.push(BBox::new(
                                    cx - bw / 2.0,
                                    cy - bh / 2.0,
                                    cx + bw / 2.0,
                                    cy + bh / 2.0,
                                    conf,
                                ));
                            }
                        }
                    }
                }
            }
        }

        // NMS to remove duplicates
        boxes.sort_by(|a, b| {
            b.conf
                .partial_cmp(&a.conf)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut kept = Vec::new();
        for bbox in boxes {
            let dominated = kept
                .iter()
                .any(|k: &BBox| crate::utils::calc_iou(&bbox.as_array(), &k.as_array()) > 0.3);
            if !dominated {
                kept.push(bbox);
            }
        }

        Ok(kept)
    }
}
