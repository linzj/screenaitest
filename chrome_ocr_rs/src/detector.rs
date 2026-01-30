use anyhow::{anyhow, Result};
use image::{GrayImage, ImageBuffer, Luma};
use std::path::Path;
use tflitec::interpreter::{Interpreter, Options};
use tflitec::model::Model;

use crate::utils::BBox;

const TARGET_SIZE: u32 = 4096;

// Anchor values from protobuf config:
// gocr_group_rpn_text_detection_config_2024_q4_chrome.binarypb
// These are base box sizes in 4096x4096 coordinate space
const ANCHOR_WIDTHS: [f32; 6] = [16.0, 64.0, 16.0, 64.0, 64.0, 64.0];
const ANCHOR_HEIGHTS: [f32; 6] = [16.0, 64.0, 16.0, 64.0, 64.0, 64.0];

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
            let mut options = Options::default();
            options.is_xnnpack_enabled = true; // Enable XNNPACK
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

        // Resize if needed and paste
        if self.scale != 1.0 {
            let resized =
                image::imageops::resize(image, new_w, new_h, image::imageops::FilterType::Lanczos3);
            image::imageops::overlay(
                &mut canvas,
                &resized,
                self.offset_x as i64,
                self.offset_y as i64,
            );
        } else {
            image::imageops::overlay(
                &mut canvas,
                image,
                self.offset_x as i64,
                self.offset_y as i64,
            );
        }

        canvas
    }

    /// Detect text regions in image
    pub fn detect(&mut self, image: &GrayImage, threshold: f32) -> Result<Vec<BBox>> {
        let preprocessed = self.preprocess(image);

        // Create interpreter for this detection with multi-threading and XNNPACK
        let mut options = Options::default();
        options.thread_count = 4; // Use 4 threads
        options.is_xnnpack_enabled = true; // Enable XNNPACK acceleration
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
        // Channel meanings (from IDA analysis):
        // - Channel 0: Confidence score
        // - Channels 1-2: Center offsets (dx, dy)
        // - Channels 3-4: Size deltas (log_width, log_height) - need exp()
        // - Channels 5-6: Rotation (cos, sin) - ignored for now
        let mut boxes = Vec::new();
        let output_count = interpreter.output_tensor_count();

        // For debugging channel statistics
        let mut channel_stats: Vec<(f32, f32)> = vec![(f32::MAX, f32::MIN); 7];
        let mut high_conf_count = 0;

        // Chrome config has 6 anchors matching 6 outputs from one FPN pass.
        // The model produces 11 outputs total (two FPN passes), but only the first 6
        // have corresponding anchors. Outputs 6-10 are the second pass and would need
        // separate anchor assignment; we skip them to avoid over-merging.
        let mut anchor_idx = 0usize;

        for output_idx in 0..output_count {
            let tensor = interpreter.output(output_idx)?;
            let shape = tensor.shape();
            let dims = shape.dimensions();

            if dims.len() == 4 && dims[3] == 7 {
                // Only process outputs that have corresponding anchors
                if anchor_idx >= ANCHOR_WIDTHS.len() {
                    break;
                }

                let feat_h = dims[1];
                let feat_w = dims[2];

                // Calculate stride (subsampling factor)
                let stride = TARGET_SIZE as f32 / feat_h as f32;

                // Get anchor values for this output scale from config
                let anchor_w = ANCHOR_WIDTHS[anchor_idx];
                let anchor_h = ANCHOR_HEIGHTS[anchor_idx];
                anchor_idx += 1;

                if std::env::var("CHROME_OCR_DEBUG").is_ok() {
                    println!(
                        "  Output[{}]: {}x{} stride={:.1} anchor=({:.0},{:.0})",
                        output_idx, feat_h, feat_w, stride, anchor_w, anchor_h
                    );
                }

                // Skip very coarse feature maps (stride > 200)
                if stride > 200.0 {
                    continue;
                }

                // Get output data as f32
                let data: &[f32] = tensor.data();

                for y in 0..feat_h {
                    for x in 0..feat_w {
                        let idx = (y * feat_w + x) * 7;
                        // Apply sigmoid to convert logit to probability
                        let logit = data[idx];
                        let conf = 1.0 / (1.0 + (-logit).exp());

                        // Update channel statistics for high-confidence detections
                        if conf > threshold {
                            high_conf_count += 1;
                            for ch in 0..7 {
                                let val = data[idx + ch];
                                if val < channel_stats[ch].0 {
                                    channel_stats[ch].0 = val;
                                }
                                if val > channel_stats[ch].1 {
                                    channel_stats[ch].1 = val;
                                }
                            }
                        }

                        if conf > threshold {
                            // Extract channel values
                            let dx = data[idx + 1]; // Center x offset
                            let dy = data[idx + 2]; // Center y offset
                            let log_w = data[idx + 3]; // Log width delta
                            let log_h = data[idx + 4]; // Log height delta
                            let rot_cos = data[idx + 5]; // Rotation cosine
                            let rot_sin = data[idx + 6]; // Rotation sine

                            // Calculate center position
                            // Formula: (grid + 0.5 + offset) * stride
                            let cx = (x as f32 + 0.5 + dx) * stride;
                            let cy = (y as f32 + 0.5 + dy) * stride;

                            // Calculate box size
                            // Chrome config: width = exp(clamp(log_w, -4, 4)) * anchor_w
                            // anchor values per output scale from config binarypb
                            let log_w_clamped = log_w.clamp(-4.0, 4.0);
                            let log_h_clamped = log_h.clamp(-4.0, 4.0);
                            let bw = log_w_clamped.exp() * anchor_w;
                            let bh = log_h_clamped.exp() * anchor_h;

                            // Calculate rotation angle from cos/sin
                            let angle = rot_sin.atan2(rot_cos);

                            // NOTE: Chrome does NOT filter by absolute angle here.
                            // The 30° threshold in decompiled_0x180476920.txt line 696 is
                            // for angle DIFFERENCE between two boxes during line grouping,
                            // not individual box rotation. Vertical text (~90°) must pass through.
                            // Angle compatibility is checked during merge_boxes_to_lines.

                            // Check if within content area
                            let scaled_w = image.width() as f32 * self.scale;
                            let scaled_h = image.height() as f32 * self.scale;

                            if cx > self.offset_x
                                && cx < self.offset_x + scaled_w
                                && cy > self.offset_y
                                && cy < self.offset_y + scaled_h
                            {
                                boxes.push(BBox::with_angle(
                                    cx - bw / 2.0,
                                    cy - bh / 2.0,
                                    cx + bw / 2.0,
                                    cy + bh / 2.0,
                                    conf,
                                    angle,
                                ));
                            }
                        }
                    }
                }
            }
        }

        // Optionally print statistics (controlled by env var)
        if std::env::var("CHROME_OCR_DEBUG").is_ok() && high_conf_count > 0 {
            println!(
                "  Detection: {} high-conf points, channel ranges:",
                high_conf_count
            );
            for (ch, (min, max)) in channel_stats.iter().enumerate() {
                println!("    ch{}: [{:.3}, {:.3}]", ch, min, max);
            }

            if !boxes.is_empty() {
                let widths: Vec<f32> = boxes.iter().map(|b| b.x2 - b.x1).collect();
                let heights: Vec<f32> = boxes.iter().map(|b| b.y2 - b.y1).collect();
                let min_w = widths.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_w = widths.iter().cloned().fold(0.0, f32::max);
                let min_h = heights.iter().cloned().fold(f32::INFINITY, f32::min);
                let max_h = heights.iter().cloned().fold(0.0, f32::max);
                let avg_w = widths.iter().sum::<f32>() / widths.len() as f32;
                let avg_h = heights.iter().sum::<f32>() / heights.len() as f32;
                println!(
                    "  Box sizes: w=[{:.0}..{:.0}], avg={:.0}; h=[{:.0}..{:.0}], avg={:.0}",
                    min_w, max_w, avg_w, min_h, max_h, avg_h
                );
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
            // Chrome config: detection NMS IoU threshold = 0.5
            let dominated = kept
                .iter()
                .any(|k: &BBox| crate::utils::calc_iou(&bbox.as_array(), &k.as_array()) > 0.5);
            if !dominated {
                kept.push(bbox);
            }
        }

        Ok(kept)
    }
}
