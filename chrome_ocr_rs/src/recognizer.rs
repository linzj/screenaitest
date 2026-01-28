use anyhow::{anyhow, Result};
use image::{GrayImage, ImageBuffer, Luma};
use std::path::Path;
use tflitec::interpreter::{Interpreter, Options};
use tflitec::model::Model;

const MODEL_HEIGHT: u32 = 32;
const MODEL_WIDTH: u32 = 168;
const LEFT_MARGIN: u32 = 12;
const EFFECTIVE_WIDTH: u32 = MODEL_WIDTH - LEFT_MARGIN; // 156
const CHAR_CONF_THRESHOLD: f32 = 0.15; // Filter low-confidence characters
const BLANK_IDX: usize = 8178; // Blank token index

pub struct LineRecognizer {
    model: Model<'static>,
    vocab: Vec<String>,
    time_steps: usize,
    vocab_size: usize,
    output_idx: usize,
    scale_val: f32,
    zero_point: i32,
}

impl LineRecognizer {
    pub fn new(model_dir: &Path) -> Result<Self> {
        let model_path = model_dir
            .join("gocr")
            .join("gocr_models")
            .join("line_recognition_mobile_convnext320_omni")
            .join("hanijpan.tflite");

        // Look for vocab file - first in current directory, then in model directory
        let vocab_path = std::path::PathBuf::from("hanijpan_char_map.json");

        if !model_path.exists() {
            return Err(anyhow!(
                "Recognition model not found: {}",
                model_path.display()
            ));
        }

        let model = Model::new(model_path.to_str().unwrap())?;

        // Try to load vocab from various locations
        let vocab = crate::utils::load_vocab(&vocab_path)?;
        println!("  LineRecognizer: {} chars in vocab", vocab.len());

        // Get output shape info and quantization params
        let (time_steps, vocab_size, output_idx, scale_val, zero_point) = {
            let mut options = Options::default();
            options.is_xnnpack_enabled = true; // Enable XNNPACK
            let interpreter = Interpreter::new(&model, Some(options))?;
            interpreter.allocate_tensors()?;

            let mut ts = 42usize;
            let mut vs = 8179usize;
            let mut out_idx = 0usize;
            let mut sv = 1.0f32;
            let mut zp = 0i32;

            let output_count = interpreter.output_tensor_count();
            for idx in 0..output_count {
                let tensor = interpreter.output(idx)?;
                let shape = tensor.shape();
                let dims = shape.dimensions();
                if dims.len() == 3 && dims[2] > 1000 {
                    ts = dims[1];
                    vs = dims[2];
                    out_idx = idx;
                    if let Some(q) = tensor.quantization_parameters() {
                        sv = q.scale;
                        zp = q.zero_point;
                    }
                    break;
                }
            }
            (ts, vs, out_idx, sv, zp)
        };

        Ok(Self {
            model,
            vocab,
            time_steps,
            vocab_size,
            output_idx,
            scale_val,
            zero_point,
        })
    }

    /// Create a new interpreter for batch processing
    pub fn create_interpreter(&self) -> Result<Interpreter> {
        let mut options = Options::default();
        options.thread_count = 8; // Use 8 threads
        options.is_xnnpack_enabled = true; // Enable XNNPACK acceleration
        let interpreter = Interpreter::new(&self.model, Some(options))?;
        interpreter.allocate_tensors()?;
        Ok(interpreter)
    }

    /// Recognize text from a line image using provided interpreter
    pub fn recognize_with_interpreter(
        &self,
        image: &GrayImage,
        interpreter: &Interpreter,
    ) -> Result<(String, f32)> {
        let (w, h) = (image.width(), image.height());

        if w < 5 || h < 5 {
            return Ok((String::new(), 0.0));
        }

        // Calculate ideal width when scaled to height 32
        let ideal_w = (w as f32 * MODEL_HEIGHT as f32 / h as f32) as u32;

        if ideal_w <= EFFECTIVE_WIDTH {
            // Short line: direct recognition
            self.recognize_segment_with_interpreter(image, interpreter)
        } else {
            // Long line: segment recognition with overlap deduplication
            let seg_w = (EFFECTIVE_WIDTH as f32 * h as f32 / MODEL_HEIGHT as f32) as u32;
            let step = (seg_w as f32 * 0.7) as u32;

            let mut segments = Vec::new();
            let mut confs = Vec::new();
            let mut x = 0u32;

            while x < w.saturating_sub(seg_w / 2) {
                let x2 = (x + seg_w).min(w);
                let seg = image::imageops::crop_imm(image, x, 0, x2 - x, h).to_image();

                let (text, conf) = self.recognize_segment_with_interpreter(&seg, interpreter)?;
                if !text.is_empty() {
                    segments.push(text);
                    confs.push(conf);
                }

                x += step;
            }

            let combined = self.merge_overlapping_segments(&segments);
            let avg_conf = if confs.is_empty() {
                0.0
            } else {
                confs.iter().sum::<f32>() / confs.len() as f32
            };

            Ok((combined, avg_conf))
        }
    }

    /// Recognize text from a line image (creates new interpreter - slower)
    pub fn recognize(&self, image: &GrayImage) -> Result<(String, f32)> {
        let (w, h) = (image.width(), image.height());

        if w < 5 || h < 5 {
            return Ok((String::new(), 0.0));
        }

        // Calculate ideal width when scaled to height 32
        let ideal_w = (w as f32 * MODEL_HEIGHT as f32 / h as f32) as u32;

        if ideal_w <= EFFECTIVE_WIDTH {
            // Short line: direct recognition
            self.recognize_segment(image)
        } else {
            // Long line: segment recognition with overlap deduplication
            let seg_w = (EFFECTIVE_WIDTH as f32 * h as f32 / MODEL_HEIGHT as f32) as u32;
            let step = (seg_w as f32 * 0.7) as u32;

            let mut segments = Vec::new();
            let mut confs = Vec::new();
            let mut x = 0u32;

            while x < w.saturating_sub(seg_w / 2) {
                let x2 = (x + seg_w).min(w);
                let seg = image::imageops::crop_imm(image, x, 0, x2 - x, h).to_image();

                let (text, conf) = self.recognize_segment(&seg)?;
                if !text.is_empty() {
                    segments.push(text);
                    confs.push(conf);
                }

                x += step;
            }

            let combined = self.merge_overlapping_segments(&segments);
            let avg_conf = if confs.is_empty() {
                0.0
            } else {
                confs.iter().sum::<f32>() / confs.len() as f32
            };

            Ok((combined, avg_conf))
        }
    }

    /// Recognize a single segment
    fn recognize_segment(&self, image: &GrayImage) -> Result<(String, f32)> {
        let interpreter = self.create_interpreter()?;
        self.recognize_segment_with_interpreter(image, &interpreter)
    }

    /// Recognize a single segment using provided interpreter
    fn recognize_segment_with_interpreter(
        &self,
        image: &GrayImage,
        interpreter: &Interpreter,
    ) -> Result<(String, f32)> {
        let (w, h) = (image.width(), image.height());

        // Scale to height 32
        let scale = MODEL_HEIGHT as f32 / h as f32;
        let new_w = ((w as f32 * scale) as u32).min(MODEL_WIDTH - LEFT_MARGIN);

        let scaled = image::imageops::resize(
            image,
            new_w,
            MODEL_HEIGHT,
            image::imageops::FilterType::Lanczos3,
        );

        // Create canvas with left margin
        let mut canvas: GrayImage =
            ImageBuffer::from_pixel(MODEL_WIDTH, MODEL_HEIGHT, Luma([255u8]));
        image::imageops::overlay(&mut canvas, &scaled, LEFT_MARGIN as i64, 0);

        self.recognize_canvas_with_interpreter(&canvas, interpreter)
    }

    /// Recognize a 168x32 canvas (creates new interpreter - slower)
    fn recognize_canvas(&self, canvas: &GrayImage) -> Result<(String, f32)> {
        let interpreter = self.create_interpreter()?;
        self.recognize_canvas_with_interpreter(canvas, &interpreter)
    }

    /// Recognize a 168x32 canvas using provided interpreter
    fn recognize_canvas_with_interpreter(
        &self,
        canvas: &GrayImage,
        interpreter: &Interpreter,
    ) -> Result<(String, f32)> {
        // Prepare input data [1, 32, 168, 1]
        let input_data: Vec<u8> = canvas.as_raw().to_vec();

        // Set input tensor
        let input_tensor = interpreter.input(0)?;
        input_tensor.set_data(&input_data)?;

        // Run inference
        interpreter.invoke()?;

        // Get logits output using cached index
        let output_tensor = interpreter.output(self.output_idx)?;
        let logits_raw: &[u8] = output_tensor.data();

        // Dequantize logits using cached params
        let mut logits = vec![0.0f32; self.time_steps * self.vocab_size];
        for t in 0..self.time_steps {
            let base = t * self.vocab_size;
            for i in 0..self.vocab_size {
                let raw = logits_raw[base + i] as i32;
                logits[base + i] = (raw - self.zero_point) as f32 * self.scale_val;
            }
        }

        // Decode using CTC greedy decoding with softmax confidence
        let mut result = String::new();
        let mut prev_idx: Option<usize> = None;
        let mut conf_scores = Vec::new();

        for t in 0..self.time_steps {
            let base = t * self.vocab_size;

            // Find max logit and compute softmax
            let mut max_idx = 0usize;
            let mut max_logit = f32::NEG_INFINITY;
            let mut max_for_softmax = f32::NEG_INFINITY;

            for i in 0..self.vocab_size {
                let val = logits[base + i];
                if val > max_logit {
                    max_logit = val;
                    max_idx = i;
                }
                if val > max_for_softmax {
                    max_for_softmax = val;
                }
            }

            // Compute softmax for max element
            let mut exp_sum = 0.0f32;
            for i in 0..self.vocab_size {
                exp_sum += (logits[base + i] - max_for_softmax).exp();
            }
            let char_conf = 1.0 / exp_sum; // softmax(max_logit) = exp(0) / sum = 1 / sum

            // CTC decoding: skip index 0, blank (8178), repeated, and low-confidence
            if max_idx != 0 && max_idx != BLANK_IDX && Some(max_idx) != prev_idx {
                if char_conf >= CHAR_CONF_THRESHOLD && max_idx < self.vocab.len() {
                    let c = &self.vocab[max_idx];
                    if !c.is_empty() {
                        result.push_str(c);
                        conf_scores.push(char_conf);
                    }
                }
            }
            prev_idx = Some(max_idx);
        }

        let avg_conf = if conf_scores.is_empty() {
            0.0
        } else {
            conf_scores.iter().sum::<f32>() / conf_scores.len() as f32
        };

        Ok((result, avg_conf))
    }

    /// Merge overlapping text segments
    fn merge_overlapping_segments(&self, segments: &[String]) -> String {
        if segments.is_empty() {
            return String::new();
        }
        if segments.len() == 1 {
            return segments[0].clone();
        }

        let mut result = segments[0].clone();
        for next_seg in segments.iter().skip(1) {
            let overlap_len = self.find_overlap(&result, next_seg);
            if overlap_len > 0 {
                // Skip the overlapping part
                let chars: Vec<char> = next_seg.chars().collect();
                result.extend(chars.iter().skip(overlap_len));
            } else {
                result.push_str(next_seg);
            }
        }
        result
    }

    /// Find overlap between end of s1 and start of s2 (with fuzzy matching)
    fn find_overlap(&self, s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let max_overlap = chars1.len().min(chars2.len());

        // First try exact match
        for overlap_len in (1..=max_overlap).rev() {
            let suffix: String = chars1[chars1.len() - overlap_len..].iter().collect();
            let prefix: String = chars2[..overlap_len].iter().collect();
            if suffix == prefix {
                return overlap_len;
            }
        }

        // Try anchor-based matching: find a short distinctive substring
        // This handles cases where OCR inserts extra chars like "标准" vs "e标准"
        let anchor_len = 2; // Look for 2-char anchors (more flexible)
        if chars1.len() >= anchor_len && chars2.len() >= anchor_len {
            // Search backwards from end of s1 for anchors
            let search_range = max_overlap.min(12); // Search in last 12 chars
                                                    // Include all possible anchor positions up to the end of s1
            for i in 0..=search_range.saturating_sub(anchor_len) {
                let anchor_start = chars1.len() - search_range + i;
                let anchor: String = chars1[anchor_start..anchor_start + anchor_len]
                    .iter()
                    .collect();

                // Skip common punctuation anchors
                if anchor
                    .chars()
                    .all(|c| c.is_ascii_punctuation() || c == ',' || c == '。')
                {
                    continue;
                }

                // Look for this anchor at start of s2
                let s2_search_range = (search_range + 5).min(chars2.len());
                for j in 0..s2_search_range.saturating_sub(anchor_len) {
                    let prefix: String = chars2[j..j + anchor_len].iter().collect();
                    if anchor == prefix {
                        // Found anchor, calculate overlap
                        // s1 ends at anchor_start + anchor_len, s2 has anchor at position j
                        // We want to skip j + (chars1.len() - anchor_start) chars from s2
                        let overlap_in_s2 = j + (chars1.len() - anchor_start);
                        if overlap_in_s2 <= chars2.len() && overlap_in_s2 >= 2 {
                            return overlap_in_s2;
                        }
                    }
                }
            }
        }

        // Fallback: estimate overlap based on typical 30% image overlap
        // Skip ~25% of s2 characters if s2 is reasonably long
        if chars2.len() >= 4 {
            let estimated_overlap = chars2.len() * 25 / 100;
            if estimated_overlap >= 2 {
                return estimated_overlap;
            }
        }

        0
    }
}
