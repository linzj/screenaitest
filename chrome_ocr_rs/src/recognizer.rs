use anyhow::{anyhow, Result};
use image::{GrayImage, ImageBuffer, Luma};
use std::path::Path;
use tflitec::interpreter::{Interpreter, Options};
use tflitec::model::Model;

const MODEL_HEIGHT: u32 = 32;
const MODEL_WIDTH: u32 = 168;
// LEFT_MARGIN is needed! Testing showed LEFT_MARGIN=0 causes MORE first char losses.
// The model expects some padding on the left for proper character alignment.
const LEFT_MARGIN: u32 = 12; // 12 pixels left margin for TFLite model
const EFFECTIVE_WIDTH: u32 = MODEL_WIDTH - LEFT_MARGIN; // 156 effective width
const CHAR_CONF_THRESHOLD: f32 = 0.15; // Filter low-confidence characters
const BLANK_IDX: usize = 8178; // Blank token index
const FRAME_WIDTH: u32 = 4; // 168 pixels / 42 time steps = 4 pixels per frame

/// Character with position information for Chrome-style deduplication
/// From IDA: Chrome uses x0, x1 coordinates to detect duplicates
#[derive(Clone, Debug)]
struct CharWithPosition {
    char: String,     // The recognized character
    conf: f32,        // Confidence score
    x0: u32,          // Start x position in original image
    x1: u32,          // End x position in original image
    time_step: usize, // Time step in CTC output (for debugging)
}

/// Chunk information for tensor-level merging
struct ChunkInfo {
    logits: Vec<f32>,
    x_start: u32, // Start pixel in original image
    x_end: u32,   // End pixel in original image
}

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
        options.thread_count = 4; // Use 4 threads
        options.is_xnnpack_enabled = true; // Enable XNNPACK acceleration
        let interpreter = Interpreter::new(&self.model, Some(options))?;
        interpreter.allocate_tensors()?;
        Ok(interpreter)
    }

    /// Recognize text from a line image using provided interpreter
    /// Chrome-style method based on IDA analysis:
    /// - Overlapping chunks to ensure complete characters
    /// - Character position-based deduplication (from IDA: x0, x1 overlap detection)
    /// - All 42 time steps are valid (no edge trimming)
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
            // Long line: Chrome's method - merge logits at tensor level, then unified CTC decode
            // From IDA analysis:
            // - chunk_border_left = 0.3, chunk_border_right = 0.3 (30% overlap each side)
            // - MergeChunkResults: direct copy using chunk_lengths array
            // - Key insight: Chrome tracks exact pixel boundaries, not just ratios

            let chunk_w = (EFFECTIVE_WIDTH as f32 * h as f32 / MODEL_HEIGHT as f32) as u32;

            // Chrome uses 30% border on each side, so step = 40% of chunk_width
            let step = (chunk_w as f32 * 0.4).max(1.0) as u32; // 40% step = 60% overlap

            // Collect chunks with their pixel boundaries
            let mut chunks: Vec<ChunkInfo> = Vec::new();
            let mut x = 0u32;

            while x < w.saturating_sub(step / 2) {
                let x2 = (x + chunk_w).min(w);
                let actual_w = x2 - x;

                if actual_w < 10 {
                    break;
                }

                let seg = image::imageops::crop_imm(image, x, 0, actual_w, h).to_image();
                let logits = self.get_chunk_logits(&seg, interpreter)?;

                chunks.push(ChunkInfo {
                    logits,
                    x_start: x,
                    x_end: x2,
                });

                x += step;
            }

            if chunks.is_empty() {
                return Ok((String::new(), 0.0));
            }

            if chunks.len() == 1 {
                let (text, avg_conf) = self.ctc_decode_logits(&chunks[0].logits);
                return Ok((text, avg_conf));
            }

            // Merge logits based on non-overlapping regions
            // Calculate "valid" region for each chunk (excluding overlap)
            let merged_logits = self.merge_chunk_logits_by_boundary(&chunks, w);

            // Unified CTC decode on merged logits
            let (text, avg_conf) = self.ctc_decode_logits(&merged_logits);

            Ok((text, avg_conf))
        }
    }

    /// Recognize text from a line image (creates new interpreter - slower)
    pub fn recognize(&self, image: &GrayImage) -> Result<(String, f32)> {
        let interpreter = self.create_interpreter()?;
        self.recognize_with_interpreter(image, &interpreter)
    }

    /// Recognize a single segment
    fn recognize_segment(&self, image: &GrayImage) -> Result<(String, f32)> {
        let interpreter = self.create_interpreter()?;
        self.recognize_segment_with_interpreter(image, &interpreter)
    }

    /// Get raw logits from a chunk (for tensor-level merging)
    fn get_chunk_logits(&self, image: &GrayImage, interpreter: &Interpreter) -> Result<Vec<f32>> {
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

        // Create canvas with left margin for model alignment
        // Note: Chrome does NOT invert images - uses original grayscale values
        let mut canvas: GrayImage =
            ImageBuffer::from_pixel(MODEL_WIDTH, MODEL_HEIGHT, Luma([255u8]));
        image::imageops::overlay(&mut canvas, &scaled, LEFT_MARGIN as i64, 0);

        // Run inference
        let input_data: Vec<u8> = canvas.as_raw().to_vec();
        let input_tensor = interpreter.input(0)?;
        input_tensor.set_data(&input_data)?;
        interpreter.invoke()?;

        // Get and dequantize logits
        let output_tensor = interpreter.output(self.output_idx)?;
        let logits_raw: &[u8] = output_tensor.data();

        let mut logits = vec![0.0f32; self.time_steps * self.vocab_size];
        for t in 0..self.time_steps {
            let base = t * self.vocab_size;
            for i in 0..self.vocab_size {
                let raw = logits_raw[base + i] as i32;
                logits[base + i] = (raw - self.zero_point) as f32 * self.scale_val;
            }
        }

        Ok(logits)
    }

    /// Merge chunk logits using Chrome's MergeChunkResults approach
    ///
    /// From IDA analysis of chrome_screen_ai.dll:
    /// - Source: ocr/photo/segmentation/tensor_lstm_client.cc
    /// - Assertion: chunk_lengths.at(ind) % frame_width == 0
    /// - Chrome pre-calculates chunk_lengths to ensure perfect frame alignment
    ///
    /// Key insight: Chrome uses a chunk_lengths array where each element specifies
    /// exactly how many frames to take from that chunk, and sum(chunk_lengths) == total_frames
    fn merge_chunk_logits_by_boundary(&self, chunks: &[ChunkInfo], _total_w: u32) -> Vec<f32> {
        if chunks.is_empty() {
            return Vec::new();
        }
        if chunks.len() == 1 {
            return chunks[0].logits.clone();
        }

        // Pre-calculate chunk_lengths like Chrome does
        // This ensures the frames tile perfectly without gaps
        let chunk_lengths = self.calculate_chunk_lengths(chunks);

        let mut merged = Vec::new();

        for (idx, chunk) in chunks.iter().enumerate() {
            let frames_to_take = chunk_lengths[idx];

            // Calculate start offset within this chunk
            // For first chunk: start at 0
            // For subsequent chunks: start at (border_left * time_steps) = 30% * 42 ≈ 13 frames
            let start_t = if idx == 0 {
                0
            } else {
                // Chrome uses 30% border on each side
                // Start reading from 30% into this chunk's logits
                ((self.time_steps as f32 * 0.3).round() as usize).min(self.time_steps)
            };

            let end_t = (start_t + frames_to_take).min(self.time_steps);

            // Copy the valid time steps
            for t in start_t..end_t {
                let base = t * self.vocab_size;
                if base + self.vocab_size <= chunk.logits.len() {
                    merged.extend_from_slice(&chunk.logits[base..base + self.vocab_size]);
                }
            }
        }

        merged
    }

    /// Calculate chunk_lengths array like Chrome does
    /// Each chunk contributes a specific number of frames, and the sum equals total expected frames
    fn calculate_chunk_lengths(&self, chunks: &[ChunkInfo]) -> Vec<usize> {
        let n = chunks.len();
        if n == 0 {
            return Vec::new();
        }
        if n == 1 {
            return vec![self.time_steps];
        }

        // Chrome uses border_left = 0.3, border_right = 0.3
        // Each chunk's content is 42 time steps
        // - First chunk: contributes frames 0 to 70% = 0 to 29.4 ≈ 29 frames
        // - Middle chunks: contribute frames 30% to 70% = 12.6 to 29.4 ≈ 17 frames
        // - Last chunk: contributes frames 30% to 100% = 12.6 to 42 ≈ 29 frames

        let first_contribution = (self.time_steps as f32 * 0.7).floor() as usize; // 29 frames
        let middle_contribution = (self.time_steps as f32 * 0.4).floor() as usize; // 16 frames
        let last_contribution = self.time_steps - (self.time_steps as f32 * 0.3).ceil() as usize; // 29 frames

        let mut chunk_lengths = Vec::with_capacity(n);

        // First chunk
        chunk_lengths.push(first_contribution);

        // Middle chunks (if any)
        for _ in 1..n - 1 {
            chunk_lengths.push(middle_contribution);
        }

        // Last chunk - adjust to ensure perfect tiling
        // Calculate expected total frames from the image width
        let total_image_w = chunks.last().unwrap().x_end;
        let expected_total = (total_image_w as f32 * MODEL_HEIGHT as f32
            / (chunks[0].x_end - chunks[0].x_start) as f32
            * self.time_steps as f32
            / (MODEL_HEIGHT as f32))
            .round() as usize;

        // For simplicity, use the calculated last contribution
        // The key is that we use floor/ceil consistently to avoid gaps
        chunk_lengths.push(last_contribution);

        chunk_lengths
    }

    /// Merge chunk logits at tensor level (Chrome's MergeChunkResults) - ratio based
    /// Each chunk has 30% overlap on each side. We keep only the non-overlapping center.
    #[allow(dead_code)]
    fn merge_chunk_logits(&self, chunks: &[Vec<f32>], border_ratio: f32) -> Vec<f32> {
        if chunks.is_empty() {
            return Vec::new();
        }
        if chunks.len() == 1 {
            return chunks[0].clone();
        }

        let total_time_steps = self.time_steps;
        let border_frames = (total_time_steps as f32 * border_ratio) as usize;

        let mut merged = Vec::new();

        for (idx, chunk) in chunks.iter().enumerate() {
            let is_first = idx == 0;
            let is_last = idx == chunks.len() - 1;

            // Determine which time steps to keep from this chunk
            // First chunk: keep from 0 to (total - border)
            // Last chunk: keep from border to total
            // Middle chunks: keep from border to (total - border)
            let (start_t, end_t) = if is_first && is_last {
                (0, total_time_steps)
            } else if is_first {
                (0, total_time_steps - border_frames)
            } else if is_last {
                (border_frames, total_time_steps)
            } else {
                (border_frames, total_time_steps - border_frames)
            };

            // Copy the valid time steps
            for t in start_t..end_t {
                let base = t * self.vocab_size;
                merged.extend_from_slice(&chunk[base..base + self.vocab_size]);
            }
        }

        merged
    }

    /// CTC decode on merged logits with post-processing to remove boundary duplicates
    fn ctc_decode_logits(&self, logits: &[f32]) -> (String, f32) {
        if logits.is_empty() {
            return (String::new(), 0.0);
        }

        let total_frames = logits.len() / self.vocab_size;
        let mut chars: Vec<(String, f32)> = Vec::new();
        let mut prev_idx: Option<usize> = None;

        for t in 0..total_frames {
            let base = t * self.vocab_size;

            // Find max logit
            let mut max_idx = 0usize;
            let mut max_logit = f32::NEG_INFINITY;
            for i in 0..self.vocab_size {
                let val = logits[base + i];
                if val > max_logit {
                    max_logit = val;
                    max_idx = i;
                }
            }

            // Compute softmax confidence
            let mut exp_sum = 0.0f32;
            for i in 0..self.vocab_size {
                exp_sum += (logits[base + i] - max_logit).exp();
            }
            let char_conf = 1.0 / exp_sum;

            // CTC decoding: skip blank, repeated, and low-confidence
            if max_idx != 0 && max_idx != BLANK_IDX && Some(max_idx) != prev_idx {
                if char_conf >= CHAR_CONF_THRESHOLD && max_idx < self.vocab.len() {
                    let c = &self.vocab[max_idx];
                    if !c.is_empty() {
                        chars.push((c.clone(), char_conf));
                    }
                }
            }
            prev_idx = Some(max_idx);
        }

        // With round() boundaries, we should have minimal duplicates
        // Skip aggressive dedup to preserve legitimate repeated characters
        let cleaned = chars;

        let avg_conf = if cleaned.is_empty() {
            0.0
        } else {
            cleaned.iter().map(|(_, c)| c).sum::<f32>() / cleaned.len() as f32
        };

        let result: String = cleaned.into_iter().map(|(s, _)| s).collect();
        (result, avg_conf)
    }

    /// Remove consecutive duplicate characters (Chrome's CJK merge behavior)
    /// This handles duplicates caused by chunk boundary overlap
    fn remove_consecutive_duplicates(&self, chars: &[(String, f32)]) -> Vec<(String, f32)> {
        if chars.len() < 2 {
            return chars.to_vec();
        }

        let mut result: Vec<(String, f32)> = Vec::new();

        for (c, conf) in chars {
            // Check if this character is a duplicate of the previous one
            if let Some((prev_c, prev_conf)) = result.last() {
                if c == prev_c {
                    // Duplicate - keep the one with higher confidence
                    if conf > prev_conf {
                        result.pop();
                        result.push((c.clone(), *conf));
                    }
                    // Otherwise skip this one
                    continue;
                }
            }
            result.push((c.clone(), *conf));
        }

        result
    }

    /// Recognize a single segment using provided interpreter
    fn recognize_segment_with_interpreter(
        &self,
        image: &GrayImage,
        interpreter: &Interpreter,
    ) -> Result<(String, f32)> {
        self.recognize_segment_trimmed(image, interpreter, 0, self.time_steps)
    }

    /// Recognize a segment with time step trimming (Chrome's TrimOutputScores behavior)
    fn recognize_segment_trimmed(
        &self,
        image: &GrayImage,
        interpreter: &Interpreter,
        start_t: usize,
        end_t: usize,
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

        // Debug: check average pixel values
        if std::env::var("CHROME_OCR_DEBUG").is_ok() {
            let avg: u32 = scaled.pixels().map(|p| p.0[0] as u32).sum::<u32>()
                / (scaled.width() * scaled.height());
            let first_col_avg: u32 = (0..scaled.height())
                .map(|y| scaled.get_pixel(0, y).0[0] as u32)
                .sum::<u32>()
                / scaled.height();
            eprintln!(
                "  [REC] scaled {}x{}, avg_pixel={}, first_col_avg={}",
                new_w, MODEL_HEIGHT, avg, first_col_avg
            );
        }

        // Create canvas with left margin for model alignment
        // Note: Chrome does NOT invert images - uses original grayscale values (from IDA analysis)
        let mut canvas: GrayImage =
            ImageBuffer::from_pixel(MODEL_WIDTH, MODEL_HEIGHT, Luma([255u8]));
        image::imageops::overlay(&mut canvas, &scaled, LEFT_MARGIN as i64, 0);

        self.recognize_canvas_with_trim(&canvas, interpreter, start_t, end_t)
    }

    /// Recognize a segment and return characters with position information
    /// This is Chrome's exact method: each character has x0, x1 position for deduplication
    /// From IDA: frame_width = 4 pixels per time step
    fn recognize_segment_with_positions(
        &self,
        image: &GrayImage,
        interpreter: &Interpreter,
        chunk_x_offset: u32,
    ) -> Result<Vec<CharWithPosition>> {
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

        // Create canvas with left margin for model alignment
        // Note: Chrome does NOT invert images - uses original grayscale values (from IDA analysis)
        let mut canvas: GrayImage =
            ImageBuffer::from_pixel(MODEL_WIDTH, MODEL_HEIGHT, Luma([255u8]));
        image::imageops::overlay(&mut canvas, &scaled, LEFT_MARGIN as i64, 0);

        // Run inference
        let input_data: Vec<u8> = canvas.as_raw().to_vec();
        let input_tensor = interpreter.input(0)?;
        input_tensor.set_data(&input_data)?;
        interpreter.invoke()?;

        // Get logits
        let output_tensor = interpreter.output(self.output_idx)?;
        let logits_raw: &[u8] = output_tensor.data();

        // Dequantize logits
        let mut logits = vec![0.0f32; self.time_steps * self.vocab_size];
        for t in 0..self.time_steps {
            let base = t * self.vocab_size;
            for i in 0..self.vocab_size {
                let raw = logits_raw[base + i] as i32;
                logits[base + i] = (raw - self.zero_point) as f32 * self.scale_val;
            }
        }

        // CTC decode with position tracking
        // Position calculation:
        // - Time step t corresponds to canvas pixel range [t*FRAME_WIDTH, (t+1)*FRAME_WIDTH)
        // - Canvas has LEFT_MARGIN (12px) of padding, so content starts at pixel 12
        // - Content ends at LEFT_MARGIN + new_w
        // - Map time step to content position, then scale to original image coords
        let mut result = Vec::new();
        let mut prev_idx: Option<usize> = None;

        // Calculate pixels per original image pixel in the scaled content
        let content_pixels = new_w as f32; // Content width in scaled image
        let orig_w = w as f32; // Original image width

        for t in 0..self.time_steps {
            let base = t * self.vocab_size;

            // Find max logit
            let mut max_idx = 0usize;
            let mut max_logit = f32::NEG_INFINITY;
            for i in 0..self.vocab_size {
                let val = logits[base + i];
                if val > max_logit {
                    max_logit = val;
                    max_idx = i;
                }
            }

            // Compute softmax confidence
            let max_for_softmax = max_logit;
            let mut exp_sum = 0.0f32;
            for i in 0..self.vocab_size {
                exp_sum += (logits[base + i] - max_for_softmax).exp();
            }
            let char_conf = 1.0 / exp_sum;

            // CTC decoding: skip blank, repeated, and low-confidence
            if max_idx != 0 && max_idx != BLANK_IDX && Some(max_idx) != prev_idx {
                if char_conf >= CHAR_CONF_THRESHOLD && max_idx < self.vocab.len() {
                    let c = &self.vocab[max_idx];
                    if !c.is_empty() {
                        // Calculate character position in original image coordinates
                        // Time step t corresponds to canvas pixel [t*FRAME_WIDTH, (t+1)*FRAME_WIDTH)
                        // Subtract LEFT_MARGIN to get content-relative position
                        let canvas_x0 = t as f32 * FRAME_WIDTH as f32;
                        let canvas_x1 = (t + 1) as f32 * FRAME_WIDTH as f32;

                        // Convert to content-relative (subtract LEFT_MARGIN, clamp to content bounds)
                        let content_x0 = (canvas_x0 - LEFT_MARGIN as f32)
                            .max(0.0)
                            .min(content_pixels);
                        let content_x1 = (canvas_x1 - LEFT_MARGIN as f32)
                            .max(0.0)
                            .min(content_pixels);

                        // Convert from scaled content to original image coordinates
                        let orig_x0 =
                            (content_x0 / content_pixels * orig_w) as u32 + chunk_x_offset;
                        let orig_x1 =
                            (content_x1 / content_pixels * orig_w) as u32 + chunk_x_offset;

                        result.push(CharWithPosition {
                            char: c.clone(),
                            conf: char_conf,
                            x0: orig_x0,
                            x1: orig_x1,
                            time_step: t,
                        });
                    }
                }
            }
            prev_idx = Some(max_idx);
        }

        Ok(result)
    }

    /// Deduplicate characters by position - Chrome's CJK merge algorithm
    /// From IDA at 0x18017E8B0: "found duplicate: x0=%d, x1=%d"
    /// Uses position overlap to identify and remove duplicate characters
    fn deduplicate_by_position(&self, mut chars: Vec<CharWithPosition>) -> Vec<CharWithPosition> {
        if chars.len() <= 1 {
            return chars;
        }

        // Sort by x0 position (left edge)
        chars.sort_by(|a, b| {
            a.x0.cmp(&b.x0).then_with(|| {
                b.conf
                    .partial_cmp(&a.conf)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
        });

        let mut result: Vec<CharWithPosition> = Vec::new();

        for ch in chars {
            // Check if this character overlaps with existing characters in result
            let mut is_duplicate = false;

            // Check against all recent characters (not just the last one)
            // This handles cases where characters might be slightly out of order
            for existing in result.iter().rev().take(5) {
                // Calculate position difference
                let x0_diff = (ch.x0 as i32 - existing.x0 as i32).abs();
                let x1_diff = (ch.x1 as i32 - existing.x1 as i32).abs();

                // Characters are duplicates if:
                // 1. Same character AND positions are within tolerance
                // 2. Different character but significant position overlap (OCR error at boundary)
                let char_width = (ch.x1 - ch.x0).max(1) as i32;
                let existing_width = (existing.x1 - existing.x0).max(1) as i32;
                let avg_width = (char_width + existing_width) / 2;

                // Position tolerance: 50% of average character width
                let tolerance = (avg_width as f32 * 0.5) as i32;

                if ch.char == existing.char && x0_diff < tolerance {
                    // Same character at nearly same position - definitely duplicate
                    is_duplicate = true;
                    break;
                }

                // Check overlap for different characters (boundary errors)
                let overlap = existing.x1.min(ch.x1) as i32 - existing.x0.max(ch.x0) as i32;
                if overlap > 0 {
                    let overlap_ratio = overlap as f32 / avg_width as f32;
                    // If >50% overlap, consider duplicate (keep higher confidence)
                    if overlap_ratio > 0.5 {
                        is_duplicate = true;
                        break;
                    }
                }
            }

            if !is_duplicate {
                result.push(ch);
            }
        }

        result
    }

    /// Improved segment merging with better duplicate detection
    /// Uses sliding window to find best overlap even with OCR errors at boundaries
    fn merge_segments_improved(&self, segments: &[String]) -> String {
        if segments.is_empty() {
            return String::new();
        }
        if segments.len() == 1 {
            return segments[0].clone();
        }

        let mut result = segments[0].clone();

        for next_seg in segments.iter().skip(1) {
            if next_seg.is_empty() {
                continue;
            }

            let result_chars: Vec<char> = result.chars().collect();
            let next_chars: Vec<char> = next_seg.chars().collect();

            // Expected overlap is about 25% of segment length
            // But due to OCR errors, actual overlap position may vary
            let expected_overlap = next_chars.len() / 4;
            let search_range = expected_overlap.max(5);

            // Find best overlap by checking multiple positions
            let mut best_overlap = 0;
            let mut best_score = 0;

            // Check overlap lengths from expected_overlap-range to expected_overlap+range
            let min_check = 1.max(expected_overlap.saturating_sub(search_range));
            let max_check = (expected_overlap + search_range)
                .min(result_chars.len())
                .min(next_chars.len());

            for overlap_len in min_check..=max_check {
                if overlap_len > result_chars.len() || overlap_len > next_chars.len() {
                    continue;
                }

                let suffix_start = result_chars.len() - overlap_len;
                let mut matches = 0;

                for i in 0..overlap_len {
                    if result_chars[suffix_start + i] == next_chars[i] {
                        matches += 1;
                    }
                }

                // Score: matches weighted by overlap length (prefer longer overlaps with good match rate)
                let match_rate = matches as f32 / overlap_len as f32;
                if match_rate >= 0.6 {
                    // At least 60% match
                    let score = (matches * overlap_len) as i32;
                    if score > best_score {
                        best_score = score;
                        best_overlap = overlap_len;
                    }
                }
            }

            // Apply the overlap
            if best_overlap > 0 {
                result.extend(next_chars[best_overlap..].iter());
            } else {
                // No good overlap found - just concatenate
                result.push_str(next_seg);
            }
        }

        // Post-process: remove obvious duplicates like "XX" patterns
        self.remove_boundary_duplicates(&result)
    }

    /// Remove obvious duplicate patterns at segment boundaries
    /// Post-process to remove obvious duplicate patterns from merged text
    fn post_process_duplicates(&self, text: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < 3 {
            return text.to_string();
        }

        let mut result = Vec::new();
        let mut i = 0;

        while i < chars.len() {
            // Check for duplicate sequences of 2-5 characters
            let mut found_dup = false;

            // Try to find duplicate pattern of length 2-5
            for dup_len in (2..=5).rev() {
                if i + dup_len * 2 <= chars.len() {
                    let seq1: String = chars[i..i + dup_len].iter().collect();
                    let seq2: String = chars[i + dup_len..i + dup_len * 2].iter().collect();

                    if seq1 == seq2 {
                        // Found exact duplicate - keep first, skip second
                        result.extend(chars[i..i + dup_len].iter());
                        i += dup_len * 2;
                        found_dup = true;
                        break;
                    }
                }
            }

            if found_dup {
                continue;
            }

            // Check for pattern "XYX" where Y is noise (e.g., "防:防" -> "防")
            if i + 2 < chars.len() && chars[i] == chars[i + 2] {
                let middle = chars[i + 1];
                if middle.is_ascii_punctuation()
                    || middle == ':'
                    || middle == ','
                    || middle == '.'
                    || middle == '。'
                    || middle == '、'
                    || middle == '♦'
                {
                    result.push(chars[i]);
                    i += 3;
                    continue;
                }
            }

            result.push(chars[i]);
            i += 1;
        }

        result.into_iter().collect()
    }

    #[allow(dead_code)]
    fn remove_boundary_duplicates(&self, text: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < 4 {
            return text.to_string();
        }

        let mut result = Vec::new();
        let mut i = 0;

        while i < chars.len() {
            result.push(chars[i]);

            // Check for duplicate patterns: "XYX" or "XYXY" where second part repeats first
            if i + 3 < chars.len() {
                // Check for 2-char duplicate: "ABAB" pattern
                if chars[i] == chars[i + 2] && chars[i + 1] == chars[i + 3] {
                    // Skip the duplicate pair
                    i += 2;
                    continue;
                }
            }

            if i + 2 < chars.len() {
                // Check for single char duplicate in pattern like "防芮防" -> "防"
                // where middle char is OCR error
                if chars[i] == chars[i + 2] && !chars[i].is_ascii_punctuation() {
                    let middle = chars[i + 1];
                    // If middle char looks like OCR noise (rare char or punctuation-like)
                    if Self::is_likely_ocr_noise(middle) {
                        i += 2; // Skip "芮防", keep just the first "防"
                        continue;
                    }
                }
            }

            i += 1;
        }

        result.into_iter().collect()
    }

    /// Check if a character is likely OCR noise at segment boundary
    fn is_likely_ocr_noise(c: char) -> bool {
        // Characters that are rare and often appear as OCR errors
        matches!(c, '芮' | 'í' | 'ì' | '兹' | '戎' | '讨' | '˙' | 'D' | ':' | '1'
            if c.is_ascii_digit() || c.is_ascii_punctuation())
            || c == '芮'
            || c == 'í'
            || c == 'ì'
            || c == '兹'
            || c == '戎'
            || c == '讨'
            || c == '˙'
    }

    /// Merge text segments by finding overlapping suffix/prefix
    /// Simple and robust approach: find longest common overlap between segments
    #[allow(dead_code)]
    fn merge_segments_by_text(&self, segments: &[String]) -> String {
        if segments.is_empty() {
            return String::new();
        }
        if segments.len() == 1 {
            return segments[0].clone();
        }

        let mut result = segments[0].clone();

        for next_seg in segments.iter().skip(1) {
            if next_seg.is_empty() {
                continue;
            }

            let result_chars: Vec<char> = result.chars().collect();
            let next_chars: Vec<char> = next_seg.chars().collect();

            // Find longest overlap (suffix of result == prefix of next_seg)
            // Check up to 50% of the shorter segment length
            let max_overlap = result_chars.len().min(next_chars.len()).min(20);
            let mut best_overlap = 0;

            for overlap_len in (1..=max_overlap).rev() {
                let suffix_start = result_chars.len() - overlap_len;
                let mut matches = true;

                for i in 0..overlap_len {
                    if result_chars[suffix_start + i] != next_chars[i] {
                        matches = false;
                        break;
                    }
                }

                if matches {
                    best_overlap = overlap_len;
                    break;
                }
            }

            // Append the non-overlapping part of next_seg
            if best_overlap > 0 {
                result.extend(next_chars[best_overlap..].iter());
            } else {
                // No overlap found - check for near-overlap (1 char tolerance)
                let near_overlap = self.find_near_overlap(&result_chars, &next_chars);
                if near_overlap > 0 {
                    result.extend(next_chars[near_overlap..].iter());
                } else {
                    // Just concatenate
                    result.push_str(next_seg);
                }
            }
        }

        result
    }

    /// Find near-overlap allowing 1 character mismatch
    fn find_near_overlap(&self, s1: &[char], s2: &[char]) -> usize {
        let max_overlap = s1.len().min(s2.len()).min(15);

        for overlap_len in (3..=max_overlap).rev() {
            let suffix_start = s1.len() - overlap_len;
            let mut mismatches = 0;

            for i in 0..overlap_len {
                if s1[suffix_start + i] != s2[i] {
                    mismatches += 1;
                    if mismatches > 1 {
                        break;
                    }
                }
            }

            if mismatches <= 1 {
                return overlap_len;
            }
        }

        0
    }

    /// Recognize a 168x32 canvas (creates new interpreter - slower)
    fn recognize_canvas(&self, canvas: &GrayImage) -> Result<(String, f32)> {
        let interpreter = self.create_interpreter()?;
        self.recognize_canvas_with_interpreter(canvas, &interpreter)
    }

    /// Recognize a 168x32 canvas using provided interpreter
    /// Based on Chrome's TrimOutputScores from IDA analysis:
    /// Chrome trims edge frames before CTC decoding to remove boundary artifacts
    fn recognize_canvas_with_interpreter(
        &self,
        canvas: &GrayImage,
        interpreter: &Interpreter,
    ) -> Result<(String, f32)> {
        self.recognize_canvas_with_trim(canvas, interpreter, 0, self.time_steps)
    }

    /// Recognize canvas with optional time step trimming
    /// start_t and end_t define the valid region of CTC output to decode
    fn recognize_canvas_with_trim(
        &self,
        canvas: &GrayImage,
        interpreter: &Interpreter,
        start_t: usize,
        end_t: usize,
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
        // Only decode from start_t to end_t (Chrome's TrimOutputScores behavior)
        let mut result = String::new();
        let mut prev_idx: Option<usize> = None;
        let mut conf_scores = Vec::new();

        let actual_end = end_t.min(self.time_steps);
        for t in start_t..actual_end {
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

    /// Merge overlapping text segments with fuzzy matching
    /// Based on Chrome's duplicate removal algorithm from IDA analysis
    fn merge_overlapping_segments(&self, segments: &[String]) -> String {
        if segments.is_empty() {
            return String::new();
        }
        if segments.len() == 1 {
            return self.clean_repeated_chars(&segments[0]);
        }

        let mut result = segments[0].clone();
        for next_seg in segments.iter().skip(1) {
            // Try exact match first
            let exact_overlap = self.find_exact_overlap(&result, next_seg);
            if exact_overlap > 0 {
                let chars: Vec<char> = next_seg.chars().collect();
                result.extend(chars.iter().skip(exact_overlap));
            } else {
                // Try fuzzy match - look for partial overlap with tolerance
                let fuzzy_overlap = self.find_fuzzy_overlap(&result, next_seg);
                if fuzzy_overlap > 0 {
                    let chars: Vec<char> = next_seg.chars().collect();
                    result.extend(chars.iter().skip(fuzzy_overlap));
                } else {
                    // No overlap found - just concatenate
                    result.push_str(next_seg);
                }
            }
        }

        // Clean up repeated characters at segment boundaries
        self.clean_repeated_chars(&result)
    }

    /// Clean up repeated characters that may appear at segment boundaries
    /// Based on Chrome's same_char_repeat filtering from FilterJunkMutator
    fn clean_repeated_chars(&self, text: &str) -> String {
        let chars: Vec<char> = text.chars().collect();
        if chars.len() < 2 {
            return text.to_string();
        }

        let mut result = Vec::new();
        let mut i = 0;

        while i < chars.len() {
            let c = chars[i];

            // Check if this is a junk single character between CJK text
            // Based on Chrome's RemoveJunkWords and single alpha line removal
            if self.is_junk_char_at_boundary(&chars, i) {
                i += 1;
                continue;
            }

            // Check for repeated number patterns like "200000" -> "2000"
            // or "198187" -> "1987" (where "1981" + "1987" overlapped incorrectly)
            if c.is_ascii_digit() {
                let (cleaned_num, skip_count) = self.clean_number_at_position(&chars, i);
                if !cleaned_num.is_empty() {
                    result.extend(cleaned_num.chars());
                    i += skip_count;
                    continue;
                }
            }

            result.push(c);

            // Check for boundary artifacts: patterns like "人|人", "疫|疫", etc.
            // These appear as "char + noise + same_char" at segment boundaries
            if i + 2 < chars.len() {
                let next1 = chars[i + 1];
                let next2 = chars[i + 2];

                // If we have pattern like "X?X" where ? is a single noise char
                // and both X are the same, skip the noise and second X
                if c == next2 && !c.is_ascii_punctuation() && !c.is_whitespace() {
                    // Check if middle char is punctuation or unusual
                    if next1.is_ascii_punctuation() || next1 == '|' || next1 == '!' || next1 == ':'
                    {
                        i += 3; // Skip "X?X", we already added first X
                        continue;
                    }
                }
            }

            // Check for immediate duplicate like "XX" that might be artifact
            // But be careful with legitimate doubles (e.g., Chinese characters)
            if i + 1 < chars.len() && chars[i + 1] == c {
                // Look ahead to see if this is part of a longer pattern
                let mut repeat_count = 1;
                let mut j = i + 1;
                while j < chars.len() && chars[j] == c {
                    repeat_count += 1;
                    j += 1;
                }

                // If more than 2 consecutive same chars, likely artifact
                // But allow some legitimate cases
                if repeat_count > 2 && !c.is_ascii_digit() {
                    i = j; // Skip all repeats, we already added one
                    continue;
                }
            }

            i += 1;
        }

        result.into_iter().collect()
    }

    /// Clean up number sequences that have been duplicated at segment boundaries
    /// Returns (cleaned_number, chars_consumed)
    fn clean_number_at_position(&self, chars: &[char], start: usize) -> (String, usize) {
        // Extract the full number sequence
        let mut end = start;
        while end < chars.len() && chars[end].is_ascii_digit() {
            end += 1;
        }

        let num_len = end - start;
        if num_len < 5 {
            return (String::new(), 0); // Too short to have duplication
        }

        let num_str: String = chars[start..end].iter().collect();

        // Check for patterns like "200000" (should be "2000") - repeated trailing zeros
        // or "199414" (should be "1994") - overlapping years

        // Pattern 1: Year followed by partial repeat "198187" -> "1987"
        // This happens when "1987" is recognized as "1981" + "987" or "198" + "1987"
        if num_len >= 6 {
            // Try to find a 4-digit year pattern
            for year_start in 0..=(num_len - 4) {
                let potential_year: String = chars[start + year_start..start + year_start + 4]
                    .iter()
                    .collect();
                if let Ok(year) = potential_year.parse::<u32>() {
                    if (1900..=2100).contains(&year) {
                        // Check if there's overlap before or after
                        // Pattern: "19XX19YY" where XXYY forms a year
                        if year_start >= 2 {
                            let before: String = chars[start..start + year_start].iter().collect();
                            // Check if 'before' is prefix of potential_year
                            if potential_year.starts_with(&before) {
                                return (potential_year, year_start + 4);
                            }
                        }
                        if year_start + 4 < num_len {
                            let after: String = chars[start + year_start + 4..end].iter().collect();
                            // Check if 'after' is suffix of potential_year
                            if potential_year.ends_with(&after) {
                                return (potential_year, num_len);
                            }
                        }
                    }
                }
            }
        }

        // Pattern 2: "200000" -> "2000" (extra zeros from segment overlap)
        if num_str.ends_with("000") && num_len > 4 {
            // Count trailing zeros
            let trailing_zeros = num_str.chars().rev().take_while(|&c| c == '0').count();
            if trailing_zeros > 2 {
                // Likely has duplicate zeros, remove half
                let zeros_to_keep = (trailing_zeros + 1) / 2;
                let base_len = num_len - trailing_zeros;
                let cleaned: String = chars[start..start + base_len].iter().collect();
                let zeros: String = std::iter::repeat('0').take(zeros_to_keep).collect();
                return (cleaned + &zeros, num_len);
            }
        }

        (String::new(), 0)
    }

    /// Check if character at position is a junk artifact at segment boundary
    /// Based on Chrome's remove_single_alpha_lines and RemoveJunkWords
    fn is_junk_char_at_boundary(&self, chars: &[char], i: usize) -> bool {
        let c = chars[i];

        // Common single-char artifacts that appear at segment boundaries
        let is_single_artifact =
            matches!(c, 'e' | 'i' | 'l' | 'I' | 'o' | 'O' | '(' | ')' | '[' | ']');

        if !is_single_artifact {
            return false;
        }

        // Check context: is this isolated between CJK characters or punctuation?
        let prev_is_cjk_or_punct = if i > 0 {
            let prev = chars[i - 1];
            Self::is_cjk(prev) || prev == ',' || prev == '。' || prev == '、'
        } else {
            false
        };

        let next_is_cjk_or_punct = if i + 1 < chars.len() {
            let next = chars[i + 1];
            Self::is_cjk(next) || next == ',' || next == '。' || next == '、'
        } else {
            false
        };

        // Pattern: "CJK + artifact + CJK" or "punct + artifact + CJK"
        if prev_is_cjk_or_punct && next_is_cjk_or_punct {
            return true;
        }

        // Pattern: ",e," - isolated artifact between punctuation
        if i > 0 && i + 1 < chars.len() {
            let prev = chars[i - 1];
            let next = chars[i + 1];
            if (prev == ',' || prev == '，') && (next == ',' || next == '，') {
                return true;
            }
        }

        false
    }

    /// Check if character is CJK
    fn is_cjk(c: char) -> bool {
        matches!(c, '\u{4E00}'..='\u{9FFF}' | '\u{3400}'..='\u{4DBF}' | '\u{F900}'..='\u{FAFF}')
    }

    /// Find exact overlap between end of s1 and start of s2
    fn find_exact_overlap(&self, s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let max_overlap = chars1.len().min(chars2.len()).min(20); // Limit search

        for overlap_len in (1..=max_overlap).rev() {
            let suffix: String = chars1[chars1.len() - overlap_len..].iter().collect();
            let prefix: String = chars2[..overlap_len].iter().collect();
            if suffix == prefix {
                return overlap_len;
            }
        }
        0
    }

    /// Find fuzzy overlap allowing 1 char difference at boundaries
    fn find_fuzzy_overlap(&self, s1: &str, s2: &str) -> usize {
        let chars1: Vec<char> = s1.chars().collect();
        let chars2: Vec<char> = s2.chars().collect();
        let max_overlap = chars1.len().min(chars2.len()).min(15); // Smaller limit for fuzzy

        // Try to find overlap with tolerance for 1 different char
        for overlap_len in (3..=max_overlap).rev() {
            let suffix = &chars1[chars1.len() - overlap_len..];
            let prefix = &chars2[..overlap_len];

            // Count matching characters
            let matches = suffix
                .iter()
                .zip(prefix.iter())
                .filter(|(a, b)| a == b)
                .count();
            // Allow up to 1 mismatch for overlaps of 3+ chars
            if matches >= overlap_len - 1 {
                return overlap_len;
            }
        }

        // Special handling for number sequences at segment boundaries
        // Patterns like "1987" split across segments may appear as "198" + "87年"
        // or "1994" as "199" + "94年" causing "19994年" etc.
        self.find_numeric_overlap(&chars1, &chars2)
    }

    /// Find overlap in numeric sequences at segment boundaries
    /// Handles cases like "198" + "1987年" -> should merge to "1987年" not "1981987年"
    fn find_numeric_overlap(&self, chars1: &[char], chars2: &[char]) -> usize {
        // Check if s1 ends with digits and s2 starts with digits
        let mut s1_digit_end = 0;
        for (i, c) in chars1.iter().rev().enumerate() {
            if c.is_ascii_digit() {
                s1_digit_end = i + 1;
            } else {
                break;
            }
        }

        if s1_digit_end < 2 {
            return 0;
        }

        let mut s2_digit_start = 0;
        for c in chars2.iter() {
            if c.is_ascii_digit() {
                s2_digit_start += 1;
            } else {
                break;
            }
        }

        if s2_digit_start < 2 {
            return 0;
        }

        // Get the trailing digits of s1 and leading digits of s2
        let s1_digits: String = chars1[chars1.len() - s1_digit_end..].iter().collect();
        let s2_digits: String = chars2[..s2_digit_start].iter().collect();

        // Check if s2_digits contains s1_digits suffix (overlapping year numbers)
        // e.g., s1="199" s2="1994" -> s1 is prefix of s2, overlap=3
        if s2_digits.starts_with(&s1_digits) {
            return s1_digit_end;
        }

        // Check reverse: s1="1994" s2="94年" -> "94" is suffix of "1994"
        if s1_digits.ends_with(&s2_digits) {
            // The entire s2_digit_start should be skipped
            return s2_digit_start;
        }

        // Check for partial overlap: s1="198" s2="87年"
        // This means "1987" was split, and we have "198" + "87"
        // Find common overlap pattern
        for overlap in (2..=s1_digit_end.min(s2_digit_start)).rev() {
            let s1_suffix: String = chars1[chars1.len() - overlap..].iter().collect();
            let s2_prefix: String = chars2[..overlap].iter().collect();
            if s1_suffix == s2_prefix {
                return overlap;
            }
        }

        0
    }
}
