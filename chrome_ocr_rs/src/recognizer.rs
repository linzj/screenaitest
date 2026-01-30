use anyhow::{anyhow, Result};
use image::{GrayImage, ImageBuffer, Luma};
use std::collections::HashMap;
use std::path::Path;
use tflitec::interpreter::{Interpreter, Options};
use tflitec::model::Model;

const MODEL_HEIGHT: u32 = 32;
const MODEL_WIDTH: u32 = 168;
// LEFT_MARGIN is needed! Testing showed LEFT_MARGIN=0 causes MORE first char losses.
// The model expects some padding on the left for proper character alignment.
const LEFT_MARGIN: u32 = 12; // 12 pixels left margin for TFLite model
                             // EFFECTIVE_WIDTH is now computed per-model via self.effective_width()
const CHAR_CONF_THRESHOLD: f32 = 0.15; // Filter low-confidence characters (CJK model)
                                       // Chrome UND config: char_score_threshold=0 (no filtering)
const UND_CHAR_CONF_THRESHOLD: f32 = 0.0;
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
    blank_idx: usize,
    output_idx: usize,
    scale_val: f32,
    zero_point: i32,
    left_margin: u32,
    pub model_name: String,
}

impl LineRecognizer {
    /// Create a recognizer for a specific model
    /// model_name: "hanijpan" for CJK, "gocr_mobile_und" for universal/Latin
    pub fn new_with_model(model_dir: &Path, model_name: &str) -> Result<Self> {
        let model_path = model_dir
            .join("gocr")
            .join("gocr_models")
            .join("line_recognition_mobile_convnext320_omni")
            .join(format!("{}.tflite", model_name));

        // Look for vocab file in multiple locations
        let vocab_filename = format!("{}_char_map.json", model_name);
        let vocab_path = {
            let cwd_path = std::path::PathBuf::from(&vocab_filename);
            if cwd_path.exists() {
                cwd_path
            } else {
                // Try next to the executable
                let exe_dir = std::env::current_exe()
                    .ok()
                    .and_then(|p| p.parent().map(|d| d.to_path_buf()));
                exe_dir
                    .map(|d| d.join(&vocab_filename))
                    .filter(|p| p.exists())
                    .unwrap_or(cwd_path)
            }
        };

        if !model_path.exists() {
            return Err(anyhow!(
                "Recognition model not found: {}",
                model_path.display()
            ));
        }

        let model = Model::new(model_path.to_str().unwrap())?;

        // Try to load vocab from various locations
        let vocab = crate::utils::load_vocab(&vocab_path)?;
        println!(
            "  LineRecognizer [{}]: {} chars in vocab",
            model_name,
            vocab.len()
        );
        // Debug: print output tensor info after creation
        // (will be printed below after we read the output tensor shape)

        // Get output shape info and quantization params
        let (time_steps, vocab_size, output_idx, scale_val, zero_point) = {
            let mut options = Options::default();
            options.is_xnnpack_enabled = true;
            let interpreter = Interpreter::new(&model, Some(options))?;
            interpreter.allocate_tensors()?;

            let mut ts = 42usize;
            let mut vs = vocab.len() + 1; // default: vocab + blank
            let mut out_idx = 0usize;
            let mut sv = 1.0f32;
            let mut zp = 0i32;

            let output_count = interpreter.output_tensor_count();
            for idx in 0..output_count {
                let tensor = interpreter.output(idx)?;
                let shape = tensor.shape();
                let dims = shape.dimensions();
                if dims.len() == 3 && dims[2] > vocab.len() {
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
            // Debug: print input tensor info
            let input_count = interpreter.input_tensor_count();
            println!("  [{}] Input tensors: {}", model_name, input_count);
            for idx in 0..input_count {
                let tensor = interpreter.input(idx)?;
                let shape = tensor.shape();
                let dims = shape.dimensions();
                let quant_info = if let Some(q) = tensor.quantization_parameters() {
                    format!("scale={}, zero_point={}", q.scale, q.zero_point)
                } else {
                    "none".to_string()
                };
                println!(
                    "    input[{}]: shape={:?} type={:?} quant=[{}]",
                    idx,
                    dims,
                    tensor.data_type(),
                    quant_info
                );
            }
            // Also print all output tensors
            let output_count = interpreter.output_tensor_count();
            println!("  [{}] Output tensors: {}", model_name, output_count);
            for idx in 0..output_count {
                let tensor = interpreter.output(idx)?;
                let shape = tensor.shape();
                let dims = shape.dimensions();
                let quant_info = if let Some(q) = tensor.quantization_parameters() {
                    format!("scale={}, zero_point={}", q.scale, q.zero_point)
                } else {
                    "none".to_string()
                };
                println!(
                    "    output[{}]: shape={:?} type={:?} quant=[{}]",
                    idx,
                    dims,
                    tensor.data_type(),
                    quant_info
                );
            }

            (ts, vs, out_idx, sv, zp)
        };

        // Blank token is at vocab_size - 1 (last index)
        // For hanijpan: vocab=8178, vocab_size=8179, blank_idx=8178
        // For und: vocab=1292, vocab_size=1293, blank_idx=1292
        let blank_idx = vocab_size - 1;

        println!(
            "  [{}] Output: time_steps={}, vocab_size={}, blank_idx={}, scale={}, zero_point={}",
            model_name, time_steps, vocab_size, blank_idx, scale_val, zero_point
        );

        // LEFT_MARGIN from Chrome config protobuf:
        // hanijpan: 12 (from IDA analysis, assertion: left_padding % frame_width == 0)
        // gocr_mobile_und: 8 (from gocr_mobile_und_config.pb: field 1 varint 8)
        let left_margin = match model_name {
            "gocr_mobile_und" => 8,
            _ => LEFT_MARGIN, // 12 for hanijpan and others
        };

        Ok(Self {
            model,
            vocab,
            time_steps,
            vocab_size,
            blank_idx,
            output_idx,
            scale_val,
            zero_point,
            left_margin,
            model_name: model_name.to_string(),
        })
    }

    /// Create the default CJK (hanijpan) recognizer
    pub fn new(model_dir: &Path) -> Result<Self> {
        Self::new_with_model(model_dir, "hanijpan")
    }

    /// Effective width for content (MODEL_WIDTH - left_margin)
    fn effective_width(&self) -> u32 {
        MODEL_WIDTH - self.left_margin
    }

    /// Canvas fill (padding) value
    /// Chrome uses memset(0) for tensor initialization (ConvertPixaToTensors at 0x1802AB450)
    /// Both byte and float paths zero-fill the tensor before copying image data.
    fn canvas_fill(&self) -> Luma<u8> {
        Luma([0u8])
    }

    /// Create a new interpreter for batch processing
    pub fn create_interpreter(&self) -> Result<Interpreter> {
        let mut options = Options::default();
        options.thread_count = 4; // Use 4 threads
                                  // Try without XNNPACK if env var set, to test if it causes issues
        if std::env::var("CHROME_OCR_NO_XNNPACK").is_err() {
            options.is_xnnpack_enabled = true;
        }
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

        if ideal_w <= self.effective_width() {
            // Short line: direct recognition
            self.recognize_segment_with_interpreter(image, interpreter)
        } else {
            // Long line: Chrome's method - merge logits at tensor level, then unified CTC decode
            // From IDA analysis:
            // - chunk_border_left = 0.3, chunk_border_right = 0.3 (30% overlap each side)
            // - MergeChunkResults: direct copy using chunk_lengths array
            // - Key insight: Chrome tracks exact pixel boundaries, not just ratios

            let chunk_w = (self.effective_width() as f32 * h as f32 / MODEL_HEIGHT as f32) as u32;

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
                let (text, avg_conf) = if self.uses_beam_search() {
                    self.ctc_beam_search(&chunks[0].logits)
                } else {
                    self.ctc_decode_logits(&chunks[0].logits)
                };
                return Ok((text, avg_conf));
            }

            // Merge logits based on non-overlapping regions
            // Calculate "valid" region for each chunk (excluding overlap)
            let merged_logits = self.merge_chunk_logits_by_boundary(&chunks, w);

            // Unified CTC decode on merged logits
            let (text, avg_conf) = if self.uses_beam_search() {
                self.ctc_beam_search(&merged_logits)
            } else {
                self.ctc_decode_logits(&merged_logits)
            };

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
        let new_w = ((w as f32 * scale) as u32).min(MODEL_WIDTH - self.left_margin);

        let scaled = image::imageops::resize(
            image,
            new_w,
            MODEL_HEIGHT,
            image::imageops::FilterType::Triangle, // Chrome uses Leptonica bilinear scaling (pixScale at 0x18078E140)
        );

        // Create canvas with left margin for model alignment
        // Chrome uses memset(0) for tensor initialization (ConvertPixaToTensors at 0x1802AB450)
        // Padding is ZERO (black), NOT 255 (white)
        let mut canvas: GrayImage =
            ImageBuffer::from_pixel(MODEL_WIDTH, MODEL_HEIGHT, self.canvas_fill());
        image::imageops::overlay(&mut canvas, &scaled, self.left_margin as i64, 0);

        // Run inference
        let mut input_data: Vec<u8> = canvas.as_raw().to_vec();

        // Debug: optionally invert pixels to test if model expects different orientation
        if std::env::var("CHROME_OCR_INVERT").is_ok() {
            for px in input_data.iter_mut() {
                *px = 255 - *px;
            }
        }

        let input_tensor = interpreter.input(0)?;
        input_tensor.set_data(&input_data)?;
        interpreter.invoke()?;

        // Get and dequantize logits
        let output_tensor = interpreter.output(self.output_idx)?;
        let logits_raw: &[u8] = output_tensor.data();

        // Debug: dump first invocation's input and output for comparison
        if std::env::var("CHROME_OCR_DUMP").is_ok() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static DUMP_COUNT: AtomicUsize = AtomicUsize::new(0);
            let count = DUMP_COUNT.fetch_add(1, Ordering::Relaxed);
            if count == 0 {
                let dump_dir = std::path::Path::new("dump_debug");
                std::fs::create_dir_all(dump_dir).ok();
                std::fs::write(dump_dir.join("input_canvas.bin"), &input_data).ok();
                std::fs::write(dump_dir.join("output_raw.bin"), logits_raw).ok();
                eprintln!(
                    "[DUMP] Saved input ({} bytes) and output ({} bytes) to dump_debug/",
                    input_data.len(),
                    logits_raw.len()
                );
                eprintln!(
                    "[DUMP] Model: {}, output_idx: {}",
                    self.model_name, self.output_idx
                );
                eprintln!(
                    "[DUMP] Raw output range: [{}, {}]",
                    logits_raw.iter().cloned().min().unwrap_or(0),
                    logits_raw.iter().cloned().max().unwrap_or(0)
                );
                eprintln!(
                    "[DUMP] Values at 255: {}",
                    logits_raw.iter().filter(|&&x| x == 255).count()
                );
            }
        }

        // Calculate content-based time step range to avoid decoding padding
        let pixels_per_step = MODEL_WIDTH as f32 / self.time_steps as f32;
        let content_start_t = (self.left_margin as f32 / pixels_per_step).floor() as usize;
        let content_end_pixel = self.left_margin + new_w;
        let content_end_t =
            ((content_end_pixel as f32 / pixels_per_step).ceil() as usize + 1).min(self.time_steps);

        let mut logits = vec![0.0f32; self.time_steps * self.vocab_size];
        for t in 0..self.time_steps {
            let base = t * self.vocab_size;
            if t >= content_start_t && t < content_end_t {
                for i in 0..self.vocab_size {
                    let raw = logits_raw[base + i] as i32;
                    logits[base + i] = (raw - self.zero_point) as f32 * self.scale_val;
                }
            } else {
                // Force blank for padding region time steps
                for i in 0..self.vocab_size {
                    logits[base + i] = if i == self.blank_idx { 10.0 } else { -10.0 };
                }
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

            // CTC decoding: skip blank and repeated
            // Index 0 is space ' ' in UND model — NOT blank. Blank = vocab_size-1.
            if max_idx != self.blank_idx && Some(max_idx) != prev_idx {
                if char_conf >= self.char_conf_threshold() && max_idx < self.vocab.len() {
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

    /// CTC prefix beam search decoder (from IDA: CTCDecoder::Decode at 0x18058AFD0)
    /// NOTE: Chrome's UND config has use_beam_search=false. This is kept for reference
    /// but not used in the normal code path.
    #[allow(dead_code)]
    fn ctc_beam_search(&self, logits: &[f32]) -> (String, f32) {
        const BEAM_WIDTH: usize = 25;
        const BEAM_CHAR_THRESHOLD: f32 = 0.001;
        if logits.is_empty() {
            return (String::new(), 0.0);
        }

        let total_frames = logits.len() / self.vocab_size;

        // Each beam: (prefix as Vec<usize>, p_blank in log, p_non_blank in log)
        // Use log probabilities to avoid underflow
        let neg_inf = f64::NEG_INFINITY;

        // Initialize with empty prefix
        // key: prefix (as character indices), value: (log_p_blank, log_p_non_blank)
        let mut beams: HashMap<Vec<usize>, (f64, f64)> = HashMap::new();
        beams.insert(Vec::new(), (0.0, neg_inf)); // empty prefix, p_blank=1.0 (log=0)

        for t in 0..total_frames {
            let base = t * self.vocab_size;

            // Compute log softmax for this timestep
            let mut max_logit = f32::NEG_INFINITY;
            for i in 0..self.vocab_size {
                let val = logits[base + i];
                if val > max_logit {
                    max_logit = val;
                }
            }

            let mut log_probs = vec![0.0f64; self.vocab_size];
            let mut exp_sum = 0.0f64;
            for i in 0..self.vocab_size {
                let e = ((logits[base + i] - max_logit) as f64).exp();
                exp_sum += e;
            }
            let log_sum = exp_sum.ln();
            for i in 0..self.vocab_size {
                log_probs[i] = (logits[base + i] - max_logit) as f64 - log_sum;
            }

            let log_p_blank = log_probs[self.blank_idx];

            // Find top-K characters by probability for efficiency
            // Instead of extending with all 1293 chars, only use top ones
            let mut char_indices: Vec<usize> = (0..self.vocab_size)
                .filter(|&i| {
                    i != self.blank_idx && log_probs[i] > (BEAM_CHAR_THRESHOLD as f64).ln()
                })
                .collect();
            char_indices.sort_by(|&a, &b| {
                log_probs[b]
                    .partial_cmp(&log_probs[a])
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            // Limit to top chars to avoid explosion
            char_indices.truncate(40);

            let mut new_beams: HashMap<Vec<usize>, (f64, f64)> = HashMap::new();

            for (prefix, (pb, pnb)) in &beams {
                let p_total = log_add(*pb, *pnb);

                // 1. Extend with blank → same prefix
                let new_pb = p_total + log_p_blank;
                let entry = new_beams
                    .entry(prefix.clone())
                    .or_insert((neg_inf, neg_inf));
                entry.0 = log_add(entry.0, new_pb);

                // 2. Extend with each character
                for &c in &char_indices {
                    let log_p_c = log_probs[c];
                    let last_char = prefix.last().copied();

                    if Some(c) == last_char {
                        // Same as last character in prefix:
                        // - Via blank path: extends prefix (new character instance)
                        // - Via non-blank path: stays same prefix (CTC repeat)
                        let new_pnb_extend = *pb + log_p_c; // blank → c = new instance
                        let new_pnb_repeat = *pnb + log_p_c; // c → c = CTC repeat (same prefix)

                        // CTC repeat stays same prefix
                        let entry = new_beams
                            .entry(prefix.clone())
                            .or_insert((neg_inf, neg_inf));
                        entry.1 = log_add(entry.1, new_pnb_repeat);

                        // Blank → c creates new instance (extended prefix)
                        let mut extended = prefix.clone();
                        extended.push(c);
                        let entry = new_beams.entry(extended).or_insert((neg_inf, neg_inf));
                        entry.1 = log_add(entry.1, new_pnb_extend);
                    } else {
                        // Different character: extend prefix
                        let new_pnb = p_total + log_p_c;
                        let mut extended = prefix.clone();
                        extended.push(c);
                        let entry = new_beams.entry(extended).or_insert((neg_inf, neg_inf));
                        entry.1 = log_add(entry.1, new_pnb);
                    }
                }
            }

            // Prune to top BEAM_WIDTH beams by total log probability
            let mut beam_vec: Vec<(Vec<usize>, (f64, f64))> = new_beams.into_iter().collect();
            beam_vec.sort_by(|a, b| {
                let total_a = log_add(a.1 .0, a.1 .1);
                let total_b = log_add(b.1 .0, b.1 .1);
                total_b
                    .partial_cmp(&total_a)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            beam_vec.truncate(BEAM_WIDTH);

            beams = beam_vec.into_iter().collect();
        }

        // Find best beam
        let best = beams
            .iter()
            .max_by(|a, b| {
                let total_a = log_add(a.1 .0, a.1 .1);
                let total_b = log_add(b.1 .0, b.1 .1);
                total_a
                    .partial_cmp(&total_b)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(prefix, _)| prefix.clone())
            .unwrap_or_default();

        // Convert indices to string
        let mut result = String::new();
        let mut conf_sum = 0.0f32;
        for &idx in &best {
            if idx < self.vocab.len() {
                result.push_str(&self.vocab[idx]);
                conf_sum += 1.0; // beam search confidence is hard to compute per-char
            }
        }

        let avg_conf = if best.is_empty() {
            0.0
        } else {
            // Use the total beam probability as confidence proxy
            let (pb, pnb) = beams.get(&best).copied().unwrap_or((neg_inf, neg_inf));
            let total_log_prob = log_add(pb, pnb);
            // Normalize by number of frames to get per-frame confidence
            let per_frame = total_log_prob / total_frames as f64;
            per_frame.exp() as f32
        };

        (result, avg_conf.max(0.01)) // minimum confidence so it passes filters
    }

    /// Get the character confidence threshold for this model
    /// Chrome UND config: char_score_threshold=0 (no filtering)
    /// CJK models: use 0.15 to filter noise
    fn char_conf_threshold(&self) -> f32 {
        if self.model_name == "gocr_mobile_und" {
            UND_CHAR_CONF_THRESHOLD
        } else {
            CHAR_CONF_THRESHOLD
        }
    }

    /// Check if this recognizer should use beam search
    /// From IDA: Chrome uses CTC Beam Search via CTCDecoder::Decode (sub_18058AFD0)
    /// with NegativeLogitsScore preprocessing
    pub fn uses_beam_search(&self) -> bool {
        // Chrome UND config has use_beam_search=false (greedy CTC decode)
        // Only hanijpan uses beam search
        self.model_name == "hanijpan"
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
        // Calculate content-based time step range to avoid decoding padding
        let (w, h) = (image.width(), image.height());
        let scale = MODEL_HEIGHT as f32 / h as f32;
        let new_w = ((w as f32 * scale) as u32).min(MODEL_WIDTH - self.left_margin);
        // Content occupies pixels [left_margin, left_margin + new_w]
        // Each time step = MODEL_WIDTH / time_steps pixels
        let pixels_per_step = MODEL_WIDTH as f32 / self.time_steps as f32;
        // Start: first time step that overlaps with content
        let start_t =
            ((self.left_margin as f32 / pixels_per_step).floor() as usize).min(self.time_steps);
        // End: last time step that overlaps with content + 1 safety margin
        let content_end_pixel = self.left_margin + new_w;
        let end_t =
            ((content_end_pixel as f32 / pixels_per_step).ceil() as usize + 1).min(self.time_steps);
        self.recognize_segment_trimmed(image, interpreter, start_t, end_t)
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
        let new_w = ((w as f32 * scale) as u32).min(MODEL_WIDTH - self.left_margin);

        let scaled = image::imageops::resize(
            image,
            new_w,
            MODEL_HEIGHT,
            image::imageops::FilterType::Triangle, // Chrome uses Leptonica bilinear scaling (pixScale at 0x18078E140)
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
        // Chrome uses memset(0) for tensor padding (ConvertPixaToTensors at 0x1802AB450)
        let mut canvas: GrayImage =
            ImageBuffer::from_pixel(MODEL_WIDTH, MODEL_HEIGHT, self.canvas_fill());
        image::imageops::overlay(&mut canvas, &scaled, self.left_margin as i64, 0);

        // Debug: save first few canvases
        if std::env::var("CHROME_OCR_SAVE_CANVAS").is_ok() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static COUNTER: AtomicUsize = AtomicUsize::new(0);
            let n = COUNTER.fetch_add(1, Ordering::Relaxed);
            if n < 10 {
                let fname = format!("debug_canvas_{}_{}.png", self.model_name, n);
                canvas.save(&fname).ok();
                eprintln!(
                    "  [SAVE] {} ({}x{}, left_margin={})",
                    fname, new_w, MODEL_HEIGHT, self.left_margin
                );
            }
        }

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
        let new_w = ((w as f32 * scale) as u32).min(MODEL_WIDTH - self.left_margin);

        let scaled = image::imageops::resize(
            image,
            new_w,
            MODEL_HEIGHT,
            image::imageops::FilterType::Triangle, // Chrome uses Leptonica bilinear scaling (pixScale at 0x18078E140)
        );

        // Create canvas with left margin for model alignment
        // Chrome uses memset(0) for tensor padding (ConvertPixaToTensors at 0x1802AB450)
        let mut canvas: GrayImage =
            ImageBuffer::from_pixel(MODEL_WIDTH, MODEL_HEIGHT, self.canvas_fill());
        image::imageops::overlay(&mut canvas, &scaled, self.left_margin as i64, 0);

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

            // CTC decoding: skip blank and repeated
            // Index 0 is space ' ' in UND model — NOT blank. Blank = vocab_size-1.
            if max_idx != self.blank_idx && Some(max_idx) != prev_idx {
                if char_conf >= self.char_conf_threshold() && max_idx < self.vocab.len() {
                    let c = &self.vocab[max_idx];
                    if !c.is_empty() {
                        // Calculate character position in original image coordinates
                        // Time step t corresponds to canvas pixel [t*FRAME_WIDTH, (t+1)*FRAME_WIDTH)
                        // Subtract LEFT_MARGIN to get content-relative position
                        let canvas_x0 = t as f32 * FRAME_WIDTH as f32;
                        let canvas_x1 = (t + 1) as f32 * FRAME_WIDTH as f32;

                        // Convert to content-relative (subtract LEFT_MARGIN, clamp to content bounds)
                        let content_x0 = (canvas_x0 - self.left_margin as f32)
                            .max(0.0)
                            .min(content_pixels);
                        let content_x1 = (canvas_x1 - self.left_margin as f32)
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
        let mut input_data: Vec<u8> = canvas.as_raw().to_vec();

        // Debug: optionally invert pixels
        if std::env::var("CHROME_OCR_INVERT").is_ok() {
            for px in input_data.iter_mut() {
                *px = 255 - *px;
            }
        }

        // Set input tensor
        let input_tensor = interpreter.input(0)?;
        input_tensor.set_data(&input_data)?;

        // Run inference
        interpreter.invoke()?;

        // Get logits output using cached index
        let output_tensor = interpreter.output(self.output_idx)?;
        let logits_raw: &[u8] = output_tensor.data();

        // Dequantize logits using cached params
        // Extract only the valid time steps (start_t to end_t)
        let actual_end = end_t.min(self.time_steps);
        let trimmed_frames = actual_end - start_t;
        let mut logits = vec![0.0f32; trimmed_frames * self.vocab_size];
        for t in start_t..actual_end {
            let src_base = t * self.vocab_size;
            let dst_base = (t - start_t) * self.vocab_size;
            for i in 0..self.vocab_size {
                let raw = logits_raw[src_base + i] as i32;
                logits[dst_base + i] = (raw - self.zero_point) as f32 * self.scale_val;
            }
        }

        // Debug: dump top predictions per timestep for the first few calls
        if std::env::var("CHROME_OCR_DUMP_LOGITS").is_ok() {
            use std::sync::atomic::{AtomicUsize, Ordering};
            static DUMP_CTR: AtomicUsize = AtomicUsize::new(0);
            let dump_n = DUMP_CTR.fetch_add(1, Ordering::Relaxed);
            if dump_n < 3 {
                eprintln!(
                    "  [DUMP] Logits for call #{}, frames={}",
                    dump_n, trimmed_frames
                );
                for t in 0..trimmed_frames.min(10) {
                    let base = t * self.vocab_size;
                    let mut top3: Vec<(usize, f32)> = (0..self.vocab_size)
                        .map(|i| (i, logits[base + i]))
                        .collect();
                    top3.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
                    let top3_str: Vec<String> = top3[..3.min(top3.len())]
                        .iter()
                        .map(|(idx, val)| {
                            let ch = if *idx == self.blank_idx {
                                "<BLK>".to_string()
                            } else if *idx < self.vocab.len() {
                                self.vocab[*idx].clone()
                            } else {
                                format!("?{}", idx)
                            };
                            format!("{}({:.2})", ch, val)
                        })
                        .collect();
                    eprintln!("    t={}: {}", t, top3_str.join(" | "));
                }
            }
        }

        // Decode using beam search or greedy based on model type
        if self.uses_beam_search() {
            let (text, conf) = self.ctc_beam_search(&logits);
            Ok((text, conf))
        } else {
            let (text, conf) = self.ctc_decode_logits(&logits);
            Ok((text, conf))
        }
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

/// Log-space addition: log(exp(a) + exp(b))
/// Numerically stable implementation
fn log_add(a: f64, b: f64) -> f64 {
    if a == f64::NEG_INFINITY {
        return b;
    }
    if b == f64::NEG_INFINITY {
        return a;
    }
    let max = a.max(b);
    max + ((a - max).exp() + (b - max).exp()).ln()
}
