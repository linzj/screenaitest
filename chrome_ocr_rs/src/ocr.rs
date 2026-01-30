use anyhow::Result;
use image::{GrayImage, Luma};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use std::path::Path;
use std::time::Instant;

#[allow(unused_imports)]
use crate::cluster_sort::ClusterSort;
use crate::detector::TextDetector;
use crate::recognizer::LineRecognizer;
use crate::sorter::LayoutSorter;
use crate::utils::{calc_common_chars_pct, calc_containment, calc_iou, BBox};

const MIN_HEIGHT: u32 = 20; // Reduced: model only needs 32px input height, 40px caused cross-line merging
const TARGET_SIZE: u32 = 4096;

#[derive(Default)]
pub struct PerfStats {
    pub load_detector: f64,
    pub load_sorter: f64,
    pub load_recognizer: f64,
    pub detection: f64,
    pub merge: f64,
    pub sorting: f64,
    pub recognition_total: f64,
    pub recognition_count: usize,
    pub ocr_total: f64,
}

pub struct ChromeOCR {
    detector: TextDetector,
    recognizer: LineRecognizer,
    recognizer_und: Option<LineRecognizer>, // Universal (Latin) recognizer
    sorter: LayoutSorter,
    cluster_sort: ClusterSort,
    save_lines: bool,
    min_conf: f32,
    perf: bool,
    stats: PerfStats,
}

impl ChromeOCR {
    pub fn new(model_dir: &Path, perf: bool) -> Result<Self> {
        println!("Loading models...");

        let mut stats = PerfStats::default();

        let t0 = Instant::now();
        let detector = TextDetector::new(model_dir)?;
        stats.load_detector = t0.elapsed().as_secs_f64();
        println!("  TextDetector: loaded");

        let t1 = Instant::now();
        let sorter = LayoutSorter::new(model_dir)?;
        stats.load_sorter = t1.elapsed().as_secs_f64();
        println!("  LayoutSorter: loaded");

        let t2 = Instant::now();
        let recognizer = LineRecognizer::new(model_dir)?;
        stats.load_recognizer = t2.elapsed().as_secs_f64();
        println!("  LineRecognizer: loaded");

        // Try to load universal (Latin/und) recognizer
        let recognizer_und = match LineRecognizer::new_with_model(model_dir, "gocr_mobile_und") {
            Ok(r) => {
                println!("  LineRecognizer [und]: loaded");
                Some(r)
            }
            Err(e) => {
                println!("  LineRecognizer [und]: not available ({})", e);
                None
            }
        };

        let cluster_sort = ClusterSort::new(model_dir)?;
        println!("  ClusterSort: loaded");

        println!("All models loaded!");

        Ok(Self {
            detector,
            recognizer,
            recognizer_und,
            sorter,
            cluster_sort,
            save_lines: false,
            min_conf: 0.0, // Chrome doesn't filter by overall line confidence; uses per-char junk filter instead
            perf,
            stats,
        })
    }

    pub fn print_load_stats(&self) {
        println!(
            "  TextDetector:   {:7.1} ms",
            self.stats.load_detector * 1000.0
        );
        println!(
            "  LayoutSorter:   {:7.1} ms",
            self.stats.load_sorter * 1000.0
        );
        println!(
            "  LineRecognizer: {:7.1} ms",
            self.stats.load_recognizer * 1000.0
        );
    }

    pub fn print_ocr_stats(&self) {
        println!("  Detection:      {:7.1} ms", self.stats.detection * 1000.0);
        println!("  Merge boxes:    {:7.1} ms", self.stats.merge * 1000.0);
        println!("  Sorting:        {:7.1} ms", self.stats.sorting * 1000.0);
        let avg = if self.stats.recognition_count > 0 {
            self.stats.recognition_total / self.stats.recognition_count as f64
        } else {
            0.0
        };
        println!(
            "  Recognition:    {:7.1} ms ({} lines, avg {:.1} ms/line)",
            self.stats.recognition_total * 1000.0,
            self.stats.recognition_count,
            avg * 1000.0
        );
        println!("  Total OCR:      {:7.1} ms", self.stats.ocr_total * 1000.0);
    }

    pub fn get_ocr_total(&self) -> f64 {
        self.stats.ocr_total
    }

    pub fn set_save_lines(&mut self, save: bool) {
        self.save_lines = save;
    }

    pub fn set_min_conf(&mut self, conf: f32) {
        self.min_conf = conf;
    }

    /// Perform OCR on an image
    pub fn ocr(&mut self, image_path: &Path) -> Result<Vec<String>> {
        let ocr_start = Instant::now();

        println!("Opening image: {}", image_path.display());
        let image = image::open(image_path)
            .map_err(|e| anyhow::anyhow!("Failed to open image {}: {}", image_path.display(), e))?
            .to_luma8();
        let (width, height) = (image.width(), image.height());
        println!("Image: {}x{}", width, height);

        // Step 1: Text Detection
        println!(
            "\n[1/3] Text Detection (on {}x{})...",
            TARGET_SIZE, TARGET_SIZE
        );
        let t0 = Instant::now();
        let boxes = self.detector.detect(&image, 0.3)?;
        self.stats.detection = t0.elapsed().as_secs_f64();
        println!("  Found {} char-level regions", boxes.len());
        println!("  Scale factor: {:.2}x", self.detector.scale);

        // Merge boxes to lines
        let t1 = Instant::now();
        let merged = self.merge_boxes_to_lines(&boxes);
        self.stats.merge = t1.elapsed().as_secs_f64();
        println!("  Merged to {} lines", merged.len());

        // Print merged line dimensions
        if std::env::var("CHROME_OCR_DEBUG").is_ok() {
            let dbg_scale = self.detector.scale;
            let dbg_ox = self.detector.offset_x;
            let dbg_oy = self.detector.offset_y;
            for (i, bbox) in merged.iter().enumerate() {
                let ox1 = ((bbox.x1 - dbg_ox) / dbg_scale) as i32;
                let oy1 = ((bbox.y1 - dbg_oy) / dbg_scale) as i32;
                let ox2 = ((bbox.x2 - dbg_ox) / dbg_scale) as i32;
                let oy2 = ((bbox.y2 - dbg_oy) / dbg_scale) as i32;
                let angle_deg = bbox.angle.to_degrees();
                println!(
                    "    M{}: ({},{})→({},{}) {}x{} angle={:.1}°",
                    i + 1,
                    ox1,
                    oy1,
                    ox2,
                    oy2,
                    ox2 - ox1,
                    oy2 - oy1,
                    angle_deg
                );
            }
        }

        // Step 2: Layout Sorting
        println!("\n[2/3] Layout Sorting...");
        let t2 = Instant::now();
        let sorted_lines = self.sorter.sort(&merged, (TARGET_SIZE, TARGET_SIZE))?;
        self.stats.sorting = t2.elapsed().as_secs_f64();

        // Step 3: Line Recognition
        println!("\n[3/3] Line Recognition (from original image)...");

        let scale = self.detector.scale;
        let offset_x = self.detector.offset_x;
        let offset_y = self.detector.offset_y;

        // Create output directory for line images (in current directory like Python)
        let lines_dir = if self.save_lines {
            let stem = image_path
                .file_stem()
                .unwrap_or_default()
                .to_str()
                .unwrap_or("output");
            let dir = Path::new(".").join(format!("{}_lines", stem));
            std::fs::create_dir_all(&dir)?;
            println!("  Saving line images to: {}/", dir.display());
            Some(dir)
        } else {
            None
        };

        let mut results = Vec::new();
        let mut line_num = 0;
        let mut recognized_lines: Vec<(i32, i32, u32, u32, f32)> = Vec::new();
        let rec_start = Instant::now();
        let mut rec_count = 0usize;

        // Create interpreters once for all recognitions
        let rec_interpreter = self.recognizer.create_interpreter()?;
        let rec_interpreter_und = match &self.recognizer_und {
            Some(r) => Some(r.create_interpreter()?),
            None => None,
        };

        for bbox in &sorted_lines {
            // Convert 4096 coordinates to original image coordinates
            let mut x1 = ((bbox.x1 - offset_x) / scale) as i32;
            let mut y1 = ((bbox.y1 - offset_y) / scale) as i32;
            let mut x2 = ((bbox.x2 - offset_x) / scale) as i32;
            let mut y2 = ((bbox.y2 - offset_y) / scale) as i32;

            // Chrome padding from IDA analysis of sub_18048ACD0 (PadAndScaleBoxes):
            // Width: max(4.0, min(16.0, box_h_4096 * box_width_padding)) in 4096-space
            // Config box_width_padding=0.0 (default, not overridden in gocr config)
            // So effective padding = 4.0 in 4096-space, split 50/50 left/right
            // Height: max(1.0, min(8.0, box_h_4096 * factor)) = 1.0 in 4096-space
            let box_h_4096 = bbox.y2 - bbox.y1;
            let pad_x_4096 = (box_h_4096 * 0.0_f32).clamp(4.0, 16.0); // = 4.0
            let pad_y_4096 = (box_h_4096 * 0.0_f32).clamp(1.0, 8.0); // = 1.0
            let pad_x = (pad_x_4096 / scale) as i32;
            let pad_y = (pad_y_4096 / scale) as i32;
            x1 = (x1 - pad_x).max(0);
            y1 = (y1 - pad_y).max(0);
            x2 = (x2 + pad_x).min(width as i32);
            y2 = (y2 + pad_y).min(height as i32);

            let box_w = (x2 - x1) as f32;
            let box_h = (y2 - y1) as f32;

            // Filter minimum box dimensions
            if box_w < 3.0 || box_h < 3.0 {
                continue;
            }

            // Expand short lines - but only if wide enough (not small table cells)
            // For small boxes (both dimensions small), the recognizer scales to 32 height
            // which handles them fine. Expanding Y adds blank context that makes the text
            // proportionally tiny.
            let line_height = (y2 - y1) as u32;
            if line_height < MIN_HEIGHT && box_w > 40.0 {
                let expand = ((MIN_HEIGHT - line_height) / 2 + 3) as i32;
                y1 = (y1 - expand).max(0);
                y2 = (y2 + expand).min(height as i32);
            }

            // Crop region with extra padding for rotation if needed
            let angle = bbox.angle;
            let need_rotation = angle.abs() > 0.05; // > ~3 degrees

            // For rotation: the bbox is axis-aligned and contains the rotated text
            // We need padding to ensure the text isn't cut off during deskewing
            let (pad_x, pad_y) = if need_rotation {
                let w = (x2 - x1) as f32;
                let h = (y2 - y1) as f32;
                let sin_a = angle.abs().sin();
                let cos_a = angle.abs().cos();
                // After rotation, the bbox expands: new_w = w*cos + h*sin
                // Add generous padding to ensure nothing is cut off
                let extra_w = (h * sin_a + w * (1.0 - cos_a)) / 2.0 + 20.0;
                let extra_h = (w * sin_a + h * (1.0 - cos_a)) / 2.0 + 15.0;
                (extra_w as i32, extra_h as i32)
            } else {
                (0, 0)
            };

            let crop_x1 = (x1 - pad_x).max(0);
            let crop_y1 = (y1 - pad_y).max(0);
            let crop_x2 = (x2 + pad_x).min(width as i32);
            let crop_y2 = (y2 + pad_y).min(height as i32);

            let region = image::imageops::crop_imm(
                &image,
                crop_x1 as u32,
                crop_y1 as u32,
                (crop_x2 - crop_x1) as u32,
                (crop_y2 - crop_y1) as u32,
            )
            .to_image();

            // Deskew region if significant rotation detected
            // Use rotate_about_center, then track bbox center to extract the right region
            let region = if need_rotation {
                // Calculate where the original bbox center is within the cropped region
                let bbox_w = (x2 - x1) as f32;
                let bbox_h = (y2 - y1) as f32;
                let crop_w = region.width() as f32;
                let crop_h = region.height() as f32;

                // Bbox center relative to crop origin
                let bbox_cx_in_crop = (x1 - crop_x1) as f32 + bbox_w / 2.0;
                let bbox_cy_in_crop = (y1 - crop_y1) as f32 + bbox_h / 2.0;

                // Crop center
                let crop_cx = crop_w / 2.0;
                let crop_cy = crop_h / 2.0;

                // Offset from crop center to bbox center
                let offset_x = bbox_cx_in_crop - crop_cx;
                let offset_y = bbox_cy_in_crop - crop_cy;

                // Rotate the entire crop around its center
                let rotated =
                    rotate_about_center(&region, -angle, Interpolation::Bilinear, Luma([255u8]));
                let (rw, rh) = (rotated.width(), rotated.height());

                // After rotation, the offset from center also rotates
                let cos_a = (-angle).cos();
                let sin_a = (-angle).sin();
                let new_offset_x = offset_x * cos_a - offset_y * sin_a;
                let new_offset_y = offset_x * sin_a + offset_y * cos_a;

                // Bbox center in rotated image
                let rotated_cx = rw as f32 / 2.0;
                let rotated_cy = rh as f32 / 2.0;
                let bbox_cx_in_rotated = rotated_cx + new_offset_x;
                let bbox_cy_in_rotated = rotated_cy + new_offset_y;

                // For near-vertical text (angle > 45°), after rotation the original
                // narrow width becomes the text height. Must use rotated dimensions.
                // For moderate angles (< 45°), original dimensions are tighter and work well.
                let is_near_vertical = angle.abs() > std::f32::consts::FRAC_PI_4;
                let (crop_base_w, crop_base_h) = if is_near_vertical {
                    let rot_cos = cos_a.abs();
                    let rot_sin = sin_a.abs();
                    (
                        bbox_w * rot_cos + bbox_h * rot_sin,
                        bbox_w * rot_sin + bbox_h * rot_cos,
                    )
                } else {
                    (bbox_w, bbox_h)
                };

                let angle_factor = (angle.abs() * 3.0).min(1.0);
                let edge_margin_x = 3.0 + angle_factor * 4.0; // 3-7 pixels
                let edge_margin_y = 2.0 + angle_factor * 2.0; // 2-4 pixels
                let crop_w = crop_base_w + edge_margin_x * 2.0;
                let crop_h = crop_base_h + edge_margin_y * 2.0;
                let tx1 = ((bbox_cx_in_rotated - crop_w / 2.0).max(0.0)) as u32;
                let ty1 = ((bbox_cy_in_rotated - crop_h / 2.0).max(0.0)) as u32;
                let tx2 = ((bbox_cx_in_rotated + crop_w / 2.0).min(rw as f32)) as u32;
                let ty2 = ((bbox_cy_in_rotated + crop_h / 2.0).min(rh as f32)) as u32;

                if tx2 > tx1 + 10 && ty2 > ty1 + 5 {
                    image::imageops::crop_imm(&rotated, tx1, ty1, tx2 - tx1, ty2 - ty1).to_image()
                } else {
                    rotated
                }
            } else {
                region
            };

            // Split multi-line regions
            let sub_regions = self.split_multiline_region(&region);

            for (sub_region, sub_y_offset) in sub_regions {
                let mut actual_y = y1 + sub_y_offset as i32;
                let mut sub_h = sub_region.height();
                let sub_w = sub_region.width();

                // Expand short sub-regions (only if wide enough)
                let sub_region = if sub_h < MIN_HEIGHT && sub_w > 40 {
                    let expand_sub = ((MIN_HEIGHT - sub_h) / 2 + 3) as i32;
                    let new_y1 = (actual_y - expand_sub).max(0);
                    let new_y2 = (actual_y + sub_h as i32 + expand_sub).min(height as i32);
                    actual_y = new_y1;
                    sub_h = (new_y2 - new_y1) as u32;

                    image::imageops::crop_imm(
                        &image,
                        x1 as u32,
                        new_y1 as u32,
                        (x2 - x1) as u32,
                        sub_h,
                    )
                    .to_image()
                } else {
                    sub_region
                };

                // Check for duplicates using Chrome's RemoveOverlaps parameters from IDA:
                // RemoveOverlapsWordPruningStep: line_overlap_iou_threshold = 0.6
                // RemoveMultiByOverlap: overlap_threshold = 0.6, max_breadth_ratio = 2
                // Chrome requires breadth ratio similarity before removing contained boxes
                let is_duplicate =
                    recognized_lines
                        .iter()
                        .any(|&(prev_x, prev_y, prev_w, prev_h, _)| {
                            let b1 = [
                                x1 as f32,
                                actual_y as f32,
                                (x1 as u32 + sub_w) as f32,
                                (actual_y as u32 + sub_h) as f32,
                            ];
                            let b2 = [
                                prev_x as f32,
                                prev_y as f32,
                                (prev_x as u32 + prev_w) as f32,
                                (prev_y as u32 + prev_h) as f32,
                            ];

                            let w1 = b1[2] - b1[0];
                            let w2 = b2[2] - b2[0];
                            let breadth_ratio = if w1.max(w2) > 0.0 {
                                w1.min(w2) / w1.max(w2)
                            } else {
                                0.0
                            };

                            // Calculate IoU (Chrome: line_overlap_iou_threshold = 0.6)
                            let iou = calc_iou(&b1, &b2);
                            if iou > 0.6 {
                                return true;
                            }

                            // Calculate containment (Chrome: overlap_threshold = 0.6)
                            // Chrome only removes contained boxes when breadth ratio >= 0.5
                            // (max_breadth_ratio = 2 means min ratio = 1/2 = 0.5)
                            let containment1 = calc_containment(&b1, &b2);
                            let containment2 = calc_containment(&b2, &b1);
                            if (containment1 > 0.6 || containment2 > 0.6) && breadth_ratio > 0.5 {
                                return true;
                            }

                            // Check vertical + horizontal overlap for same-row detection
                            if breadth_ratio < 0.5 {
                                return false;
                            }

                            let y_overlap = (b1[3].min(b2[3]) - b1[1].max(b2[1])).max(0.0);
                            let min_h = (b1[3] - b1[1]).min(b2[3] - b2[1]);
                            let y_overlap_ratio = if min_h > 0.0 { y_overlap / min_h } else { 0.0 };

                            let x_overlap = (b1[2].min(b2[2]) - b1[0].max(b2[0])).max(0.0);
                            let x_overlap_ratio = if w1.min(w2) > 0.0 {
                                x_overlap / w1.min(w2)
                            } else {
                                0.0
                            };

                            y_overlap_ratio > 0.6 && x_overlap_ratio > 0.5
                        });

                if is_duplicate {
                    if std::env::var("CHROME_OCR_DEBUG").is_ok() {
                        println!(
                            "    [SKIP] y={} duplicate (x={} w={} h={})",
                            actual_y, x1, sub_w, sub_h
                        );
                    }
                    continue;
                }

                // Recognize using shared interpreter(s)
                // Try CJK (hanijpan) model first, then universal (und) model
                // Pick the result with higher confidence
                rec_count += 1;
                let (text_cjk, conf_cjk) = self
                    .recognizer
                    .recognize_with_interpreter(&sub_region, &rec_interpreter)?;

                let (text, conf) = if let (Some(ref rec_und), Some(ref interp_und)) =
                    (&self.recognizer_und, &rec_interpreter_und)
                {
                    let (text_und, conf_und) =
                        rec_und.recognize_with_interpreter(&sub_region, interp_und)?;

                    // Chrome multi-pass: UND first, then language-specific.
                    // For very short text (table cells ≤3 chars), prefer UND if it produces
                    // ASCII and CJK produces non-ASCII (garbled). CJK model is unreliable
                    // on tiny crops but correct for normal-sized Chinese text.
                    let cjk_is_ascii = text_cjk.chars().all(|c| c.is_ascii());
                    let und_is_ascii = text_und.chars().all(|c| c.is_ascii());
                    let is_short = text_cjk.chars().count() <= 3 && text_und.chars().count() <= 3;

                    // Chrome uses script detection (GocrScriptDirectionIdentificationMutator)
                    // to route text to the correct model. We detect script from output:
                    // If CJK model produces all-ASCII text without spaces, it's likely Latin text
                    // processed by the wrong model. Prefer UND which handles Latin properly.
                    let cjk_no_space = !text_cjk.contains(' ') && text_cjk.chars().count() > 5;
                    let und_has_space = text_und.contains(' ');
                    let cjk_all_ascii = text_cjk.chars().all(|c| c.is_ascii());

                    let choice =
                        if is_short && !text_und.is_empty() && und_is_ascii && !cjk_is_ascii {
                            // Short text: UND ASCII vs CJK garbled → prefer UND
                            ("und_short", text_und, conf_und)
                        } else if cjk_all_ascii && cjk_no_space && und_has_space && conf_und > 0.5 {
                            // Latin body text: CJK model gives no spaces, UND gives proper text
                            // Real CJK text would contain non-ASCII chars
                            ("und_latin", text_und, conf_und)
                        } else if conf_und > conf_cjk && text_und.chars().count() >= 1 {
                            // UND has higher confidence
                            ("und_conf", text_und, conf_und)
                        } else {
                            ("cjk", text_cjk, conf_cjk)
                        };
                    if std::env::var("CHROME_OCR_DEBUG_REC").is_ok() {
                        println!(
                            "  [REC] y={} model={} cjk_conf={:.2} und_conf={:.2} text=\"{}\"",
                            actual_y,
                            choice.0,
                            conf_cjk,
                            conf_und,
                            &choice.1[..choice.1.len().min(50)]
                        );
                    }
                    (choice.1, choice.2)
                } else {
                    (text_cjk, conf_cjk)
                };

                // Post-process to clean up artifacts
                let text = Self::post_process_text(&text);

                // Filter out empty, single-char, or low-confidence results
                // Chrome: min_line_length_to_process = 2 from FilterJunkMutator settings
                // Exception: single ASCII digits are kept (table cells like "5", "4")
                let text_trimmed = text.trim();
                let text_len = text_trimmed.chars().count();
                let is_single_digit = text_len == 1
                    && text_trimmed
                        .chars()
                        .next()
                        .map_or(false, |c| c.is_ascii_digit());
                if (text_len < 2 && !is_single_digit) || conf < self.min_conf {
                    if std::env::var("CHROME_OCR_DEBUG").is_ok() {
                        println!(
                            "    [SKIP] y={} len={} conf={:.2}: {}",
                            actual_y, text_len, conf, text
                        );
                    }
                    continue;
                }

                // Chrome-style junk filter (HeuristicLineIsJunk at 0x18022A490)
                // Filter lines where most characters are junk (punctuation, symbols)
                if Self::is_junk_line(text_trimmed) {
                    if std::env::var("CHROME_OCR_DEBUG").is_ok() {
                        println!("    [SKIP] y={} junk: {}", actual_y, text);
                    }
                    continue;
                }

                // Use trimmed text for comparison and output
                let text = text_trimmed.to_string();

                // NOTE: Chrome does NOT use text-based duplicate filtering.
                // Duplicate removal is purely spatial (IoU/containment checks above).
                // Repeated text (e.g. "Text", "72") in tables is legitimate.

                line_num += 1;
                recognized_lines.push((x1, actual_y, sub_w, sub_h, conf));

                // Save line image
                if let Some(ref dir) = lines_dir {
                    let line_path = dir.join(format!("line_{:03}.png", line_num));
                    sub_region.save(&line_path)?;
                }

                println!(
                    "  L{} (y={:4}) conf={:.2}: {}",
                    line_num, actual_y, conf, text
                );
                results.push(text);
            }
        }

        // Chrome RemoveOverlapsStep + RemoveOverlapsWordPruningStep (from IDA analysis):
        // 1. Sort lines by height*confidence descending (larger/more confident first)
        // 2. For each pair: compute IoU and containment
        // 3. If overlap > 0.6: compare text via char frequency matching
        // 4. Remove the smaller/less confident overlapping line
        // 5. Breadth ratio < 0.5 protects lines from removal (different sizes)

        // Build sort order: by (height * confidence) descending
        let mut order: Vec<usize> = (0..recognized_lines.len()).collect();
        order.sort_by(|&a, &b| {
            let (_, _, w_a, h_a, conf_a) = recognized_lines[a];
            let (_, _, w_b, h_b, conf_b) = recognized_lines[b];
            let score_a = h_a as f32 * conf_a;
            let score_b = h_b as f32 * conf_b;
            score_b
                .partial_cmp(&score_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut keep = vec![true; results.len()];

        for idx_i in 0..order.len() {
            let i = order[idx_i];
            if !keep[i] {
                continue;
            }
            let (x1_i, y1_i, w_i, h_i, _conf_i) = recognized_lines[i];
            let b_i = [
                x1_i as f32,
                y1_i as f32,
                (x1_i as u32 + w_i) as f32,
                (y1_i as u32 + h_i) as f32,
            ];

            for idx_j in (idx_i + 1)..order.len() {
                let j = order[idx_j];
                if !keep[j] {
                    continue;
                }
                let (x1_j, y1_j, w_j, h_j, _conf_j) = recognized_lines[j];
                let b_j = [
                    x1_j as f32,
                    y1_j as f32,
                    (x1_j as u32 + w_j) as f32,
                    (y1_j as u32 + h_j) as f32,
                ];

                // Chrome RemoveOverlapsStep: compute IoU and containment
                let iou = calc_iou(&b_i, &b_j);
                let cont_ij = calc_containment(&b_i, &b_j);
                let cont_ji = calc_containment(&b_j, &b_i);
                let max_overlap = iou.max(cont_ij).max(cont_ji);

                // Chrome RemoveOverlapsStep: block_different_direction_maximum = 0.3
                // Primary gate: max(IoU, containment1, containment2) > 0.3 triggers processing.
                // After gate, Chrome does symbol-level overlap validation (sub_18045D260).
                // We approximate: require high IoU (>0.6) for direct removal,
                // or medium overlap (>0.3) with high Y-overlap ratio (>0.7 = same line).
                // This prevents removing adjacent body text lines that only overlap from
                // bbox expansion.

                if max_overlap <= 0.3 {
                    continue;
                }

                // Chrome breadth ratio logic (from IDA analysis of RemoveOverlapsStep):
                // - breadth_ratio < 0.6 AND confidence >= 0.5 → protect (Gate 3)
                // - breadth_ratio <= 0.5 → trigger symbol-level overlap removal
                // We approximate symbol-level removal: very small breadth ratio
                // with reasonable overlap means fragment contained in larger line.
                let breadth_i = w_i as f32;
                let breadth_j = w_j as f32;
                let breadth_ratio = if breadth_i.max(breadth_j) > 0.0 {
                    breadth_i.min(breadth_j) / breadth_i.max(breadth_j)
                } else {
                    0.0
                };

                // Compute Y overlap ratio (substitute for Chrome's symbol-level validation)
                let y_overlap = (b_i[3].min(b_j[3]) - b_i[1].max(b_j[1])).max(0.0);
                let min_h_line = (b_i[3] - b_i[1]).min(b_j[3] - b_j[1]);
                let y_overlap_ratio = if min_h_line > 0.0 {
                    y_overlap / min_h_line
                } else {
                    0.0
                };

                // Removal logic with breadth ratio guard.
                // Chrome's RemoveOverlapsStep:
                // - breadth_ratio < 0.6 AND confidence >= 0.5 → protect (Gate 3)
                // - breadth_ratio <= 0.5 → trigger symbol-level check (can override protection)
                // We use breadth_ratio < 0.5 as general protection, but allow fragment
                // removal when breadth_ratio < 0.3 (very clear fragment in larger line).
                let should_remove = if breadth_ratio < 0.3 {
                    // Very small fragment relative to the other line.
                    // Chrome does symbol-level overlap check here.
                    // We approximate: if Y-overlap > 0.4, the fragment is on the same
                    // physical line as the larger line → remove fragment.
                    // Threshold 0.44 balances fragment removal vs table cell protection.
                    // Table cells have Y-overlap ≈ 0.33 (protected).
                    // Body text fragments have Y-overlap ≈ 0.44-1.0 (removed).
                    y_overlap_ratio > 0.44
                } else if breadth_ratio < 0.5 {
                    false // Protect: similar to Chrome's Gate 3 (different sizes)
                } else if max_overlap > 0.6 {
                    true // Chrome RemoveOverlapsWordPruningStep line_overlap_iou_threshold = 0.6
                } else if y_overlap_ratio > 0.7 {
                    true // Same-line overlaps (Chrome's symbol-level check would confirm)
                } else {
                    false // Adjacent lines: protect
                };

                if should_remove {
                    keep[j] = false;
                    if std::env::var("CHROME_OCR_DEBUG").is_ok() {
                        println!(
                            "  [DEDUP] drop j={}: \"{}\" (kept i={}: \"{}\") iou={:.2} cont={:.2}/{:.2} y_ovr={:.2}",
                            j, results[j], i, results[i], iou, cont_ij, cont_ji, y_overlap_ratio
                        );
                    }
                } else if std::env::var("CHROME_OCR_DEBUG").is_ok() && max_overlap > 0.3 {
                    println!(
                        "  [DEDUP-SKIP] i={} j={} iou={:.2} cont={:.2}/{:.2} y_ovr={:.2} br={:.2} \"{}\" vs \"{}\"",
                        i, j, iou, cont_ij, cont_ji, y_overlap_ratio, breadth_ratio,
                        &results[i][..results[i].len().min(30)],
                        &results[j][..results[j].len().min(30)]
                    );
                }
            }
        }

        let results: Vec<String> = results
            .into_iter()
            .enumerate()
            .filter(|(idx, _)| keep[*idx])
            .map(|(_, s)| s)
            .collect();

        // Update timing stats
        self.stats.recognition_total = rec_start.elapsed().as_secs_f64();
        self.stats.recognition_count = rec_count;
        self.stats.ocr_total = ocr_start.elapsed().as_secs_f64();

        Ok(results)
    }

    /// Merge detection boxes into lines using Hough Transform-based greedy expansion.
    /// Based on Chrome's GroupingBoxesHoughTransform (sub_18049E3B0) from IDA analysis.
    ///
    /// Chrome algorithm (from IDA reverse engineering):
    ///   1. Build spatial hash grid (cell_size = avg_dim * 2)
    ///   2. For each unvisited box, grow a line cluster by directional expansion
    ///   3. Neighbor search constraints (from sub_1804AEA00):
    ///      - height_ratio <= 1.5 (config+96)
    ///      - perpendicular distance <= 0.3 * min_h (config+100)
    ///      - gap along direction <= 1.5 * avg_height (config+104)
    ///   4. Merge overlapping line hypotheses (IoU >= 0.1, config+128)
    ///
    /// Key difference from pairwise Union-Find: greedy directional expansion
    /// prevents transitive cross-line merging in radial text layouts.
    fn merge_boxes_to_lines(&self, boxes: &[BBox]) -> Vec<BBox> {
        if boxes.is_empty() {
            return Vec::new();
        }

        let n = boxes.len();

        // Compute average dimensions (from sub_18049AE80)
        let avg_height: f32 = boxes.iter().map(|b| b.height()).sum::<f32>() / n as f32;
        let avg_width: f32 = boxes.iter().map(|b| b.width()).sum::<f32>() / n as f32;

        // Chrome config: cell_size_portion = 2 (config+152, config+156)
        let cell_w = (avg_width * 2.0).max(1.0);
        let cell_h = (avg_height * 2.0).max(1.0);

        // Build spatial hash grid (from sub_18049AE80)
        // Key: grid_cols * row + col -> Vec<usize>
        let grid_cols = (4096.0 / cell_w) as i32 + 1;
        let mut spatial_hash: std::collections::HashMap<i32, Vec<usize>> =
            std::collections::HashMap::new();

        for (idx, b) in boxes.iter().enumerate() {
            let (cx, cy) = b.center();
            let col = (cx / cell_w) as i32;
            let row = (cy / cell_h) as i32;
            let key = grid_cols * row + col;
            spatial_hash.entry(key).or_default().push(idx);
        }

        // Greedy line expansion (based on Chrome's main loop in sub_18049E3B0, lines 907-1330)
        let mut visited = vec![false; n];
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        // Process boxes in confidence order (Chrome sorts by score, line 435-633)
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| {
            boxes[b]
                .conf
                .partial_cmp(&boxes[a].conf)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        for &seed_idx in &order {
            if visited[seed_idx] {
                continue;
            }
            visited[seed_idx] = true;

            let seed = &boxes[seed_idx];
            let seed_angle = seed.angle;
            let cos_a = seed_angle.cos();
            let sin_a = seed_angle.sin();

            let mut cluster = vec![seed_idx];

            // Expand in both directions (+1 forward, -1 backward)
            // Chrome's sub_1804AEA00 is called twice: forward and backward
            for direction in &[1.0f32, -1.0f32] {
                // Start from the seed's position
                let (mut cur_cx, mut cur_cy) = seed.center();
                let mut cur_w = seed.width();
                let mut cur_h = seed.height();

                // Iterative expansion along line direction
                let max_steps = 200; // Safety limit
                for _step in 0..max_steps {
                    let mut found_neighbor = false;
                    let mut best_idx = 0usize;
                    let mut best_along_dist = f32::MAX;

                    // Project search position along line direction
                    let search_cx = cur_cx + direction * cos_a * (cur_w * 0.5 + avg_width);
                    let search_cy = cur_cy + direction * sin_a * (cur_w * 0.5 + avg_width);

                    // Search neighboring grid cells (3x3 around projected position)
                    let search_col = (search_cx / cell_w) as i32;
                    let search_row = (search_cy / cell_h) as i32;

                    for dr in -1..=1 {
                        for dc in -1..=1 {
                            let key = grid_cols * (search_row + dr) + (search_col + dc);
                            if let Some(cell_indices) = spatial_hash.get(&key) {
                                for &cand_idx in cell_indices {
                                    if visited[cand_idx] {
                                        continue;
                                    }

                                    let cand = &boxes[cand_idx];

                                    // Chrome config+96: height_ratio <= 1.5
                                    let h_ratio = if cur_h > cand.height() {
                                        cur_h / cand.height()
                                    } else {
                                        cand.height() / cur_h
                                    };
                                    if h_ratio > 1.5 {
                                        continue;
                                    }

                                    let (cand_cx, cand_cy) = cand.center();
                                    let dx = cand_cx - cur_cx;
                                    let dy = cand_cy - cur_cy;

                                    // Chrome config+100: perpendicular distance <= 0.3 * min_h
                                    let perp = (dx * (-sin_a) + dy * cos_a).abs();
                                    let min_h = cur_h.min(cand.height());
                                    if perp > min_h * 0.3 {
                                        continue;
                                    }

                                    // Along-line distance (signed by direction)
                                    let along = (dx * cos_a + dy * sin_a) * direction;

                                    // Must be in the expansion direction (along > 0)
                                    // and gap must be reasonable
                                    if along < -cur_w * 0.5 {
                                        continue; // Behind current box
                                    }

                                    // Chrome config+104: gap <= 1.5 * avg_height
                                    let gap = along - (cur_w + cand.width()) * 0.5;
                                    if gap > avg_height * 1.5 {
                                        continue;
                                    }

                                    // Angle check (30 degrees from Chrome config)
                                    let angle_diff = (seed_angle - cand.angle).abs().to_degrees();
                                    let angle_diff = if angle_diff > 180.0 {
                                        360.0 - angle_diff
                                    } else {
                                        angle_diff
                                    };
                                    if angle_diff > 30.0 {
                                        continue;
                                    }

                                    // Pick closest neighbor in expansion direction
                                    if along < best_along_dist {
                                        best_along_dist = along;
                                        best_idx = cand_idx;
                                        found_neighbor = true;
                                    }
                                }
                            }
                        }
                    }

                    if found_neighbor {
                        visited[best_idx] = true;
                        cluster.push(best_idx);
                        // Update current position to the newly added box
                        let added = &boxes[best_idx];
                        cur_cx = added.center().0;
                        cur_cy = added.center().1;
                        cur_w = added.width();
                        cur_h = added.height();
                    } else {
                        break; // No more neighbors in this direction
                    }
                }
            }

            clusters.push(cluster);
        }

        if std::env::var("CHROME_OCR_DEBUG").is_ok() {
            let total_assigned: usize = clusters.iter().map(|c| c.len()).sum();
            println!(
                "  [DBG] {} clusters from greedy expansion ({}/{} boxes assigned)",
                clusters.len(),
                total_assigned,
                n
            );
            // Count singleton clusters
            let singletons = clusters.iter().filter(|c| c.len() == 1).count();
            println!("  [DBG] {} singleton clusters", singletons);
            // Show largest greedy expansion clusters
            let mut sizes: Vec<(usize, usize)> = clusters
                .iter()
                .enumerate()
                .map(|(i, c)| (c.len(), i))
                .collect();
            sizes.sort_by(|a, b| b.0.cmp(&a.0));
            for &(sz, idx) in sizes.iter().take(5) {
                let cl = &clusters[idx];
                let x1 = cl
                    .iter()
                    .map(|&i| boxes[i].x1)
                    .fold(f32::INFINITY, f32::min);
                let y1 = cl
                    .iter()
                    .map(|&i| boxes[i].y1)
                    .fold(f32::INFINITY, f32::min);
                let x2 = cl
                    .iter()
                    .map(|&i| boxes[i].x2)
                    .fold(f32::NEG_INFINITY, f32::max);
                let y2 = cl
                    .iter()
                    .map(|&i| boxes[i].y2)
                    .fold(f32::NEG_INFINITY, f32::max);
                println!("  [DBG] Greedy cluster[{}]: {} members, bbox=({:.0},{:.0})→({:.0},{:.0}) h={:.0}", idx, sz, x1, y1, x2, y2, y2-y1);
            }
        }

        // Merge overlapping line hypotheses using Union-Find (Chrome config+128: IoU >= 0.1)
        // This handles cases where the same line is discovered from different seed boxes
        let num_clusters = clusters.len();
        let mut cl_parent: Vec<usize> = (0..num_clusters).collect();

        fn find_cl(parent: &mut [usize], i: usize) -> usize {
            if parent[i] != i {
                parent[i] = find_cl(parent, parent[i]);
            }
            parent[i]
        }

        // Compute bounding box and angle for each cluster
        let cluster_bboxes: Vec<[f32; 4]> = clusters
            .iter()
            .map(|cl| {
                let x1 = cl
                    .iter()
                    .map(|&i| boxes[i].x1)
                    .fold(f32::INFINITY, f32::min);
                let y1 = cl
                    .iter()
                    .map(|&i| boxes[i].y1)
                    .fold(f32::INFINITY, f32::min);
                let x2 = cl
                    .iter()
                    .map(|&i| boxes[i].x2)
                    .fold(f32::NEG_INFINITY, f32::max);
                let y2 = cl
                    .iter()
                    .map(|&i| boxes[i].y2)
                    .fold(f32::NEG_INFINITY, f32::max);
                [x1, y1, x2, y2]
            })
            .collect();

        let cluster_angles: Vec<f32> = clusters
            .iter()
            .map(|cl| {
                let (ss, sc): (f32, f32) = cl
                    .iter()
                    .map(|&i| (boxes[i].angle.sin(), boxes[i].angle.cos()))
                    .fold((0.0, 0.0), |(s, c), (ds, dc)| (s + ds, c + dc));
                ss.atan2(sc)
            })
            .collect();

        let cluster_avg_h: Vec<f32> = clusters
            .iter()
            .map(|cl| cl.iter().map(|&i| boxes[i].height()).sum::<f32>() / cl.len() as f32)
            .collect();

        // Chrome ClusterLinesSpec: maximum_breadth_gap = 0.7
        // Breadth = extent along text direction (width for horizontal text)
        let cluster_avg_w: Vec<f32> = clusters
            .iter()
            .map(|cl| cl.iter().map(|&i| boxes[i].width()).sum::<f32>() / cl.len() as f32)
            .collect();

        // Chrome config: union_box_height_percentage = 2
        // Max merged height must not exceed 2 * avg_height to prevent super-clusters.
        let max_merged_height = avg_height * 2.0;

        // Pass 1a: Merge same-line clusters first (high vertical overlap)
        // This ensures same-line fragments are merged before any cross-line merges
        // could block them through the height guard.
        for i in 0..num_clusters {
            for j in (i + 1)..num_clusters {
                let ri = find_cl(&mut cl_parent, i);
                let rj = find_cl(&mut cl_parent, j);
                if ri == rj {
                    continue;
                }

                let bi = &cluster_bboxes[i];
                let bj = &cluster_bboxes[j];

                // Check if these are same-line clusters (high vertical overlap)
                let y_overlap = (bi[3].min(bj[3]) - bi[1].max(bj[1])).max(0.0);
                let hi = bi[3] - bi[1];
                let hj = bj[3] - bj[1];
                let min_h = hi.min(hj);
                let y_overlap_ratio = if min_h > 0.0 { y_overlap / min_h } else { 0.0 };

                // Must be on the same line (>70% vertical overlap) and have some IoU
                // 70% threshold prevents adjacent-line merges (typical y_overlap ~0.3-0.5)
                // while allowing same-line segment merges (y_overlap ~0.9-1.0)
                if y_overlap_ratio > 0.7 {
                    let iou = calc_iou(bi, bj);
                    let cont_ij = calc_containment(bi, bj);
                    let cont_ji = calc_containment(bj, bi);
                    if iou >= 0.1 || cont_ij > 0.5 || cont_ji > 0.5 {
                        cl_parent[rj] = ri;
                    }
                }
            }
        }

        // Pass 1b: Merge remaining overlapping clusters with height guard
        for i in 0..num_clusters {
            for j in (i + 1)..num_clusters {
                let ri = find_cl(&mut cl_parent, i);
                let rj = find_cl(&mut cl_parent, j);
                if ri == rj {
                    continue;
                }

                let bi = &cluster_bboxes[i];
                let bj = &cluster_bboxes[j];

                let should_merge = {
                    let iou = calc_iou(bi, bj);
                    if iou >= 0.1 {
                        true
                    } else {
                        let cont_ij = calc_containment(bi, bj);
                        let cont_ji = calc_containment(bj, bi);
                        cont_ij > 0.5 || cont_ji > 0.5
                    }
                };

                if should_merge {
                    // Height guard: compute merged bounding box height
                    let mut merged_y1 = f32::INFINITY;
                    let mut merged_y2 = f32::NEG_INFINITY;
                    for k in 0..num_clusters {
                        let rk = find_cl(&mut cl_parent, k);
                        if rk == ri || rk == rj {
                            merged_y1 = merged_y1.min(cluster_bboxes[k][1]);
                            merged_y2 = merged_y2.max(cluster_bboxes[k][3]);
                        }
                    }
                    let merged_height = merged_y2 - merged_y1;

                    if merged_height <= max_merged_height {
                        cl_parent[rj] = ri;
                    }
                }
            }
        }

        // Pass 2: Merge small clusters (<=3 boxes) into nearby larger ones along the line
        // This handles singleton boxes that were processed as seeds before a larger cluster
        // could absorb them (common in horizontal text bars with gaps)
        for small_i in 0..num_clusters {
            if clusters[small_i].len() > 3 {
                continue; // Only try to merge small clusters
            }
            let ri = find_cl(&mut cl_parent, small_i);
            // Check if already merged into something bigger
            let root_size: usize = (0..num_clusters)
                .filter(|&k| find_cl(&mut cl_parent, k) == ri)
                .map(|k| clusters[k].len())
                .sum();
            if root_size > 3 {
                continue; // Already part of a bigger group
            }

            let small_bb = &cluster_bboxes[small_i];
            let small_cx = (small_bb[0] + small_bb[2]) / 2.0;
            let small_cy = (small_bb[1] + small_bb[3]) / 2.0;
            let small_h = small_bb[3] - small_bb[1];

            for big_j in 0..num_clusters {
                if clusters[big_j].len() < 3 {
                    continue; // Only merge into larger clusters
                }
                let rj = find_cl(&mut cl_parent, big_j);
                if ri == rj {
                    continue;
                }

                let big_bb = &cluster_bboxes[big_j];
                let big_h = big_bb[3] - big_bb[1];

                // Height compatibility
                let h_ratio = if small_h > big_h {
                    small_h / big_h
                } else {
                    big_h / small_h
                };
                if h_ratio > 1.5 {
                    continue;
                }

                // Angle compatibility (30 degrees)
                let angle_diff = (cluster_angles[small_i] - cluster_angles[big_j])
                    .abs()
                    .to_degrees();
                let angle_diff = if angle_diff > 180.0 {
                    360.0 - angle_diff
                } else {
                    angle_diff
                };
                if angle_diff > 30.0 {
                    continue;
                }

                // Check perpendicular distance from small cluster center to big cluster's line
                let big_cx = (big_bb[0] + big_bb[2]) / 2.0;
                let big_cy = (big_bb[1] + big_bb[3]) / 2.0;
                let big_angle = cluster_angles[big_j];
                let big_cos = big_angle.cos();
                let big_sin = big_angle.sin();

                let dx = small_cx - big_cx;
                let dy = small_cy - big_cy;
                let perp = (dx * (-big_sin) + dy * big_cos).abs();
                let min_h = small_h.min(big_h);

                if perp > min_h * 0.3 {
                    continue;
                }

                // Check gap along line direction (depth gap: Chrome maximum_depth_gap = 1.5)
                let along = (dx * big_cos + dy * big_sin).abs();
                let big_w = big_bb[2] - big_bb[0];
                let small_w = small_bb[2] - small_bb[0];
                let gap = along - (big_w + small_w) * 0.5;
                if gap > avg_height * 1.5 {
                    continue;
                }

                // Chrome ClusterLinesSpec breadth gap check: maximum_breadth_gap = 0.7
                // Gap along text direction / min_avg_symbol_breadth must be <= 0.7
                // This prevents merging clusters that are far apart along the text direction
                // (e.g., table cells in different columns, column-boundary fragments).
                let min_avg_w = cluster_avg_w[small_i].min(cluster_avg_w[big_j]);
                if min_avg_w > 0.0 && gap > 0.0 && gap / min_avg_w > 0.7 {
                    continue;
                }

                // Merge small into big
                let small_root = find_cl(&mut cl_parent, small_i);
                cl_parent[small_root] = rj;
                break; // Only merge into one cluster
            }
        }

        // Collect merged clusters
        let mut merged_clusters: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for i in 0..num_clusters {
            let root = find_cl(&mut cl_parent, i);
            merged_clusters
                .entry(root)
                .or_default()
                .extend(clusters[i].iter());
        }

        // Chrome SplitLinesStep (from IDA analysis of sub_18046FB10):
        // Split merged clusters that span multiple text lines.
        // Chrome checks space_depth between consecutive symbols.
        // If space_depth / avg_symbol_depth > maximum_space_ratio, split the line.
        // We check Y-center gaps between consecutive boxes (depth direction for horizontal text).
        let mut final_clusters: Vec<Vec<usize>> = Vec::new();
        for (_root, indices) in &merged_clusters {
            if indices.len() < 3 {
                final_clusters.push(indices.clone());
                continue;
            }

            // Compute cluster's dominant angle
            let (sum_sin, sum_cos): (f32, f32) = indices
                .iter()
                .map(|&i| (boxes[i].angle.sin(), boxes[i].angle.cos()))
                .fold((0.0, 0.0), |(s, c), (ds, dc)| (s + ds, c + dc));
            let cluster_angle = sum_sin.atan2(sum_cos);
            let angle_deg = cluster_angle.to_degrees().abs();

            // Only split near-horizontal clusters (Chrome handles all angles via depth direction,
            // but our main problem is horizontal body text merging across lines)
            if angle_deg > 15.0 && angle_deg < 165.0 {
                final_clusters.push(indices.clone());
                continue;
            }

            // Compute average box height (= avg symbol depth)
            let cluster_avg_h: f32 =
                indices.iter().map(|&i| boxes[i].height()).sum::<f32>() / indices.len() as f32;

            // Sort boxes by Y-center (depth direction for horizontal text)
            let mut sorted_indices = indices.clone();
            sorted_indices.sort_by(|&a, &b| {
                let ya = boxes[a].center().1;
                let yb = boxes[b].center().1;
                ya.partial_cmp(&yb).unwrap_or(std::cmp::Ordering::Equal)
            });

            let y_centers: Vec<f32> = sorted_indices
                .iter()
                .map(|&i| boxes[i].center().1)
                .collect();

            // Chrome: space_depth / avg_symbol_depth > maximum_space_ratio (~1.5)
            let split_threshold = cluster_avg_h * 1.5;

            let mut split_points: Vec<usize> = Vec::new();
            for k in 1..y_centers.len() {
                let gap = y_centers[k] - y_centers[k - 1];
                if gap > split_threshold {
                    split_points.push(k);
                }
            }

            if split_points.is_empty() {
                final_clusters.push(indices.clone());
            } else {
                let mut prev = 0;
                for &sp in &split_points {
                    let sub: Vec<usize> = sorted_indices[prev..sp].to_vec();
                    if !sub.is_empty() {
                        final_clusters.push(sub);
                    }
                    prev = sp;
                }
                let sub: Vec<usize> = sorted_indices[prev..].to_vec();
                if !sub.is_empty() {
                    final_clusters.push(sub);
                }
                if std::env::var("CHROME_OCR_DEBUG").is_ok() {
                    println!(
                        "  [SPLIT-LINE] {} boxes → {} sub-lines (avg_h={:.0} angle={:.1}° splits={:?})",
                        indices.len(),
                        split_points.len() + 1,
                        cluster_avg_h,
                        cluster_angle.to_degrees(),
                        split_points.iter().map(|&sp| {
                            let gap = y_centers[sp] - y_centers[sp - 1];
                            format!("y_gap={:.0}", gap)
                        }).collect::<Vec<_>>()
                    );
                }
            }
        }

        // Chrome MergeLinesStep (from IDA analysis of sub_1804410C0):
        // Merge clusters on the same physical text line.
        // Chrome MergeLinesSpec protobuf defaults (from 0x18214c094):
        //   minimum_breadth_ratio: 0.6
        //   maximum_angle_difference: 3°
        //   minimum_breadth_overlap: 0.6
        //   maximum_depth_gap: 1.5
        // Chrome uses direction-aligned breadth/depth. We use axis-aligned
        // with a tight Y-center proximity check to avoid merging PPT radial text.
        {
            // Compute bounding boxes and avg char height for each cluster
            // Using avg char height (not cluster bbox height) for Y-center proximity check
            // prevents merging PPT radial text where cluster bboxes are tall (200+px)
            // but individual chars are only 30-40px.
            let cluster_info: Vec<(f32, f32, f32, f32, f32, f32)> = final_clusters
                .iter()
                .map(|indices| {
                    let cx1 = indices
                        .iter()
                        .map(|&i| boxes[i].x1)
                        .fold(f32::INFINITY, f32::min);
                    let cy1 = indices
                        .iter()
                        .map(|&i| boxes[i].y1)
                        .fold(f32::INFINITY, f32::min);
                    let cx2 = indices
                        .iter()
                        .map(|&i| boxes[i].x2)
                        .fold(f32::NEG_INFINITY, f32::max);
                    let cy2 = indices
                        .iter()
                        .map(|&i| boxes[i].y2)
                        .fold(f32::NEG_INFINITY, f32::max);
                    let (ss, sc): (f32, f32) = indices
                        .iter()
                        .map(|&i| (boxes[i].angle.sin(), boxes[i].angle.cos()))
                        .fold((0.0, 0.0), |(s, c), (ds, dc)| (s + ds, c + dc));
                    let angle = ss.atan2(sc);
                    let avg_char_h: f32 = indices.iter().map(|&i| boxes[i].height()).sum::<f32>()
                        / indices.len().max(1) as f32;
                    (cx1, cy1, cx2, cy2, angle, avg_char_h)
                })
                .collect();

            let n = final_clusters.len();
            let mut ml_parent: Vec<usize> = (0..n).collect();

            fn find_ml(parent: &mut [usize], i: usize) -> usize {
                let mut r = i;
                while parent[r] != r {
                    parent[r] = parent[parent[r]];
                    r = parent[r];
                }
                r
            }

            let mut merged_any = true;
            while merged_any {
                merged_any = false;
                for i in 0..n {
                    let ri = find_ml(&mut ml_parent, i);
                    for j in (i + 1)..n {
                        let rj = find_ml(&mut ml_parent, j);
                        if ri == rj {
                            continue;
                        }

                        let (ax1, ay1, ax2, ay2, a_angle, a_avg_ch) = cluster_info[i];
                        let (bx1, by1, bx2, by2, b_angle, b_avg_ch) = cluster_info[j];

                        let a_w = ax2 - ax1;
                        let b_w = bx2 - bx1;
                        let a_h = ay2 - ay1;
                        let b_h = by2 - by1;
                        if a_w <= 0.0 || b_w <= 0.0 || a_h <= 0.0 || b_h <= 0.0 {
                            continue;
                        }

                        // 1. Angle check (Chrome: maximum_angle_difference = 3°)
                        let angle_diff = (a_angle - b_angle).abs();
                        let angle_diff = if angle_diff > std::f32::consts::PI {
                            2.0 * std::f32::consts::PI - angle_diff
                        } else {
                            angle_diff
                        };
                        if angle_diff > 3.0_f32.to_radians() {
                            continue;
                        }

                        // 2. Y-center proximity: must be on the same physical line.
                        // Use avg character box height (not cluster bbox height) so
                        // tall PPT clusters (~200px bbox with 30px chars) don't get
                        // a huge threshold that allows cross-line merges.
                        let a_cy = (ay1 + ay2) * 0.5;
                        let b_cy = (by1 + by2) * 0.5;
                        let min_avg_ch = a_avg_ch.min(b_avg_ch);
                        let y_center_dist = (a_cy - b_cy).abs();
                        // Same line: Y-center distance < 50% of min avg char height
                        // (widened from 30% because avg char height is tighter than bbox height)
                        if y_center_dist > min_avg_ch * 0.5 {
                            continue;
                        }

                        // 3. Breadth ratio (Chrome: minimum_breadth_ratio = 0.6)
                        let breadth_ratio = a_w.min(b_w) / a_w.max(b_w);
                        if breadth_ratio < 0.6 {
                            continue;
                        }

                        // 4. Breadth overlap (Chrome: minimum_breadth_overlap = 0.6)
                        // OR adjacent with small gap (< 1.5 * avg_height along breadth)
                        let merged_w = ax2.max(bx2) - ax1.min(bx1);
                        let breadth_overlap = if merged_w > 0.0 {
                            (a_w + b_w - merged_w) / merged_w
                        } else {
                            0.0
                        };
                        let avg_h = (a_h + b_h) * 0.5;
                        let x_gap = (ax1.max(bx1) - ax2.min(bx2)).max(0.0);
                        let is_adjacent = x_gap < avg_h * 1.5;

                        if breadth_overlap < 0.6 && !is_adjacent {
                            continue;
                        }

                        // All checks passed - merge
                        ml_parent[rj] = ri;
                        merged_any = true;

                        if std::env::var("CHROME_OCR_DEBUG").is_ok() {
                            let s = self.detector.scale;
                            let ox = self.detector.offset_x;
                            let oy = self.detector.offset_y;
                            println!(
                                "  [MERGE-LINES] br={:.2} ang={:.1}° bovr={:.2} ydist={:.0} adj={} ({:.0},{:.0})-({:.0},{:.0}) + ({:.0},{:.0})-({:.0},{:.0})",
                                breadth_ratio, angle_diff.to_degrees(),
                                breadth_overlap, y_center_dist, is_adjacent,
                                (ax1 - ox) / s, (ay1 - oy) / s, (ax2 - ox) / s, (ay2 - oy) / s,
                                (bx1 - ox) / s, (by1 - oy) / s, (bx2 - ox) / s, (by2 - oy) / s
                            );
                        }
                    }
                }
            }

            // Rebuild final_clusters with merged groups
            let mut merged_map: std::collections::HashMap<usize, Vec<usize>> =
                std::collections::HashMap::new();
            for i in 0..n {
                let root = find_ml(&mut ml_parent, i);
                merged_map
                    .entry(root)
                    .or_default()
                    .extend(final_clusters[i].iter());
            }
            let merge_count = n - merged_map.len();
            if merge_count > 0 {
                println!(
                    "  MergeLinesStep: {} clusters → {} (merged {})",
                    n,
                    merged_map.len(),
                    merge_count
                );
            }
            final_clusters = merged_map.into_values().collect();
        }

        // Debug output
        if std::env::var("CHROME_OCR_DEBUG").is_ok() {
            for (idx, indices) in final_clusters.iter().enumerate() {
                if indices.len() > 15 {
                    let x1 = indices
                        .iter()
                        .map(|&i| boxes[i].x1)
                        .fold(f32::INFINITY, f32::min);
                    let y1 = indices
                        .iter()
                        .map(|&i| boxes[i].y1)
                        .fold(f32::INFINITY, f32::min);
                    let x2 = indices
                        .iter()
                        .map(|&i| boxes[i].x2)
                        .fold(f32::NEG_INFINITY, f32::max);
                    let y2 = indices
                        .iter()
                        .map(|&i| boxes[i].y2)
                        .fold(f32::NEG_INFINITY, f32::max);
                    println!(
                        "  [DBG] Large group idx={}: {} members, bbox=({:.0},{:.0})→({:.0},{:.0})",
                        idx,
                        indices.len(),
                        x1,
                        y1,
                        x2,
                        y2
                    );
                }
            }
        }

        // Create merged boxes from clusters
        // Chrome's Hough Transform requires hough_votes_threshold=10 votes per line.
        // Our greedy expansion doesn't have this natural threshold.
        // single_box_confidence_threshold = 0.4 for 1-box clusters.
        let mut merged = Vec::new();
        for indices in &final_clusters {
            let num_boxes = indices.len();

            let x1 = indices
                .iter()
                .map(|&i| boxes[i].x1)
                .fold(f32::INFINITY, f32::min);
            let y1 = indices
                .iter()
                .map(|&i| boxes[i].y1)
                .fold(f32::INFINITY, f32::min);
            let x2 = indices
                .iter()
                .map(|&i| boxes[i].x2)
                .fold(f32::NEG_INFINITY, f32::max);
            let y2 = indices
                .iter()
                .map(|&i| boxes[i].y2)
                .fold(f32::NEG_INFINITY, f32::max);
            let conf = indices
                .iter()
                .map(|&i| boxes[i].conf)
                .fold(0.0f32, f32::max);
            let (sum_sin, sum_cos): (f32, f32) = indices
                .iter()
                .map(|&i| {
                    let b = &boxes[i];
                    (b.angle.sin() * b.conf, b.angle.cos() * b.conf)
                })
                .fold((0.0, 0.0), |(s, c), (ds, dc)| (s + ds, c + dc));
            let angle = sum_sin.atan2(sum_cos);

            // Chrome: single-box lines need confidence > 0.4
            if num_boxes == 1 && conf < 0.4 {
                if std::env::var("CHROME_OCR_DEBUG").is_ok() {
                    let s = self.detector.scale;
                    let ox = self.detector.offset_x;
                    let oy = self.detector.offset_y;
                    println!(
                        "  [SKIP single-box] conf={:.2} at ({:.0},{:.0})→({:.0},{:.0})",
                        conf,
                        (x1 - ox) / s,
                        (y1 - oy) / s,
                        (x2 - ox) / s,
                        (y2 - oy) / s
                    );
                }
                continue;
            }

            // Chrome's Hough Transform needs >=10 votes for a line.
            // Filter merged lines with very few constituent detection boxes
            // (these are typically noise from arrow/icon regions).
            // Use a threshold of 5 (Chrome's filter_min_boxes_lines_size default).
            // Exception: near-horizontal/vertical text with high confidence can have few boxes.
            let bw = x2 - x1;
            let bh = y2 - y1;
            let aspect = if bh > 0.0 { bw / bh } else { 999.0 };

            // Thin bar filter: only catch graphic elements (decorative bars, rules)
            // that have few detection boxes. Real text lines have 50+ char-level boxes.
            // Graphic bars typically have <30 scattered detections.
            let s = self.detector.scale;
            let orig_h = bh / s;
            if aspect > 15.0 && orig_h < 30.0 && num_boxes < 30 {
                if std::env::var("CHROME_OCR_DEBUG").is_ok() {
                    let ox = self.detector.offset_x;
                    let oy = self.detector.offset_y;
                    println!(
                        "  [SKIP thin-bar] {}x{:.0} aspect={:.0} boxes={} at ({:.0},{:.0})",
                        (bw / s) as i32,
                        orig_h,
                        aspect,
                        num_boxes,
                        (x1 - ox) / s,
                        (y1 - oy) / s
                    );
                }
                continue;
            }

            // Filter lines with very few boxes AND near-vertical angle
            // Chrome's Hough needs ≥10 votes; vertical icon strips typically have < 5 boxes
            let angle_deg = angle.to_degrees().abs();
            let is_near_vertical = angle_deg > 60.0 && angle_deg < 120.0;
            if num_boxes < 5 && is_near_vertical {
                if std::env::var("CHROME_OCR_DEBUG").is_ok() {
                    let ox = self.detector.offset_x;
                    let oy = self.detector.offset_y;
                    println!(
                        "  [SKIP vert-few] boxes={} angle={:.1}° at ({:.0},{:.0})→({:.0},{:.0})",
                        num_boxes,
                        angle.to_degrees(),
                        (x1 - ox) / s,
                        (y1 - oy) / s,
                        (x2 - ox) / s,
                        (y2 - oy) / s
                    );
                }
                continue;
            }

            if std::env::var("CHROME_OCR_DEBUG").is_ok() {
                let ox = self.detector.offset_x;
                let oy = self.detector.offset_y;
                println!(
                    "  [MERGED] boxes={:3} conf={:.2} at ({:.0},{:.0})→({:.0},{:.0}) angle={:.1}°",
                    num_boxes,
                    conf,
                    (x1 - ox) / s,
                    (y1 - oy) / s,
                    (x2 - ox) / s,
                    (y2 - oy) / s,
                    angle.to_degrees()
                );
            }

            merged.push(BBox::with_angle(x1, y1, x2, y2, conf, angle));
        }

        // Filter out very small boxes (likely noise)
        // Chrome uses minimum box size based on original coordinates
        // Convert thresholds to 4096-space using scale factor
        let scale = self.detector.scale;
        let min_w_4096 = 4.0 * scale; // ~4px original width minimum
        let min_h_4096 = 3.0 * scale; // ~3px original height minimum
        if std::env::var("CHROME_OCR_DEBUG").is_ok() {
            let ox = self.detector.offset_x;
            let oy = self.detector.offset_y;
            for b in &merged {
                if b.width() <= min_w_4096 || b.height() <= min_h_4096 {
                    let orig_x1 = ((b.x1 - ox) / scale) as i32;
                    let orig_y1 = ((b.y1 - oy) / scale) as i32;
                    let orig_x2 = ((b.x2 - ox) / scale) as i32;
                    let orig_y2 = ((b.y2 - oy) / scale) as i32;
                    println!(
                        "  [FILTERED] ({},{})→({},{}) {}x{} angle={:.1}°",
                        orig_x1,
                        orig_y1,
                        orig_x2,
                        orig_y2,
                        b.width() as i32,
                        b.height() as i32,
                        b.angle.to_degrees()
                    );
                }
            }
        }
        let merged: Vec<BBox> = merged
            .into_iter()
            .filter(|b| b.width() > min_w_4096 && b.height() > min_h_4096)
            .collect();

        // Apply NMS to remove overlapping boxes (iou_threshold=0.5)
        let mut result = self.nms_merged_boxes(merged, 0.5);
        if std::env::var("CHROME_OCR_DEBUG").is_ok() {
            println!("  After NMS: {} lines", result.len());
        }

        // Post-grouping: merge small "heading" boxes with adjacent text boxes on the same line
        // Custom heuristic to fix char-level grouping splitting inline headings.
        // Uses stricter criteria to avoid over-merging table cells.
        self.merge_heading_boxes(&mut result);

        result
    }

    /// Merge small heading-like boxes into adjacent text boxes on the same line.
    /// A "heading box" is a narrow box (width < median_width * 0.3) that shares Y overlap
    /// with a wider box and is horizontally adjacent (small gap).
    fn merge_heading_boxes(&self, lines: &mut Vec<BBox>) {
        if lines.len() < 2 {
            return;
        }

        let scale = self.detector.scale;

        loop {
            let mut did_merge = false;
            let n = lines.len();

            // Compute median width to identify "small" boxes
            let mut widths: Vec<f32> = lines.iter().map(|b| b.width()).collect();
            widths.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let median_w = widths[widths.len() / 2];
            // A heading box is narrow (< 30% of median width)
            let heading_threshold = median_w * 0.3;

            'outer: for i in 0..n {
                // Only consider small boxes as heading candidates
                if lines[i].width() > heading_threshold {
                    continue;
                }
                let heading = &lines[i];

                for j in 0..n {
                    if i == j {
                        continue;
                    }
                    let text = &lines[j];

                    // Must be wider than the heading
                    if text.width() < heading.width() * 2.0 {
                        continue;
                    }

                    // Text box must be a single line (not multi-line).
                    // Multi-line boxes are too tall relative to the heading.
                    if text.height() > heading.height() * 3.0 {
                        continue;
                    }

                    // Y alignment check: heading center must be within text box's Y range
                    // This prevents merging headings with text on different lines
                    let heading_cy = (heading.y1 + heading.y2) * 0.5;
                    if heading_cy < text.y1 || heading_cy > text.y2 {
                        continue;
                    }

                    // Horizontal adjacency: heading must be left-adjacent or overlapping with text
                    // Only merge if heading starts at or before text's left edge
                    // This prevents merging rightward boxes from different columns
                    let gap_limit = 50.0 * scale; // ~50 original pixels gap max

                    // Heading must start at/before text's start (left-adjacent pattern)
                    if heading.x1 > text.x1 + gap_limit {
                        continue; // Heading is too far right - likely different column
                    }

                    let x_gap = if heading.x2 < text.x1 {
                        text.x1 - heading.x2 // heading is to the left of text
                    } else {
                        0.0 // heading overlaps with text
                    };

                    if x_gap > gap_limit {
                        continue;
                    }

                    // Merge heading into text box (extend text box)
                    lines[j] = BBox::with_angle(
                        heading.x1.min(text.x1),
                        heading.y1.min(text.y1),
                        heading.x2.max(text.x2),
                        heading.y2.max(text.y2),
                        heading.conf.max(text.conf),
                        text.angle,
                    );
                    lines.remove(i);
                    did_merge = true;

                    if std::env::var("CHROME_OCR_DEBUG").is_ok() {
                        let s = self.detector.scale;
                        let ox = self.detector.offset_x;
                        let oy = self.detector.offset_y;
                        let b = &lines[if i < j { j - 1 } else { j }];
                        println!(
                            "  [HEADING-MERGE] Merged small box into line at ({:.0},{:.0})-({:.0},{:.0})",
                            (b.x1 - ox) / s, (b.y1 - oy) / s, (b.x2 - ox) / s, (b.y2 - oy) / s
                        );
                    }
                    break 'outer;
                }
            }

            if !did_merge {
                break;
            }
        }
    }

    /// Apply NMS to merged boxes - using Chrome's RemoveOverlaps parameters from IDA analysis
    /// Key thresholds: line_overlap_iou_threshold=0.6, minimum_breadth_ratio=0.6
    fn nms_merged_boxes(&self, mut boxes: Vec<BBox>, iou_threshold: f32) -> Vec<BBox> {
        if boxes.len() <= 1 {
            return boxes;
        }

        // Sort by area (larger first) - larger boxes are more likely to be complete lines
        boxes.sort_by(|a, b| {
            let area_a = a.width() * a.height();
            let area_b = b.width() * b.height();
            area_b
                .partial_cmp(&area_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        let mut kept = Vec::new();

        for bbox in boxes {
            let b = bbox.as_array();
            let b_w = b[2] - b[0];
            let b_h = b[3] - b[1];

            // Check if this box significantly overlaps with any kept box
            let is_dominated = kept.iter().any(|k: &BBox| {
                let kb = k.as_array();
                let kb_w = kb[2] - kb[0];
                let kb_h = kb[3] - kb[1];

                // Chrome's MergeLines: minimum_breadth_ratio = 0.6
                let breadth_ratio = b_w.min(kb_w) / b_w.max(kb_w);
                if breadth_ratio < 0.6 {
                    return false; // Widths too different, not same line
                }

                // Calculate vertical overlap (depth overlap)
                let y_overlap = (b[3].min(kb[3]) - b[1].max(kb[1])).max(0.0);
                let min_h = b_h.min(kb_h);
                let y_overlap_ratio = if min_h > 0.0 { y_overlap / min_h } else { 0.0 };

                // Chrome's MergeLines: minimum_breadth_overlap = 0.6
                let x_overlap = (b[2].min(kb[2]) - b[0].max(kb[0])).max(0.0);
                let merged_w = b[2].max(kb[2]) - b[0].min(kb[0]);
                let breadth_overlap = if merged_w > 0.0 {
                    (b_w + kb_w - merged_w) / merged_w
                } else {
                    0.0
                };

                // Chrome's MergeLines: maximum_depth_gap = 1.5 (relative to avg height)
                let avg_h = (b_h + kb_h) * 0.5;
                let merged_h = b[3].max(kb[3]) - b[1].min(kb[1]);
                let depth_gap = (merged_h - b_h - kb_h).max(0.0);
                let depth_gap_ratio = if avg_h > 0.0 { depth_gap / avg_h } else { 0.0 };

                // IOU and containment checks (Chrome: line_overlap_iou_threshold = 0.6)
                let iou = calc_iou(&b, &kb);
                let containment1 = calc_containment(&b, &kb);
                let containment2 = calc_containment(&kb, &b);

                // Merge if:
                // 1. High IOU (>0.6) or
                // 2. One contains the other (>0.6) or
                // 3. Same row (y_overlap > 0.6) AND significant x overlap AND small depth gap
                let same_row =
                    y_overlap_ratio > 0.6 && breadth_overlap > 0.3 && depth_gap_ratio < 1.5;

                iou > iou_threshold || containment1 > 0.6 || containment2 > 0.6 || same_row
            });

            if !is_dominated {
                kept.push(bbox);
            }
        }

        kept
    }

    /// Post-process recognized text to clean up artifacts
    /// Based on Chrome's FilterJunkMutator from IDA analysis
    fn post_process_text(text: &str) -> String {
        let mut result = String::new();
        let chars: Vec<char> = text.chars().collect();
        let mut i = 0;

        // Strip leading junk characters (detection edge artifacts)
        // Chrome's PadAndScaleBoxes uses only 4px padding in 4096-space,
        // but detection boxes may still include edge content like borders, bullets
        while i < chars.len() {
            let c = chars[i];
            // Keep if it's a CJK char or alphanumeric (strict for leading position)
            // Hiragana/Katakana at leading position are likely artifacts in CJK text
            if Self::is_strong_content_char(c) {
                break;
            }
            // Keep if it's an opening bracket with matching closer nearby
            if Self::has_matching_bracket(&chars, i) {
                break;
            }
            // Strip this leading junk character
            i += 1;
        }

        while i < chars.len() {
            let c = chars[i];

            // Skip isolated punctuation artifacts between CJK characters
            // Based on Chrome's RemoveJunkWords
            if i > 0 && i + 1 < chars.len() {
                let prev = chars[i - 1];
                let next = chars[i + 1];

                let prev_is_cjk = prev >= '\u{4E00}' && prev <= '\u{9FFF}';
                let next_is_cjk = next >= '\u{4E00}' && next <= '\u{9FFF}';

                // Artifact characters that commonly appear between CJK
                let c_is_artifact = c == '|' || c == '!' || c == '$';

                if prev_is_cjk && next_is_cjk && c_is_artifact {
                    i += 1;
                    continue;
                }
            }

            // Clean up doubled punctuation like ",," or ".."
            if i + 1 < chars.len()
                && c == chars[i + 1]
                && (c == ',' || c == '.' || c == '。' || c == '，')
            {
                result.push(c);
                i += 2;
                continue;
            }

            result.push(c);
            i += 1;
        }

        // Strip trailing junk characters
        while result.ends_with(|c: char| {
            !Self::is_content_char(c)
                && c != '。'
                && c != '.'
                && c != ')'
                && c != '）'
                && c != '」'
                && c != '】'
                && c != '%'
        }) {
            result.pop();
        }

        result
    }

    /// Check if a character is meaningful content (not edge artifact)
    fn is_content_char(c: char) -> bool {
        c.is_alphanumeric()
            || (c >= '\u{4E00}' && c <= '\u{9FFF}')  // CJK Unified
            || (c >= '\u{3400}' && c <= '\u{4DBF}')  // CJK Extension A
            || (c >= '\u{3040}' && c <= '\u{309F}')  // Hiragana
            || (c >= '\u{30A0}' && c <= '\u{30FF}')  // Katakana
            || (c >= '\u{AC00}' && c <= '\u{D7AF}')  // Korean
            || c == '\u{3000}' // Ideographic space
    }

    /// Strict content check for leading position - only CJK and ASCII alphanumeric
    /// Hiragana/Katakana at leading position are often artifacts in Chinese text
    fn is_strong_content_char(c: char) -> bool {
        c.is_ascii_alphanumeric()
            || (c >= '\u{4E00}' && c <= '\u{9FFF}')  // CJK Unified
            || (c >= '\u{3400}' && c <= '\u{4DBF}')  // CJK Extension A
            || (c >= '\u{AC00}' && c <= '\u{D7AF}') // Korean
    }

    /// Check if character at position has a matching bracket in the text
    fn has_matching_bracket(chars: &[char], pos: usize) -> bool {
        let c = chars[pos];
        let closer = match c {
            '(' => ')',
            '（' => '）',
            '「' => '」',
            '【' => '】',
            '『' => '』',
            '[' => ']',
            '{' => '}',
            _ => return false,
        };
        // Check if closer exists within next 5 chars (for patterns like "(二)")
        chars[pos + 1..].iter().take(5).any(|&ch| ch == closer)
    }

    /// Chrome-style junk line filter (HeuristicLineIsJunk at 0x18022A490)
    /// Filters lines that are mostly junk characters (symbols, punctuation, whitespace)
    fn is_junk_line(text: &str) -> bool {
        let chars: Vec<char> = text.chars().collect();
        let total = chars.len();
        if total == 0 {
            return true;
        }

        // Count meaningful characters (CJK, alphabetic, numeric)
        let meaningful_count = chars
            .iter()
            .filter(|c| {
                c.is_alphanumeric()
                    || (**c >= '\u{4E00}' && **c <= '\u{9FFF}') // CJK Unified
                    || (**c >= '\u{3400}' && **c <= '\u{4DBF}') // CJK Extension A
                    || (**c >= '\u{3040}' && **c <= '\u{30FF}') // Hiragana/Katakana
                    || (**c >= '\u{AC00}' && **c <= '\u{D7AF}') // Korean
            })
            .count();

        // If less than half the characters are meaningful, it's junk
        // Chrome checks "stripped fraction" - ratio of junk characters
        if meaningful_count * 2 < total {
            return true;
        }

        // Mixed-script check: CJK + Japanese kana is often garbled OCR
        // (CJK model produces kana artifacts on non-text regions)
        let has_cjk = chars.iter().any(|c| {
            (*c >= '\u{4E00}' && *c <= '\u{9FFF}') || (*c >= '\u{3400}' && *c <= '\u{4DBF}')
        });
        let has_kana = chars.iter().any(|c| {
            (*c >= '\u{30A0}' && *c <= '\u{30FF}') // Katakana
                || (*c >= '\u{3040}' && *c <= '\u{309F}') // Hiragana
        });
        let has_latin = chars.iter().any(|c| c.is_ascii_alphabetic());
        let has_digit = chars.iter().any(|c| c.is_ascii_digit());

        // Short text mixing kana with CJK/Latin+digits is almost certainly garbled
        if total <= 8 && has_kana && (has_cjk || has_latin || has_digit) {
            return true;
        }

        // Check for repeated characters (Chrome: repeated char detection)
        if total >= 3 {
            let mut max_repeat = 1;
            let mut cur_repeat = 1;
            for i in 1..chars.len() {
                if chars[i] == chars[i - 1] {
                    cur_repeat += 1;
                    if cur_repeat > max_repeat {
                        max_repeat = cur_repeat;
                    }
                } else {
                    cur_repeat = 1;
                }
            }
            // If more than 2/3 of the line is repeated same character
            if max_repeat * 3 > total * 2 {
                return true;
            }
        }

        false
    }

    /// Split multi-line region using horizontal projection
    fn split_multiline_region(&self, region: &GrayImage) -> Vec<(GrayImage, u32)> {
        let (w, h) = (region.width(), region.height());

        // Don't split short regions (must be > ~2 lines to have multi-line content)
        if h < 30 {
            return vec![(region.clone(), 0)];
        }

        // Calculate horizontal projection
        let mut h_proj = vec![0u32; h as usize];
        for y in 0..h {
            for x in 0..w {
                let pixel = region.get_pixel(x, y).0[0];
                if pixel < 200 {
                    h_proj[y as usize] += 1;
                }
            }
        }

        // Find line regions
        let threshold = (*h_proj.iter().max().unwrap_or(&1) as f32 * 0.1).max(1.0) as u32;
        let mut lines = Vec::new();
        let mut in_line = false;
        let mut line_start = 0;

        for (i, &val) in h_proj.iter().enumerate() {
            if val > threshold && !in_line {
                in_line = true;
                line_start = i;
            } else if val <= threshold && in_line {
                in_line = false;
                if i - line_start >= 10 {
                    lines.push((line_start as u32, i as u32));
                }
            }
        }

        if in_line && h as usize - line_start >= 10 {
            lines.push((line_start as u32, h));
        }

        // If only one line, return as-is
        if lines.len() <= 1 {
            return vec![(region.clone(), 0)];
        }

        // Split into sub-regions
        lines
            .iter()
            .map(|&(y1, y2)| {
                let y1_pad = y1.saturating_sub(2);
                let y2_pad = (y2 + 2).min(h);
                let sub =
                    image::imageops::crop_imm(region, 0, y1_pad, w, y2_pad - y1_pad).to_image();
                (sub, y1_pad)
            })
            .collect()
    }
}
