use anyhow::Result;
use image::GrayImage;
use std::path::Path;
use std::time::Instant;

use crate::detector::TextDetector;
use crate::recognizer::LineRecognizer;
use crate::sorter::LayoutSorter;
use crate::utils::{calc_containment, calc_iou, calc_xy_overlap, BBox};

const MIN_HEIGHT: u32 = 40;

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
    sorter: LayoutSorter,
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

        println!("All models loaded!");

        Ok(Self {
            detector,
            recognizer,
            sorter,
            save_lines: false,
            min_conf: 0.3,
            perf,
            stats,
        })
    }

    pub fn print_load_stats(&self) {
        println!("  TextDetector:   {:7.1} ms", self.stats.load_detector * 1000.0);
        println!("  LayoutSorter:   {:7.1} ms", self.stats.load_sorter * 1000.0);
        println!("  LineRecognizer: {:7.1} ms", self.stats.load_recognizer * 1000.0);
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
        let t0 = Instant::now();
        let boxes = self.detector.detect(&image, 0.5)?;
        self.stats.detection = t0.elapsed().as_secs_f64();
        let target_size = self.detector.target_size;
        println!(
            "\n[1/3] Text Detection (on {}x{})...",
            target_size, target_size
        );
        println!("  Found {} char-level regions", boxes.len());

        // Merge boxes to lines
        let t1 = Instant::now();
        let merged = self.merge_boxes_to_lines(&boxes);
        self.stats.merge = t1.elapsed().as_secs_f64();
        println!("  Merged to {} lines", merged.len());

        // Step 2: Layout Sorting
        println!("\n[2/3] Layout Sorting...");
        let t2 = Instant::now();
        let sorted_lines = self.sorter.sort(&merged, (target_size, target_size))?;
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

        // Create interpreter once for all recognitions
        let rec_interpreter = self.recognizer.create_interpreter()?;

        for bbox in &sorted_lines {
            // Convert padded coordinates to original image coordinates
            // Subtract offset, then divide by scale
            let mut x1 = ((bbox.x1 - offset_x) / scale) as i32;
            let mut y1 = ((bbox.y1 - offset_y) / scale) as i32;
            let mut x2 = ((bbox.x2 - offset_x) / scale) as i32;
            let mut y2 = ((bbox.y2 - offset_y) / scale) as i32;

            x1 = x1.max(0);
            y1 = y1.max(0);
            x2 = x2.min(width as i32);
            y2 = y2.min(height as i32);

            if x2 - x1 < 10 || y2 - y1 < 5 {
                continue;
            }

            // Expand short lines
            let line_height = (y2 - y1) as u32;
            if line_height < MIN_HEIGHT {
                let expand = ((MIN_HEIGHT - line_height) / 2 + 3) as i32;
                y1 = (y1 - expand).max(0);
                y2 = (y2 + expand).min(height as i32);
            }

            // Crop region
            let region = image::imageops::crop_imm(
                &image,
                x1 as u32,
                y1 as u32,
                (x2 - x1) as u32,
                (y2 - y1) as u32,
            )
            .to_image();

            // Split multi-line regions
            let sub_regions = self.split_multiline_region(&region);

            for (sub_region, sub_y_offset) in sub_regions {
                let mut actual_y = y1 + sub_y_offset as i32;
                let mut sub_h = sub_region.height();
                let sub_w = sub_region.width();

                // Expand short sub-regions
                let sub_region = if sub_h < MIN_HEIGHT {
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

                // Check for duplicates
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

                            let y_overlap = (b1[3].min(b2[3]) - b1[1].max(b2[1])).max(0.0);
                            let x_overlap = (b1[2].min(b2[2]) - b1[0].max(b2[0])).max(0.0);
                            let min_h = (b1[3] - b1[1]).min(b2[3] - b2[1]);
                            let min_w = (b1[2] - b1[0]).min(b2[2] - b2[0]);

                            min_h > 0.0
                                && min_w > 0.0
                                && y_overlap / min_h > 0.5
                                && x_overlap / min_w > 0.5
                        });

                if is_duplicate {
                    continue;
                }

                // Recognize using shared interpreter
                rec_count += 1;
                let (text, conf) = self.recognizer.recognize_with_interpreter(&sub_region, &rec_interpreter)?;

                // Filter out very short fragments (likely noise)
                let text_len = text.chars().count();
                if text_len >= 2 && conf >= self.min_conf {
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
        }

        // Update timing stats
        self.stats.recognition_total = rec_start.elapsed().as_secs_f64();
        self.stats.recognition_count = rec_count;
        self.stats.ocr_total = ocr_start.elapsed().as_secs_f64();

        Ok(results)
    }

    /// Merge detection boxes into lines
    fn merge_boxes_to_lines(&self, boxes: &[BBox]) -> Vec<BBox> {
        if boxes.is_empty() {
            return Vec::new();
        }

        // Calculate center and size for each box
        let mut boxes_with_info: Vec<_> = boxes
            .iter()
            .map(|b| {
                let (cx, cy) = b.center();
                (b.clone(), cx, cy, b.width(), b.height())
            })
            .collect();

        // Sort by y then x
        boxes_with_info.sort_by(|a, b| {
            a.2.partial_cmp(&b.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        // Merge adjacent boxes
        let mut merged = Vec::new();
        let mut used = vec![false; boxes_with_info.len()];

        for i in 0..boxes_with_info.len() {
            if used[i] {
                continue;
            }

            let mut group = vec![i];
            used[i] = true;

            // Iteratively expand group
            let mut changed = true;
            while changed {
                changed = false;
                for j in 0..boxes_with_info.len() {
                    if used[j] {
                        continue;
                    }

                    let other = &boxes_with_info[j];

                    // Check if adjacent to any box in group
                    for &g_idx in &group {
                        let g = &boxes_with_info[g_idx];

                        let x_dist = (other.1 - g.1).abs();
                        let y_dist = (other.2 - g.2).abs();
                        let x_thresh = other.3 + g.3; // Same as Python
                        let y_thresh = (other.4 + g.4) * 0.3; // Same as Python

                        if x_dist < x_thresh && y_dist < y_thresh {
                            group.push(j);
                            used[j] = true;
                            changed = true;
                            break;
                        }
                    }
                }
            }

            // Merge group into single box
            if !group.is_empty() {
                let x1 = group
                    .iter()
                    .map(|&i| boxes_with_info[i].0.x1)
                    .fold(f32::INFINITY, f32::min);
                let y1 = group
                    .iter()
                    .map(|&i| boxes_with_info[i].0.y1)
                    .fold(f32::INFINITY, f32::min);
                let x2 = group
                    .iter()
                    .map(|&i| boxes_with_info[i].0.x2)
                    .fold(f32::NEG_INFINITY, f32::max);
                let y2 = group
                    .iter()
                    .map(|&i| boxes_with_info[i].0.y2)
                    .fold(f32::NEG_INFINITY, f32::max);
                let conf = group
                    .iter()
                    .map(|&i| boxes_with_info[i].0.conf)
                    .fold(0.0f32, f32::max);

                merged.push(BBox::new(x1, y1, x2, y2, conf));
            }
        }

        // Apply NMS to remove overlapping boxes (like Python: iou_threshold=0.5)
        self.nms_merged_boxes(merged, 0.5)
    }

    /// Apply NMS to merged boxes
    fn nms_merged_boxes(&self, mut boxes: Vec<BBox>, iou_threshold: f32) -> Vec<BBox> {
        if boxes.len() <= 1 {
            return boxes;
        }

        // Sort by confidence (higher first), then by area (larger first)
        boxes.sort_by(|a, b| {
            b.conf
                .partial_cmp(&a.conf)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    let area_a = a.width() * a.height();
                    let area_b = b.width() * b.height();
                    area_b
                        .partial_cmp(&area_a)
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
        });

        let mut kept = Vec::new();

        for bbox in boxes {
            let b = bbox.as_array();
            let is_dominated = kept.iter().any(|k: &BBox| {
                let kb = k.as_array();
                let iou = calc_iou(&b, &kb);
                let containment1 = calc_containment(&b, &kb);
                let containment2 = calc_containment(&kb, &b);
                let xy_overlap = calc_xy_overlap(&b, &kb);

                iou > iou_threshold || containment1 > 0.6 || containment2 > 0.6 || xy_overlap > 0.5
            });

            if !is_dominated {
                kept.push(bbox);
            }
        }

        kept
    }

    /// Split multi-line region using horizontal projection
    fn split_multiline_region(&self, region: &GrayImage) -> Vec<(GrayImage, u32)> {
        let (w, h) = (region.width(), region.height());

        // Don't split short regions (50px can contain ~2 lines of 25px each)
        if h < 50 {
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
