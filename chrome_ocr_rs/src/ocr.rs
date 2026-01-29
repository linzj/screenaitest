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
use crate::utils::{calc_containment, calc_iou, BBox};

const MIN_HEIGHT: u32 = 40;
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

        let cluster_sort = ClusterSort::new(model_dir)?;
        println!("  ClusterSort: loaded");

        println!("All models loaded!");

        Ok(Self {
            detector,
            recognizer,
            sorter,
            cluster_sort,
            save_lines: false,
            min_conf: 0.7,
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

        // Create interpreter once for all recognitions
        let rec_interpreter = self.recognizer.create_interpreter()?;

        for bbox in &sorted_lines {
            // Convert 4096 coordinates to original image coordinates
            let mut x1 = ((bbox.x1 - offset_x) / scale) as i32;
            let mut y1 = ((bbox.y1 - offset_y) / scale) as i32;
            let mut x2 = ((bbox.x2 - offset_x) / scale) as i32;
            let mut y2 = ((bbox.y2 - offset_y) / scale) as i32;

            // Chrome pads boxes before recognition: clamp(height * scale_factor, 4.0, 16.0)
            // From IDA analysis of sub_18048ACD0 (region_proposal_text_detector.cc)
            // Since our char-level grouping may miss edge chars, use box_height/3
            // as padding (approximately one char width) to compensate
            let box_h = (y2 - y1) as f32;
            let pad_x = (box_h / 3.0).clamp(10.0, 40.0) as i32;
            let pad_y = (box_h / 6.0).clamp(4.0, 16.0) as i32;
            x1 = (x1 - pad_x).max(0);
            y1 = (y1 - pad_y).max(0);
            x2 = (x2 + pad_x).min(width as i32);
            y2 = (y2 + pad_y).min(height as i32);

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

                // Check for duplicates using Chrome's RemoveOverlaps parameters from IDA:
                // line_overlap_iou_threshold = 0.6, minimum_breadth_ratio = 0.6
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

                            // Calculate IoU (Chrome: line_overlap_iou_threshold = 0.6)
                            let iou = calc_iou(&b1, &b2);
                            if iou > 0.6 {
                                return true;
                            }

                            // Calculate containment
                            let containment1 = calc_containment(&b1, &b2);
                            let containment2 = calc_containment(&b2, &b1);
                            if containment1 > 0.6 || containment2 > 0.6 {
                                return true;
                            }

                            // Check breadth ratio (Chrome: minimum_breadth_ratio = 0.6)
                            let w1 = b1[2] - b1[0];
                            let w2 = b2[2] - b2[0];
                            let breadth_ratio = w1.min(w2) / w1.max(w2);

                            // Only compare if similar width (same line type)
                            if breadth_ratio < 0.6 {
                                return false;
                            }

                            // Check vertical overlap
                            let y_overlap = (b1[3].min(b2[3]) - b1[1].max(b2[1])).max(0.0);
                            let min_h = (b1[3] - b1[1]).min(b2[3] - b2[1]);
                            let y_overlap_ratio = if min_h > 0.0 { y_overlap / min_h } else { 0.0 };

                            // Chrome: minimum_breadth_overlap = 0.6 for horizontal overlap
                            let x_overlap = (b1[2].min(b2[2]) - b1[0].max(b2[0])).max(0.0);
                            let x_overlap_ratio = if w1.min(w2) > 0.0 {
                                x_overlap / w1.min(w2)
                            } else {
                                0.0
                            };

                            // Same row with significant x overlap
                            y_overlap_ratio > 0.6 && x_overlap_ratio > 0.5
                        });

                if is_duplicate {
                    if std::env::var("CHROME_OCR_DEBUG").is_ok() {
                        println!("    [SKIP] y={} duplicate", actual_y);
                    }
                    continue;
                }

                // Recognize using shared interpreter
                rec_count += 1;
                let (text, conf) = self
                    .recognizer
                    .recognize_with_interpreter(&sub_region, &rec_interpreter)?;

                // Post-process to clean up artifacts
                let text = Self::post_process_text(&text);

                // Filter out empty, single-char, or low-confidence results
                // Chrome's GroupDetectionBoxes merges chars to lines - single chars are noise
                let text_len = text.chars().count();
                if text_len < 2 || conf < self.min_conf {
                    if std::env::var("CHROME_OCR_DEBUG").is_ok() {
                        println!(
                            "    [SKIP] y={} len={} conf={:.2}: {}",
                            actual_y, text_len, conf, text
                        );
                    }
                    continue;
                }

                // Check for text-based duplicates (same or very similar text)
                let is_text_duplicate = results.iter().any(|prev_text: &String| {
                    // Exact match
                    if prev_text == &text {
                        return true;
                    }
                    // One is substring of the other (for partial matches)
                    let shorter = if prev_text.len() < text.len() {
                        prev_text
                    } else {
                        &text
                    };
                    let longer = if prev_text.len() >= text.len() {
                        prev_text
                    } else {
                        &text
                    };
                    if shorter.chars().count() >= 3 && longer.contains(shorter.as_str()) {
                        return true;
                    }
                    false
                });

                if is_text_duplicate {
                    if std::env::var("CHROME_OCR_DEBUG").is_ok() {
                        println!("    [SKIP] y={} text duplicate: {}", actual_y, text);
                    }
                    continue;
                }

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

        // Pass 1: Merge overlapping clusters (IoU >= 0.1 or containment > 0.5)
        for i in 0..num_clusters {
            for j in (i + 1)..num_clusters {
                let ri = find_cl(&mut cl_parent, i);
                let rj = find_cl(&mut cl_parent, j);
                if ri == rj {
                    continue;
                }
                // Chrome config+128: grouping_box_overlap = 0.1
                let iou = calc_iou(&cluster_bboxes[i], &cluster_bboxes[j]);
                if iou >= 0.1 {
                    cl_parent[rj] = ri;
                    continue;
                }
                // Also merge if one cluster is mostly contained in another
                let cont_ij = calc_containment(&cluster_bboxes[i], &cluster_bboxes[j]);
                let cont_ji = calc_containment(&cluster_bboxes[j], &cluster_bboxes[i]);
                if cont_ij > 0.5 || cont_ji > 0.5 {
                    cl_parent[rj] = ri;
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

                // Check gap along line direction
                let along = (dx * big_cos + dy * big_sin).abs();
                let big_w = big_bb[2] - big_bb[0];
                let small_w = small_bb[2] - small_bb[0];
                let gap = along - (big_w + small_w) * 0.5;
                if gap > avg_height * 1.5 {
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

        // Debug output
        if std::env::var("CHROME_OCR_DEBUG").is_ok() {
            for (root, indices) in &merged_clusters {
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
                        "  [DBG] Large group root={}: {} members, bbox=({:.0},{:.0})→({:.0},{:.0})",
                        root,
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
        let mut merged = Vec::new();
        for indices in merged_clusters.values() {
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

            merged.push(BBox::with_angle(x1, y1, x2, y2, conf, angle));
        }

        // Filter out very small boxes (likely noise)
        // Chrome: width >= 4 && height > 3 (in original coordinates)
        if std::env::var("CHROME_OCR_DEBUG").is_ok() {
            let scale = self.detector.scale;
            let ox = self.detector.offset_x;
            let oy = self.detector.offset_y;
            for b in &merged {
                if b.width() <= 80.0 || b.height() <= 30.0 {
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
            .filter(|b| b.width() > 80.0 && b.height() > 30.0)
            .collect();

        // Apply NMS to remove overlapping boxes (iou_threshold=0.5)
        self.nms_merged_boxes(merged, 0.5)
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

        result
    }

    /// Split multi-line region using horizontal projection
    fn split_multiline_region(&self, region: &GrayImage) -> Vec<(GrayImage, u32)> {
        let (w, h) = (region.width(), region.height());

        // Don't split short regions
        if h < 60 {
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
