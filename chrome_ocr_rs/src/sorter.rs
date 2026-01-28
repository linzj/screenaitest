use anyhow::Result;
use std::path::Path;

use crate::utils::BBox;

pub struct LayoutSorter;

impl LayoutSorter {
    pub fn new(_model_dir: &Path) -> Result<Self> {
        // Using simple row-based sorting, no model needed
        Ok(Self)
    }

    /// Sort boxes according to reading order using Chrome's ClusterLines parameters
    /// From IDA: minimum_breadth_ratio=0.6, maximum_depth_gap=1.5, minimum_breadth_overlap=0.6
    pub fn sort(&self, boxes: &[BBox], _image_size: (u32, u32)) -> Result<Vec<BBox>> {
        if boxes.is_empty() {
            return Ok(Vec::new());
        }

        // Calculate center, height, and width for each box
        let mut boxes_with_info: Vec<_> = boxes
            .iter()
            .map(|b| {
                let cy = (b.y1 + b.y2) / 2.0;
                let cx = (b.x1 + b.x2) / 2.0;
                let h = b.y2 - b.y1;
                let w = b.x2 - b.x1;
                (b.clone(), cx, cy, h, w)
            })
            .collect();

        // Sort by y1 (top of box) first
        boxes_with_info.sort_by(|a, b| {
            a.0.y1
                .partial_cmp(&b.0.y1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Group into rows based on Chrome's ClusterLines parameters
        let mut rows: Vec<Vec<(BBox, f32, f32, f32, f32)>> = Vec::new();
        let mut current_row: Vec<(BBox, f32, f32, f32, f32)> = vec![boxes_with_info[0].clone()];

        for item in boxes_with_info.into_iter().skip(1) {
            // Get current row bounds
            let row_y1 = current_row
                .iter()
                .map(|x| x.0.y1)
                .fold(f32::INFINITY, f32::min);
            let row_y2 = current_row
                .iter()
                .map(|x| x.0.y2)
                .fold(f32::NEG_INFINITY, f32::max);
            let row_h = row_y2 - row_y1;

            let item_y1 = item.0.y1;
            let item_y2 = item.0.y2;
            let item_h = item.3;

            // Chrome's maximum_depth_gap = 1.5 (relative to average height)
            let avg_h = (row_h + item_h) * 0.5;
            let merged_h = row_y2.max(item_y2) - row_y1.min(item_y1);
            let depth_gap = (merged_h - row_h - item_h).max(0.0);
            let depth_gap_ratio = if avg_h > 0.0 { depth_gap / avg_h } else { 0.0 };

            // Calculate vertical overlap ratio
            let overlap = (row_y2.min(item_y2) - row_y1.max(item_y1)).max(0.0);
            let min_h = row_h.min(item_h);
            let overlap_ratio = if min_h > 0.0 { overlap / min_h } else { 0.0 };

            // Same row if: significant vertical overlap
            // Chrome uses minimum_breadth_overlap = 0.6, but we need stricter here
            // Only consider same row if overlap > 0.5 (stricter than Chrome's 0.6 threshold)
            let same_row = overlap_ratio > 0.5;

            if same_row {
                current_row.push(item);
            } else {
                // Sort current row by x and add to rows
                current_row
                    .sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
                rows.push(current_row);
                current_row = vec![item];
            }
        }

        // Don't forget the last row
        current_row.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        rows.push(current_row);

        // Sort rows by their minimum y1 position (top of row)
        rows.sort_by(|a, b| {
            let min_y1_a = a.iter().map(|x| x.0.y1).fold(f32::INFINITY, f32::min);
            let min_y1_b = b.iter().map(|x| x.0.y1).fold(f32::INFINITY, f32::min);
            min_y1_a
                .partial_cmp(&min_y1_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Flatten rows into result
        let result: Vec<BBox> = rows
            .into_iter()
            .flat_map(|row| row.into_iter().map(|(b, _, _, _, _)| b))
            .collect();

        Ok(result)
    }

    /// Simple sorting fallback (by y then x)
    #[allow(dead_code)]
    pub fn sort_simple(boxes: &[BBox]) -> Vec<BBox> {
        let mut sorted: Vec<_> = boxes.to_vec();
        sorted.sort_by(|a, b| {
            let cy_a = (a.y1 + a.y2) / 2.0;
            let cy_b = (b.y1 + b.y2) / 2.0;
            let cx_a = (a.x1 + a.x2) / 2.0;
            let cx_b = (b.x1 + b.x2) / 2.0;
            cy_a.partial_cmp(&cy_b)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| cx_a.partial_cmp(&cx_b).unwrap_or(std::cmp::Ordering::Equal))
        });
        sorted
    }
}
