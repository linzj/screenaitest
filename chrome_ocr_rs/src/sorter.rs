use anyhow::Result;
use std::path::Path;

use crate::utils::BBox;

pub struct LayoutSorter;

impl LayoutSorter {
    pub fn new(_model_dir: &Path) -> Result<Self> {
        // Using simple row-based sorting, no model needed
        Ok(Self)
    }

    /// Sort boxes according to reading order using row-based grouping
    pub fn sort(&self, boxes: &[BBox], image_size: (u32, u32)) -> Result<Vec<BBox>> {
        if boxes.is_empty() {
            return Ok(Vec::new());
        }

        let (img_w, img_h) = image_size;

        // Calculate normalized center for each box
        let mut boxes_with_center: Vec<_> = boxes
            .iter()
            .map(|b| {
                let cx = (b.x1 + b.x2) / 2.0 / img_w as f32;
                let cy = (b.y1 + b.y2) / 2.0 / img_h as f32;
                (b.clone(), cx, cy)
            })
            .collect();

        // Sort by y (row) first
        boxes_with_center
            .sort_by(|a, b| a.2.partial_cmp(&b.2).unwrap_or(std::cmp::Ordering::Equal));

        // Group into rows
        let row_threshold = 0.03f32;
        let mut rows: Vec<Vec<(BBox, f32, f32)>> = Vec::new();
        let mut current_row: Vec<(BBox, f32, f32)> = vec![boxes_with_center[0].clone()];

        for item in boxes_with_center.into_iter().skip(1) {
            if (item.2 - current_row.last().unwrap().2).abs() < row_threshold {
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

        // Flatten rows into result
        let result: Vec<BBox> = rows
            .into_iter()
            .flat_map(|row| row.into_iter().map(|(b, _, _)| b))
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
