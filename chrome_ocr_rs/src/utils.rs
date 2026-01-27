use anyhow::{anyhow, Result};
use std::path::PathBuf;

/// Find Chrome Screen AI model directory
pub fn find_model_dir() -> Result<PathBuf> {
    let local_app_data =
        dirs::data_local_dir().ok_or_else(|| anyhow!("Cannot find local app data directory"))?;

    let screen_ai_dir = local_app_data
        .join("Google")
        .join("Chrome")
        .join("User Data")
        .join("screen_ai");

    if !screen_ai_dir.exists() {
        return Err(anyhow!(
            "Chrome Screen AI directory not found: {}",
            screen_ai_dir.display()
        ));
    }

    // Find the latest version directory
    let mut versions: Vec<_> = std::fs::read_dir(&screen_ai_dir)?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.path().is_dir()
                && e.file_name()
                    .to_str()
                    .map(|s| {
                        s.chars()
                            .next()
                            .map(|c| c.is_ascii_digit())
                            .unwrap_or(false)
                    })
                    .unwrap_or(false)
        })
        .collect();

    versions.sort_by(|a, b| {
        let parse_version =
            |s: &str| -> Vec<i32> { s.split('.').filter_map(|p| p.parse().ok()).collect() };
        let va = parse_version(a.file_name().to_str().unwrap_or(""));
        let vb = parse_version(b.file_name().to_str().unwrap_or(""));
        vb.cmp(&va)
    });

    versions
        .first()
        .map(|e| e.path())
        .ok_or_else(|| anyhow!("No version directory found in {}", screen_ai_dir.display()))
}

/// Load vocabulary from JSON char map file
/// The file should be a JSON object with numeric keys (as strings) and character values
pub fn load_vocab(path: &std::path::Path) -> Result<Vec<String>> {
    // Try JSON format first (hanijpan_char_map.json)
    let json_path = path.with_extension("json");
    let json_path = if json_path.exists() {
        json_path
    } else {
        // Also check in the same directory with a different name
        path.parent()
            .map(|p| p.join("hanijpan_char_map.json"))
            .filter(|p| p.exists())
            .unwrap_or_else(|| {
                // Check in current directory
                std::path::PathBuf::from("hanijpan_char_map.json")
            })
    };

    if json_path.exists() {
        let content = std::fs::read_to_string(&json_path)?;
        let map: std::collections::HashMap<String, String> =
            serde_json::from_str(&content)
            .map_err(|e| anyhow!("Failed to parse JSON: {}", e))?;

        // Find max index
        let max_idx = map.keys()
            .filter_map(|k| k.parse::<usize>().ok())
            .max()
            .unwrap_or(0);

        // Build vocab vector
        let mut vocab = vec![String::new(); max_idx + 1];
        for (k, v) in map {
            if let Ok(idx) = k.parse::<usize>() {
                if idx < vocab.len() {
                    vocab[idx] = v;
                }
            }
        }
        return Ok(vocab);
    }

    // Fall back to plain text format
    if path.exists() {
        let content = std::fs::read_to_string(path)?;
        return Ok(content.lines().map(|s| s.to_string()).collect());
    }

    Err(anyhow!("Vocabulary file not found: {}", path.display()))
}

/// Calculate IoU between two bounding boxes [x1, y1, x2, y2]
pub fn calc_iou(b1: &[f32; 4], b2: &[f32; 4]) -> f32 {
    let x1 = b1[0].max(b2[0]);
    let y1 = b1[1].max(b2[1]);
    let x2 = b1[2].min(b2[2]);
    let y2 = b1[3].min(b2[3]);

    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let a1 = (b1[2] - b1[0]) * (b1[3] - b1[1]);
    let a2 = (b2[2] - b2[0]) * (b2[3] - b2[1]);

    inter / (a1 + a2 - inter + 1e-6)
}

/// Calculate containment ratio of b1 in b2
pub fn calc_containment(b1: &[f32; 4], b2: &[f32; 4]) -> f32 {
    let x1 = b1[0].max(b2[0]);
    let y1 = b1[1].max(b2[1]);
    let x2 = b1[2].min(b2[2]);
    let y2 = b1[3].min(b2[3]);

    let inter = (x2 - x1).max(0.0) * (y2 - y1).max(0.0);
    let a1 = (b1[2] - b1[0]) * (b1[3] - b1[1]);

    inter / (a1 + 1e-6)
}

/// Calculate y-overlap ratio with x-overlap check
pub fn calc_xy_overlap(b1: &[f32; 4], b2: &[f32; 4]) -> f32 {
    let y1 = b1[1].max(b2[1]);
    let y2 = b1[3].min(b2[3]);
    let y_overlap = (y2 - y1).max(0.0);
    let h1 = b1[3] - b1[1];
    let h2 = b2[3] - b2[1];
    let min_h = h1.min(h2);
    let y_ratio = if min_h > 0.0 { y_overlap / min_h } else { 0.0 };

    let x1 = b1[0].max(b2[0]);
    let x2 = b1[2].min(b2[2]);
    let x_overlap = (x2 - x1).max(0.0);
    let w1 = b1[2] - b1[0];
    let w2 = b2[2] - b2[0];
    let min_w = w1.min(w2);
    let x_ratio = if min_w > 0.0 { x_overlap / min_w } else { 0.0 };

    if y_ratio > 0.5 && x_ratio > 0.5 {
        y_ratio.max(x_ratio)
    } else {
        0.0
    }
}

#[derive(Clone, Debug)]
pub struct BBox {
    pub x1: f32,
    pub y1: f32,
    pub x2: f32,
    pub y2: f32,
    pub conf: f32,
}

impl BBox {
    pub fn new(x1: f32, y1: f32, x2: f32, y2: f32, conf: f32) -> Self {
        Self {
            x1,
            y1,
            x2,
            y2,
            conf,
        }
    }

    pub fn width(&self) -> f32 {
        self.x2 - self.x1
    }

    pub fn height(&self) -> f32 {
        self.y2 - self.y1
    }

    pub fn center(&self) -> (f32, f32) {
        ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)
    }

    pub fn as_array(&self) -> [f32; 4] {
        [self.x1, self.y1, self.x2, self.y2]
    }
}
