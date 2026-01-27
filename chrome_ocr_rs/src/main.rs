use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

mod detector;
mod ocr;
mod recognizer;
mod sorter;
mod utils;

use ocr::ChromeOCR;

#[derive(Parser, Debug)]
#[command(name = "chrome_ocr")]
#[command(about = "Chrome Screen AI OCR - Rust implementation")]
struct Args {
    /// Input image path
    image_path: PathBuf,

    /// Save line images to directory
    #[arg(long)]
    save_lines: bool,

    /// Show performance statistics
    #[arg(long)]
    perf: bool,

    /// Model directory (default: auto-detect Chrome Screen AI models)
    #[arg(long)]
    model_dir: Option<PathBuf>,

    /// Minimum confidence threshold
    #[arg(long, default_value = "0.3")]
    min_conf: f32,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("==================================================");
    println!("Chrome Screen AI OCR (Rust)");
    println!("==================================================");

    let model_dir = match args.model_dir {
        Some(dir) => dir,
        None => utils::find_model_dir()?,
    };

    println!("Model directory: {}", model_dir.display());

    let load_start = Instant::now();
    let mut ocr = ChromeOCR::new(&model_dir, args.perf)?;
    let load_time = load_start.elapsed();

    ocr.set_save_lines(args.save_lines);
    ocr.set_min_conf(args.min_conf);

    let results = ocr.ocr(&args.image_path)?;

    println!("\n==================================================");
    println!("Results ({} lines):", results.len());
    println!("==================================================");
    for (i, line) in results.iter().enumerate() {
        println!("  {}: {}", i + 1, line);
    }

    if args.perf {
        println!("\n==================================================");
        println!("Performance Statistics");
        println!("==================================================");
        println!("Model Loading:");
        ocr.print_load_stats();
        println!("  Total:          {:7.1} ms", load_time.as_secs_f64() * 1000.0);
        println!("\nOCR Processing:");
        ocr.print_ocr_stats();
        let total = load_time.as_secs_f64() + ocr.get_ocr_total();
        println!("\nGrand Total:      {:7.1} ms", total * 1000.0);
    }

    Ok(())
}
