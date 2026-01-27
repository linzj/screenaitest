use anyhow::Result;
use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

mod detector;
mod native_ocr;
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

    /// Use native chrome_screen_ai.dll instead of TFLite
    #[arg(long)]
    native: bool,

    /// Run both TFLite and Native modes for comparison (AB test)
    #[arg(long)]
    ab: bool,
}

fn main() -> Result<()> {
    env_logger::init();

    let args = Args::parse();

    println!("==================================================");
    println!("Chrome Screen AI OCR (Rust)");
    println!("==================================================");

    let model_dir = match &args.model_dir {
        Some(dir) => dir.clone(),
        None => utils::find_model_dir()?,
    };

    println!("Model directory: {}", model_dir.display());

    if args.ab {
        // AB test: run both modes
        run_ab_test(&args, &model_dir)
    } else if args.native {
        // Use native chrome_screen_ai.dll
        run_native_ocr(&args, &model_dir)
    } else {
        // Use TFLite implementation
        run_tflite_ocr(&args, &model_dir)
    }
}

fn run_ab_test(args: &Args, model_dir: &PathBuf) -> Result<()> {
    println!("=== AB Test: TFLite vs Native ===\n");

    // Run TFLite first
    println!(">>> Running TFLite mode...\n");
    let tflite_start = Instant::now();
    let tflite_result = run_tflite_ocr_silent(args, model_dir);
    let tflite_time = tflite_start.elapsed();

    // Run Native
    println!("\n>>> Running Native mode...\n");
    let native_start = Instant::now();
    let native_result = run_native_ocr_silent(args, model_dir);
    let native_time = native_start.elapsed();

    // Compare results
    println!("\n==================================================");
    println!("AB Test Results");
    println!("==================================================");

    println!("\n[TFLite] Time: {:.1} ms", tflite_time.as_secs_f64() * 1000.0);
    match &tflite_result {
        Ok(lines) => println!("  Lines: {}", lines.len()),
        Err(e) => println!("  Error: {}", e),
    }

    println!("\n[Native] Time: {:.1} ms", native_time.as_secs_f64() * 1000.0);
    match &native_result {
        Ok(lines) => println!("  Lines: {}", lines.len()),
        Err(e) => println!("  Error: {}", e),
    }

    Ok(())
}

fn run_tflite_ocr_silent(args: &Args, model_dir: &PathBuf) -> Result<Vec<String>> {
    let mut ocr = ChromeOCR::new(model_dir, false)?;
    ocr.set_min_conf(args.min_conf);
    ocr.ocr(&args.image_path)
}

fn run_native_ocr_silent(args: &Args, model_dir: &PathBuf) -> Result<Vec<native_ocr::OcrLine>> {
    let native_ocr = native_ocr::NativeOCR::new(model_dir)?;
    let image = image::open(&args.image_path)?.to_luma8();
    native_ocr.perform_ocr(&image)
}

fn run_native_ocr(args: &Args, model_dir: &PathBuf) -> Result<()> {
    println!("Mode: Native (chrome_screen_ai.dll)");

    let load_start = Instant::now();
    let native_ocr = native_ocr::NativeOCR::new(model_dir)?;
    let load_time = load_start.elapsed();

    println!("Opening image: {}", args.image_path.display());
    let image = image::open(&args.image_path)?.to_luma8();
    println!("Image: {}x{}", image.width(), image.height());

    let ocr_start = Instant::now();
    let lines = native_ocr.perform_ocr(&image)?;
    let ocr_time = ocr_start.elapsed();

    println!("\n==================================================");
    println!("Results ({} lines):", lines.len());
    println!("==================================================");
    for (i, line) in lines.iter().enumerate() {
        println!("  {}: {} (conf={:.2})", i + 1, line.text, line.confidence);
    }

    if args.perf {
        println!("\n==================================================");
        println!("Performance Statistics (Native)");
        println!("==================================================");
        println!("  Loading:   {:7.1} ms", load_time.as_secs_f64() * 1000.0);
        println!("  OCR:       {:7.1} ms", ocr_time.as_secs_f64() * 1000.0);
        println!("  Total:     {:7.1} ms", (load_time + ocr_time).as_secs_f64() * 1000.0);
    }

    Ok(())
}

fn run_tflite_ocr(args: &Args, model_dir: &PathBuf) -> Result<()> {
    println!("Mode: TFLite");

    let load_start = Instant::now();
    let mut ocr = ChromeOCR::new(model_dir, args.perf)?;
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
