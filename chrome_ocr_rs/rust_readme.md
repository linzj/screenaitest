# Chrome OCR Rust Implementation

## Build

```bash
cd chrome_ocr_rs
./build.sh
```

The build script uses pre-built TFLite C library. The output binary will be at `target/release/chrome_ocr.exe`.

## Run

Basic usage (run from screenaitest directory where `hanijpan_char_map.json` is located):

```bash
cd /d/linzjUbuntu2204/quark-local-ocr/screenaitest
./chrome_ocr_rs/target/release/chrome_ocr.exe <image_path>
```

### Options

- `--perf` - Show performance statistics
- `--save-lines` - Save cropped line images to `<image>_lines/` directory
- `--min-conf <value>` - Set minimum confidence threshold (default: 0.3)

### Examples

```bash
# Basic OCR
./chrome_ocr_rs/target/release/chrome_ocr.exe ../resource/test_imgs/general_ocr_002.png

# With performance stats
./chrome_ocr_rs/target/release/chrome_ocr.exe --perf ../resource/test_imgs/general_ocr_002.png

# Save line images for debugging
./chrome_ocr_rs/target/release/chrome_ocr.exe --save-lines ../resource/test_imgs/general_ocr_002.png
```

## Requirements

- The vocab file `hanijpan_char_map.json` must be in the current working directory
- Chrome Screen AI models must be installed at `%LOCALAPPDATA%\Google\Chrome\User Data\screen_ai\`

## Comparison with Python

| Feature | Python | Rust |
|---------|--------|------|
| Detection | TFLite | TFLite |
| Recognition | ONNX | TFLite |
| Multi-threading | No | Yes (4 threads) |
