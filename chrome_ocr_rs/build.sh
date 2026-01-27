#!/bin/bash
# Build script for chrome_ocr_rs
# One-click build with prebuilt TFLite library

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TFLITEC_PREBUILT_PATH="$SCRIPT_DIR/libs/tensorflowlite_c.dll"
export TFLITEC_HEADER_DIR="$SCRIPT_DIR/include"

cd "$SCRIPT_DIR"
cargo build --release

# Copy runtime DLLs to output directory
if [ -f target/release/chrome_ocr.exe ]; then
    cp libs/tensorflowlite_c.dll target/release/
    echo "Build complete: target/release/chrome_ocr.exe"
fi
