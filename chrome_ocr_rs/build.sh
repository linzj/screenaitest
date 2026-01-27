#!/bin/bash
# Build script for chrome_ocr_rs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export TFLITEC_PREBUILT_PATH="$SCRIPT_DIR/libs/tensorflowlite_c.dll"
export TFLITEC_HEADER_DIR="$SCRIPT_DIR/include"

cd "$SCRIPT_DIR"
cargo build --release
