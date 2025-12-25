# -*- coding: utf-8 -*-
"""Convert Chrome Screen AI TFLite models to ONNX format"""

import os
import sys
from pathlib import Path

def convert_models():
    try:
        import tflite2onnx
    except ImportError:
        print("Please install tflite2onnx: pip install tflite2onnx")
        return

    # Find Chrome Screen AI models
    screen_ai_path = Path(os.environ.get('LOCALAPPDATA', '')) / 'Google/Chrome/User Data/screen_ai'
    versions = sorted(
        [d for d in screen_ai_path.iterdir() if d.is_dir() and d.name[0].isdigit()],
        key=lambda x: [int(p) for p in x.name.split('.')], reverse=True
    )
    model_dir = versions[0]
    print(f"Source: {model_dir}")

    # Output directory
    output_dir = Path(__file__).parent / 'onnx_models'
    output_dir.mkdir(exist_ok=True)
    print(f"Output: {output_dir}")

    # Models to convert
    models = [
        ('detection', model_dir / 'gocr/gocr_models/detection/gocr_group_rpn_text_detection_model_2024_q4.tflite'),
        ('layout_sorter', model_dir / 'gocr/layout/cluster_sort/model_v2.tflite'),
        ('hanijpan', model_dir / 'gocr/gocr_models/line_recognition_mobile_convnext320_omni/hanijpan.tflite'),
    ]

    for name, tflite_path in models:
        if not tflite_path.exists():
            print(f"[SKIP] {name}: {tflite_path} not found")
            continue

        onnx_path = output_dir / f'{name}.onnx'
        print(f"\n[{name}]")
        print(f"  Input:  {tflite_path}")
        print(f"  Output: {onnx_path}")
        print(f"  Size:   {tflite_path.stat().st_size / 1024 / 1024:.1f} MB")

        try:
            tflite2onnx.convert(str(tflite_path), str(onnx_path))
            print(f"  Status: OK ({onnx_path.stat().st_size / 1024 / 1024:.1f} MB)")
        except Exception as e:
            print(f"  Status: FAILED - {e}")

    print("\nDone!")

if __name__ == '__main__':
    convert_models()
