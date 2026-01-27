# -*- coding: utf-8 -*-
"""
Chrome Screen AI OCR - ONNX DirectML Version
支持ONNX和TFLite两种模式的性能对比
"""

import os
import sys
import io
import json
import time
import numpy as np
from pathlib import Path
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

# 修复Windows UTF-8输出
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# 全局选项
USE_ONNX_DETECTION = '--onnx-detect' in sys.argv


class TextDetectorONNX:
    """文本检测模型 - ONNX版本，支持 DirectML"""

    def __init__(self, onnx_path, verbose=False):
        import onnxruntime as ort
        self.ort = ort

        # 选择执行器
        providers = ort.get_available_providers()
        self.use_dml = 'DmlExecutionProvider' in providers and '--cpu' not in sys.argv
        if self.use_dml:
            self.session = ort.InferenceSession(
                str(onnx_path),
                providers=['DmlExecutionProvider', 'CPUExecutionProvider']
            )
            self.device = 'DirectML'
        else:
            self.session = ort.InferenceSession(
                str(onnx_path),
                providers=['CPUExecutionProvider']
            )
            self.device = 'CPU'

        self.input_names = {inp.shape[1]: inp.name for inp in self.session.get_inputs()}
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.verbose = verbose
        print(f"  TextDetector [ONNX/{self.device}]: scales {sorted(self.input_names.keys())}")

    def detect(self, image, threshold=0.3):
        """同步检测（兼容旧接口）"""
        inputs, metadata = self._preprocess_image(image)
        io_binding = self._submit_inference(inputs)
        return self._get_result_and_decode(io_binding, threshold, metadata)

    def _preprocess_image(self, image):
        """预处理图像，返回模型输入和元数据"""
        gray = image.convert('L')
        orig_w, orig_h = image.size

        max_dim = max(orig_w, orig_h)
        scale = 4096 / max_dim
        scaled_w = int(orig_w * scale)
        scaled_h = int(orig_h * scale)
        scaled_img = gray.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

        img_4096 = Image.new('L', (4096, 4096), 255)
        offset_x = (4096 - scaled_w) // 2
        offset_y = (4096 - scaled_h) // 2
        img_4096.paste(scaled_img, (offset_x, offset_y))

        # 保存到 self 供后续使用
        self.img_4096 = img_4096
        self.scale = scale
        self.offset_x = offset_x
        self.offset_y = offset_y

        # 准备输入
        inputs = {}
        for size, name in self.input_names.items():
            resized = img_4096.resize((size, size), Image.Resampling.BILINEAR)
            arr = np.array(resized, dtype=np.uint8).reshape(1, size, size, 1)
            arr = np.ascontiguousarray(arr)
            inputs[name] = arr

        metadata = {
            'scale': scale,
            'scaled_w': scaled_w,
            'scaled_h': scaled_h,
            'offset_x': offset_x,
            'offset_y': offset_y
        }
        return inputs, metadata

    def _submit_inference(self, inputs):
        """Phase 1: 提交推理请求，返回 IOBinding"""
        io_binding = self.session.io_binding()

        # 绑定所有输入
        for name, arr in inputs.items():
            io_binding.bind_cpu_input(name, arr)

        # 绑定输出
        if self.use_dml:
            for output_name in self.output_names:
                io_binding.bind_output(output_name, 'dml', 0)
        else:
            for output_name in self.output_names:
                io_binding.bind_output(output_name, 'cpu', 0)

        # 运行推理
        self.session.run_with_iobinding(io_binding)
        return io_binding

    def _get_result_and_decode(self, io_binding, threshold, metadata):
        """Phase 2+3: 获取输出并解码为检测框"""
        if self.use_dml:
            io_binding.synchronize_outputs()
            outputs = io_binding.copy_outputs_to_cpu()
        else:
            outputs = io_binding.copy_outputs_to_cpu()

        return self._decode_boxes(outputs, threshold, metadata)

    def _decode_boxes(self, outputs, threshold, metadata):
        """解码输出为检测框"""
        offset_x = metadata['offset_x']
        offset_y = metadata['offset_y']
        scaled_w = metadata['scaled_w']
        scaled_h = metadata['scaled_h']

        boxes = []
        for tensor in outputs:
            if len(tensor.shape) == 4 and tensor.shape[-1] == 7:
                feat_h, feat_w = tensor.shape[1], tensor.shape[2]
                conf = tensor[0, :, :, 0]
                ys, xs = np.where(conf > threshold)
                for y, x in zip(ys, xs):
                    c = float(conf[y, x])
                    cx = (x + 0.5) * 4096 / feat_w
                    cy = (y + 0.5) * 4096 / feat_h
                    bw = 4096 / feat_w * 1.2
                    bh = 4096 / feat_h * 1.2
                    if offset_x < cx < offset_x + scaled_w and offset_y < cy < offset_y + scaled_h:
                        boxes.append({
                            'bbox': [cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2],
                            'conf': c, 'in_4096': True
                        })

        boxes = sorted(boxes, key=lambda x: -x['conf'])
        kept = []
        for box in boxes:
            overlap = False
            for k in kept:
                if self._iou(box['bbox'], k['bbox']) > 0.3:
                    overlap = True
                    break
            if not overlap:
                kept.append(box)
        return kept

    def _iou(self, b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)


class TextDetector:
    """文本检测模型 - TFLite版本"""

    def __init__(self, model_dir, verbose=False):
        import tensorflow as tf
        model_path = model_dir / 'gocr/gocr_models/detection/gocr_group_rpn_text_detection_model_2024_q4.tflite'
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_sizes = {inp['shape'][1]: inp for inp in self.input_details}
        self.verbose = verbose
        print(f"  TextDetector [TFLite/CPU]: scales {sorted(self.input_sizes.keys())}")

    def detect(self, image, threshold=0.3):
        """检测文字区域"""
        gray = image.convert('L')
        orig_w, orig_h = image.size

        max_dim = max(orig_w, orig_h)
        scale = 4096 / max_dim
        scaled_w = int(orig_w * scale)
        scaled_h = int(orig_h * scale)
        scaled_img = gray.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

        img_4096 = Image.new('L', (4096, 4096), 255)
        offset_x = (4096 - scaled_w) // 2
        offset_y = (4096 - scaled_h) // 2
        img_4096.paste(scaled_img, (offset_x, offset_y))

        self.img_4096 = img_4096
        self.scale = scale
        self.offset_x = offset_x
        self.offset_y = offset_y

        for size, inp in self.input_sizes.items():
            resized = img_4096.resize((size, size), Image.Resampling.BILINEAR)
            arr = np.array(resized, dtype=np.uint8).reshape(1, size, size, 1)
            self.interpreter.set_tensor(inp['index'], arr)

        self.interpreter.invoke()

        boxes = []
        for out in self.output_details:
            tensor = self.interpreter.get_tensor(out['index'])
            if len(tensor.shape) == 4 and tensor.shape[-1] == 7:
                feat_h, feat_w = tensor.shape[1], tensor.shape[2]
                conf = tensor[0, :, :, 0]
                ys, xs = np.where(conf > threshold)
                for y, x in zip(ys, xs):
                    c = float(conf[y, x])
                    cx = (x + 0.5) * 4096 / feat_w
                    cy = (y + 0.5) * 4096 / feat_h
                    bw = 4096 / feat_w * 1.2
                    bh = 4096 / feat_h * 1.2
                    if offset_x < cx < offset_x + scaled_w and offset_y < cy < offset_y + scaled_h:
                        boxes.append({
                            'bbox': [cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2],
                            'conf': c, 'in_4096': True
                        })

        boxes = sorted(boxes, key=lambda x: -x['conf'])
        kept = []
        for box in boxes:
            overlap = False
            for k in kept:
                if self._iou(box['bbox'], k['bbox']) > 0.3:
                    overlap = True
                    break
            if not overlap:
                kept.append(box)
        return kept

    def _iou(self, b1, b2):
        x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
        x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
        a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
        return inter / (a1 + a2 - inter + 1e-6)


class LayoutSorter:
    """布局排序模型 - TFLite版本"""

    def __init__(self, model_dir, verbose=False):
        import tensorflow as tf
        model_path = model_dir / 'gocr/layout/cluster_sort/model_v2.tflite'
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.verbose = verbose
        print(f"  LayoutSorter [TFLite/CPU]: loaded")

    def sort(self, boxes, image_size):
        if not boxes:
            return []
        for box in boxes:
            b = box['bbox']
            box['cx'] = (b[0] + b[2]) / 2 / image_size[0]
            box['cy'] = (b[1] + b[3]) / 2 / image_size[1]

        sorted_boxes = sorted(boxes, key=lambda x: (x['cy'], x['cx']))
        row_threshold = 0.02
        rows = [[sorted_boxes[0]]]
        current_row = rows[0]
        for box in sorted_boxes[1:]:
            if abs(box['cy'] - current_row[-1]['cy']) < row_threshold:
                current_row.append(box)
            else:
                rows.append(sorted(current_row, key=lambda x: x['cx']))
                current_row = [box]
        rows.append(sorted(current_row, key=lambda x: x['cx']))

        result = []
        for row in rows:
            result.extend(row)
        return result


class LineRecognizerONNX:
    """行识别模型 - ONNX DirectML版本，支持分阶段计算"""

    def __init__(self, onnx_path, char_map_path, verbose=False):
        import onnxruntime as ort
        self.ort = ort

        # 选择执行器
        providers = ort.get_available_providers()
        self.use_dml = 'DmlExecutionProvider' in providers and '--cpu' not in sys.argv
        if self.use_dml:
            self.session = ort.InferenceSession(
                str(onnx_path),
                providers=['DmlExecutionProvider', 'CPUExecutionProvider']
            )
            self.device = 'DirectML'
        else:
            self.session = ort.InferenceSession(
                str(onnx_path),
                providers=['CPUExecutionProvider']
            )
            self.device = 'CPU'

        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [out.name for out in self.session.get_outputs()]
        self.char_map = self._load_char_map(char_map_path)
        self.verbose = verbose

        # Warmup - 第一次推理通常较慢
        warmup_input = np.zeros((1, 32, 168, 1), dtype=np.uint8)
        for _ in range(3):
            self.session.run(None, {self.input_name: warmup_input})

        print(f"  LineRecognizer [ONNX/{self.device}]: {len(self.char_map)} chars (warmed up)")

    def _load_char_map(self, path):
        with open(path, 'r', encoding='utf-8') as f:
            return {int(k): v for k, v in json.load(f).items()}

    def recognize(self, image, return_conf=False):
        """识别单行文字"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != 'L':
            image = image.convert('L')

        w, h = image.size
        if w < 5 or h < 5:
            return ("", 0.0) if return_conf else ""

        ideal_w = int(w * 32 / h)
        if ideal_w <= 168:
            text, conf = self._recognize_segment(image)
            return (text, conf) if return_conf else text
        else:
            seg_w = int(168 * h / 32)
            step = int(seg_w * 0.7)
            results = []
            confs = []
            for x in range(0, w - seg_w // 2, step):
                x2 = min(x + seg_w, w)
                seg = image.crop((x, 0, x2, h))
                text, conf = self._recognize_segment(seg)
                if text:
                    results.append(text)
                    confs.append(conf)
            combined = ' '.join(results)
            avg_conf = float(np.mean(confs)) if confs else 0.0
            return (combined, avg_conf) if return_conf else combined

    def _recognize_segment(self, image):
        w, h = image.size
        scale = 32 / h
        new_w = min(int(w * scale), 168)
        scaled = image.resize((new_w, 32), Image.Resampling.LANCZOS)
        canvas = Image.new('L', (168, 32), 255)
        canvas.paste(scaled, (0, 0))
        return self._recognize_canvas(canvas)

    def _prepare_canvas(self, image):
        """将图像预处理为 canvas，用于批量识别"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != 'L':
            image = image.convert('L')

        w, h = image.size
        if w < 5 or h < 5:
            return None

        scale = 32 / h
        new_w = min(int(w * scale), 168)
        scaled = image.resize((new_w, 32), Image.Resampling.LANCZOS)
        canvas = Image.new('L', (168, 32), 255)
        canvas.paste(scaled, (0, 0))
        return canvas

    def recognize_batch(self, images):
        """批量识别：Phase 1 批量提交，Phase 2+3 逐个获取解码"""
        # Phase 1: 预处理并批量提交
        bindings = []
        valid_indices = []
        for i, image in enumerate(images):
            canvas = self._prepare_canvas(image)
            if canvas is not None:
                binding = self._submit_inference(canvas)
                bindings.append(binding)
                valid_indices.append(i)

        # Phase 2+3: 逐个获取结果并解码
        results = [("", 0.0)] * len(images)
        for idx, binding in zip(valid_indices, bindings):
            text, conf = self._get_result_and_decode(binding)
            results[idx] = (text, conf)

        return results

    def _recognize_canvas(self, canvas):
        """同步识别（兼容旧接口）"""
        io_binding = self._submit_inference(canvas)
        return self._get_result_and_decode(io_binding)

    def _submit_inference(self, canvas):
        """Phase 1: 提交推理请求，返回 IOBinding 供后续获取结果"""
        input_data = np.array(canvas, dtype=np.uint8).reshape(1, 32, 168, 1)
        input_data = np.ascontiguousarray(input_data)

        # 为每次推理创建独立的 IOBinding
        io_binding = self.session.io_binding()

        # 绑定输入（CPU numpy array）
        io_binding.bind_cpu_input(self.input_name, input_data)

        if self.use_dml:
            # DirectML: 绑定输出到 GPU，避免自动复制
            for output_name in self.output_names:
                io_binding.bind_output(output_name, 'dml', 0)
        else:
            # CPU: 绑定输出到 CPU
            for output_name in self.output_names:
                io_binding.bind_output(output_name, 'cpu', 0)

        # 运行推理（DirectML 时快速返回，输出留在 GPU）
        self.session.run_with_iobinding(io_binding)

        return io_binding

    def _get_result_and_decode(self, io_binding):
        """Phase 2+3: 获取输出并立即进行 CTC 解码"""
        if self.use_dml:
            # DirectML: 同步等待 GPU 完成，然后复制到 CPU
            io_binding.synchronize_outputs()
            outputs = io_binding.copy_outputs_to_cpu()
        else:
            # CPU: 直接获取输出
            outputs = io_binding.copy_outputs_to_cpu()

        # Phase 3: CTC 解码（与 Phase 2 融合）
        return self._ctc_decode(outputs)

    def _ctc_decode(self, outputs):
        """CTC 解码"""
        # 找到logits输出 (shape: [1, 42, 8179])
        logits = None
        for out in outputs:
            if len(out.shape) == 3 and out.shape[-1] > 1000:
                logits = out[0].astype(np.float32)
                break

        if logits is None:
            return "", 0.0

        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / e_x.sum(axis=-1, keepdims=True)

        probs = softmax(logits)
        pred = np.argmax(logits, axis=-1)
        max_probs = np.max(probs, axis=-1)

        char_conf_threshold = 0.15
        decoded = []
        conf_scores = []
        prev = -1
        for i, idx in enumerate(pred):
            idx = int(idx)
            char_conf = max_probs[i]
            if idx not in [0, 8178] and idx != prev and char_conf >= char_conf_threshold:
                c = self.char_map.get(idx, '')
                if c:
                    decoded.append(c)
                    conf_scores.append(char_conf)
            prev = idx

        text = ''.join(decoded)
        avg_conf = float(np.mean(conf_scores)) if conf_scores else 0.0
        return text, avg_conf


class ChromeOCR_ONNX:
    """Chrome Screen AI OCR - ONNX DirectML版本"""

    def __init__(self, verbose=False, perf=False):
        self.verbose = verbose
        self.perf = perf
        self.perf_stats = {}

        # TFLite模型路径
        screen_ai_path = Path(os.environ.get('LOCALAPPDATA', '')) / 'Google/Chrome/User Data/screen_ai'
        versions = sorted(
            [d for d in screen_ai_path.iterdir() if d.is_dir() and d.name[0].isdigit()],
            key=lambda x: [int(p) for p in x.name.split('.')], reverse=True
        )
        self.model_dir = versions[0]

        # ONNX模型路径
        script_dir = Path(__file__).parent
        onnx_path = script_dir / 'onnx_models' / 'hanijpan.onnx'
        char_map_path = script_dir / 'hanijpan_char_map.json'

        print(f"Model version: {self.model_dir.name}")
        print("Loading models...")

        t0 = time.perf_counter()
        if USE_ONNX_DETECTION:
            onnx_detect_path = script_dir / 'onnx_models' / 'detection_uint8.onnx'
            self.detector = TextDetectorONNX(onnx_detect_path, verbose)
            self.detector_type = 'ONNX'
        else:
            self.detector = TextDetector(self.model_dir, verbose)
            self.detector_type = 'TFLite'
        t1 = time.perf_counter()
        self.sorter = LayoutSorter(self.model_dir, verbose)
        t2 = time.perf_counter()
        self.recognizer = LineRecognizerONNX(onnx_path, char_map_path, verbose)
        t3 = time.perf_counter()

        if self.perf:
            self.perf_stats['load_detector'] = t1 - t0
            self.perf_stats['load_sorter'] = t2 - t1
            self.perf_stats['load_recognizer'] = t3 - t2
            self.perf_stats['load_total'] = t3 - t0

        print("All models loaded!")

    def print_perf_stats(self):
        if not self.perf_stats:
            return
        print("\n" + "=" * 50)
        print(f"Performance Statistics (Detection: {self.detector_type})")
        print("=" * 50)
        if 'load_total' in self.perf_stats:
            print("\n[Model Loading]")
            detector_device = getattr(self.detector, 'device', 'CPU')
            print(f"  TextDetector:   {self.perf_stats.get('load_detector', 0)*1000:7.1f} ms ({self.detector_type}/{detector_device})")
            print(f"  LayoutSorter:   {self.perf_stats.get('load_sorter', 0)*1000:7.1f} ms (TFLite/CPU)")
            print(f"  LineRecognizer: {self.perf_stats.get('load_recognizer', 0)*1000:7.1f} ms (ONNX/{self.recognizer.device})")
            print(f"  Total:          {self.perf_stats.get('load_total', 0)*1000:7.1f} ms")
        if 'ocr_total' in self.perf_stats:
            print("\n[OCR Processing]")
            print(f"  Detection:      {self.perf_stats.get('detection', 0)*1000:7.1f} ms")
            print(f"  Merge boxes:    {self.perf_stats.get('merge', 0)*1000:7.1f} ms")
            print(f"  Sorting:        {self.perf_stats.get('sorting', 0)*1000:7.1f} ms")
            print(f"  Recognition:    {self.perf_stats.get('recognition_total', 0)*1000:7.1f} ms " +
                  f"({self.perf_stats.get('recognition_count', 0)} lines, " +
                  f"avg {self.perf_stats.get('recognition_avg', 0)*1000:.1f} ms/line)")
            print(f"  Total OCR:      {self.perf_stats.get('ocr_total', 0)*1000:7.1f} ms")
        total = self.perf_stats.get('load_total', 0) + self.perf_stats.get('ocr_total', 0)
        print(f"\n[Total]           {total*1000:7.1f} ms ({total:.2f} s)")

    def ocr(self, image_path, use_detection=True):
        ocr_start = time.perf_counter()
        image = Image.open(image_path)
        print(f"Image: {image.size}")

        if use_detection:
            detector_device = getattr(self.detector, 'device', 'CPU')
            print(f"\n[1/3] Text Detection ({self.detector_type}/{detector_device})...")
            t0 = time.perf_counter()
            boxes = self.detector.detect(image, threshold=0.5)
            t1 = time.perf_counter()
            print(f"  Found {len(boxes)} char-level regions")
            print(f"  Scale factor: {self.detector.scale:.2f}x")
            if self.perf:
                self.perf_stats['detection'] = t1 - t0

            if len(boxes) == 0:
                print("  No text detected")
                return []

            t0 = time.perf_counter()
            lines = self._merge_boxes_to_lines(boxes, y_threshold=20)
            t1 = time.perf_counter()
            print(f"  Merged to {len(lines)} lines")
            if self.perf:
                self.perf_stats['merge'] = t1 - t0

            print("\n[2/3] Layout Sorting (TFLite/CPU)...")
            t0 = time.perf_counter()
            sorted_lines = self.sorter.sort(lines, (4096, 4096))
            t1 = time.perf_counter()
            if self.perf:
                self.perf_stats['sorting'] = t1 - t0

            batch_mode = '--batch' in sys.argv
            print(f"\n[3/3] Line Recognition (ONNX/{self.recognizer.device}, {'batch' if batch_mode else 'sequential'})...")
            results = []
            orig_gray = Image.open(image_path).convert('L')
            scale = self.detector.scale
            offset_x = self.detector.offset_x
            offset_y = self.detector.offset_y
            min_conf = 0.3

            # 预处理所有区域
            regions = []
            region_info = []  # (index, y1)
            for i, box in enumerate(sorted_lines):
                b = box['bbox']
                x1 = int((b[0] - offset_x) / scale)
                y1 = int((b[1] - offset_y) / scale)
                x2 = int((b[2] - offset_x) / scale)
                y2 = int((b[3] - offset_y) / scale)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_gray.width, x2), min(orig_gray.height, y2)

                if x2 - x1 < 10 or y2 - y1 < 5:
                    continue

                region = orig_gray.crop((x1, y1, x2, y2))
                regions.append(region)
                region_info.append((i, y1))

            t0 = time.perf_counter()
            if batch_mode:
                # 批量模式：Phase 1 全部提交，Phase 2+3 逐个解码
                batch_results = self.recognizer.recognize_batch(regions)
                for (i, y1), (text, conf) in zip(region_info, batch_results):
                    if text.strip() and conf >= min_conf:
                        print(f"  L{i+1} (y={y1:4d}) conf={conf:.2f}: {text}")
                        results.append(text)
            else:
                # 顺序模式：逐个识别
                for (i, y1), region in zip(region_info, regions):
                    text, conf = self.recognizer.recognize(region, return_conf=True)
                    if text.strip() and conf >= min_conf:
                        print(f"  L{i+1} (y={y1:4d}) conf={conf:.2f}: {text}")
                        results.append(text)
            t1 = time.perf_counter()

            ocr_end = time.perf_counter()
            if self.perf:
                self.perf_stats['recognition_total'] = t1 - t0
                self.perf_stats['recognition_count'] = len(regions)
                self.perf_stats['recognition_avg'] = (t1 - t0) / len(regions) if regions else 0
                self.perf_stats['ocr_total'] = ocr_end - ocr_start

            return results
        return []

    def _merge_boxes_to_lines(self, boxes, y_threshold=50):
        if not boxes:
            return []
        for box in boxes:
            b = box['bbox']
            box['cx'] = (b[0] + b[2]) / 2
            box['cy'] = (b[1] + b[3]) / 2
            box['w'] = b[2] - b[0]
            box['h'] = b[3] - b[1]

        sorted_boxes = sorted(boxes, key=lambda x: (x['cy'], x['cx']))
        merged = []
        used = set()

        for i, box in enumerate(sorted_boxes):
            if i in used:
                continue
            group = [box]
            used.add(i)
            changed = True
            while changed:
                changed = False
                for j, other in enumerate(sorted_boxes):
                    if j in used:
                        continue
                    for g in group:
                        x_dist = abs(other['cx'] - g['cx'])
                        y_dist = abs(other['cy'] - g['cy'])
                        x_thresh = (other['w'] + g['w'])
                        y_thresh = (other['h'] + g['h']) * 0.3
                        if x_dist < x_thresh and y_dist < y_thresh:
                            group.append(other)
                            used.add(j)
                            changed = True
                            break
            if group:
                merged.append(self._merge_row(group))
        return merged

    def _merge_row(self, row_boxes):
        x1 = min(b['bbox'][0] for b in row_boxes)
        y1 = min(b['bbox'][1] for b in row_boxes)
        x2 = max(b['bbox'][2] for b in row_boxes)
        y2 = max(b['bbox'][3] for b in row_boxes)
        conf = max(b['conf'] for b in row_boxes)
        return {'bbox': [x1, y1, x2, y2], 'conf': conf}


def main():
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print("Chrome Screen AI OCR - ONNX DirectML Version")
        print("=" * 50)
        print("使用ONNX Runtime进行推理，支持TFLite/ONNX检测模型对比")
        print()
        print("Usage:")
        print("  python chrome_ocr_onnx.py <image>")
        print("  python chrome_ocr_onnx.py <image> --perf")
        print("  python chrome_ocr_onnx.py <image> --onnx-detect --perf")
        print()
        print("Options:")
        print("  --perf         显示性能统计")
        print("  --cpu          强制识别使用CPU")
        print("  --onnx-detect  使用ONNX检测模型 (默认TFLite)")
        return

    image_path = sys.argv[1]
    perf = '--perf' in sys.argv

    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return

    print("=" * 50)
    print("Chrome Screen AI OCR (ONNX DirectML)")
    print("=" * 50)

    ocr = ChromeOCR_ONNX(perf=perf)
    results = ocr.ocr(image_path)

    output_file = Path(image_path).stem + '_ocr.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))

    print(f"\n{'=' * 50}")
    print(f"Saved: {output_file}")
    print(f"Lines: {len(results)}")

    if perf:
        ocr.print_perf_stats()


if __name__ == '__main__':
    main()
