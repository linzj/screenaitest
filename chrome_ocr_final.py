# -*- coding: utf-8 -*-
"""
Chrome Screen AI OCR - 完整版
使用Chrome内置的所有OCR相关模型:
1. 文本检测模型 (RPN) - 检测文字区域
2. 布局排序模型 - 确定阅读顺序
3. 行识别模型 (Hanijpan) - 识别文字
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


class TextDetector:
    """文本检测模型 - 检测图像中的文字区域"""

    def __init__(self, model_dir, verbose=False):
        import tensorflow as tf
        model_path = model_dir / 'gocr/gocr_models/detection/gocr_group_rpn_text_detection_model_2024_q4.tflite'
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_sizes = {inp['shape'][1]: inp for inp in self.input_details}
        self.verbose = verbose
        print(f"  TextDetector: scales {sorted(self.input_sizes.keys())}")

    def print_model_info(self):
        """打印模型详细信息"""
        print("\n" + "=" * 60)
        print("TextDetector (RPN) - gocr_group_rpn_text_detection_model_2024_q4.tflite")
        print("=" * 60)
        print("\n[输入 Inputs]")
        for i, inp in enumerate(self.input_details):
            quant = inp.get('quantization_parameters', {})
            print(f"  Input {i}:")
            print(f"    Name:  {inp['name']}")
            print(f"    Shape: {inp['shape']}")
            print(f"    Type:  {inp['dtype']}")
            if quant and quant.get('scales') is not None and len(quant.get('scales', [])) > 0:
                print(f"    Quantization: scale={quant['scales'][0]:.6f}, zero_point={quant['zero_points'][0]}")

        print("\n[输出 Outputs]")
        for i, out in enumerate(self.output_details):
            quant = out.get('quantization_parameters', {})
            print(f"  Output {i}:")
            print(f"    Name:  {out['name']}")
            print(f"    Shape: {out['shape']}")
            print(f"    Type:  {out['dtype']}")
            if quant and quant.get('scales') is not None and len(quant.get('scales', [])) > 0:
                print(f"    Quantization: scale={quant['scales'][0]:.6f}, zero_point={quant['zero_points'][0]}")

    def detect(self, image, threshold=0.3):
        """检测文字区域，返回边界框列表和放大后的图片"""
        gray = image.convert('L')
        orig_w, orig_h = image.size

        # 放大到4096并padding
        max_dim = max(orig_w, orig_h)
        scale = 4096 / max_dim
        scaled_w = int(orig_w * scale)
        scaled_h = int(orig_h * scale)
        scaled_img = gray.resize((scaled_w, scaled_h), Image.Resampling.LANCZOS)

        img_4096 = Image.new('L', (4096, 4096), 255)
        offset_x = (4096 - scaled_w) // 2
        offset_y = (4096 - scaled_h) // 2
        img_4096.paste(scaled_img, (offset_x, offset_y))

        # 保存放大图和参数供后续裁剪使用
        self.img_4096 = img_4096
        self.scale = scale
        self.offset_x = offset_x
        self.offset_y = offset_y

        # 创建多尺度输入
        for size, inp in self.input_sizes.items():
            resized = img_4096.resize((size, size), Image.Resampling.BILINEAR)
            arr = np.array(resized, dtype=np.uint8).reshape(1, size, size, 1)
            self.interpreter.set_tensor(inp['index'], arr)

        self.interpreter.invoke()

        # 收集检测结果 (坐标在4096系统中)
        boxes = []
        for out in self.output_details:
            tensor = self.interpreter.get_tensor(out['index'])
            if len(tensor.shape) == 4 and tensor.shape[-1] == 7:
                feat_h, feat_w = tensor.shape[1], tensor.shape[2]
                conf = tensor[0, :, :, 0]

                # 找高置信度位置
                ys, xs = np.where(conf > threshold)
                for y, x in zip(ys, xs):
                    c = float(conf[y, x])
                    cx = (x + 0.5) * 4096 / feat_w
                    cy = (y + 0.5) * 4096 / feat_h
                    bw = 4096 / feat_w * 1.2
                    bh = 4096 / feat_h * 1.2
                    # 检查是否在内容区域内
                    if offset_x < cx < offset_x + scaled_w and offset_y < cy < offset_y + scaled_h:
                        boxes.append({
                            'bbox': [cx - bw/2, cy - bh/2, cx + bw/2, cy + bh/2],
                            'conf': c,
                            'in_4096': True  # 标记坐标在4096系统中
                        })

        # NMS去重
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
    """布局排序模型 - 确定文字区域的阅读顺序"""

    def __init__(self, model_dir, verbose=False):
        import tensorflow as tf
        model_path = model_dir / 'gocr/layout/cluster_sort/model_v2.tflite'
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.verbose = verbose
        print(f"  LayoutSorter: loaded")

    def print_model_info(self):
        """打印模型详细信息"""
        print("\n" + "=" * 60)
        print("LayoutSorter - cluster_sort/model_v2.tflite")
        print("=" * 60)
        print("\n[输入 Inputs]")
        for i, inp in enumerate(self.input_details):
            quant = inp.get('quantization_parameters', {})
            print(f"  Input {i}:")
            print(f"    Name:  {inp['name']}")
            print(f"    Shape: {inp['shape']}")
            print(f"    Type:  {inp['dtype']}")
            if quant and quant.get('scales') is not None and len(quant.get('scales', [])) > 0:
                print(f"    Quantization: scale={quant['scales'][0]:.6f}, zero_point={quant['zero_points'][0]}")

        print("\n[输出 Outputs]")
        for i, out in enumerate(self.output_details):
            quant = out.get('quantization_parameters', {})
            print(f"  Output {i}:")
            print(f"    Name:  {out['name']}")
            print(f"    Shape: {out['shape']}")
            print(f"    Type:  {out['dtype']}")
            if quant and quant.get('scales') is not None and len(quant.get('scales', [])) > 0:
                print(f"    Quantization: scale={quant['scales'][0]:.6f}, zero_point={quant['zero_points'][0]}")

    def sort(self, boxes, image_size):
        """对检测框排序，返回阅读顺序"""
        if not boxes:
            return []

        w, h = image_size

        # 计算中心点
        for box in boxes:
            b = box['bbox']
            box['cx'] = (b[0] + b[2]) / 2 / w
            box['cy'] = (b[1] + b[3]) / 2 / h

        # 按行分组
        sorted_boxes = sorted(boxes, key=lambda x: x['cy'])
        rows = []
        row_threshold = 0.03
        current_row = [sorted_boxes[0]]

        for box in sorted_boxes[1:]:
            if abs(box['cy'] - current_row[-1]['cy']) < row_threshold:
                current_row.append(box)
            else:
                rows.append(sorted(current_row, key=lambda x: x['cx']))
                current_row = [box]
        rows.append(sorted(current_row, key=lambda x: x['cx']))

        # 展平
        result = []
        for row in rows:
            result.extend(row)
        return result


class LineRecognizer:
    """行识别模型 - 识别单行文字"""

    def __init__(self, model_dir, verbose=False):
        import tensorflow as tf
        model_path = model_dir / 'gocr/gocr_models/line_recognition_mobile_convnext320_omni/hanijpan.tflite'
        self.interpreter = tf.lite.Interpreter(model_path=str(model_path))
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.char_map = self._load_char_map(model_dir)
        self.verbose = verbose
        print(f"  LineRecognizer: {len(self.char_map)} chars")

    def print_model_info(self):
        """打印模型详细信息"""
        print("\n" + "=" * 60)
        print("LineRecognizer (Hanijpan) - hanijpan.tflite")
        print("=" * 60)
        print("\n[输入 Inputs]")
        for i, inp in enumerate(self.input_details):
            quant = inp.get('quantization_parameters', {})
            print(f"  Input {i}:")
            print(f"    Name:  {inp['name']}")
            print(f"    Shape: {inp['shape']}")
            print(f"    Type:  {inp['dtype']}")
            if quant and quant.get('scales') is not None and len(quant.get('scales', [])) > 0:
                print(f"    Quantization: scale={quant['scales'][0]:.6f}, zero_point={quant['zero_points'][0]}")

        print("\n[输出 Outputs]")
        for i, out in enumerate(self.output_details):
            quant = out.get('quantization_parameters', {})
            print(f"  Output {i}:")
            print(f"    Name:  {out['name']}")
            print(f"    Shape: {out['shape']}")
            print(f"    Type:  {out['dtype']}")
            if quant and quant.get('scales') is not None and len(quant.get('scales', [])) > 0:
                print(f"    Quantization: scale={quant['scales'][0]:.6f}, zero_point={quant['zero_points'][0]}")

        # 字符映射统计
        print(f"\n[字符映射 Character Map]")
        print(f"  Total chars: {len(self.char_map)}")
        # 统计字符类型
        cjk_count = 0
        ascii_count = 0
        digit_count = 0
        punct_count = 0
        for idx, char in self.char_map.items():
            if '\u4e00' <= char <= '\u9fff' or '\u3400' <= char <= '\u4dbf':
                cjk_count += 1
            elif char.isascii() and char.isalpha():
                ascii_count += 1
            elif char.isdigit():
                digit_count += 1
            elif not char.isalnum():
                punct_count += 1
        print(f"  CJK chars:   {cjk_count}")
        print(f"  ASCII:       {ascii_count}")
        print(f"  Digits:      {digit_count}")
        print(f"  Punctuation: {punct_count}")

    def _load_char_map(self, model_dir):
        cache_path = Path(__file__).parent / 'hanijpan_char_map.json'
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                return {int(k): v for k, v in json.load(f).items()}

        label_path = model_dir / 'gocr/gocr_models/line_recognition_mobile_convnext320_omni/hanijpan_label_map.pb'
        with open(label_path, 'rb') as f:
            data = f.read()

        char_map = {}
        i = 0
        while i < len(data):
            if data[i:i+1] == b'\x0a':
                entry_len = data[i+1]
                if i + 2 + entry_len <= len(data):
                    entry = data[i+2:i+2+entry_len]
                    if len(entry) >= 4 and entry[0] == 0x0a:
                        char_len = entry[1]
                        if 2 + char_len < len(entry):
                            try:
                                char = entry[2:2+char_len].decode('utf-8')
                                idx_pos = 2 + char_len
                                if idx_pos < len(entry) and entry[idx_pos] == 0x10:
                                    idx = 0
                                    shift = 0
                                    for j in range(idx_pos+1, len(entry)):
                                        b = entry[j]
                                        idx |= (b & 0x7f) << shift
                                        shift += 7
                                        if not (b & 0x80):
                                            break
                                    char_map[idx] = char
                            except:
                                pass
                    i += 2 + entry_len
                    continue
            i += 1

        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in char_map.items()}, f, ensure_ascii=False)
        return char_map

    def recognize(self, image, return_conf=False):
        """识别单行文字，保持宽高比缩放后padding到168x32，长行分段识别"""
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        if image.mode != 'L':
            image = image.convert('L')

        w, h = image.size
        if w < 5 or h < 5:
            return ("", 0.0) if return_conf else ""

        # Calculate ideal width when scaled to height 32
        # Use 160 as effective width (168 - 8px left margin for edge recognition)
        effective_width = 160
        ideal_w = int(w * 32 / h)

        if ideal_w <= effective_width:
            # Short line: direct recognition
            text, conf = self._recognize_segment(image)
            return (text, conf) if return_conf else text
        else:
            # Long line: segment recognition with overlap deduplication
            seg_w = int(effective_width * h / 32)
            step = int(seg_w * 0.7)

            segments = []
            confs = []
            for x in range(0, w - seg_w // 2, step):
                x2 = min(x + seg_w, w)
                seg = image.crop((x, 0, x2, h))
                text, conf = self._recognize_segment(seg)
                if text:
                    segments.append(text)
                    confs.append(conf)

            # Merge segments with overlap deduplication
            combined = self._merge_overlapping_segments(segments)
            avg_conf = float(np.mean(confs)) if confs else 0.0
            return (combined, avg_conf) if return_conf else combined

    def _merge_overlapping_segments(self, segments):
        """Merge overlapping text segments by finding and removing duplicates"""
        if not segments:
            return ""
        if len(segments) == 1:
            return segments[0]

        result = segments[0]
        for i in range(1, len(segments)):
            next_seg = segments[i]
            # Find the overlap between end of result and start of next_seg
            overlap_len = self._find_overlap(result, next_seg)
            if overlap_len > 0:
                # Append only the non-overlapping part
                result += next_seg[overlap_len:]
            else:
                # No overlap found, just concatenate
                result += next_seg
        return result

    def _find_overlap(self, s1, s2):
        """Find the length of overlap between end of s1 and start of s2"""
        # Try to find the longest suffix of s1 that matches a prefix of s2
        max_overlap = min(len(s1), len(s2))
        for overlap_len in range(max_overlap, 0, -1):
            if s1[-overlap_len:] == s2[:overlap_len]:
                return overlap_len
        return 0

    def _recognize_segment(self, image):
        """识别单个片段，返回(文字, 置信度)"""
        w, h = image.size

        # Scale to height 32, width proportionally
        scale = 32 / h
        # Leave 8px left margin to avoid edge recognition issues
        left_margin = 8
        new_w = min(int(w * scale), 168 - left_margin)
        new_h = 32

        scaled = image.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Padding to 168x32 with left margin
        canvas = Image.new('L', (168, 32), 255)
        canvas.paste(scaled, (left_margin, 0))

        return self._recognize_canvas(canvas)

    def _recognize_canvas(self, canvas):
        """识别168x32的画布，返回(文字, 置信度)"""
        input_data = np.array(canvas, dtype=np.uint8).reshape(1, 32, 168, 1)
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        for out in self.output_details:
            tensor = self.interpreter.get_tensor(out['index'])
            if len(tensor.shape) == 3 and tensor.shape[-1] > 1000:
                logits = tensor[0]
                # 反量化
                quant = out.get('quantization_parameters', {})
                if quant and quant.get('scales'):
                    logits = (logits.astype(np.float32) - quant['zero_points'][0]) * quant['scales'][0]
                break
        else:
            return "", 0.0

        # Softmax计算置信度
        def softmax(x):
            e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
            return e_x / e_x.sum(axis=-1, keepdims=True)

        probs = softmax(logits)
        pred = np.argmax(logits, axis=-1)
        max_probs = np.max(probs, axis=-1)

        # 单字符置信度阈值 - 低于此值的字符视为噪声
        # 0.15可以过滤掉边界噪声如 "|" (12.11%)
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


class ChromeOCR:
    """Chrome Screen AI OCR Pipeline - 使用所有模型"""

    def __init__(self, verbose=False, perf=False, save_lines=False):
        self.verbose = verbose
        self.perf = perf
        self.save_lines = save_lines
        self.perf_stats = {}  # 存储性能统计

        screen_ai_path = Path(os.environ.get('LOCALAPPDATA', '')) / 'Google/Chrome/User Data/screen_ai'
        versions = sorted(
            [d for d in screen_ai_path.iterdir() if d.is_dir() and d.name[0].isdigit()],
            key=lambda x: [int(p) for p in x.name.split('.')], reverse=True
        )
        self.model_dir = versions[0]
        print(f"Model version: {self.model_dir.name}")
        print(f"Model directory: {self.model_dir}")
        print("Loading models...")

        t0 = time.perf_counter()
        self.detector = TextDetector(self.model_dir, verbose)
        t1 = time.perf_counter()
        self.sorter = LayoutSorter(self.model_dir, verbose)
        t2 = time.perf_counter()
        self.recognizer = LineRecognizer(self.model_dir, verbose)
        t3 = time.perf_counter()

        if self.perf:
            self.perf_stats['load_detector'] = t1 - t0
            self.perf_stats['load_sorter'] = t2 - t1
            self.perf_stats['load_recognizer'] = t3 - t2
            self.perf_stats['load_total'] = t3 - t0

        print("All models loaded!")

    def print_perf_stats(self):
        """打印性能统计"""
        if not self.perf_stats:
            return

        print("\n" + "=" * 50)
        print("Performance Statistics")
        print("=" * 50)

        # 模型加载
        if 'load_total' in self.perf_stats:
            print("\n[Model Loading]")
            print(f"  TextDetector:   {self.perf_stats.get('load_detector', 0)*1000:7.1f} ms")
            print(f"  LayoutSorter:   {self.perf_stats.get('load_sorter', 0)*1000:7.1f} ms")
            print(f"  LineRecognizer: {self.perf_stats.get('load_recognizer', 0)*1000:7.1f} ms")
            print(f"  Total:          {self.perf_stats.get('load_total', 0)*1000:7.1f} ms")

        # OCR处理
        if 'ocr_total' in self.perf_stats:
            print("\n[OCR Processing]")
            print(f"  Detection:      {self.perf_stats.get('detection', 0)*1000:7.1f} ms")
            print(f"  Merge boxes:    {self.perf_stats.get('merge', 0)*1000:7.1f} ms")
            print(f"  Sorting:        {self.perf_stats.get('sorting', 0)*1000:7.1f} ms")
            print(f"  Recognition:    {self.perf_stats.get('recognition_total', 0)*1000:7.1f} ms " +
                  f"({self.perf_stats.get('recognition_count', 0)} lines, " +
                  f"avg {self.perf_stats.get('recognition_avg', 0)*1000:.1f} ms/line)")
            print(f"  Total OCR:      {self.perf_stats.get('ocr_total', 0)*1000:7.1f} ms")

        # 总计
        total = self.perf_stats.get('load_total', 0) + self.perf_stats.get('ocr_total', 0)
        print(f"\n[Total]           {total*1000:7.1f} ms ({total:.2f} s)")

    def print_all_model_info(self):
        """打印所有模型的详细信息"""
        print("\n" + "#" * 70)
        print("#" + " " * 20 + "Chrome Screen AI 模型详情" + " " * 21 + "#")
        print("#" * 70)

        self.detector.print_model_info()
        self.sorter.print_model_info()
        self.recognizer.print_model_info()

        # 列出不能使用的模型
        print("\n" + "=" * 60)
        print("其他模型 (不能直接使用)")
        print("=" * 60)
        print("\n[1] tflite_langid.tflite - 语言识别")
        print("  状态: 无法使用")
        print("  原因: 需要自定义算子 NGramHash")
        print("\n[2] screen2x_model.tflite - 内容提取")
        print("  状态: 非OCR模型")
        print("  用途: 网页无障碍功能，提取主要内容")
        print("  输入: 10个图神经网络张量")
        print("  输出: 3类 (NON_ESSENTIAL, HEADLINE, MAIN_CONTENT)")

    def ocr(self, image_path, use_detection=True):
        """
        完整OCR流程
        1. 文本检测 - 找到文字区域 (在4096x4096放大图上)
        2. 布局排序 - 确定阅读顺序
        3. 行识别 - 从4096图中裁剪并识别
        """
        ocr_start = time.perf_counter()

        image = Image.open(image_path)
        print(f"Image: {image.size}")

        if use_detection:
            # 使用检测模型 (会创建4096x4096放大图)
            print("\n[1/3] Text Detection (on 4096x4096)...")
            t0 = time.perf_counter()
            boxes = self.detector.detect(image, threshold=0.5)
            t1 = time.perf_counter()
            print(f"  Found {len(boxes)} char-level regions")
            print(f"  Scale factor: {self.detector.scale:.2f}x")

            if self.perf:
                self.perf_stats['detection'] = t1 - t0

            if len(boxes) == 0:
                print("  No text detected, using fallback")
                return self._ocr_fallback(image)

            # 合并小框为行 (在4096坐标系中，阈值20更保守)
            t0 = time.perf_counter()
            lines = self._merge_boxes_to_lines(boxes, y_threshold=20)
            t1 = time.perf_counter()
            print(f"  Merged to {len(lines)} lines")

            if self.perf:
                self.perf_stats['merge'] = t1 - t0

            print("\n[2/3] Layout Sorting...")
            t0 = time.perf_counter()
            sorted_lines = self.sorter.sort(lines, (4096, 4096))
            t1 = time.perf_counter()

            if self.perf:
                self.perf_stats['sorting'] = t1 - t0

            print("\n[3/3] Line Recognition (from original image)...")
            results = []
            # 从原图裁剪，不是从4096图
            orig_gray = Image.open(image_path).convert('L')
            scale = self.detector.scale
            offset_x = self.detector.offset_x
            offset_y = self.detector.offset_y

            min_conf = 0.3  # 置信度阈值
            rec_times = []

            # Create output directory for line images if save_lines is enabled
            lines_dir = None
            if self.save_lines:
                lines_dir = Path(image_path).stem + '_lines'
                os.makedirs(lines_dir, exist_ok=True)
                print(f"  Saving line images to: {lines_dir}/")

            line_num = 0
            for i, box in enumerate(sorted_lines):
                b = box['bbox']
                # 4096坐标转回原图坐标
                x1 = int((b[0] - offset_x) / scale)
                y1 = int((b[1] - offset_y) / scale)
                x2 = int((b[2] - offset_x) / scale)
                y2 = int((b[3] - offset_y) / scale)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(orig_gray.width, x2), min(orig_gray.height, y2)

                if x2 - x1 < 10 or y2 - y1 < 5:
                    continue

                # 从原图裁剪
                region = orig_gray.crop((x1, y1, x2, y2))
                t0 = time.perf_counter()
                text, conf = self.recognizer.recognize(region, return_conf=True)
                t1 = time.perf_counter()
                rec_times.append(t1 - t0)

                if text.strip() and conf >= min_conf:
                    line_num += 1
                    # Save line image if save_lines is enabled
                    if self.save_lines and lines_dir:
                        line_path = Path(lines_dir) / f'line_{line_num:03d}.png'
                        region.save(str(line_path))
                    print(f"  L{line_num} (y={y1:4d}) conf={conf:.2f}: {text}")
                    results.append(text)

            ocr_end = time.perf_counter()

            if self.perf:
                self.perf_stats['recognition_total'] = sum(rec_times)
                self.perf_stats['recognition_count'] = len(rec_times)
                self.perf_stats['recognition_avg'] = sum(rec_times) / len(rec_times) if rec_times else 0
                self.perf_stats['ocr_total'] = ocr_end - ocr_start

            return results
        else:
            return self._ocr_fallback(image)

    def _merge_boxes_to_lines(self, boxes, y_threshold=50):
        """合并检测框为词组，同时考虑x和y距离"""
        if not boxes:
            return []

        # 计算每个框的中心和尺寸
        for box in boxes:
            b = box['bbox']
            box['cx'] = (b[0] + b[2]) / 2
            box['cy'] = (b[1] + b[3]) / 2
            box['w'] = b[2] - b[0]
            box['h'] = b[3] - b[1]

        # 按y排序，再按x排序
        sorted_boxes = sorted(boxes, key=lambda x: (x['cy'], x['cx']))

        # 合并相邻的框（x和y都接近）
        merged = []
        used = set()

        for i, box in enumerate(sorted_boxes):
            if i in used:
                continue

            # 找所有与当前框相邻的框
            group = [box]
            used.add(i)

            # 迭代扩展组
            changed = True
            while changed:
                changed = False
                for j, other in enumerate(sorted_boxes):
                    if j in used:
                        continue
                    # 检查是否与组内任何框相邻
                    for g in group:
                        # x方向：间距小于框宽度的2倍
                        # y方向：间距小于框高度的0.5倍
                        x_dist = abs(other['cx'] - g['cx'])
                        y_dist = abs(other['cy'] - g['cy'])
                        x_thresh = (other['w'] + g['w'])  # 允许相邻
                        y_thresh = (other['h'] + g['h']) * 0.3  # y方向更严格

                        if x_dist < x_thresh and y_dist < y_thresh:
                            group.append(other)
                            used.add(j)
                            changed = True
                            break

            # 合并组
            if group:
                merged.append(self._merge_row(group))

        return merged

    def _merge_row(self, row_boxes):
        """合并一行的所有框"""
        x1 = min(b['bbox'][0] for b in row_boxes)
        y1 = min(b['bbox'][1] for b in row_boxes)
        x2 = max(b['bbox'][2] for b in row_boxes)
        y2 = max(b['bbox'][3] for b in row_boxes)
        conf = max(b['conf'] for b in row_boxes)
        return {'bbox': [x1, y1, x2, y2], 'conf': conf}

    def _ocr_fallback(self, image):
        """备选方案：投影法检测"""
        print("Using projection-based detection...")
        gray = image.convert('L')
        gray_array = np.array(gray)

        # 检测列
        columns = detect_columns(gray_array)
        print(f"Columns: {len(columns)}")

        results = []
        for col_idx, (x1, x2) in enumerate(columns):
            col_array = gray_array[:, x1:x2]
            col_img = gray.crop((x1, 0, x2, image.height))

            lines = detect_lines(col_array)
            for y1, y2 in lines:
                line_img = col_img.crop((0, y1, x2-x1, y2))
                if line_img.height < 10:
                    continue
                text = self.recognizer.recognize(line_img)
                if text.strip():
                    print(f"  C{col_idx+1}: {text}")
                    results.append(text)

        return results


def detect_lines(gray_array, min_gap=5, min_height=10):
    """自动检测文本行"""
    h, w = gray_array.shape

    # 二值化
    threshold = np.mean(gray_array) - 10
    binary = (gray_array < threshold).astype(np.uint8)

    # 水平投影
    h_proj = np.sum(binary, axis=1)

    # 平滑
    kernel = np.ones(3) / 3
    h_proj = np.convolve(h_proj, kernel, mode='same')

    # 找文本行
    lines = []
    in_line = False
    line_start = 0
    threshold_val = np.max(h_proj) * 0.02

    for i, val in enumerate(h_proj):
        if val > threshold_val and not in_line:
            in_line = True
            line_start = i
        elif val <= threshold_val and in_line:
            in_line = False
            if i - line_start >= min_height:
                y1 = max(0, line_start - 2)
                y2 = min(h, i + 2)
                lines.append((y1, y2))

    if in_line and h - line_start >= min_height:
        lines.append((line_start, h))

    # 合并太近的行
    merged = []
    for y1, y2 in lines:
        if merged and y1 - merged[-1][1] < min_gap:
            merged[-1] = (merged[-1][0], y2)
        else:
            merged.append((y1, y2))

    return merged


def detect_columns(gray_array):
    """自动检测列"""
    h, w = gray_array.shape

    # 二值化
    threshold = np.mean(gray_array) - 10
    binary = (gray_array < threshold).astype(np.uint8)

    # 垂直投影
    v_proj = np.sum(binary, axis=0)

    # 平滑
    kernel = np.ones(w // 30) / (w // 30)
    v_proj_smooth = np.convolve(v_proj, kernel, mode='same')

    # 找列分隔（投影值低的区域）
    threshold_val = np.max(v_proj_smooth) * 0.05

    # 在中间区域找最低点作为分隔
    mid_start = int(w * 0.3)
    mid_end = int(w * 0.7)
    mid_region = v_proj_smooth[mid_start:mid_end]

    if len(mid_region) > 0 and np.min(mid_region) < threshold_val:
        split_pos = mid_start + np.argmin(mid_region)
        return [(0, split_pos), (split_pos, w)]

    return [(0, w)]


def main():
    if len(sys.argv) < 2 or '--help' in sys.argv or '-h' in sys.argv:
        print("Chrome Screen AI OCR - 完整版")
        print("=" * 50)
        print("使用所有Chrome OCR模型:")
        print("  1. TextDetector (RPN) - 文本检测")
        print("  2. LayoutSorter - 布局排序")
        print("  3. LineRecognizer (Hanijpan) - 行识别")
        print()
        print("Usage:")
        print("  python chrome_ocr_final.py <image>       # OCR识别图片")
        print("  python chrome_ocr_final.py --info       # 打印模型详细信息")
        print("  python chrome_ocr_final.py <image> -v   # 详细模式OCR")
        print("  python chrome_ocr_final.py <image> --perf  # 显示性能统计")
        print()
        print("Options:")
        print("  --info, -i     只打印模型详细信息，不进行OCR")
        print("  --fallback     使用投影法代替检测模型")
        print("  -v, --verbose  显示详细处理信息")
        print("  --perf         显示关键步骤耗时统计")
        print("  --save-lines   保存每行裁剪图片到 <image>_lines/ 目录")
        return

    # 检查是否只打印模型信息
    if '--info' in sys.argv or '-i' in sys.argv:
        print("=" * 50)
        print("Chrome Screen AI OCR - 模型信息")
        print("=" * 50)
        ocr = ChromeOCR(verbose=True)
        ocr.print_all_model_info()
        return

    image_path = sys.argv[1]
    use_detection = '--fallback' not in sys.argv
    verbose = '-v' in sys.argv or '--verbose' in sys.argv
    perf = '--perf' in sys.argv
    save_lines = '--save-lines' in sys.argv

    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found")
        return

    print("=" * 50)
    print("Chrome Screen AI OCR")
    print("=" * 50)

    ocr = ChromeOCR(verbose=verbose, perf=perf, save_lines=save_lines)

    # 如果是详细模式，先打印模型信息
    if verbose:
        ocr.print_all_model_info()
        print("\n" + "#" * 70)
        print("#" + " " * 25 + "开始 OCR 处理" + " " * 28 + "#")
        print("#" * 70)

    results = ocr.ocr(image_path, use_detection=use_detection)

    # 保存结果
    output_file = Path(image_path).stem + '_ocr.txt'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results))

    print(f"\n{'=' * 50}")
    print(f"Saved: {output_file}")
    print(f"Lines: {len(results)}")

    # 打印性能统计
    if perf:
        ocr.print_perf_stats()


if __name__ == '__main__':
    main()
