# 识别模块逆向分析

## 1. 完整调用链路

```
GocrLineRecognizer::Recognize (sub_1805EAD50)
  ├── CreateLineLayouts (sub_1805E5980) - 准备行布局
  │     └── sub_1805E99E0 - 单行识别调用
  ├── sub_1805E6330 - 最终结果组装
  └── sub_1805E7650 - 可选后处理

TensorLstmClient::RunModelOnPixa (sub_1802A65B0)  ← 核心
  ├── 1. ConvertPixaToTensors (sub_1802AB450)
  │     - Pixa → TFLite 输入张量
  │     - float32 tensor, shape [batch, height, width, channels]
  ├── 2. TfliteLstmClientBase::RunSessionWithTargets (sub_1802A1530)
  │     - 准备批次目标
  │     - 调用 TFLite interpreter invoke
  │     - 获取输出张量 (dequantized float32)
  ├── 3. CopyTensorToScores (sub_1802A5360)
  │     - tensor_lstm_client.cc:475+
  │     - 从输出张量复制得分到 scores 向量
  └── 4. MobileLstmRecognizer::DecodeBestPath (sub_180290AC0)
        - mobile_lstm_recognizer.cc:597+
```

## 2. CTC Decoding 逆向分析

### 完整调用链路

```
GocrCTCDecoderRecognizer::DecodeLines (sub_1805837D0)
  └── CTCDecoder::Decode (sub_18058AFD0) - ctc_decoder.cc
        ├── 入口验证: sub_18058AF30 - batch_size must be 1
        ├── 初始化: sub_18058CB70 - 创建 beam state, 设置 blank_id
        ├── Per-timestep loop:
        │     ├── 构建 (label, neg_logit) pair 列表
        │     │   - is_sorted_ctc: 使用排序
        │     │   - 否则: XOR 0x80000000 取反 float (NegativeLogitsScore)
        │     ├── CTCBeamOps::Step (vtable+8)
        │     ├── CTCBeamOps::TopPaths (vtable+16)
        │     └── sub_18058CD70 - beam step 结果写入
        ├── Final: sub_18058CFB0 - 回溯 beam path, 构建字符序列
        └── Confidence: sub_180589420 → sub_18059C840
```

### 模型输出到概率的转换

- **模型输出**: shape [1, 42, 1293], dtype UInt8
- TFLite 内部自动 dequantize: `float_value = (uint8_value - zero_point) * scale`
- Logits 预处理: XOR 0x80000000 = 取反符号位 (NegativeLogitsScore)
- **不做 softmax**。CTC beam search 直接使用原始 logits (negated)

### Beam Search 算法

**每个 timestep**:
1. 对每个现有 beam 候选和每个非 blank 标签:
   - 扩展代价: `neg_logit_score + lm_cost * lm_weight`
   - 重复字符: `duplicate_cost` (config offset 48)
   - blank: `blank_cost` (config offset 48)
   - 新字符: `insertion_cost` (offset 32) + `prior_weight * prior[label]`
2. 合并候选，按总代价排序
3. Beam pruning (保留 top-k)

**Beam 状态结构** (40 bytes):
| 偏移 | 字段 | 类型 |
|------|------|------|
| 0 | label_id | int32 |
| 8 | language_model_state | ptr |
| 16 | neg_logit_sum | float |
| 20 | lm_cost_sum | float |
| 32 | total_score | float |
| 36 | beam_index | int32 |

### Confidence 评分

**CtcDecoderConfidenceScorer_AvgLogits** (sub_18059C840):
- `sum_exp = sum(exp(logit[i]))` for all classes
- `confidence = 1.0 / sum_exp` per timestep
- 等价于 softmax 分母的倒数
- **CTC decoder 不做 confidence 阈值过滤**

### 字符映射

- `blank_id = vocab_size - 1` (UND: 1292, hanijpan: 8178)
- 非 blank: 查找 label_map (map<int, string>)
- CTC 合并: 连续相同 label 且中间无 blank → 合并

## 3. TensorLstmClient 结构偏移

| 偏移 | 类型 | 字段 |
|------|------|------|
| 192 | unsigned int | left_padding |
| 196 | int | right_padding / frame_width |
| 200 | unsigned int | max_line_height |
| 205 | byte | use_padding flag |
| 206 | byte | use_batch_multiplier |
| 208 | int | num_threads |
| 220 | int | data_type_flag (0=float, 1=uint8) |
| 232 | int | max_crop_width |
| 240 | int | default_batch_height |
| 244 | int | batch_multiplier |
| 248 | float | scale_factor |

## 4. UND 模型 11 个输出 Tensor

| 输出索引 | 用途 | 使用者 |
|----------|------|--------|
| 0 | **Logits** (主 CTC logits) | CTCDecoder::Decode |
| 1 | Prior probabilities | beam search 先验 |
| 2 | Math heatmap | 公式区域检测 |
| 3 | Symbol bounding boxes | 字符位置 |
| 4 | Script classification | 脚本识别 |
| 5 | Content type | 内容类型 (文本/手写/图像) |
| 6 | Direction | 文本方向 (LTR/RTL) |
| 7 | Font style | 字体风格 |
| 8-10 | Auxiliary | 条码/方向等辅助输出 |

**关键**: logits 可嵌入 script/style/direction 分类，由 `script_id_logits_first_index` 等字段指示分界。

## 5. 多脚本识别架构

### 管线流程

```
1. rpn_text_detection_mutator        → 文本检测
2. gocr_group_rpn_detector_mutator   → 检测框分组
3. joint_detector_mutator            → 联合检测器
4. script_direction_identification   → 脚本/方向识别
5. script_supported_mutator          → 脚本支持检查
6. taser_line_recognition_mutator    → 行识别 (MultiPass)
7. taser_line_selection_mutator      → 行选择
```

### 脚本→模型映射 (21 个脚本)

| 脚本 | TFLite 模型 | 语言模型 |
|------|-------------|---------|
| Hani (中文) | hanijpan.tflite | hani_lm.fst |
| Jpan (日文) | hanijpan.tflite (同 Hani) | jpan_lm.fst |
| und (通用) | gocr_mobile_und.tflite | - |
| Arab/Cyrl/Grek/... | 各自专用模型 | 各自 FST |

### 模型选择机制

1. `MobileLangIdV2::IdentifyLanguage` (sub_180EC28A0) 检测语言
2. 使用 `tflite_langid.tflite` 模型
3. 返回语言代码 → 映射到脚本标识符
4. 选择**单一**识别器 (非 try-all-pick-best)

## 6. Junk 过滤 (FilterJunkMutator)

### HeuristicLineIsJunk (0x18022A490)

**源码**: `filter_junk_mutator.cc`

后处理过滤:
1. 最小字符串长度检查
2. 单字符行过滤
3. 重复字符检测 (如 "aaaaaa")
4. 逐字符 confidence vs 阈值
5. "垃圾"字符比例检查
6. 最大词长检查
7. 连续标点检测

### Chrome 置信度阈值

| 参数 | 值 | 来源 |
|------|-----|------|
| layout_min_confidence | 0.35 | protobuf |
| junk_min_confidence | 0.4 | protobuf |
| junk_latin_min_confidence | 0.5 | protobuf |
| junk_cjk_min_confidence | 0 | protobuf |
| min_line_length_to_process | 2 | protobuf |

## 7. 已确认的参数

| 参数 | 值 | 状态 |
|------|-----|------|
| LEFT_MARGIN (hanijpan) | 12 | ✅ |
| LEFT_MARGIN (und) | 8 | ✅ |
| BLANK_IDX (hanijpan) | 8178 | ✅ |
| BLANK_IDX (und) | 1292 | ✅ |
| frame_width | 4 (168/42) | ✅ |
| Canvas size | 168 × 32 | ✅ |

## 8. 关键地址索引

| 函数 | 地址 | 说明 |
|------|------|------|
| GocrLineRecognizer::Recognize | 0x1805EAD50 | 行识别入口 |
| TensorLstmClient::RunModelOnPixa | 0x1802A65B0 | 核心推理 |
| ConvertPixaToTensors | 0x1802AB450 | Pixa → Tensor |
| RunSessionWithTargets | 0x1802A1530 | LSTM 两阶段推理 |
| CopyTensorToScores | 0x1802A5360 | 输出复制 |
| CTCDecoder::Decode | 0x18058AFD0 | CTC 解码主函数 |
| GocrCTCDecoderRecognizer | 0x18058ADC0 | 识别器入口 |
| AvgLogitsConfidenceScorer | 0x18059C840 | 置信度评分 |
| FilterJunkMutator | 0x18022A490 | 垃圾过滤 |
| MobileLangIdV2 | 0x180EC28A0 | 语言检测 |
