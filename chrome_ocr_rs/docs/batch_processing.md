# Chrome Screen AI 批处理架构 (IDA 逆向分析)

## 核心结论

Chrome Screen AI **不会**将相邻行图片在像素层面合并为一张大图。性能优化策略是沿 **batch 维度堆叠**多张行图片，一次推理调用处理多张图。

---

## 1. 检测侧批处理 (TensorDetectorClient)

### 1.1 ConvertTensorVecAndRotate (0x180278250)

**源文件**: `ocr/photo/detection/tensorflow/tensor_detector_client.cc`

将每行图片 tensor (shape `[1, H, W, C]`) 通过 memcpy 逐个复制到输出 tensor 的对应偏移位置。

- 输出 tensor shape: **`[N, H, W, C]`**, N = 行数
- 每张图片独占一个 batch slot，**无空间合并**
- 支持多线程并行复制

**核心 memcpy 逻辑** (sub_18028A180):
```c
// dest: output_data + element_size * batch_offset
// src: source tensor data
// size: element_size (H * W * C per image)
memcpy(output_data + element_size * batch_index, tile_tensor_data, element_size);
```

可选 space-to-depth 变换 (当 scale > 0):
```c
scaled_height = height / scale;
scaled_width  = width / scale;
scaled_channels = channels * scale * scale;
```
这是逐图变换，不是拼接。

### 1.2 三种 Batch 分割策略 (0x180274920)

| 策略 | 条件 | 算法 |
|------|------|------|
| **Trivial** | `a7 == true` (RPC 调用) | 固定大小 `min(max_batch, total)` |
| **Greedy** | 有首选大小列表，非空且 >0 | 按首选大小列表逆序贪心分配 |
| **Uniform** | 默认 fallback | `ceil(total / target_batches)` 均匀分割 |

### 1.3 检测 TFLite 模型验证

**模型**: `gocr_group_rpn_text_detection_model_2024_q4.tflite`

| 输入 | Shape | Signature |
|------|-------|-----------|
| input_features | [1, 4096, 4096, 1] | [1, 4096, 4096, 1] (固定) |
| input_features_1 | [1, 1280, 1280, 1] | [1, 1280, 1280, 1] (固定) |
| input_features_2 | [1, 640, 640, 1] | [1, 640, 640, 1] (固定) |
| input_features_3 | [1, 160, 160, 1] | [1, 160, 160, 1] (固定) |

检测模型 batch 维度固定为 1。逆向中的 batch 分割是将大图切分为多个 tile 分批推理。

---

## 2. 识别侧批处理 (TensorLstmClient)

### 2.1 RunModelOnPixa (0x1802A65B0)

**源文件**: `ocr/photo/segmentation/tensor_lstm_client.cc`

将多行 Pix 图片组成 batch，每张图片被切分为固定宽度的垂直条带 ("frames")。

**Tensor 准备** (sub_1802AB450):
1. 读取第一张图片高度（所有图片必须同高）
2. 计算帧参数:
   ```
   left_padding_frames = left_padding / frame_width
   right_padding_frames = right_padding / frame_width
   frame_pixel_count = frame_width * height
   ```
3. 同一 batch 中较短图片零填充至最宽图片的帧数
4. 输出 tensor shape: **`[batch_size, num_frames, frame_width * height]`**

**数据类型处理**:
- uint8 模式: 直接复制像素值
- float32 模式: `pixel_float = (float)pixel_byte / scale_factor`

### 2.2 RunSessionWithTargets (0x1802A1530)

**两阶段推理**:
1. **CNN 阶段**: 一次批量推理处理所有 batch 行图片
2. **LSTM 阶段** (如存在): 沿时间步逐步迭代，每步同时处理所有 batch 元素

```
for timestep in 0..sequence_length:
    for batch_item in 0..batch_size:
        copy CNN_output[batch_item][timestep] → LSTM_input[batch_item]
    run LSTM one step
    copy state tensors for next step
    extract output per batch item
```

### 2.3 识别 TFLite 模型验证

**模型**: `gocr_mobile_und.tflite` / `hanijpan.tflite`

| 属性 | 值 |
|------|-----|
| 输入 shape | `[1, 32, 168, 1]` |
| shape_signature | **`[-1, 32, 168, 1]`** (batch 动态) |
| 高度 | 32 像素 (固定) |
| 宽度 | 168 像素 (固定) |
| 通道 | 1 (灰度) |
| 输出 shape | `[batch, 42, num_classes]` |
| 帧数验证 | 168 / 4 = 42 帧 ✅ |

**实测验证**:
```
batch=4 → 输入 [4, 32, 168, 1] → 输出 [4, 42, 1293]  ✅
batch=8 → 输入 [8, 32, 168, 1] → 输出 [8, 42, 1293]  ✅
```

---

## 3. PhotoOcrEngine 行分组 (OcrLineBatch)

### 3.1 OcrLineBatch (0x180167E30)

**源文件**: `ocr/photo/engine/photo_ocr_engine.cc`

按识别模型索引将检测行分组 (abseil hash map)：

1. **阶段 1 - 桶分配**: 每个检测的 `model_index` (offset +168) 作为 key，分配到对应桶
2. **阶段 2 - 列表合并**: 对同一模型的行，通过 `pixaJoin` 合并 Pixa 列表 (非像素级)
3. **阶段 3 - 批量推理**: 调用 `sub_1802AF6A0` 处理合并后的 Pixa
4. **阶段 4 - 结果分发**: 将批量结果拆分回各个检测

### 3.2 merge_line_boxes 设置

在 `sub_180168DA0` (`photo_ocr_engine.cc:1302`) 中:
```c
DCHECK(settings_.merge_line_boxes() == false);    // offset 300
DCHECK(settings_.group_line_boxes() == false);     // offset 168
DCHECK(absl::GetFlag(FLAGS_enable_assist_filtering) == false);
```
代码中存在合并行框的路径，但**默认禁用**。

---

## 4. TensorDetectorClient::Process 调度 (0x180275BB0)

**源文件**: `ocr/photo/detection/tensorflow/tensor_detector_client.cc`

根据配置选择不同的推理路径:

| 条件 | 调用的方法 | vtable 偏移 |
|------|-----------|------------|
| `*(a1+179) == 1` | RunHorizontalVerticalModelOnPixa | +64 |
| 默认 (无旋转支持) | **RunModelOnPixa** | +40 |
| 需要旋转，tile 数超阈值 | RunModelOnPixaWithRotate90 | +48 |
| 需要旋转，tile 数未超阈值 | RunModelOnPixaHorizontalSingleCallWithRotation90 | +56 |

---

## 5. 性能优化总结

| 层级 | 优化方式 |
|------|---------|
| 检测 CNN | N 张 tile → 1 次批量推理，shape `[N,H,W,C]` |
| 识别 CNN | batch_size 张行图 → 1 次批量 CNN 特征提取 |
| 识别 LSTM | 每个时间步同时处理 batch_size 个序列 |
| 多线程 | batch tensor 创建和推理支持线程池并行 |
| 模型分组 | 按识别模型索引分组，减少模型切换开销 |

---

## 6. 关键地址索引

| 函数 | 地址 | 说明 |
|------|------|------|
| TensorDetectorClient::Process | 0x180275BB0 | 检测推理调度 |
| ConvertTensorVecAndRotate | 0x180278250 | 检测 batch tensor 创建 |
| BatchSplitStrategy | 0x180274920 | 三种 batch 分割策略 |
| RunModelOnPixa (检测) | 0x1802762F0 | 检测批量推理 |
| TensorLstmClient::Process | 0x1802A6110 | 识别入口 |
| RunModelOnPixa (识别) | 0x1802A65B0 | 识别批量推理 |
| RunSessionWithTargets | 0x1802A1530 | LSTM 两阶段推理 |
| ConvertPixaToTensors | 0x1802AB450 | Pixa → Tensor 转换 |
| OcrLineBatch | 0x180167E30 | 行分组批处理 |
| OcrDetectionBoxes | 0x180168DA0 | 按模型分组识别 |
