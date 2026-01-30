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

**源文件**: `ocr/photo/detection/tensorflow/tensor_detector_client.cc`

```c
void BatchSplitStrategy(
    this,           // a1
    int total,      // a2: 总行数
    int** preferred_sizes, // a3: 首选 batch 大小列表 (vector<int>*)
    int target_batches,    // a4: 目标批次数 (通常 = num_threads)
    int max_batch,  // a5: 最大 batch 大小 (<=0 表示不限)
    char flag,      // a6: 未知标志
    char is_rpc,    // a7: 是否 RPC 调用
    output_vec      // a8: 输出 batch 大小列表
);
```

#### 策略 1: Trivial (RPC 调用)

条件: `is_rpc == true` (a7)

```c
// tensor_detector_client.cc:202
// "Trivial batch split appropriate for RPC calls."
batch_size = min(max_batch, total);  // 如果 max_batch <= 0, 则 = total
remaining = total;
while (remaining > 0) {
    emit(min(batch_size, remaining));
    remaining -= batch_size;
}
```

**效果**: 所有行尽量放入一个大 batch，或按 max_batch 上限切分。

#### 策略 2: Greedy (有首选大小列表)

条件: `preferred_sizes` 非空，首元素 > 0，`flag == false`

```c
// tensor_detector_client.cc:214
// "Greedily batch split."
min_batch = max(2, total / target_batches);
remaining = total;
while (remaining > 0) {
    // 从最大首选尺寸向最小遍历 (逆序扫描)
    for (i = len(preferred_sizes)-1; i >= 0; i--) {
        size = preferred_sizes[i];
        if (size <= remaining) {
            // 如果是最后一个首选尺寸 (最小的)，或 size <= min_batch:
            if (i == len-1 || size <= min_batch) {
                emit(size);
                remaining -= size;
                break;
            }
        } else if (i == len-1) {
            // 最大尺寸仍超过 remaining，强制使用
            emit(size);
            remaining -= size;
            break;
        }
    }
}
```

**效果**: 优先使用大的首选尺寸，确保每个 batch 不小于 `total/target_batches`。

#### 策略 3: Uniform (默认 fallback)

条件: 以上两个策略都不适用

```c
// tensor_detector_client.cc:217
// "Uniform batch split."
batch_size = ceil(total / target_batches);
if (max_batch > 0 && batch_size > max_batch)
    batch_size = max_batch;
remaining = total;
while (remaining > 0) {
    emit(min(batch_size, remaining));
    remaining -= batch_size;
}
```

**效果**: 均匀切分，每个 batch 大小相近。

#### 示例

假设 100 行，num_threads=4，max_batch=0 (不限):

| 策略 | 结果 |
|------|------|
| Trivial | [100] (全部一个 batch) |
| Uniform | [25, 25, 25, 25] |
| Greedy (preferred=[8,16,32]) | 取决于 min_batch=max(2,25)=25, 会用 32: [32, 32, 32, 4] |

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

### 2.1 Batch 大小计算 (RunModelOnPixa 0x1802A65B0)

**源文件**: `ocr/photo/segmentation/tensor_lstm_client.cc`

RunModelOnPixa 是识别侧批处理的核心函数 (~3300 行伪代码)。它根据配置参数决定如何将 N 行图片分成若干 batch 进行推理。

#### TensorLstmClient 配置结构

| 偏移 | 类型 | 字段 | 说明 |
|------|------|------|------|
| 192 | uint32 | left_padding | 左 padding (像素) |
| 196 | int32 | right_padding / frame_width | 右 padding / 帧宽 |
| 200 | uint32 | max_line_height | 最大行高 (>0 启用高度拆分路径) |
| 205 | byte | use_padding | 是否启用 padding |
| 206 | byte | use_batch_multiplier | 是否启用 batch 乘数 |
| 208 | int32 | num_threads | 线程数 (< 2 时强制为 1) |
| 216 | int32 | max_batch_height | batch 上限 (>0 时限制每批大小) |
| 220 | int32 | data_type_flag | 0=float32, 1=uint8 |
| 232 | int32 | max_crop_width | 最大裁剪宽度 |
| 240 | int32 | default_batch_height | 默认 batch 大小 (0=自动计算) |
| 244 | int32 | batch_multiplier | batch 乘数 |
| 248 | float | scale_factor | 缩放因子 |
| 344 | ptr | thread_pool | 线程池指针 |

#### 路径 A: 简单路径 (max_line_height <= 0)

所有行直接分批，不做高度拆分:

```c
// Step 1: 确定 batch_height (每批的行数)
if (default_batch_height != 0) {       // offset 240
    batch_height = default_batch_height;
} else {
    batch_height = ceil(total_lines / num_threads);  // offset 208
}

// Step 2: 对齐到首选大小 (如果提供了 preferred_sizes 列表)
if (preferred_sizes != NULL && preferred_sizes.size() > 0) {
    // 升序扫描，找到第一个 >= batch_height 的首选大小
    for (i = 0; i < preferred_sizes.size(); i++) {
        if (preferred_sizes[i] >= batch_height) {
            batch_height = preferred_sizes[i];
            goto done;
        }
    }
    // 没找到：使用最大的首选大小
    batch_height = preferred_sizes[last];
}

// Step 3: 应用 batch_multiplier
if (use_batch_multiplier)   // offset 206
    effective_multiplier = batch_multiplier;  // offset 244
else
    effective_multiplier = 1;

// Step 4: 调用 ConvertPixaToTensors 创建 batch tensor
// 内部将 total_lines 按 batch_height 分组
num_batches = ConvertPixaToTensors(pixa, batch_height, multiplier, ...);
```

#### 路径 B: 高度约束路径 (max_line_height > 0)

当行图片高度不一致时，先拆分为固定高度的子块:

```c
// Step 1: 先获取每行尺寸 (batch_height=1)
ConvertPixaToTensors(pixa, batch_height=1, ...);

// Step 2: 计算总子块数
total_chunks = 0;
for each line:
    total_chunks += ceil(line_height / max_line_height);

// Step 3: 计算最优批次数
if (total_chunks <= max_batches_config * (num_threads - 1)) {
    num_batches = min(num_threads, ceil(total_chunks / max_batches_config));
} else {
    num_batches = num_threads;
}
batch_height = ceil(total_chunks / num_batches);

// Step 4: 对齐到首选大小 (同路径 A)

// Step 5: 应用 max_batch_height 上限
if (max_batch_height > 0 && batch_height > max_batch_height) {
    adjusted_batches = ceil(total_chunks / max_batch_height);
    // 如果 num_threads 是偶数，可能会调整为偶数批次
    batch_height = ceil(total_chunks / adjusted_batches);
}
```

#### 示例: 24 行图片，num_threads=4，default_batch_height=0

| 参数 | 值 |
|------|-----|
| total_lines | 24 |
| num_threads | 4 |
| batch_height | ceil(24/4) = 6 |
| num_batches | 4 |
| 每批 | [6, 6, 6, 6] |

如果 default_batch_height=8:

| 参数 | 值 |
|------|-----|
| batch_height | 8 (直接使用) |
| num_batches | ceil(24/8) = 3 |
| 每批 | [8, 8, 8] |

### 2.2 线程调度 (RunModelOnPixa)

计算出 num_batches 后，推理按以下方式调度:

```c
thread_pool = *(this + 344);
num_threads = *(this + 208);

if (thread_pool && num_threads > 0 && pool_size >= 2) {
    // 多线程路径
    barrier = create_barrier(num_batches);

    // 前 (num_batches - ceil(num_batches/num_threads)) 个 batch → 线程池
    // 剩余 batch → 主线程
    main_thread_start = num_batches - ceil(num_batches / num_threads);

    for (k = 0; k < num_batches; k++) {
        if (k < main_thread_start) {
            thread_pool.enqueue(RunSessionWithTargets, batch[k]);
        } else {
            RunSessionWithTargets(batch[k]);  // 主线程直接执行
            barrier.signal();
        }
    }
    barrier.wait();  // 等待所有线程完成

} else {
    // 单线程路径: 顺序处理所有 batch
    for (k = 0; k < num_batches; k++) {
        RunSessionWithTargets(batch[k]);
    }
}
```

**线程分配示例** (num_batches=4, num_threads=4):
- main_thread_start = 4 - ceil(4/4) = 3
- batch 0,1,2 → 线程池 (并行)
- batch 3 → 主线程
- 总共 4 个线程同时工作

### 2.3 ConvertPixaToTensors (0x1802AB450)

将多行 Pix 图片组成 batch，每张图片被切分为固定宽度的垂直条带 ("frames")。

**Tensor 准备**:
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

### 2.4 RunSessionWithTargets (0x1802A1530)

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

### 2.5 结果收集

```c
// 检查每个 batch 的状态码
for (k = 0; k < num_batches; k++) {
    if (status[k] != 1) {
        log_error("batch failed");
        break;
    }
    // 将 batch 输出追加到最终结果向量
    append(final_output, batch_output[k]);
}
// 验证: final_output.size() == total_lines
```

### 2.6 识别 TFLite 模型验证

**模型**: `gocr_mobile_und.tflite` / `hanijpan.tflite`

| 属性 | 值 |
|------|-----|
| 输入 shape | `[1, 32, 168, 1]` |
| shape_signature | **`[-1, 32, 168, 1]`** (batch 动态) |
| 高度 | 32 像素 (固定) |
| 宽度 | 168 像素 (固定) |
| 通道 | 1 (灰度) |
| 输出 shape | `[batch, 42, num_classes]` |
| 帧数验证 | 168 / 4 = 42 帧 |

**实测验证**:
```
batch=4 → 输入 [4, 32, 168, 1] → 输出 [4, 42, 1293]  ✅
batch=8 → 输入 [8, 32, 168, 1] → 输出 [8, 42, 1293]  ✅
```

---

## 3. PhotoOcrEngine 行分组

### 3.1 OcrDetectionBoxes (0x180168DA0)

**源文件**: `ocr/photo/engine/photo_ocr_engine.cc:1302`

这是识别管线的入口，负责将检测到的行分配给识别器。

**默认模式** (`merge_line_boxes=false`, `group_line_boxes=false`):
```c
DCHECK(settings_.merge_line_boxes() == false);    // offset 300
DCHECK(settings_.group_line_boxes() == false);     // offset 168

// 线程池: num_recognizer_models >= 2 时创建
if (num_recognizer_models >= 2)
    thread_pool = new ThreadPool(num_recognizer_models);

// 对每个检测结果:
for (det_idx = 0; det_idx < num_detections; det_idx++) {
    num_pixs = pixaGetCount(detection_pixa[det_idx]);
    DCHECK(num_pixs <= 2);  // 每个检测最多 2 个 pix (line 1360)

    for (i = 0; i < num_pixs; i++) {
        // 填充 line_data 结构 (152 字节):
        line_data.pix = pix;                // +0
        line_data.uncropped_pix = uncrop;   // +8
        line_data.box = box;                // +16
        line_data.detection_data = det;     // +24
        line_data.bbox = copy(bbox);        // +32 (56 bytes)
        line_data.dpi = dpi;                // +88
        line_data.det_index = det_idx;      // +92
        line_data.recognizer = recognizer;  // +96

        // 默认模式: 逐行直接处理
        sub_180164400(settings, &line_data);
    }
}
```

**分组模式** (`group_line_boxes=true`):
```c
// 按模型分桶，使用 width/height 比例做负载均衡
num_buckets = max(1, num_recognizer_models);
float scores[num_buckets] = {0};

for each line_data:
    // 找 score 最小的桶 (贪心负载均衡)
    min_bucket = argmin(scores);
    bucket[min_bucket].push(line_data);
    scores[min_bucket] += (float)box.width / box.height;

// 并行处理各桶
for (i = 0; i < num_buckets; i++) {
    if (thread_pool && i < num_buckets - 1)
        thread_pool.enqueue(process_bucket, bucket[i]);
    else
        process_bucket(bucket[i]);  // 最后一个桶在主线程
}
```

### 3.2 OcrLineBatch (0x180167E30)

**源文件**: `ocr/photo/engine/photo_ocr_engine.cc`

按识别模型索引将检测行分组 (abseil hash map)：

1. **阶段 1 - 桶分配**: 每个检测的 `model_index` (offset +168) 作为 key，分配到对应桶
2. **阶段 2 - 列表合并**: 对同一模型的行，通过 `pixaJoin` 合并 Pixa 列表 (非像素级)
3. **阶段 3 - 批量推理**: 调用 `sub_1802AF6A0` 处理合并后的 Pixa
4. **阶段 4 - 结果分发**: 将批量结果拆分回各个检测

### 3.3 批量推理入口 (0x1802AF6A0)

```c
// 参数:
// a1: TensorLstmClient this
// a2: batch_count_ptr (指向批次数量)
// a3: pixa_arrays (offset+16 存 boxa 列表)
// a4: output_vector (vector of recognition results)

// Step 1: 裁剪 output_vector 到 *batch_count 个元素
trim_output(a4, *batch_count);

// Step 2: 创建空容器
Pixa* combined_pixa = pixaCreate(0);
Pixa* combined_pixa2 = pixaCreate(0);
Boxa* combined_boxa = boxaCreate(0);

// Step 3: 合并所有批次的 Pixa/Boxa
for (i = 0; i < *batch_count; i++) {
    pixa_src = batch_pixas[i];     // a2 + offset
    boxa_src = batch_boxas[i];     // a3 + offset

    // sub_18078D340: 裁剪/准备图片
    crop_result = CropAndPrepare(pixa_src, 0, boxa_src, &out_pixa, &out_pixa2, &out_boxa);

    // 追加到合并容器
    pixaJoin(combined_pixa, out_pixa, 1);
    pixaJoin(combined_pixa2, out_pixa2, 1);
    boxaJoin(combined_boxa, out_boxa, 1);
}

// Step 4: 调用虚函数 → TensorLstmClient::Process
//   vtable+16 → Process(combined_pixa, combined_pixa2, combined_boxa, output_vector)
return this->Process(combined_pixa, combined_pixa2, combined_boxa, output_vector);
```

**关键**: 所有行的 Pixa 被合并成一个大列表后，一次性传入 `TensorLstmClient::Process`，由 `RunModelOnPixa` 内部决定如何分批。

### 3.4 TensorLstmClient::Process (0x1802A6110)

**源文件**: `ocr/photo/segmentation/tensor_lstm_client.cc:630+`

```c
Status TensorLstmClient::Process(pixa, &status, &pixa_count, ...) {
    // 调用 vtable+40 → RunModelOnPixa
    RunModelOnPixa(this, &status, &pixa_count,
                   &batch_outputs, &score_outputs,
                   &batch_size, NULL /*preferred_sizes*/);

    if (status != OK) {
        LOG(ERROR) << "Error running tensorflow model";
        return status;
    }

    DCHECK(batch_size > 0);  // line 633

    // CopyTensorToScores: 从 batch 输出提取最终得分
    CopyTensorToScores(this, batch_outputs, score_outputs,
                       output_count, batch_size, flag, results);

    DCHECK(results.size() == pixa_count);  // line 646
}
```

**注意**: `preferred_sizes` 在识别路径中传入 NULL，所以识别侧始终走 Uniform 分割策略:
`batch_height = ceil(total_lines / num_threads)` 或使用 `default_batch_height`。

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

## 5. 完整调用链路

```
PhotoOcrEngine
  ├── OcrDetectionBoxes (0x180168DA0)
  │     ├── 默认: 逐行处理 → sub_180164400
  │     └── group_line_boxes: 按模型分桶 → 并行处理
  │
  └── OcrLineBatch (0x180167E30)
        ├── 按 model_index 分组 (abseil hash map)
        ├── pixaJoin 合并同组 Pixa 列表
        └── sub_1802AF6A0 (批量推理入口)
              ├── CropAndPrepare 每批 → 合并为大 Pixa
              └── TensorLstmClient::Process (0x1802A6110)
                    └── RunModelOnPixa (0x1802A65B0)
                          ├── 计算 batch_height:
                          │     ├── 使用 default_batch_height (偏移240)
                          │     ├── 或 ceil(total / num_threads) (偏移208)
                          │     ├── 对齐到 preferred_sizes (如有)
                          │     └── 限制在 max_batch_height (偏移216)
                          ├── ConvertPixaToTensors (0x1802AB450)
                          │     └── 将行图片打包为 [batch, frames, pixels] tensor
                          ├── 多线程/单线程 调度
                          │     ├── 前 N-ceil(N/T) 批 → 线程池
                          │     └── 剩余批 → 主线程
                          └── RunSessionWithTargets (0x1802A1530)
                                ├── CNN 阶段: 批量推理
                                └── LSTM 阶段: 逐时间步, 所有 batch 并行
```

---

## 6. 性能优化总结

| 层级 | 优化方式 |
|------|---------|
| 检测 CNN | N 张 tile → 1 次批量推理，shape `[N,H,W,C]` |
| 识别 CNN | batch_size 张行图 → 1 次批量 CNN 特征提取 |
| 识别 LSTM | 每个时间步同时处理 batch_size 个序列 |
| 多线程 | batch tensor 创建和推理支持线程池并行 |
| 模型分组 | 按识别模型索引分组，减少模型切换开销 |
| 负载均衡 | 分组模式下按 width/height 比贪心分配到各桶 |

---

## 7. 关键地址索引

| 函数 | 地址 | 说明 |
|------|------|------|
| TensorDetectorClient::Process | 0x180275BB0 | 检测推理调度 |
| ConvertTensorVecAndRotate | 0x180278250 | 检测 batch tensor 创建 |
| BatchSplitStrategy | 0x180274920 | 三种 batch 分割策略 |
| RunModelOnPixa (检测) | 0x1802762F0 | 检测批量推理 |
| TfliteLstmClient 构造函数 | 0x18029DD20 | TFLite LSTM 客户端初始化 |
| TensorLstmClient::Process | 0x1802A6110 | 识别入口 |
| RunModelOnPixa (识别) | 0x1802A65B0 | 识别批量推理 (核心) |
| RunSessionWithTargets | 0x1802A1530 | LSTM 两阶段推理 |
| ConvertPixaToTensors | 0x1802AB450 | Pixa → Tensor 转换 |
| CopyTensorToScores | 0x1802A5360 | 输出 → 得分提取 |
| OcrLineBatch | 0x180167E30 | 行分组批处理 |
| OcrDetectionBoxes | 0x180168DA0 | 按模型分组识别 |
| 批量推理入口 | 0x1802AF6A0 | Pixa 合并 → Process |
