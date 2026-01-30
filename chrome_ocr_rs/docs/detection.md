# 检测模块逆向分析

## 1. 检测模型 (7 通道输出)

| 通道 | 含义 | 处理方式 |
|------|------|----------|
| ch0 | confidence (logit) | `sigmoid(ch0)` 后阈值过滤 (>0.3) |
| ch1 | center x offset | `cx = (grid_x + 0.5 + ch1) * stride` |
| ch2 | center y offset | `cy = (grid_y + 0.5 + ch2) * stride` |
| ch3 | log(width) | `width = exp(ch3) * stride` |
| ch4 | log(height) | `height = exp(ch4) * stride` |
| ch5 | rotation cos | `angle = atan2(ch6, ch5)` |
| ch6 | rotation sin | 用于旋转校正 |

## 2. 过滤条件

- 检测置信度阈值: 0.3 (sigmoid 后)
- 旋转角度限制: **无** (Chrome 不过滤单框绝对角度, IDA 确认)
- NMS IoU 阈值: 0.3
- 最小框尺寸: width > 150, height > 40 (4096 空间)

## 3. Detection Confidence Threshold 分析

### 三层检测 score 过滤体系

| 过滤层 | 函数 | 阈值 | 来源 |
|--------|------|------|------|
| GroupDetectionBoxes 行 score | sub_18048CE30 @ 0x18048D747 | **0.0f** | settings+0x11C, 默认值 |
| 单 box 行 score | sub_18048CE30 | **0.4f** | settings+328, `single_box_confidence_threshold` |
| IoU 阈值 | sub_18048CE30 | **0.7f** | settings+332, `intersection_over_union_threshold` |

主 score 阈值默认为 0.0f — 几乎所有具有正 score 的行都会通过。

## 4. Detection Postprocess (0x180240130)

**源码**: `region_proposal_text_detector.cc`

- 调用 `CropFromRGBImage` (sub_1806B6D70) 提取行图像
- 包含 Average height 计算逻辑

## 5. GroupDetectionBoxes (sub_18048CE30)

**源文件**: `ocr/photo/detection/region_proposal_text_detector_util.cc` (0x3097 字节)

### 主要流程

```
1. 获取 tensor 形状: image_height = shape[0], image_width = shape[1]
2. 调用 GroupingBoxesHoughTransform (sub_18049E3B0)
3. 对每个合并行:
   a. 跳过 width<=0 或 height<=0 的空行
   b. 如果 scale != 1.0: 缩放边界框
   c. 如果 !config.skip_padding: 调用 PadAndScaleBoxes (sub_18048ACD0)
   d. 跳过 width<4 或 height<=3 的过小行
   e. 计算行统计 (平均score, score方差, 平均角度)
   f. 过滤: score/cluster_size 比值检查
   g. 过滤: 单box行的score阈值检查
   h. 如果 use_min_height: 过滤小于 config.min_height 的行
```

### PadAndScaleBoxes (sub_18048ACD0)

**核心算法**:
```c
float padding = clamp(height * scale_factor, 4.0f, 16.0f);
float top_padding = vertical_pad_fraction * padding;
float bottom_padding = padding - top_padding;
box->width += padding;  // 宽度增加 padding
// 中心坐标根据旋转角度调整以补偿 padding 偏移
```

**调用参数**:
| 参数 | 来源 | 说明 |
|------|------|------|
| scale_factor | config+0xF8 | 高度缩放因子 |
| min_box_h | config+0xFC | 最小框高 |
| max_box_h | config+0x100 | 最大框高 |
| vertical_pad_frac | config+0x1B4 | 垂直 padding 分配比例 |
| max_padding | 16.0 | 最大 padding |
| min_padding | 4.0 | 最小 padding |

## 6. CropMultiScaleDetections (0x180495AC0)

**源码**: `region_proposal_text_detector_util.cc`

- 验证输入图像为灰度
- 迭代检测结果 (每个检测 240 字节)
- 调用 sub_180494C30 处理每个检测框

## 7. 裁剪实现 (0x180493A60)

**两条裁剪路径**:

| 路径 | 条件 | 实现 |
|------|------|------|
| CropZero | `angle == 0.0` | 简单矩形裁剪 (sub_1806B7BD0) |
| 旋转裁剪 | `angle != 0.0` | 仿射变换 warp (sub_1806BAE90) |

**旋转裁剪流程**:
```
1. sub_1807E3EE0(bbox) - 获取旋转信息 (角点旋转, pivot = 第一个角点)
2. sub_180931A30() - 创建旋转矩阵
3. sub_180E92000() - 初始化变换矩阵
4. sub_1807E6D90(-x1, -y1, matrix) - 平移到 bbox 原点
5. sub_18090A080(height, width, 1) - 创建精确尺寸输出 tensor
6. sub_1806BAE90() - 执行仿射变换 warp
```

关键: 输出尺寸 = 原始 bbox 尺寸 (不扩展), Chrome 使用仿射变换直接从原图采样。

## 8. CropFromRGBImage (0x1806B6D70)

**源码**: `ocr/google_ocr/image/image_utils.cc`

包装函数，根据通道数选择路径:
- 单通道 → `sub_1809092C0` (简单复制)
- 多通道 → `sub_18090A080` + `sub_1806B72A0` (ConvertToGray)

### ConvertToGray (0x1806B72A0)

```c
gray = (307*R + 512*G + 205*B + 512) >> 10
// 约等于: gray = 0.299*R + 0.500*G + 0.200*B
```

## 9. RemoveOverlaps 完整逆向分析

Chrome 的 28 步 layout pipeline 有两个不同的重叠去除步骤:
- **Step 6: RemoveOverlapsWordPruningStep** -- 基于词级文本比较的早期剪枝
- **Step 19: RemoveOverlapsStep** -- 基于几何重叠的行级去除

---

### 9.1 RemoveOverlapsStep (Step 19) -- 行级几何重叠去除

**源码**: `research/ocr/api/internal/layout_analyzer/remove_overlaps_step.cc`
**vtable**: `0x1819431e8` (RemoveOverlapsStep)
**构造**: `sub_18041EA00`, **Execute**: `sub_18045D800` (0x181d 字节)

#### Protobuf Spec: RemoveOverlapsSpec

```protobuf
message RemoveOverlapsSpec {
  bool enabled = 1;                          // default: true
  double symbol_removal_breadth_ratio = 2;   // default: 0.5
  double block_different_direction_maximum = 3; // default: 0.3
  PageLayoutOverlappingRemoverSpec word_overlap = 4;
  PageLayoutOverlappingRemoverSpec symbol_overlap = 5;
  double min_low_block_confidence = 6;       // default: 0.5
  double max_low_block_symbol_breadth_ratio = 7; // default: 0.6
  PageLayoutOverlappingRemoverSpec line_overlap = 8;
}
```

**Struct field layout** (object at `this`):
| Offset | Type | Field |
|--------|------|-------|
| 96 | ptr | PageLayout* (context) |
| 104-... | OverlappingRemoverSpec | Internal spec storage |
| 120 | uint32 | flags (bit0=word, bit1=symbol, bit2=line) |
| 128 | ptr | word_overlap RemoverSpec |
| 136 | ptr | symbol_overlap RemoverSpec |
| 144 | ptr | line_overlap RemoverSpec |
| 152 | double | symbol_removal_breadth_ratio (default 0.5) |
| 160 | double | block_different_direction_maximum (default 0.3) |
| 168 | double | min_low_block_confidence (default 0.5) |
| 176 | double | max_low_block_symbol_breadth_ratio (default 0.6) |
| 192 | ptr | current PageLayout* |

#### Execute 主流程 (sub_18045D800)

```
1. 初始化三个 OverlapRemover: line_overlap (type=2), word_overlap (type=0), symbol_overlap (type=5)
2. 获取页面所有 line 的有序列表 (sub_18045F020)
3. 如果 flags & 4 (line): 对每条 line 构建 OverlapRegion, 存入 line_overlap remover
4. 如果 flags & 1 (word): 对每条 line 构建 OverlapRegion, 存入 word_overlap remover
5. 对所有 line 建立 hash map (line_id -> {line_ptr, symbol_breadth})
6. 排序 line 列表 (sub_18045F310, IntroSort)
7. 对于排好序的每对 (line_i, line_j) where j < i:
   a. 加载 line_j 的 bbox
   b. 查找 line_j 在 hash_map 中的条目
   c. 计算 line_i 与 line_j 的 box overlap metrics:
      - sub_180430990: 计算 IoU, containment_i, containment_j
   d. 三重重叠阈值检查 (见下文)
   e. 如果两条线属于相同 block/direction/orientation:
      - 对两条线的所有 word 临时加 +100 confidence boost (flag 0x20000)
      - 运行 word_overlap remover (sub_180468050)
      - 运行 symbol-level 重叠检查 (sub_18045D260)
      - 对 word 恢复 confidence (-100)
      - 如果 symbol breadth ratio < threshold, 运行 symbol_overlap remover
   f. 否则如果重叠超过阈值:
      - 标记被覆盖的 line 为待删除
8. 收集待删除 line 列表
9. 调用 sub_180703AA0 从 PageLayout 中删除这些 line
10. 重建各级结构 (sub_180466D70, sub_1806FC640, sub_1806FC7C0, sub_1806FC840)
```

#### 核心重叠判定逻辑

**Box Overlap Calculator (sub_180430990)**:
```c
// 计算两个 box 的重叠区域面积 (intersection_area)
// 支持旋转矩形: 如果任一 box 有 rotation (angle != 0),
//   使用旋转多边形交集 (sub_18071CD00 + sub_18071CD70)
// 否则使用轴对齐矩形交集 (sub_180435300)

if (intersection_area == 0.0) return false;

float area_a = width_a * height_a;
float area_b = width_b * height_b;

// IoU (Intersection over Union)
*iou = intersection_area / (area_a + area_b - intersection_area);

// Containment of A in B (what fraction of A is covered)
*containment_a = intersection_area / area_a;

// Containment of B in A (what fraction of B is covered)
*containment_b = intersection_area / area_b;
```

**三重阈值检查 (Execute 主体)**:
```c
// 条件1: 粗筛 -- max(IoU, containment_a, containment_b) > threshold
if (fmaxf(containment_b, fmaxf(containment_a, iou)) <= this->block_different_direction_maximum)
    goto skip;  // 重叠太小, 跳过

// 条件2: 细检 -- 计算 word-level 重叠分数
double word_overlap_score = sub_18045D260(page, line_a, line_b);
// sub_18045D260:
//   1. 获取 line_a 和 line_b 各自的所有 word bbox
//   2. 构建一个 N*M 重叠矩阵 (sub_180458410)
//   3. 对每对 (word_i in line_a, word_j in line_b):
//      计算 word_i_area * overlap_fraction + word_j_area * overlap_fraction
//   4. 分别除以 total_area_a 和 total_area_b
//   5. 返回 max(score_a, score_b) -- 两个方向中较大的
if (word_overlap_score <= this->block_different_direction_maximum)
    goto skip;  // word 级重叠也不够, 跳过

// 条件3: confidence 和 breadth 保护
// 检查是否应该保护高置信度短行:
if (this->min_low_block_confidence <= line_confidence
    && min(breadth_a, breadth_b) / max(breadth_a, breadth_b) <= this->max_low_block_symbol_breadth_ratio)
    goto skip;  // 高置信度且 breadth 差异大, 不删除
```

#### "谁被删除" 的判定

在 `RemoveOverlappingInternal` 回调 (sub_1804699B0) 中:

```c
// 来自 page_layout_overlapping_remover.cc line 190
// 比较两条重叠线的文本内容
string text_a = GetWordText(word_a);
string text_b = GetWordText(word_b);

if (text_a == text_b) {
    // 文本相同时: 比较 word 索引 (sub_1806FF220)
    int idx_a = GetWordIndex(layout, word_a);
    int idx_b = GetWordIndex(layout, word_b);
    if (idx_a > idx_b) {
        // 索引较大的 word 被替换为索引较小的
        // 实质: 保留较早出现的 line, 删除后来的
        ReplaceBoxes(line_from=a, line_to=b);
    }
} else {
    // 文本不同时: 被标记删除的 line 被记录到 removal list
    // (由 DetectOverlaps 的 callback 处理)
}
```

#### 同 block/direction 特殊处理

当两条重叠线属于同一 block 且有相同 direction 和 orientation 时:
```c
// 1. 临时给所有 word 加 +100 confidence boost
for (word in line_words) {
    if (word->flags & 0x20000) {
        word->confidence += 100.0f;
    }
}

// 2. 运行 word_overlap remover
// 3. 检查 symbol breadth ratio
double ratio = min(breadth_a, breadth_b) / max(breadth_a, breadth_b);
if (ratio <= this->symbol_removal_breadth_ratio) {
    // "Small symbol breadth detected; performing symbol-level overlap removal."
    // 运行 symbol_overlap remover
}

// 4. 恢复 confidence
for (word in line_words) {
    if (word->flags & 0x20000) {
        word->confidence -= 100.0f;
    }
}
```

---

### 9.2 RemoveOverlapsWordPruningStep (Step 6) -- 词级早期剪枝

**源码**: 嵌入在 layout_analyzer 流水线中
**vtable**: `0x181941f08` (RemoveOverlapsWordPruningStep)
**构造**: `sub_18041D250`, **Execute**: `sub_180460B70` (非常大, ~0x2000+ 字节)

#### Protobuf Spec: RemoveOverlapsWordPruningStep

```protobuf
message RemoveOverlapsWordPruningStep {
  bool enabled = 1;                                        // default: true
  float line_overlap_iou_threshold = 2;                    // default: 0.6
  float line_overlap_threshold = 3;                        // default: 0.6
  bool skip_curved_boxes = 4;                              // default: true
  float complete_overlap_word_overlap_threshold = 5;       // default: 0.1
  float complete_word_confidence_difference_threshold = 6; // default: 0.2
  double word_overlap_threshold = 7;                       // default: 0.75
  float word_confidence_difference_threshold = 8;          // default: 0.05
  float common_characters_percentage_threshold = 9;        // default: 0.6
  float exact_text_match_overlap_threshold = 10;           // default: 0.5
  bool retain_empty_paragraphs = 11;
  bool apply_accumulated_different_orientation_threshold = 12;
  float different_orientation_support_threshold_multiplier = 13; // default: 4
  int32 accumulated_number_of_overlaps = 14;               // default: 10
}
```

#### 关键区别: Step 6 vs Step 19

| 特征 | Step 6 (WordPruning) | Step 19 (RemoveOverlaps) |
|------|---------------------|--------------------------|
| 阶段 | 早期 (识别前) | 晚期 (行组装后) |
| 粒度 | 词级文本比较 | 行级几何重叠 |
| 文本比较 | 是 (common chars, exact match) | 仅同文本替换 |
| 重叠指标 | IoU, containment, word overlap | IoU, containment, word+symbol overlap |
| 置信度 | word 置信度差异 | block 置信度 + confidence boost |
| 方向处理 | 累积不同方向阈值 | block direction 检查 |
| IoU 阈值 | 0.6 (line level) | 由 OverlapSpec 指定 |
| 关键阈值 | word_overlap=0.75 | 由三级 Remover 各自配置 |

#### Step 6 核心逻辑

```
1. 对所有行对 (i, j):
   a. 计算行级 IoU
   b. 如果 IoU > line_overlap_iou_threshold (0.6):
      - 计算行级 containment
      - 如果 containment > line_overlap_threshold (0.6):
        - 获取两行的所有 word 文本
        - 计算 common_characters_percentage
        - 如果 common_chars > common_characters_percentage_threshold (0.6):
          → 比较 word 置信度差异
          → 如果 confidence_diff > threshold: 删除低置信度行
        - 否则如果 exact_text_match:
          → 如果 overlap > exact_text_match_overlap_threshold (0.5):
            删除低置信度行
   c. 否则: 检查 word 级重叠
      - 如果 word_overlap > word_overlap_threshold (0.75):
        - 比较 confidence difference
        - 如果 diff > word_confidence_difference_threshold (0.05):
          删除低置信度行
      - 如果 complete overlap (containment > 0.1):
        - 如果 confidence_diff > complete_word_confidence_difference_threshold (0.2):
          删除低置信度行
```

---

### 9.3 PageLayoutOverlappingRemoverSpec (共用 Remover)

```protobuf
message PageLayoutOverlappingRemoverSpec {
  bool skip_curved_boxes = 1;
  OverlapSpec overlap_spec = 2;
  bool replace_big_box = 3;
}

message OverlapSpec {
  double maximum_overlap = 1;                    // IoU threshold
  double maximum_duplicate = 2;                  // containment threshold (accumulated)
  double maximum_overwrite = 3;                  // overwrite threshold
  double minimum_breadth_ratio = 4;              // breadth ratio filter
  double maximum_different_direction_overlap = 5; // different direction threshold
}
```

OverlapSpec 默认值全为 0.0 (不过滤), 实际值由上层 RemoveOverlapsSpec 配置注入。

### 9.4 DetectOverlaps 核心函数 (sub_18046A1D0)

**源码**: `research/ocr/layout/detect_overlaps.cc`

```c
void DetectOverlaps(OverlapSpec* spec, Span<Region> regions_a, Span<Region> regions_b,
                    function<void(Region*, Region*)> callback) {
    // 对 regions_a 中的每个 region:
    for (auto& a : regions_a) {
        double total_duplicate = 0.0;  // 累积 containment

        for (auto& b : regions_b) {
            if (a.top() > b.top())  // 按 Y 坐标排序, 提前退出
                break;
            if (a == b) continue;   // 跳过自身

            // 检查 b 是否已在 "已处理" 集合中
            if (b in processed_set)
                continue;

            // 获取两个 region 的 direction (writing direction)
            int dir_a = a.GetDirection();
            int dir_b = b.GetDirection();

            // 计算几何重叠面积 (支持旋转矩形)
            double intersection = ComputeIntersection(a, b);
            if (intersection == 0.0)
                continue;

            double area_a = a.width() * a.height();
            double area_b = b.width() * b.height();

            // IoU
            double iou = intersection / (area_a + area_b - intersection);
            // containment (fraction of A covered by B)
            double containment = intersection / area_a;
            // overwrite (fraction of B covered by A)
            double overwrite = intersection / area_b;

            // 累积 duplicate 分数
            total_duplicate += containment;

            // 判断重叠类型 (4种, 按优先级):
            // 1. kDifferentDirectionOverlap: dir_a != dir_b && spec.flags & 0x10
            //    && iou > spec.maximum_different_direction_overlap
            if (dir_a != dir_b && (spec.flags & 0x10) && iou > spec.offset_56) {
                callback(a, b);  // type=2
            }
            // 2. kOverlap: spec.flags & 1 && iou > spec.maximum_overlap
            else if ((spec.flags & 1) && iou > spec.offset_24) {
                callback(a, b);  // type=1
            }
            // 3. kDuplicate: spec.flags & 2 && total_duplicate > spec.maximum_duplicate
            else if ((spec.flags & 2) && total_duplicate > spec.offset_32) {
                callback(a, b);  // type=3
            }
            // 4. kOverwrite: spec.flags & 4 && overwrite > spec.maximum_overwrite
            else if ((spec.flags & 4) && overwrite > spec.offset_40) {
                // 额外检查: 如果 spec.flags & 8 (minimum_breadth_ratio)
                //   计算两个 region 的 breadth 中心, 检查距离/最大高度比
                //   如果比值 < spec.minimum_breadth_ratio, 则不触发
                if (!(spec.flags & 8) || breadth_distance_ok) {
                    callback(a, b);  // type=4
                }
            }
        }
    }
}
```

#### 重叠类型枚举

| Type | Name | Condition | 含义 |
|------|------|-----------|------|
| 1 | kOverlap | IoU > maximum_overlap | 标准 IoU 重叠 |
| 2 | kDifferentDirectionOverlap | dir_a != dir_b && IoU > threshold | 不同书写方向重叠 |
| 3 | kDuplicate | accumulated_containment > threshold | 累积包含度超限 |
| 4 | kOverwrite | overwrite > threshold | 单次覆盖超限 |

### 9.5 RemoveOverlapsSpec 默认值汇总

| 字段 | Offset (struct) | Default | 说明 |
|------|--------|---------|------|
| symbol_removal_breadth_ratio | 152 | 0.5 | symbol 宽度比阈值 |
| block_different_direction_maximum | 160 | 0.3 | 不同方向最大重叠 |
| min_low_block_confidence | 168 | 0.5 | 低 block 置信度阈值 |
| max_low_block_symbol_breadth_ratio | 176 | 0.6 | 低 block symbol 宽度比 |
| enabled | 80 | true | 是否启用 |

### 9.6 关键地址索引 (RemoveOverlaps)

| 函数 | 地址 | 说明 |
|------|------|------|
| RemoveOverlapsStep 构造 | 0x18041EA00 | 创建步骤对象 (200 bytes) |
| RemoveOverlapsStep Execute | 0x18045D800 | 主执行函数 (0x181d bytes) |
| RemoveOverlapsStep Init | 0x18045D630 | 从 spec 初始化 |
| RemoveOverlapsWordPruningStep 构造 | 0x18041D250 | 创建步骤对象 (184 bytes) |
| RemoveOverlapsWordPruningStep Execute | 0x180460B70 | 主执行函数 (very large) |
| Box Overlap Calculator | 0x180430990 | 计算 IoU/containment |
| Word-level Overlap Score | 0x18045D260 | 计算 word 级重叠分数 |
| DetectOverlaps | 0x18046A1D0 | 核心重叠检测算法 |
| RemoveOverlappingInternal callback | 0x1804699B0 | 处理重叠 pair |
| OverlappingRemover::Process | 0x180467890 | 执行检测+删除 |
| BuildOverlapRegions | 0x180467BC0 | 从 line 构建 region |
| DetectOverlaps (template) | 0x1804679F0 | 模板化重叠检测入口 |
| Overlap Area (rotated) | 0x18046AD40 | 计算旋转矩形交集面积 |
| RemoveOverlapsSpec constructor | 0x180726570 | Spec 默认值初始化 |
| OverlapSpec constructor | 0x180738790 | OverlapSpec 拷贝构造 |
| RemoveOverlapsSpec vtable | 0x1819D8978 | PageLayoutAnalyzerSpec |

## 10. BoundingBox 结构体 (56 字节)

```c
struct BoundingBox {
    void* vtable;     // +0
    uint64_t ref;     // +8
    uint32_t flags;   // +16
    void* metadata;   // +24
    int32_t x;        // +32
    int32_t y;        // +36
    int32_t width;    // +40
    int32_t height;   // +44
    float   angle;    // +48 (度)
    int32_t score;    // +52
};
```

## 11. 关键地址索引

| 函数 | 地址 | 说明 |
|------|------|------|
| Detection Postprocess | 0x180240130 | 检测后处理主入口 |
| GroupDetectionBoxes | 0x18048CE30 | 行分组主管线 |
| PadAndScaleBoxes | 0x18048ACD0 | Padding 计算 |
| CropMultiScaleDetections | 0x180495AC0 | 多尺度裁剪 |
| 单框裁剪 | 0x180494C30 | 单检测框处理 |
| 裁剪实现 | 0x180493A60 | CropZero / 旋转裁剪 |
| CropFromRGBImage | 0x1806B6D70 | RGB→灰度包装 |
| ConvertToGray | 0x1806B72A0 | 灰度转换 |
| CropZero | 0x1806B7BD0 | 无旋转裁剪 |
| Warp | 0x1806BAE90 | 仿射变换裁剪 |
| RemoveOverlapsStep | 0x18045D800 | 重叠去除 |
| Box Overlap Calculator | 0x180430990 | IoU/containment 计算 |
