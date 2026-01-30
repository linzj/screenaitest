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

## 9. RemoveOverlaps 逆向分析

### RemoveOverlapsStep::Process (sub_18045D800)

**源码**: `research/ocr/api/internal/layout_analyzer/remove_overlaps_step.cc`

**双重检查**:
1. 计算 IoU、containment1、containment2
2. 检查 `max(IoU, containment1, containment2) > threshold`
3. 再运行 symbol-level 重叠检查 (sub_18045D260)
4. **两个检查都通过**才考虑删除

### 配置参数

| 参数 | 值 | 说明 |
|------|-----|------|
| line_overlap_iou_threshold | 0.6 | 行 IoU 阈值 |
| line_overlap_threshold | 0.6 | 行重叠 (containment) 阈值 |
| overlap_threshold | 0.6 | 重叠阈值 |
| max_breadth_ratio | 2 | 最大宽度比 (min ratio = 0.5) |
| symbol_removal_breadth_ratio | 0.5 | 符号级去除宽度比 |
| min_low_block_confidence | 0.5 | 低置信度块最小阈值 |

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
