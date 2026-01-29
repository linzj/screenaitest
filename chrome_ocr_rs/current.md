# Chrome OCR Rust 实现 - 当前状态

## ⚠️ 重要提醒

**不要参考 Python 实现** - 只参考 IDA 逆向分析的 DLL 原生实现。
必须使用ida-pro-mcp 逆向，参考真正的解决方案

---

## 当前对比结果 (PPT.png) - 2026-01-29 最新

| 指标 | DLL (Native) | Rust (TFLite) |
|------|--------------|---------------|
| 行数 | 24 行 | **24 行** |
| 正确率 | 100% | **100% (24/24 完全正确)** |
| 匹配率 | - | **24/24 = 100%** |
| 时间 | 447 ms | ~1682 ms |

**重大进展 (2026-01-29)**:
- ✅ 实现了基于 Chrome Hough Transform 的行分组算法 (从 IDA 逆向 sub_18049E3B0)
- ✅ 消除了 Union-Find 跨行合并问题 (M1 cross-line merge)
- ✅ 所有 24 行的识别结果完全正确
- ✅ 修复垂直文本检测: 移除了错误的 30° 绝对角度过滤 (从 IDA 确认 sub_180476920 中 30° 是行间角度差阈值，不是单框绝对角度)
- ✅ 修复垂直文本识别: 近垂直文本 (>45°) 使用旋转后的 bbox 尺寸进行裁剪

### Native 完整输出 (24 lines):
```
1: 趣味话题
2: 激发学习兴趣
3: 归纳总结能力
4: 章节总结
5: 热点话题
6: 发现问题能力
7: 分析问题能力
8: 解决问题能力
9: 深度学习能力
10: 辩论赛
11: 质疑思辨能力
12: 动脑动手动口能力
13: 终生学习能力
14: 提升高阶能力
15: 小组合作学习+成绩小组互评
16: 查阅文献能力
17: 追踪前沿文献
18: 科技写作能力
19: 课程论文
20: 有效拓宽视野
21: 培养学科自信
22: 升学大幅提高
23: 专家讲座
24: 培养高阶思维
```

### TFLite 当前输出 (24 lines):
```
1: 热点话题 ✅
2: 辩论赛 ✅
3: 质疑思辨能力 ✅
4: 动脑动手动口能力 ✅
5: 终生学习能力 ✅
6: 提升高阶能力 ✅ (NEW)
7: 趣味话题 ✅
8: 激发学习兴趣 ✅
9: 发现问题能力 ✅
10: 分析问题能力 ✅
11: 解决问题能力 ✅
12: 深度学习能力 ✅
13: 小组合作学习+成绩小组互评 ✅
14: 章节总结 ✅
15: 归纳总结能力 ✅
16: 课程论文 ✅
17: 查阅文献能力 ✅
18: 追踪前沿文献 ✅
19: 科技写作能力 ✅
20: 专家讲座 ✅
21: 有效拓宽视野 ✅
22: 培养学科自信 ✅
23: 升学大幅提高 ✅
24: 培养高阶思维 ✅ (NEW)
```

### 修复历程

**2025-01-28 修复:**
| Native | 之前 TFLite | 现在 TFLite |
|--------|-------------|-------------|
| 辩论赛 | 论赛 | ✅ 辩论赛 |
| 动脑动手动口能力 | 边脑动手动口能大 | ✅ 动脑动手动口能力 |
| 归纳总结能力 | 日纳总结能大 | ✅ 归纳总结能力 |
| 质疑思辨能力 | 贡疑思辨能力 | ✅ 质疑思辨能力 |
| 热点话题 | で热点话题 | ✅ 热点话题 |

**2025-01-29 修复 (Hough Transform 重构):**
| Native | 之前 TFLite | 现在 TFLite |
|--------|-------------|-------------|
| 课程论文 | 课程论 | ✅ 课程论文 |
| 升学大幅提高 | 升学大幅摄 | ✅ 升学大幅提高 |
| 专家讲座 | 家讲座 | ✅ 专家讲座 |
| 小组合作学习+成绩小组互评 | 小组合作学+成绩小年百 | ✅ 小组合作学习+成绩小组互评 |

**2026-01-29 修复 (垂直文本检测与识别):**
| Native | 之前 TFLite | 现在 TFLite |
|--------|-------------|-------------|
| 提升高阶能力 | (缺失) | ✅ 提升高阶能力 |
| 培养高阶思维 | (缺失) | ✅ 培养高阶思维 |

**根因**: detector.rs 中有 `if angle.abs() > FRAC_PI_6 { continue; }` 过滤掉了 >30° 的检测框。
IDA 逆向 sub_180476920 确认: 30° 阈值是**行间角度差**阈值，不是单框绝对角度过滤。
Chrome 不会过滤单个检测框的绝对旋转角度，垂直文本 (~90°) 必须通过。

**修复内容**:
1. `detector.rs`: 移除绝对角度过滤
2. `ocr.rs`: 近垂直文本 (>45°) 使用旋转后的 bbox 尺寸 (`rotated_w = w*|cos| + h*|sin|`)

---

## ✅ 已解决的问题

### 1. 长行识别的 Tensor 合并
- 实现 Chrome 的 chunk_lengths 预计算方法
- "办公室" 等边界字符正确保留

### 2. 旋转文本检测与校正
- 利用 ch5/ch6 (cos/sin) 获取旋转角度
- 使用 `imageproc::rotate_about_center` 进行旋转校正
- Chrome 不过滤单框绝对角度 (从 IDA 确认: sub_180476920 中 30° 是行间角度差)
- 近垂直文本 (>45°) 使用旋转后 bbox 尺寸进行裁剪
- 后旋转裁剪（pixel < 180 阈值）

### 3. 文本重复去除
- 添加文本相似度检查
- 过滤相同或包含关系的文本

### 4. 置信度过滤
- 默认阈值改为 0.7，过滤低置信度噪声

### 5. 小框过滤
- 过滤 4096 空间中 width < 150 或 height < 40 的检测框

### 6. 旋转文本 padding 增强 (2025-01-28)
- 增加 x 方向 padding 因子 (0.6 → 0.9)
- 增加基础 padding (20 → 30)
- 修复 "课程论文"、"升学大幅提高" 等边缘字符截断问题

### 7. 最小文本长度调整 (2025-01-28)
- 从 3 字符降为 2 字符
- 允许更多短文本通过过滤（如 "论赛"）

---

## 待解决问题

### ✅ 全部已解决 (PPT.png 24/24 行匹配)

以下问题已在之前的修复中全部解决:
- ✅ 旋转校正 padding - 使用角度相关 margin
- ✅ 检测框合并 - Hough Transform 重构
- ✅ 部分文本未检测 - 移除绝对角度过滤 + 垂直文本裁剪修复
- ✅ 识别器问题 - 长行 chunk 分段 + LEFT_MARGIN=12

### 潜在改进方向 (非阻塞)
- 性能: TFLite ~1682ms vs Native ~447ms, 有优化空间
- 更多测试图片验证泛化能力

---

## 需要 IDA 逆向调查的问题

### 1. 旋转校正的具体实现
- Chrome 如何计算旋转后的裁剪区域？
- 是否使用 padding？如何计算？
- 是否有后处理步骤清理边界噪声？

### 2. 检测框合并算法
- Chrome 的 MergeLines 具体参数？
- 如何避免不同行的文本被合并？

### 3. 行图像预处理
- 在送入识别模型前有哪些预处理步骤？
- 是否有文本边界检测/调整？

---

## 关键算法参数

### 检测模型 (7 通道输出)

| 通道 | 含义 | 处理方式 |
|------|------|----------|
| ch0 | confidence (logit) | `sigmoid(ch0)` 后阈值过滤 (>0.3) |
| ch1 | center x offset | `cx = (grid_x + 0.5 + ch1) * stride` |
| ch2 | center y offset | `cy = (grid_y + 0.5 + ch2) * stride` |
| ch3 | log(width) | `width = exp(ch3) * stride` |
| ch4 | log(height) | `height = exp(ch4) * stride` |
| ch5 | rotation cos | `angle = atan2(ch6, ch5)` |
| ch6 | rotation sin | 用于旋转校正 |

### 过滤条件
- 检测置信度阈值: 0.3 (sigmoid 后)
- 识别置信度阈值: 0.7
- 旋转角度限制: 无 (Chrome 不过滤单框绝对角度, 从 IDA 确认)
- 最小文本长度: 3 字符
- NMS IoU 阈值: 0.3
- 最小框尺寸: width > 150, height > 40 (4096 空间)

---

## 相关文件

| 文件 | 功能 |
|------|------|
| `detector.rs` | 检测器，7通道解码，旋转角度提取，极端角度过滤 |
| `ocr.rs` | 合并逻辑，旋转校正，文本去重，置信度过滤 |
| `recognizer.rs` | 识别器，长行分段合并 |
| `utils.rs` | BBox 结构体（含旋转角度） |

---

## 调试命令

```bash
# 启用详细输出
CHROME_OCR_DEBUG=1 ./chrome_ocr PPT.png

# 保存行图像
./chrome_ocr PPT.png --save-lines

# 对比 Native 和 TFLite
./chrome_ocr PPT.png --ab

# 指定置信度阈值
./chrome_ocr PPT.png --min-conf 0.5
```

---

## Hot Path 地址映射

从 profiler 数据计算的基址偏移：
- 基址偏移: `0x7ffe3211ea14`
- 示例映射: `0x7fffb235eb44` → IDA `0x180240130` (Detection postprocess)

---

## IDA 逆向调查进展

### Hot Path 地址映射

**基址偏移**: `0x7FFE3211EA14`
**公式**: `IDA_addr = runtime_addr - 0x7FFE3211EA14`

| Hot Path Runtime | IDA Address | 功能 | CPU% |
|------------------|-------------|------|------|
| 0x7fffb235eb44 | 0x180240130 | Detection postprocess | 33.54% |
| 0x7fffb2360b83 | 0x18024216F | Detection postprocess内部 | 33.54% |
| 0x7fffb23ee491 | 0x1802CFA7D | Recognition entry | 27.17% |
| 0x7fffb23f69f7 | 0x1802D7FE3 | Recognition core | 22.42% |
| 0x7fffb260b043 | 0x1804EC62F | TFLite invoke | 22.02% |

### 已分析的函数

#### 1. Detection Postprocess (0x180240130) ✅
**文件**: `detection_postprocess.c`
**源码**: `region_proposal_text_detector.cc`

关键发现：
- 调用 `CropFromRGBImage` (sub_1806B6D70) 提取行图像
- 包含 "Average height" 计算逻辑（见下文）

#### 2. 角度阈值 ✅ (已确认 - 修正理解)
**文件**: `decompiled_0x180476920.txt`
**源码**: `cluster_sort_gcn_step.cc`

关键代码（第 695-696 行）：
```c
if ( COERCE_FLOAT(...) <= 30.0 )  // 30 度阈值
```

**重要发现 (2026-01-29 修正)**: 这个 30° 是**行间角度差**阈值 (`v49 - v48`),
用于判断两个 box 是否角度兼容以合并到同一行, **不是**单框绝对角度过滤。
Chrome 不会按绝对角度过滤单个检测框, 垂直文本 (~90°) 可以通过检测阶段。

#### 3. 角度归一化 ✅ (已确认)
**文件**: `decompiled_0x180476920.txt`

角度归一化到 [-180, 180] 范围：
```c
// 第 703-705 行
for ( ; v49 <= -180.0; v49 = v49 + 360.0 )
  ;
for ( ; v49 > 180.0; v49 = v49 + -360.0 )
  ;
```

#### 4. Average Height 计算 ✅ (已确认)
**文件**: `detection_postprocess.c` (第 642-662 行)

计算逻辑：
```c
v83 = count_of_boxes;           // 检测框数量
v85 = 0.0;                      // 高度总和
for each box:
    v85 += box->height;         // 在 offset +44 位置
average_height = scale * v85 / count;  // 缩放后的平均高度
```

#### 5. TensorDetectorClient 结构 ✅
**文件**: `TensorDetectorClient_constructor.c`

关键偏移量：
| 偏移量 | 字段 | 说明 |
|--------|------|------|
| +432 | anchor_widths_ | vector<float> |
| +456 | anchor_heights_ | vector<float> |
| +392 | heatmap_scaling | vector<int> |

断言验证：
- `anchor_heights_.size() == anchor_widths_.size()`
- `anchor_heights_.size() == settings.model_output_size()`

#### 6. CropFromRGBImage (0x1806B6D70) ✅ 已反编译
**文件**: `CropFromRGBImage.c`
**源码**: `ocr/google_ocr/image/image_utils.cc`

这是一个**包装函数**，根据图像通道数选择处理路径：

```c
__int64 __fastcall sub_1806B6D70(__int64 a1, __int64 a2, unsigned int a3)
// a1: 输出缓冲区 (112字节)
// a2: 输入tensor (图像数据)
// a3: 标志位 (通常=1)
```

**关键逻辑**:
1. 第78行: 检查第三维度 `v5[2] == 1` (是否单通道)
2. 单通道 → 调用 `sub_1809092C0` (简单复制)
3. 多通道 → 调用 `sub_18090A080` + `sub_1806B72A0` (ConvertToGray)

**重要发现**: 此函数主要是 RGB→灰度转换，真正的裁剪逻辑不在这里！

#### 7. ConvertToGray (0x1806B72A0) ✅ 已反编译
**文件**: `sub_1806B72A0.c`
**源码**: `ocr/google_ocr/image/image_utils.cc` line 371

RGB 转灰度函数，转换公式：
```c
gray = (307*R + 512*G + 205*B + 512) >> 10
// 约等于: gray = 0.299*R + 0.500*G + 0.200*B
```

#### 8. CropMultiScaleDetections (0x180495AC0) ✅ 已反编译
**文件**: `sub_180495AC0.c`
**源码**: `ocr/photo/detection/region_proposal_text_detector_util.cc`

处理多尺度检测结果：
- 验证输入图像为灰度 (error: "Input image must be grayscale to be cropped.")
- 迭代检测结果 (每个检测 240 字节)
- 调用 `sub_180494C30` 处理每个检测框

#### 9. 单框裁剪 (0x180494C30) ✅ 已反编译
**文件**: `sub_180494C30.c`
**源码**: `ocr/photo/detection/region_proposal_text_detector_util.cc`

处理单个检测框的裁剪：
- 调用 `sub_180493A60` 执行实际裁剪
- 调用 `sub_1806CA820` 将 tensor 转为 Pix (Leptonica 图像)
- 错误信息: "Cant crop", "Pix nullptr after cropping"

#### 10. 裁剪实现 (0x180493A60) ✅ 已反编译 🔴 关键函数
**文件**: `sub_180493A60.c`
**源码**: `ocr/photo/detection/region_proposal_text_detector_util.cc`

**两条裁剪路径**:

1. **CropZero** (旋转角度=0): 第172行 `*(float *)(a3 + 48) == 0.0`
   - 调用 `sub_18095B350` 创建边界框
   - 调用 `sub_18095B0C0` 裁剪边界框到图像尺寸
   - 调用 `sub_1806B7BD0` (ImageClipRectangle) 执行裁剪

2. **旋转裁剪** (旋转角度≠0):
   - 第336行: `sub_1807E3EE0(a3)` 获取旋转信息
   - 第337行: `sub_180931A30` 处理旋转矩阵
   - 第371行: `sub_180E92000` 初始化变换矩阵
   - 第372行: `sub_1807E6D90` 应用平移 (-x1, -y1)
   - 第486行: `sub_18090A080` 创建输出 tensor
   - **第539行**: `sub_1806BAE90` 执行旋转裁剪 (仿射变换)

**关键参数** (第532行):
```c
v76 = (LPVOID)0x200000001LL;  // 某些变换参数
```

#### 11. CropZero/ImageClipRectangle (0x1806B7BD0) ✅ 已反编译
**文件**: `CropZero.c`
**源码**: `ocr/google_ocr/image/image_utils.cc` line 443

简单裁剪函数（无旋转）：
```c
__int64 sub_1806B7BD0(__int64 a1, _QWORD *a2, int a3, int a4, int a5, int a6)
// a2: 源图像 tensor
// a3, a4: Y, X 坐标 (裁剪原点)
// a5, a6: Height, Width (裁剪尺寸)
```

**关键逻辑**:
- 第104行: 边界验证
- 第109行: 创建输出 tensor `sub_18090A080(lpMem, height, width, channels, 1, 0)`
- 第151行: 计算源地址 `offset = channels * (x + width * y)`
- 第166行: 逐行复制 `sub_1817F9680(dst, src + row_offset, row_size)`

**重要发现**: 这是纯裁剪，**没有 padding 逻辑**！

#### 12. 旋转裁剪/Warp (0x1806BAE90) ✅ 已反编译
**文件**: `sub_1806BAE90.c`
**源码**: `ocr/photo/detection/anigauss/warp.cc` line 880

仿射变换函数，使用 SSE/SIMD 优化：
- 大量 `_mm_*` 指令处理像素坐标变换
- 处理旋转、缩放等几何变换
- **重要**: 这里也没有显式的 padding 逻辑

---

## 🔴 Chrome 旋转裁剪方法 (从 IDA 分析 sub_180493A60)

### Chrome 的旋转裁剪流程

**关键发现**: Chrome 使用仿射变换矩阵直接从原图采样，**不是**先裁剪后旋转！

```
1. 检查旋转角度 (offset 48)
   if (angle == 0.0) -> CropZero 路径 (简单裁剪)
   else -> 旋转裁剪路径

2. 旋转裁剪路径:
   a. sub_1807E3EE0(bbox) - 获取旋转信息
   b. sub_180931A30() - 创建旋转矩阵
   c. sub_18095B0C0() - 裁剪边界框到图像尺寸
   d. sub_180E92000() - 初始化变换矩阵
   e. sub_1807E6D90(-x1, -y1, matrix) - 应用平移到bbox原点
   f. sub_18090A080(height, width, 1) - 创建精确尺寸输出tensor
   g. sub_1806BAE90() - 执行仿射变换warp

3. 关键参数:
   - 输出尺寸 = 原始bbox尺寸 (不扩展!)
   - v76 = 0x200000001LL - warp参数 (2个32位值)
   - 变换包含: 平移(-x1,-y1) + 旋转
```

### IDA 代码片段 (sub_180493A60)

```c
// Line 172: 检查旋转角度
if ( *(float *)(a3 + 48) == 0.0 )
{
    // CropZero - 无旋转的简单裁剪
    strcpy((char *)lpMem, "CropZero");
    sub_1806B7BD0(lpMem, v6, *v12, v12[1], v12[2], v12[3]);
}
else
{
    // 旋转裁剪
    v83 = sub_1807E3EE0(a3);           // Line 336: 获取旋转信息
    v20 = sub_180931A30(v83);          // Line 337: 创建旋转矩阵
    v25 = sub_18095B0C0(v20, ...);     // Line 364: 裁剪到图像边界

    sub_180E92000(v84, 0, v19, v26);   // Line 371: 初始化变换矩阵
    sub_1807E6D90(-*v72, -v72[1], v84); // Line 372: 平移 (-x1, -y1)

    // Line 486: 创建输出tensor (精确尺寸)
    sub_18090A080(lpMem, v36, v35, 1, 1, 0);

    // Line 532-539: 执行warp
    v76 = (LPVOID)0x200000001LL;       // warp参数
    sub_1806BAE90(&v73, &v76, v82, &v98);
}
```

### 与当前 Rust 实现的差异

| 步骤 | Chrome | Rust (当前改进后) |
|------|--------|-------------------|
| 裁剪区域 | 精确bbox尺寸 | 扩展padding + bbox中心追踪 |
| 旋转方式 | 仿射变换warp | rotate_about_center + 边缘修正 |
| 输出尺寸 | bbox尺寸 | bbox尺寸 + 角度相关边缘margin |
| 采样中心 | bbox中心 | bbox中心 (通过偏移量追踪) |
| 后处理 | 无需裁剪 | 需要裁剪到bbox中心区域 |

### Chrome 的 sub_1807E3EE0 - 多边形角点旋转

**2025-01-28 IDA 分析发现**: Chrome 的旋转中心是 bbox 的**第一个角点** (x1, y1)，不是中心！

```c
// sub_1807E3EE0 - 获取旋转后的多边形
v4 = *v3[3];   // First y coordinate (pivot y)
v5 = *v3[2];   // First x coordinate (pivot x)

// 对每个角点应用旋转
v12 = v11[v10] - v5;  // dx = x - pivot_x
v13 = v3[3][v10] - v4;  // dy = y - pivot_y

// 旋转公式: x' = pivot_x + dx*cos - dy*sin
v11[v10] = (v12 * v9 + v5) - (v13 * v7);
// y' = pivot_y + dy*cos + dx*sin
v3[3][v10++] = (v13 * v9) + (v12 * v7 + v4);
```

### 当前 Rust 实现改进

基于 IDA 分析，当前实现使用以下策略：

1. **带 padding 裁剪**: 根据旋转角度计算所需 padding
2. **中心点追踪**: 追踪 bbox 中心在旋转过程中的位置
   ```rust
   let offset_x = bbox_cx_in_crop - crop_cx;
   let offset_y = bbox_cy_in_crop - crop_cy;
   // 旋转后偏移量也旋转
   let new_offset_x = offset_x * cos_a - offset_y * sin_a;
   let new_offset_y = offset_x * sin_a + offset_y * cos_a;
   ```
3. **角度相关边缘 margin**: 旋转角度越大，添加越多边缘 margin
   ```rust
   let angle_factor = (angle.abs() * 3.0).min(1.0);
   let edge_margin_x = 3.0 + angle_factor * 4.0;  // 3-7 pixels
   let edge_margin_y = 2.0 + angle_factor * 2.0;  // 2-4 pixels
   ```

这种实现虽然不是 Chrome 的精确方法，但能有效处理大多数旋转文本行。

### 识别器相关函数 (tensor_lstm)

#### 13. TensorLstmClient 函数
**主要函数列表**:
| 地址 | 大小 | 说明 |
|------|------|------|
| 0x18029F750 | 0x1211 | **主输入准备函数** - 验证 dtype (float/uint8) |
| 0x1802A65B0 | 0x4e91 | RunModelOnPixa |
| 0x1802A5360 | 0xbbf | TensorLstmClient 函数 |
| 0x1802A1530 | 0x24b2 | TfliteLstmClientBase::RunSessionWithTargets |

**关键字符串**:
- `ocr/photo/segmentation/tensor_lstm_client.cc`
- `tf_input_tensor.dtype() == tf::DT_FLOAT`
- `tf_input_tensor.dtype() == tf::DT_UINT8`
- `input_row_size * num_rows == input_tensor->bytes`
- `lstm_input_tensor->type == TfLiteType::kTfLiteFloat32`

### 待完成任务

#### ✅ 优先级 1: 反编译 CropFromRGBImage
已完成，发现是 RGB→灰度转换的包装函数

#### ✅ 优先级 2: 反编译 识别器输入准备 (0x18029F750)
已反编译，发现是 `TfliteLstmClientBase::CachedConvolutionStep`

**关键发现**:
1. 识别模型参数位于 TensorLstmClient 对象的固定偏移量
2. offset 192: left_padding (unsigned int)
3. offset 196: right_padding (int)
4. offset 205: use_padding flag (byte)
5. DLL 断言: `"left_padding % frame_width == 0"` - 允许 0 或 12

#### ✅ 优先级 2: 修正角度过滤逻辑
已移除单框绝对角度过滤（从 IDA 确认 30° 是行间角度差，不是单框过滤）

#### ✅ 优先级 3: "辩论赛" 首字符丢失问题 - 已修复

**问题**: "辩论赛"→"论赛"

**根本原因**: `rotate_about_center` 围绕裁剪区域的几何中心旋转，而不是 bbox 中心。
当 bbox 不在裁剪区域中心时，旋转后内容会偏移。

**解决方案** (2025-01-28):
1. 计算 bbox 中心在裁剪区域中的位置
2. 旋转后追踪 bbox 中心的新位置
3. 以 bbox 尺寸居中裁剪 bbox 中心位置

**关键代码**:
```rust
// bbox中心在裁剪区域中的位置
let bbox_cx_in_crop = (x1 - crop_x1) as f32 + bbox_w / 2.0;
let bbox_cy_in_crop = (y1 - crop_y1) as f32 + bbox_h / 2.0;

// 旋转后bbox中心位置
let new_offset_x = offset_x * cos_a - offset_y * sin_a;
let new_offset_y = offset_x * sin_a + offset_y * cos_a;
let bbox_cx_in_rotated = rotated_cx + new_offset_x;

// 以bbox尺寸裁剪
let tx1 = (bbox_cx_in_rotated - bbox_w / 2.0) as u32;
```

**测试结果**:
- ✅ "辩论赛" 正确识别
- ✅ 检测到更多行 (22行 vs 之前的20行)
- ⚠️ 部分行仍有首字符错误 (动→边, 归→日, 课→果)

**待调查的剩余问题**:
1. 长行chunk处理的首字符问题 - 可能是检测框不包含完整首字符
2. 尝试移除chunk的LEFT_MARGIN会导致更多问题，说明模型确实需要padding
3. 需要进一步IDA分析Chrome的chunk处理参数

#### ✅ 优先级 4: 分析 Recognition Entry (0x1802CFA7D)
反编译发现这是 `visionkit::SearcherOptions` 构造函数，不是识别入口

#### ✅ 优先级 5: 分析 Recognition Core (0x1802D7FE3)
反编译发现这是 protobuf 序列化大小计算函数，不是 CTC 解码

#### 优先级 6: 反编译识别器输入准备 (0x18029F750) 🔴
这是最可能包含 line image 预处理逻辑的函数

---

## Hough Transform 行分组算法分析 (sub_18049E3B0)

### 概述

Chrome 的 `GroupingBoxesHoughTransform` 函数 (位于 `ocr/photo/detection/detector_box_merging.cc`) 使用 Hough Transform 将字符级检测框分组为文本行。这是一个多阶段算法：

1. **预处理**: 可选的排序/过滤、GCN/textflow聚类
2. **空间哈希**: 建立角度/距离网格的 Hough 累加器
3. **投票**: 每个字符向 Hough 空间投票
4. **行假设提取**: 从累加器中提取最佳行假设
5. **行验证与合并**: 验证行假设并使用 Union-Find 合并
6. **后过滤**: 过滤并输出最终行聚类

### 阶段 1: 预处理

```
sub_18049E3B0 参数:
  a1: config 对象 (GroupingConfig)
  a2: boxes 向量 (vector<BoundingBox>, 每个元素56字节)
  a3: score 向量 (vector<int>) - 可选的排序索引
  a4: output cluster 向量
  a5: output result 向量
  a6: image_width
  a7: image_height
```

**排序逻辑** (行 435-633):
- 如果有 score 向量 (a3 != null) 且其大小与 box 数量匹配:
  - 创建 (score, box_ptr) 对
  - 使用 MergeSort (sub_1804B2C10) 按 score 排序
  - 重排 boxes 和 scores
- 否则如果 config 指定随机化:
  - 使用 MTRandom (sub_1804B3C30) 随机打乱

**GCN/Textflow 聚类** (行 647-736):
- 如果 config+276 == 1 (textflow/GCN 模式):
  - 调用 sub_1804A3A00 进行基于 GCN 的聚类
  - 这是一个独立的聚类路径

### 阶段 2: 空间哈希表建立 (sub_18049AE80)

```c
// sub_18049AE80 - 计算网格参数并建立空间哈希
// 输入: boxes, config, image_width, image_height
// 输出: cell_w, cell_h, grid_cols, grid_rows, avg_height

// 1. 计算平均高度和平均宽度
avg_height = sum(box.height) / num_boxes;  // offset +44
avg_width  = sum(box.width)  / num_boxes;  // offset +40

// 2. 计算网格单元大小
cell_w = avg_width  * config.cell_horizontal_size_portion;  // config+156
cell_h = avg_height * config.cell_vertical_size_portion;     // config+152

// 3. 计算网格尺寸
grid_cols = (int)(image_width  / cell_w) + 1;
grid_rows = (int)(image_height / cell_h) + 1;
```

**空间哈希** (使用 abseil flat_hash_map):
- 键: `grid_cols * row + col` (单个 int32)
- 值: `vector<int>` (该网格单元中的 box 索引列表)
- 对每个 box, 计算其旋转中心点所在的网格单元
- 中心点计算考虑旋转角度:
  ```
  // 使用旋转后的中心坐标
  angle_rad = box.angle * 0.017453292  // deg to rad
  cos_a = cos(angle_rad), sin_a = sin(angle_rad)

  // 中心点 = 原点 + 旋转偏移
  cx = box.x + (-height/2 * cos_a + width/2 * sin_a)
  cy = box.y + (height/2 * cos_a + width/2 * sin_a) // 近似

  cell_key = grid_cols * (int)(cy / cell_h) + (int)(cx / cell_w)
  ```

### 阶段 3: Hough 角度表预计算 (行 840-906)

```c
num_angle_steps = config.hough_angle_steps;  // config+120, 默认36
distance_step   = config.hough_distance_step; // config+116, 默认8

// 预计算角度表: 每个角度步的 (angle, sin/cos_ratio_pair)
angle_step = PI / num_angle_steps;
angle_one_third = avg_height / 3.0;

// 为每个角度步创建条目: (angle, sin_a/one_third, cos_a/one_third)
for i in 0..num_angle_steps:
    angle = i * angle_step
    sin_a = sin(angle)
    cos_a = cos(angle)
    table[i] = (angle, sin_a / angle_one_third, cos_a / angle_one_third)
    // 注: 除以 one_third 实际上是乘以 3/avg_height
```

### 阶段 4: Hough 投票 (sub_1804AC270)

这是核心函数 - 每个 box 向 Hough 累加器投票:

```c
// sub_1804AC270 - 单个 box 的 Hough 投票
// 参数:
//   a1: box 数据 (56字节结构, 56*box_idx + base)
//   a2: box_index
//   a3: config 对象
//   a4: distance_step (hough_distance_step)
//   a5: angle_table (指向预计算的角度表)
//   a6: hough_accumulator (flat_hash_map<int, float>)
//   a7: spatial_hash (用于邻居查找)
//   a8: per_box_trees (每个box的邻居树)
//   a9: out_best_score
//   a10: out_best_distance
//   a11: out_best_angle_idx
//   a12: vote_weight (+1 正向, -1 反向)

// 1. 计算box的旋转角度 (标准化到 [0, PI))
box_angle = atan2(box.sin, box.cos);  // 使用 ch5/ch6
if (box_angle > PI) box_angle -= PI;

// 2. 预计算角度映射表
// 将当前box的角度映射到角度表的各个步
angle_map[i] = (base_idx + i) % num_angle_steps
// 其中 base_idx = (int)(box_angle / angle_step) + angle_min

// 3. 遍历所有角度步, 计算 Hough 空间中的投票
for angle_idx in 0..num_angle_steps:
    actual_angle_idx = angle_map[angle_idx]

    // Hough 变换: r = x*cos(theta) + y*sin(theta)
    // 使用预计算的 sin/cos 值
    distance = (int)(
        angle_table[actual_angle_idx].sin_ratio * cx +
        angle_table[actual_angle_idx].cos_ratio * cy
    ) + half_distance_step

    // 累加器键 = num_angle_steps * distance + actual_angle_idx
    hough_key = num_angle_steps * distance + actual_angle_idx

    // 投票: 权重 = vote_weight (1 或 -1)
    accumulator[hough_key] += (float)vote_weight

    // 记录邻居关系
    spatial_hash.insert(hough_key -> box_list)

    // 追踪最佳得分
    if (accumulator[hough_key] > best_score):
        best_score = accumulator[hough_key]
        best_distance = distance
        best_angle_idx = actual_angle_idx
```

其中 `half_distance_step = (distance_step + (distance_step-1)>>31 - 1) >> 1`:
- 对于 distance_step=8: half = 3
- 这是整数除法 `(distance_step - 1) / 2`

### 阶段 5: 主循环 - 行假设提取与验证 (行 907-1330)

对每个未处理的 box:

```
visited = bitset(num_boxes)  // 标记已处理的box

for each box_i (未被visited标记):
    // 1. 为 box_i 进行 Hough 投票 (正向, weight=+1)
    sub_1804AC270(box_i, ..., vote_weight=+1)

    // 标记 box_i 为已访问
    visited[box_i] = true

    // 2. 检查是否找到有效行假设
    if (config.min_cluster_size > best_score):
        continue  // 分数太低, 跳过

    // 3. 计算 box_i 的旋转中心和方向向量
    cx, cy = rotated_center(box_i)
    angle  = box_i.angle
    cos_a, sin_a = cos(angle), sin(angle)

    // 方向向量 (沿文本方向, 归一化)
    dir = normalize(cos_a, sin_a)
    // 如果 dir 指向负方向, 翻转
    rho = cx * dir.x + cy * dir.y
    if (-rho > rho): dir = -dir
    rho = max(-rho, rho)

    // 4. 正向搜索邻居 (sub_1804AEA00)
    forward_neighbors = sub_1804AEA00(
        boxes, box_i, box_width, box_height, box_angle,
        spatial_hash, config,
        grid_rows, grid_cols,
        hough_angle_idx, hough_distance,
        vote_weight=+1,  // 正向
        line_hypothesis, tree, accumulator
    )

    // 5. 反向搜索邻居 (sub_1804AEA00)
    backward_neighbors = sub_1804AEA00(
        boxes, box_i, box_width, box_height, box_angle,
        spatial_hash, config,
        grid_rows, grid_cols,
        hough_angle_idx, hough_distance,
        vote_weight=-1,  // 反向
        line_hypothesis, tree, accumulator
    )

    // 6. 收集所有邻居
    all_neighbors = collect from tree traversal

    // 7. 检查行假设大小 >= min_cluster_size
    if (len(all_neighbors) >= config.min_cluster_size):
        clusters.push(all_neighbors)

        // 对邻居的角度进行统计
        // 检查是否有 >45度 或 <-45度 的角度偏差

        // 对每个邻居:
        for neighbor_idx in all_neighbors:
            visited[neighbor_idx] = true

            // 如果邻居已被投票过, 进行反向投票以抵消
            if (was_voted[neighbor_idx]):
                sub_1804AC270(neighbor_idx, ..., vote_weight=-1)
```

### 阶段 6: 邻居搜索 (sub_1804AEA00)

这是沿文本行方向搜索相邻字符的核心函数:

```c
// sub_1804AEA00 - 沿方向搜索邻居
// 参数:
//   a1: boxes (span)
//   a2: current_box_idx
//   a3: box_width_ptr
//   a4: box_height_ptr
//   a5: box_angle
//   a6: spatial_hash
//   a7: config
//   a8: grid_rows
//   a9: grid_cols
//   a10: hough_angle_idx
//   a11: hough_distance
//   a12: direction (+1 正向, -1 反向)
//   a13: line_state (方向向量等)
//   a14: tree (邻居树)
//   a15: accumulator

// 关键约束参数 (从 config 读取):
// config+96:  grouping_max_height_ratio (1.5)
// config+100: grouping_max_strict_vertical_distance (0.3) [实际偏移需确认]
// config+128: grouping_box_overlap (0.1)
// config+104: grouping_max_gap_portion (1.5)
// config+108: grouping_max_gap (10)

// 搜索步骤:
// 1. 从当前box出发, 沿方向向量搜索
// 2. 计算搜索区域的网格单元
// 3. 对网格中的候选box检查:
//    a. 高度比 <= max_height_ratio (1.5)
//    b. 垂直距离 <= max_strict_vertical_distance * height
//    c. 水平重叠 >= overlap 阈值
//    d. 间距 <= max_gap_portion * avg_height 且 <= max_gap
// 4. 通过检查的候选box加入行假设
```

### 阶段 7: 行验证与 Union-Find 合并 (行 1499-2025)

```
// 对每个行假设中的 box:
// 1. 标准化角度到 [0, 360) 范围
// 2. 检查角度是否在 [-45, 45] 或 [135, 225] 范围
//    (即近似水平方向)
// 3. 如果 is_nearly_horizontal 且 cluster_count >= 3:
//    使用正交旋转 (sub_1807E5250) 标准化box方向
//    如果角度绝对值 >= config+136 (约 135度), 进一步旋转

// 行合并检查:
// 对每个行假设中的 box_pair:
//   1. 从空间哈希中查找同一网格单元的其他已分配box
//   2. 计算两个box的边界框
//   3. 计算 IoU (Intersection over Union) 或重叠面积
//   4. 如果重叠 >= config.grouping_box_overlap (0.1):
//      使用 Union-Find 合并两个行假设
//      union(head_i, head_j)
```

### 阶段 8: 聚类输出 (sub_1804ACB00)

```c
// sub_1804ACB00 - 最终聚类输出
// 使用 Union-Find 结果, 将 boxes 分组到最终聚类中

// 1. 标记所有 head 节点 (parent[i] == i)
for i in 0..num_boxes:
    if parent[i] == i:
        label_map[i] = next_label++

// 2. 路径压缩: 从后向前遍历
for i in (num_boxes-1)..=0:
    head = find_root(parent, i)
    parent[i] = label_map[head]

// 3. 按标签分组
for i in 0..num_boxes:
    cluster[parent[i]].push(boxes[i])

// 4. 检查最大聚类大小
// 日志: "Large cluster size: N"
// 日志: "Num clusters in list: N max_size: M"
```

### BoundingBox 结构体 (56 字节)

从 IDA 分析确认的结构:

```c
struct BoundingBox {  // 56 bytes total
    void* vtable;     // +0:  vtable pointer
    uint64_t ref;     // +8:  reference/flags
    uint32_t flags;   // +16: flags (bit fields for rotated/swapped/etc)
    void* metadata;   // +24: optional metadata
    int32_t x;        // +32: x coordinate (top-left)
    int32_t y;        // +36: y coordinate (top-left)
    int32_t width;    // +40: width
    int32_t height;   // +44: height
    float   angle;    // +48: rotation angle in degrees
    int32_t score;    // +52: detection score (或其他)
};
```

### 关键配置参数偏移 (config 对象)

```
config+96:  grouping_max_height_ratio       = 1.5
config+100: grouping_max_strict_vertical_distance (未确认偏移)
config+116: hough_distance_step             = 8
config+120: hough_angle_steps               = 36
config+128: grouping_box_overlap            = 0.1
config+132: random_seed (用于 MTRandom)
config+136: min_cluster_size (约 135.0?)
config+152: cell_vertical_size_portion      = 2
config+156: cell_horizontal_size_portion    = 2
config+276: use_textflow_gcn (boolean)
config+224: optional config override pointer
```

### 算法总结

Chrome 的字符行分组算法核心流程:

1. **网格化**: 根据平均字符大小将图像分成网格单元
2. **角度量化**: 将 [0, PI) 分成 36 步 (每步 5 度)
3. **Hough 投票**: 每个字符向 (angle, distance) 空间投票
   - distance = x*sin(theta)/scale + y*cos(theta)/scale
   - scale = avg_height/3
4. **贪心搜索**: 从未处理的字符开始, 找投票最高的行假设
5. **方向扩展**: 沿行方向正反两个方向搜索邻居
   - 约束: 高度比 <= 1.5, 垂直偏移 <= 0.3*h, 间距 <= 1.5*avg_h
6. **合并**: 使用 Union-Find 合并重叠的行假设
   - 重叠阈值: IoU >= 0.1
7. **输出**: 分组后的聚类

---

## GroupDetectionBoxes (sub_18048CE30) - 完整分析 (2025-01-29)

### 概述

`GroupDetectionBoxes` (sub_18048CE30, 大小 0x3097 字节) 是检测后处理的主管线函数。
源文件: `ocr/photo/detection/region_proposal_text_detector_util.cc`

**函数签名**:
```c
unsigned __int64 __fastcall GroupDetectionBoxes(
    __int64 config,          // a1: RegionProposalConfig 对象 (r15)
    __int64 tensor_shape,    // a2: 检测tensor形状信息
    const __m128i *a3,       // a3: 某些输入数据
    float scale,             // a4 (xmm3): 缩放因子 (xmm6 in function body)
    int model_scale,         // a5: 模型尺度标识
    char use_min_height,     // a6: 是否使用最小高度过滤
    const __m128i *a7,       // a7: 输入数据
    __int64 a8,              // a8: 附加数据
    _QWORD *boxes_span,     // a9: boxes 数据 (span-like)
    int **scores_span,       // a10: scores 数据 (span-like)
    __int64 output_vec       // a11: 输出向量
);
```

### 主要流程

```
1. 获取 tensor 形状: image_height = shape[0], image_width = shape[1]
2. 调用 GroupingBoxesHoughTransform (sub_18049E3B0)
   - 将字符级检测框分组为行
   - 输出到 v273 (merged_boxes) 和 v288 (cluster_indices)
3. 日志: "Box Density: {num_boxes / (height * width/8 / 8)}"
4. 对每个合并后的行 (v273 中的元素, 每个 240 字节):
   a. 跳过 width<=0 或 height<=0 的空行
   b. 如果 scale != 1.0: 缩放行的边界框
   c. 如果 !config.skip_padding (config+432 == 0):
      调用 PadAndScaleBoxes (sub_18048ACD0)
   d. 跳过 width<4 或 height<=3 的过小行
   e. 调用 sub_18048BB50 计算行统计 (平均score, score方差, 平均角度)
   f. 过滤: score/cluster_size 比值检查
   g. 过滤: 单box行的score阈值检查
   h. 如果 use_min_height: 过滤小于 config.min_height 的行
   i. 将通过过滤的行添加到输出向量
```

### PadAndScaleBoxes (sub_18048ACD0) - 详细分析

**函数签名** (校正后的类型):
```c
float* PadAndScaleBoxes(
    float* output,           // rcx: 输出 float[2] = {top_padding, bottom_padding}
    float scale_factor,      // xmm1: config[0xF8] - 缩放因子
    float min_box_h,         // xmm2: config[0xFC] - 最小框高
    float max_box_h,         // xmm3: config[0x100] - 最大框高
    float vertical_pad_frac, // stack[0x20]: config[0x1B4] (=config+436) - 垂直padding分配比例
    float max_padding,       // stack[0x28]: 16.0 - 最大padding
    float min_padding,       // stack[0x30]: 4.0 - 最小padding
    float a8,                // stack[0x38]: 8.0
    float a9,                // stack[0x40]: 8.0
    float a10,               // stack[0x48]: 1.0
    BoundingBox* box         // stack[0x50]: box指针 (会被修改)
);
```

**调用时的常量参数**:
```c
PadAndScaleBoxes(
    output_buf,               // rcx
    config[0xF8],             // xmm1 = scale_factor
    config[0xFC],             // xmm2 = min_box_h
    config[0x100],            // xmm3 = max_box_h
    config[0x1B4],            // 从 config+436, vertical_padding_fraction
    16.0f,                    // max_padding
    4.0f,                     // min_padding
    8.0f,                     // param
    8.0f,                     // param
    1.0f,                     // param
    box_ptr                   // box to modify
);
```

**核心算法 (从汇编精确还原)**:

```c
// 输入: box 结构体 (中心坐标 int32, 尺寸 int32, 角度 float 度)
int cx = box->center_x;      // +0x20
int cy = box->center_y;      // +0x24
int w  = box->width;         // +0x28
int h  = box->height;        // +0x2C
float angle_deg = box->angle; // +0x30

float angle_rad = angle_deg * 0.017453292f; // deg to rad
float sin_a = sinf(angle_rad);
float cos_a = cosf(angle_rad);

// Step 1: 计算padding量
// padding = clamp(height * scale_factor, min_padding, max_padding)
//         = clamp(h * scale_factor, 4.0, 16.0)
float raw_pad = (float)h * scale_factor;  // scale_factor from config+0xF8
float padding = fmaxf(min_padding, fminf(max_padding, raw_pad));
// => padding = clamp(h * config[0xF8], 4.0, 16.0)

// Step 2: 垂直padding分配
float v_pad = vertical_pad_frac * padding;  // config[0x1B4] * padding
// output[0] = v_pad          (top padding)
// output[1] = padding - v_pad (bottom padding)

// Step 3: 修改box - 加入宽度方向的padding
// box 的 width 和 height 会被修改为包含 padding 的新值
// 新的 center_x, center_y 也会根据旋转角度进行调整

// 计算旋转后的偏移 (精确公式从汇编):
// 首先计算旧的角点位置:
float old_x1 = (float)cx + sin_a * (-0.5f * (float)h);
float old_x2 = old_x1 + cos_a * (0.5f * (float)w);
float old_y1 = (float)cy + cos_a * (0.5f * (float)w);
// ... (中间涉及多个 sin/cos 变换步骤)

// 宽度方向的padding:
float w_pad = fmaxf(min_box_h, fminf(max_box_h, (float)h * scale_factor));
// 注: 这里 min_box_h 和 max_box_h 用于裁剪宽度padding

// 计算新的带padding的位置 (考虑旋转):
// 沿旋转后的高度方向偏移 padding/2
float half_h_ext = 0.5f * (float)h;
float w_ext = v_pad;

// 计算新的中心点 (考虑旋转后的偏移):
// new_cx 和 new_cy 通过旋转矩阵从原始中心+padding偏移计算

// Step 4: 更新box
box->center_x = roundf(new_cx);   // +0x20
box->center_y = roundf(new_cy);   // +0x24
box->width = roundf((float)old_w + padding);  // +0x28: 宽度增加padding
box->height = roundf(new_h);      // +0x2C: 高度可能也调整
```

**返回值使用** (在 GroupDetectionBoxes 中):
```c
// PadAndScaleBoxes 返回后:
// output[0] = top_padding_amount (float)
// output[1] = bottom_padding_amount (float)

// 这些值被存储到行的扩展属性中:
extended_box->field_40 = output[0];  // top padding
extended_box->field_44 = output[1];  // bottom padding
```

### sub_18048BB50 - 行统计计算

这个函数计算每个合并行的统计信息:

```c
// 参数:
// a1: boxes_span (56字节/box)
// a2: scores_span (float)
// a3: cluster_indices_span (int[])
// a4: output struct

int num_boxes = cluster_indices->size;
if (num_boxes == 0) return;

// 计算: 平均score, score方差, 平均角度
double sum_score = 0.0;
double sum_score_sq = 0.0;
double sum_angle = 0.0;
double sum_angle_sq = 0.0;

for (int i = 0; i < num_boxes; i++) {
    int box_idx = cluster_indices[i];
    float score = scores[box_idx];
    int angle = boxes[box_idx].height;  // offset +44 = angle/score

    sum_score += score;
    sum_score_sq += score * score;
    sum_angle += (double)angle;
    sum_angle_sq += (double)(angle * angle);
}

// 方差计算
float variance = 0.0;
if (num_boxes > 1) {
    variance = (sum_score_sq - sum_score*sum_score/num_boxes) / num_boxes;
}

// 存储结果到输出结构
output->avg_score = (float)(sum_score);        // +24
output->score_variance = (float)variance;       // +28
output->avg_angle = (float)(sum_angle / num_boxes); // +36
output->cluster_size = num_boxes;               // +52
output->flags |= 0x8F;                         // +16
```

### 过滤逻辑

合并后的行需要通过以下过滤:

```c
// 1. 尺寸过滤: width >= 4 && height > 3
void*** merged_box = boxes[v32]; // offset +168
if (merged_box->width <= 0 || merged_box->height <= 0)
    skip("Skipping box");

// 2. 最小尺寸过滤 (padding后): width >= 4, height > 3
if (merged_box->width < 4 || merged_box->height <= 3)
    remove_box();

// 3. 最大旋转角度: |angle| < config+376
if (config->max_angle_threshold > fabsf(merged_box->angle))
    clear_angle_to_zero();

// 4. Score/cluster_size 比值过滤:
float score_ratio = extended_box->field_24 / (float)extended_box->field_52;
// field_24 = sum_score, field_52 = cluster_size
// => score_ratio = avg_score_per_box
if (config->min_score_ratio > score_ratio) {
    // 额外检查: 如果 config->min_cluster_for_removal == 0
    // 或者 cluster_size < min_cluster_for_removal
    // => 移除这个行
    log("Removing box: ... Score: ... cluster size: ...");
    remove_box();
}

// 5. 单box行的score阈值:
if (cluster_size == 4 && config->single_box_score_threshold > score_ratio)
    log("Skipping single box");
    remove_box();

// 6. 最小高度过滤 (仅当 use_min_height=true):
if (use_min_height && (float)config->min_height_pixels > extended_box->height)
    log("Removing small box.");
    remove_box();
```

### 合并行的数据结构 (240 字节)

每个合并后的行在 v273 中占 240 字节:
```
offset +0:    基本数据
offset +16:   flags (uint32)
offset +17:   更多flags (byte)
offset +168:  BoundingBox* (merged bounding box, 指向56字节结构)
offset +184:  ExtendedInfo* (扩展信息, 含padding等)
```

ExtendedInfo 结构:
```
offset +16:  flags (byte)
offset +24:  avg_score / sum_score (float)
offset +28:  score_variance (float)
offset +32:  new_center_x (int, from PadAndScaleBoxes)
offset +36:  new_center_y / avg_angle (float)
offset +40:  new_width / top_padding (int/float)
offset +44:  new_height / bottom_padding (int/float)
offset +48:  angle (float, degrees)
offset +52:  cluster_size (int)
offset +56:  model_scale (int)
```

### 关键发现总结

1. **Padding 计算是自适应的**: `padding = clamp(height * scale_factor, 4.0, 16.0)`
   - 小字符(height小)得到较少padding (最少4px)
   - 大字符(height大)得到较多padding (最多16px)
   - scale_factor 来自 config+0xF8

2. **Padding 分为上下两部分**: 由 vertical_pad_fraction (config+0x1B4=config+436) 控制
   - top_padding = fraction * total_padding
   - bottom_padding = (1-fraction) * total_padding

3. **Box 被原地修改**: PadAndScaleBoxes 会修改 box 的中心坐标、宽度、高度
   - 新宽度 = 旧宽度 + padding
   - 新高度 = 旧高度 + padding (需要进一步确认)
   - 中心坐标会根据旋转角度调整以补偿padding偏移

4. **过滤流程严格**: 合并后的行要经过 5-6 层过滤
   - 尺寸、score、cluster_size、角度、最小高度
   - 这解释了为什么 Rust 实现缺少某些行

---

## Detection Confidence Threshold 分析 (2026-01-29)

### 核心发现

**GroupDetectionBoxes (sub_18048CE30) 中的 score 过滤阈值 = 0.0f**

在 GroupDetectionBoxes 函数中, offset 0x11C (284) 处的浮点值用于过滤合并行的平均 score:

```asm
; 地址 0x18048D737-0x18048D754
movss   xmm11, dword ptr [rax+18h]    ; 加载 score_sum (BoundingBox offset 0x18)
cvtsi2ss xmm2, dword ptr [rax+34h]    ; 加载 count (BoundingBox offset 0x34)
divss   xmm11, xmm2                    ; avg_score = score_sum / count
movss   xmm2, dword ptr [r15+11Ch]    ; 加载阈值 (settings offset 0x11C = 284)
ucomiss xmm2, xmm11                    ; 比较: threshold vs avg_score
jbe     loc_18048D8E9                  ; if threshold <= score => 保留该行
```

r15 指向 `RegionProposalTextDetectorSettings` 结构体。

### 阈值来源分析

1. **构造函数 (sub_1806A8620)**: offset 272-287 被零初始化 (包含 offset 284)
   ```c
   *(_OWORD *)(a1 + 272) = 0;  // 零初始化 offsets 272-287, 包含 0x11C=284
   ```

2. **静态默认实例 (0x181f22440)**: 确认 offset 0x11C 处的值 = 0x00000000 (0.0f)
   ```
   地址 0x181f2255c (= 0x181f22440 + 0x11C): u32 值 = 0 => float 0.0
   ```

3. **Protobuf 描述符**: `RegionProposalTextDetectorSettings` 的 proto 定义中
   **没有** `global_score_threshold` 字段。这个字段属于 `DetectionCascadeOptions` 和
   `DetectionFilterCalculatorOptions`。

4. **InitOCRUsingCallback (sub_180045220)**: Chrome 的 OCR 初始化函数只设置了:
   - PassThroughCoarseClassifier
   - OCR 选项 (page layout, model name)
   - 模型名: "gocr_mobile_chrome_multiscript_2024_q4"
   - **没有** 显式设置任何检测分数阈值

5. **Settings 反序列化**: settings 通过 `sub_1814D3940` 从序列化 protobuf 数据解析。
   由于 proto 中无此字段, 默认值 = 0.0f (protobuf float 默认值)。

### 三层检测 score 过滤体系

| 过滤层 | 函数 | 阈值 | 来源 |
|--------|------|------|------|
| 1. GroupDetectionBoxes 行 score | sub_18048CE30 @ 0x18048D747 | **0.0f** | settings+0x11C (284), 默认值 |
| 2. 单 box 行 score | sub_18048CE30 | **0.4f** | settings+328, proto field 34 `single_box_confidence_threshold` default "0.4" |
| 3. IoU 阈值 | sub_18048CE30 | **0.7f** | settings+332, proto field 35 `intersection_over_union_threshold` default "0.7" |

### 具体过滤逻辑

```c
// 1. 主行 score 过滤 (offset 0x11C = 0.0f, 实际上不过滤任何行)
float avg_score = box->score_sum / (float)box->count;
if (settings->global_score_threshold <= avg_score)  // 0.0 <= any_positive_score
    keep_box();  // 几乎所有行都会通过

// 2. 单 box 行过滤 (cluster_size == 4 的特殊情况)
if (cluster_size == 4 && settings->single_box_threshold > avg_score)
    // single_box_threshold = 0.4f
    remove_box();

// 3. min_cluster_for_removal 检查 (offset 0x17C = 380)
movsxd  rcx, dword ptr [r15+17Ch]  // min_cluster_for_removal
// 如果 cluster_size < min_cluster_for_removal 且 score 低于阈值, 移除
```

### 结论

**Chrome 的 GroupDetectionBoxes 中, 主 score 阈值默认为 0.0f**, 这意味着:
- 在行合并阶段, 几乎所有具有正 score 的行都会通过
- 真正的过滤来自 `single_box_confidence_threshold` (0.4f) - 仅影响单独一个 box 组成的"行"
- `intersection_over_union_threshold` (0.7f) 用于 NMS 去重

对于 Rust 实现, 这意味着:
- 不应在 GroupDetectionBoxes 等效逻辑中使用严格的 score 阈值
- 单 box 行应使用 0.4f 阈值过滤
- 缺失的 "提升高阶能力" 和 "培养高阶思维" 行可能不是因为 score 过滤,
  而是因为检测模型本身没有输出足够的检测框, 或者行分组算法中的其他约束

### DetectionFilterCalculator 补充说明

`DetectionCascadeOptions` 中的 `global_score_threshold` (field 12) 是在
VisionKit 管线层面由 `DetectionFilterCalculator` 使用的, 与
`RegionProposalTextDetectorSettings` 中的过滤是不同层级的:

- **RegionProposalTextDetector 层**: 在字符框合并为行之后进行过滤 (offset 0x11C = 0.0f)
- **VisionKit Pipeline 层**: DetectionFilterCalculator 可能在管线中进一步过滤

但在 Chrome Screen AI 的初始化中, 没有设置 `global_score_threshold`, 所以两层过滤的阈值都默认为 0.0f。

---

## 待 IDA 逆向调查的具体问题

> **当前状态**: PPT.png 24/24 行完全匹配。以下为进一步优化方向。

### 1. 仿射变换裁剪 (sub_1806BAE90) 🟡
当前 Rust 使用 `rotate_about_center` + 后裁剪，而 Chrome 使用仿射变换直接从原图采样。
功能等价但实现不同，可能在某些边缘情况下产生差异。

### 2. PadAndScaleBoxes 精确参数 🟡
config+0xF8 (scale_factor), config+0xFC (min_box_h), config+0x100 (max_box_h),
config+0x1B4 (vertical_pad_fraction) 的具体数值尚未从 protobuf 配置中提取。
当前功能正常，但精确参数可能有助于处理其他测试图像。

### 3. 性能优化 🟢
TFLite ~1682ms vs Native ~447ms。可能的优化:
- 减少不必要的图像复制
- 批量识别优化
- XNNPACK 参数调优

---

## 已确认的参数 (可直接使用)

| 参数 | 当前值 | Chrome 值 | 状态 |
|------|--------|-----------|------|
| 单框角度过滤 | 无 (不过滤) | 无 (不过滤) | ✅ 已修复 (之前错误地用30°过滤) |
| 行间角度差阈值 | 30° | 30° | ✅ 符合 (用于行分组兼容性检查) |
| 角度归一化 | [-180, 180] | [-180, 180] | ✅ 符合 |
| LEFT_MARGIN | 12 | 12 | ✅ 确认正确 (0会更差) |
| BLANK_IDX | 8178 | 8178 | ✅ 符合 |
| frame_width | 4 | 4 (168/42) | ✅ 符合 |

## 关键 IDA 发现 (2025-01-28)

### 识别器 Canvas 布局
**测试结论**: LEFT_MARGIN=12 是正确的
- LEFT_MARGIN=0 导致更多首字符丢失
- LEFT_MARGIN=12 只有特定行（如"辩论赛"）有问题

**Rust 实现 (正确)**:
```rust
let mut canvas = ImageBuffer::from_pixel(168, 32, Luma([255u8]));
overlay(&mut canvas, &scaled, 12, 0);  // LEFT_MARGIN=12 正确
```

### 图像反转
**测试结论**: Chrome 不做图像反转
- 从 IDA 分析: CropFromRGBImage 和 ConvertToGray 不做反转
- 使用原始灰度值

### TensorLstmClient 结构偏移
| 偏移量 | 类型 | 字段 |
|--------|------|------|
| 192 | unsigned int | left_padding |
| 196 | int | right_padding |
| 200 | unsigned int | model height |
| 205 | byte | use_padding flag |
| 208 | int | thread count |
| 232 | int | max width |
| 240 | int | batch size |
| 244 | int | related to batch |
| 248 | float | score threshold |
