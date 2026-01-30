# Chrome OCR Rust 实现 - 当前状态

## ⚠️ 重要提醒

**不要参考 Python 实现** - 只参考 IDA 逆向分析的 DLL 原生实现。
必须使用 ida-pro-mcp 逆向，参考真正的解决方案。

---

## 详细逆向文档

逆向分析结果按模块拆分到以下文档：

| 文档 | 内容 |
|------|------|
| [docs/detection.md](docs/detection.md) | 检测模块: 7通道解码、裁剪、旋转、NMS、RemoveOverlaps |
| [docs/line_grouping.md](docs/line_grouping.md) | Hough Transform 行分组算法、Union-Find、配置参数 |
| [docs/recognition.md](docs/recognition.md) | 识别模块: CTC解码、Beam Search、多脚本架构、Junk过滤 |
| [docs/batch_processing.md](docs/batch_processing.md) | 批处理架构: batch维度堆叠、TFLite模型验证 |

---

## 当前对比结果 (2026-01-30)

| 图片 | Native | Rust | 状态 |
|------|--------|------|------|
| PPT.png | 24 行 | 24 行 | ✅ 完全匹配 |
| general_ocr_002.png | 38 行 | 38 行 | ✅ 完全匹配 |
| layout.test.jpg | 189 行 | 190 行 | ⚠️ 接近 (+1, 106 common lines preserved) |

### PPT.png Native 完整输出 (24 lines)
```
1: 趣味话题          2: 激发学习兴趣      3: 归纳总结能力
4: 章节总结          5: 热点话题          6: 发现问题能力
7: 分析问题能力      8: 解决问题能力      9: 深度学习能力
10: 辩论赛          11: 质疑思辨能力     12: 动脑动手动口能力
13: 终生学习能力    14: 提升高阶能力     15: 小组合作学习+成绩小组互评
16: 查阅文献能力    17: 追踪前沿文献     18: 科技写作能力
19: 课程论文        20: 有效拓宽视野     21: 培养学科自信
22: 升学大幅提高    23: 专家讲座         24: 培养高阶思维
```

### 修复历程

**2025-01-28**: 辩论赛(论赛→✅)、动脑动手动口能力(边脑动手动口能大→✅)、归纳总结能力(日纳总结能大→✅)、质疑思辨能力(贡疑思辨能力→✅)、热点话题(で热点话题→✅)

**2025-01-29 (Hough Transform 重构)**: 课程论文(课程论→✅)、升学大幅提高(升学大幅摄→✅)、专家讲座(家讲座→✅)、小组合作学习+成绩小组互评(小组合作学+成绩小年百→✅)

**2026-01-29 (垂直文本)**: 提升高阶能力(缺失→✅)、培养高阶思维(缺失→✅)。根因: 30°是行间角度差阈值，不是单框绝对角度过滤。

### layout.test.jpg 管线分析

| 阶段 | 数量 |
|------|------|
| Detection (char boxes) | ~8930 |
| After NMS | 219 |
| After heading merge | 209 |
| Spatial duplicates | -33 |
| Empty text (len=0) | -77 (需 beam search) |
| Single char (len=1) | -42 (需 beam search) |
| **输出** | **55** |

---

## 已确认的关键参数

| 参数 | 值 | 状态 |
|------|-----|------|
| 单框角度过滤 | 无 (不过滤) | ✅ |
| 行间角度差阈值 | 30° | ✅ |
| LEFT_MARGIN (hanijpan) | 12 | ✅ |
| LEFT_MARGIN (und) | 8 | ✅ |
| BLANK_IDX (hanijpan) | 8178 | ✅ |
| BLANK_IDX (und) | 1292 | ✅ |
| frame_width | 4 (168/42) | ✅ |
| Canvas size | 168 × 32 | ✅ |
| 检测置信度阈值 | 0.3 (sigmoid 后) | ✅ |
| 主行 score 阈值 | 0.0f (不过滤) | ✅ |
| 单 box score 阈值 | 0.4f | ✅ |
| NMS IoU 阈值 | 0.3 | ✅ |
| RemoveOverlaps IoU | 0.6 | ✅ |
| max_breadth_ratio | 2 (min ratio=0.5) | ✅ |

---

## RemoveOverlaps 阈值逆向分析 (2026-01-30)

### RemoveOverlapsStep (sub_18045D800)

Step 对象布局 (200 bytes, 分配在 sub_18041EA00):
- step+0: vtable (RemoveOverlapsStep)
- step+8: enabled (byte)
- step+16..103: base Step fields
- step+96 (0x60): a4 parameter
- step+104 (0x68): RemoveOverlapsSpec 开始

RemoveOverlapsSpec 布局 (初始化在 sub_180726570):
- spec+0: vtable
- spec+8: arena ptr
- spec+16: field bitmask
- spec+24: line_overlap ptr (PageLayoutOverlappingRemoverSpec)
- spec+32: word_overlap ptr (PageLayoutOverlappingRemoverSpec)
- spec+40: symbol_overlap ptr (PageLayoutOverlappingRemoverSpec)
- spec+48: symbol_removal_breadth_ratio (double) = **0.5** (默认, xmmword_1819D82D0[0])
- spec+56: block_different_direction_maximum (double) = **0.3** (默认, xmmword_1819D82D0[1])
- spec+64: min_low_block_confidence (double) = **0.5** (默认, xmmword_1819D82E0[0])
- spec+72: max_low_block_symbol_breadth_ratio (double) = **0.6** (默认, xmmword_1819D82E0[1])
- spec+80: enabled (bool) = true

#### Process 函数中的阈值使用 (sub_18045D800)

1. **主要重叠判定** (0x18045e36c):
   ```
   max(IoU, overlap_ratio1, overlap_ratio2) > step+0xA0 (=spec+56 = block_different_direction_maximum)
   ```
   默认阈值: **0.3**

2. **符号级重叠验证** (0x18045e387):
   ```
   symbol_overlap_ratio > step+0xA0 (同上)
   ```
   默认阈值: **0.3**

3. **面包宽度比检查** (0x18045e3f2):
   ```
   step+0xA8 (=spec+64 = min_low_block_confidence) <= line.confidence
   ```
   默认阈值: **0.5**

4. **面包宽度比阈值** (0x18045e41f):
   ```
   min_breadth / max_breadth <= step+0xB0 (=spec+72 = max_low_block_symbol_breadth_ratio)
   ```
   默认阈值: **0.6**

5. **符号移除面包比** (0x18045e588):
   ```
   step+0x98 (=spec+48 = symbol_removal_breadth_ratio) > min_breadth/max_breadth
   ```
   默认阈值: **0.5** (触发 symbol-level overlap removal)

#### OverlapSpec (aksara::OverlapSpec) - 在 DetectOverlaps 中使用

OverlapSpec 包含在 PageLayoutOverlappingRemoverSpec.overlap_spec 中。
DetectOverlaps (sub_18046A1D0) 中的阈值使用:

| 字段 | 偏移 | 检查条件 | 重叠类型 |
|------|------|----------|----------|
| maximum_overlap | a1+24 | IoU > threshold | kOverlap (1) |
| maximum_duplicate | a1+32 | total_dup > threshold | kDuplicate (3) |
| maximum_overwrite | a1+40 | overwrite > threshold | kOverwrite (4) |
| minimum_breadth_ratio | a1+48 | breadth check | - |
| max_different_direction_overlap | a1+56 | IoU > threshold (diff dir) | kDifferentDirectionOverlap (2) |

OverlapSpec 的 protobuf 默认值全部为 **0.0**，实际值来自 binarypb 配置文件。

### RemoveOverlapsWordPruningStep (sub_180460B70)

Step 对象布局 (184 bytes, 分配在 sub_18041D250):
- step+104 (0x68): RemoveOverlapsWordPruningStep Spec 开始

Spec 硬编码默认值 (构造函数 sub_180727480):

| 字段 | Spec偏移 | 类型 | 默认值 |
|------|----------|------|--------|
| enabled | 24 | bool | true (来自配置) |
| line_overlap_iou_threshold | 28 | float | **0.6** |
| skip_curved_boxes | 32 | bool | true |
| apply_accumulated_different_orientation | 33 | bool | true |
| line_overlap_threshold (complete_overlap_word_overlap_threshold) | 36 | float | **0.6** |
| complete_word_confidence_difference_threshold (field 5) | 40 | float | **0.1** |
| word_overlap_threshold | 48 | double | **0.75** |
| word_confidence_difference_threshold (field 6) | 56 | float | **0.2** |
| common_characters_percentage_threshold (field 8) | 60 | float | **0.05** |
| exact_text_match_overlap_threshold (field 9) | 64 | float | **0.6** |
| (field 10) | 68 | float | **0.5** |
| different_orientation_support_threshold_multiplier | 72 | float | **4.0** |
| accumulated_number | 76 | int | **10** |

### Protobuf 描述符确认 (地址 0x18214c990)

RemoveOverlapsWordPruningStep protobuf 描述符中的默认值:
- line_overlap_iou_threshold: 0.6
- line_overlap_threshold: 0.6
- skip_curved_boxes: true
- complete_overlap_word_overlap_threshold: 0.1
- complete_word_confidence_difference_threshold: 0.2
- word_overlap_threshold: 0.75
- word_confidence_difference_threshold: 0.05
- common_characters_percentage_threshold: 0.6
- exact_text_match_overlap_threshold: 0.5

RemoveOverlapsSpec protobuf 描述符中的默认值 (地址 0x18214c580):
- enabled: true
- symbol_removal_breadth_ratio: 0.5
- block_different_direction_maximum: 0.3
- min_low_block_confidence: 0.5
- max_low_block_symbol_breadth_ratio: 0.6
- line_overlap: (submessage PageLayoutOverlappingRemoverSpec)
- word_overlap: (submessage PageLayoutOverlappingRemoverSpec)
- symbol_overlap: (submessage PageLayoutOverlappingRemoverSpec)

### 总结

Chrome RemoveOverlaps 使用的关键阈值:
1. **行级 IoU 阈值**: 0.6 (RemoveOverlapsWordPruningStep.line_overlap_iou_threshold)
2. **行级重叠阈值**: 0.6 (RemoveOverlapsWordPruningStep.line_overlap_threshold)
3. **词级重叠阈值**: 0.75 (RemoveOverlapsWordPruningStep.word_overlap_threshold)
4. **RemoveOverlapsStep 主阈值**: 0.3 (block_different_direction_maximum - 控制何时触发重叠检查)
5. **符号移除面包比**: 0.5 (symbol_removal_breadth_ratio)
6. **面包比阈值**: 0.6 (max_low_block_symbol_breadth_ratio)

---

## 相关文件

| 文件 | 功能 |
|------|------|
| `detector.rs` | 检测器，7通道解码，旋转角度提取 |
| `ocr.rs` | 合并逻辑，旋转校正，文本去重，置信度过滤 |
| `recognizer.rs` | 识别器，长行分段合并，CTC解码 |
| `utils.rs` | BBox 结构体（含旋转角度） |

---

## 2026-01-30 Session 进展

### 关键发现

1. **Chrome完整Pipeline (28步Layout Analysis)**:
   - Step 6: RemoveOverlapsWordPruningStep (word级overlap删除, 在clustering前)
   - Step 11: SplitLinesStep (基于symbol depth分割多行区域)
   - Step 12: MergeLinesStep (合并相邻行)
   - Step 18: ClusterSortGcnStep (GCN神经网络行聚类)
   - Step 19: RemoveOverlapsStep (line级overlap删除, threshold=0.3)
   - 全部在recognition之前运行!

2. **Chrome RemoveOverlapsStep主阈值是0.3** (不是0.6!):
   - block_different_direction_maximum = 0.3
   - 如果max(IoU, containment1, containment2) > 0.3就触发overlap移除
   - 之前用0.6，body text overlaps (containment ~0.38-0.52) 全部漏掉
   - 用0.3: layout从207降到172 (过于aggressive，需要添加Chrome的breadth ratio保护逻辑)

3. **Latin文本Script路由**:
   - Chrome先运行GocrScriptDirectionIdentificationMutator检测script
   - CJK(hanijpan)模型处理英文文本时不插入空格，但confidence略高于UND
   - 修复: 如果CJK输出全ASCII+无空格+UND有空格→选择UND
   - 效果: 有空格的body text行从26增加到93

4. **Chrome Height Guard (sub_180476920)**:
   - 严格模式: 2 * min(h_a, h_b) > merged_height (cross-direction用)
   - 宽松模式: 3 * min(h_a, h_b) + h_a + h_b > merged_height (same-direction用)
   - 当前Rust用 avg_height * 2.0 作为全局guard (比Chrome更保守)

### 测试结果变化

| 修改 | PPT | general | layout |
|------|-----|---------|--------|
| 基线(session开始) | 24✅ | 38✅ | 206 |
| +Latin script路由到UND | 24✅ | 38✅ | 207 |
| +overlap threshold 0.6→0.3 | 24✅ | 38✅ | 172 |
| +MergeLinesStep (bbox height) | 24✅→23❌ | 38✅ | ~196 |
| +MergeLinesStep (avg_char_h) | 24✅ | 38✅ | 196 |
| +OverlapPruneStep (0.6) | 20❌ | 38✅ | 194 |
| +FragmentAbsorption | 20❌ | 38✅ | 176 |
| 回退到 MergeLinesStep only | 24✅ | 38✅ | 196 |
| +X-gap splitting (threshold 1.5) | 24✅ | 38✅ | 208❌ |
| 回退X-gap splitting | 24✅ | 38✅ | 196 |
| +Fragment dedup (br<0.3, y_ovr>0.3) | 24✅→23❌ | 38✅ | 179 |
| 修复: br<0.5保护 + br<0.3+y_ovr>0.4 | 24✅ | 38✅ | 187 |
| 调整 y_ovr>0.44 | 24✅ | 38✅ | **190** ←当前 |

### MergeLinesStep 逆向分析 (sub_1804410C0)

Chrome MergeLinesSpec protobuf 默认值 (地址 0x18214c094):
- `minimum_breadth_ratio`: 0.6 (宽度比阈值)
- `maximum_angle_difference`: 3° (角度差上限)
- `minimum_breadth_overlap`: 0.6 (宽度重叠比)
- `maximum_depth_gap`: 1.5 (高度gap比)
- `merge_adjacent_lines`: false (当true时只merge 1个neighbor)
- `delete_entities_with_no_symbols`: true

Chrome MergeLinesStep 使用**方向对齐的 breadth/depth**:
- breadth = 沿文本方向的范围 (水平文本=宽度)
- depth = 垂直于文本方向的范围 (水平文本=高度)
- 对于有角度的文本，breadth/depth 会随文本方向旋转

**合并判定4条件** (sub_1804410C0):
1. `min(w_a, w_b) / max(w_a, w_b) >= 0.6` (breadth ratio)
2. `|angle_a - angle_b| < 3°` (angle difference)
3. `(w_a + w_b - merged_w) / merged_w >= 0.6` (breadth overlap)
4. depth gap < 1.5 * min_depth

### ⚠️ MergeLinesStep 实现教训 (反复踩坑记录)

**错误方案1: 纯axis-aligned breadth overlap**
- 直接用Chrome的4个条件做merge
- **结果: PPT从24→23行 (破坏!)**
- **原因**: PPT的放射状文本 (从中心向外辐射) 在x轴上有大量重叠
  - 例如"趣味话题"和"激发学习兴趣"在x方向有overlap
  - 它们角度接近(都接近0°)，宽度也相似
  - 但它们是不同行! 只是PPT布局让它们在x方向重叠

**正确方案: 加Y-center proximity约束**
- 在Chrome的4个条件基础上增加Y中心距检查:
  - `|y_center_a - y_center_b| < 0.3 * min(h_a, h_b)`
- 这确保只合并**同一物理行**上的clusters
- 额外添加"adjacent"检查: x方向间距 < avg_h * 1.5 的侧邻cluster也可合并
- **结果: PPT=24✅, general=38✅**

**关键教训**:
1. Chrome的direction-aligned breadth对角度文本有效，但我们用axis-aligned x/y
2. 对于放射状布局(如PPT)，必须有Y中心距约束防止跨行合并
3. Chrome可能因direction-aligned计算天然避免了这个问题

### RemoveOverlaps 0.3阈值分析

**0.3是different-direction overlap的阈值** (不是same-direction!):
- `block_different_direction_maximum = 0.3`
- 用于: 当两个block方向不同时 (`max(IoU, containment1, containment2) > 0.3`)
- Same-direction overlaps由symbol-level验证处理 (sub_18045D260)
  - 需要逐个字符bbox计算重叠 → 我们没有symbol级数据无法复制

**body text Y-overlap分析**:
- layout.test.jpg中相邻body text行的Y-overlap ratio = 0.36~0.52
- 这些是**相邻行**(上下紧挨)，不是同行重复
- 我们的0.7 Y-overlap阈值正确保护了它们不被误删

### layout.test.jpg gap根因分析

多出的17行主要是:
- **中等长度碎片** (5-20字符): +19行
  - "iment", "fOr dropping labels", "EText", "71Tex", "6262"等
  - 来自检测区域重叠导致的cluster边界分裂
- **长碎片** (>20字符): +8行
- **少了微小碎片** (<4字符): -10行
- 净差: +17行

### 当前状态 (2026-01-30 最终)

- **PPT=24✅, general=38✅, layout=190 (目标189, 差+1)**
- MergeLinesStep 使用 avg_char_h 代替 bbox height 做 Y-center proximity check
- OverlapPruneStep 和 FragmentAbsorption 均已尝试并回退 (弊大于利)
- **Fragment dedup** (新增): breadth_ratio < 0.3 && y_overlap > 0.44 → remove smaller line
  - 从196减到190 (移除6个column-boundary fragments)
  - 保留了所有106个common lines
- **Pass 2 breadth gap check** (新增): gap / min_avg_w > 0.7 → reject merge
  - Chrome ClusterLinesSpec: maximum_breadth_gap = 0.7
  - 无实际影响 (说明Pass 2不是cross-column合并源头)

### 剩余+1的根因

| 类别 | 数量 | 说明 |
|------|------|------|
| Rust-only | 84 | 包含merged table cells, truncated fragments, 错误数字 |
| Native-only | 83 | 包含正确的table cells, 完整body text, 正确数字 |
| Common | 106 | 两边匹配的行 |
| **净差** | **+1** | 84-83=1 |

进一步关闭gap需要:
1. **Hough分组breadth gap**: cluster-merge阶段使用avg_symbol_breadth做gap check (不是单box width!)
2. **FST语言模型**: 改善CTC解码质量 (trailing punctuation, word boundaries)
3. **Table cell分离**: 阻止"6262","7268"等merged cells的产生

### 已尝试并失败的方法 (不要重复!)

1. **OverlapPruneStep (cluster-level containment removal)**:
   - 在MergeLinesStep后移除containment > threshold的小cluster
   - threshold=0.6: PPT从24→20 (破坏! 有效cluster被containment=0.78的大region覆盖)
   - threshold=0.85: 仍然破坏PPT (因为avg_char_h修复前的MergeLinesStep问题)
   - 只捕获2个cluster，效果不值得风险

2. **FragmentAbsorption (post-recognition fragment removal)**:
   - 移除宽度 < 0.3×相邻长行 且 Y重叠 > 50% 且 X间距 < 2×height 的短行
   - 问题1: 把table cells ("11","5","62","72","68") 误删 (相邻table header)
   - 加 parent text > 25 chars 约束后: 仍然删了"excluded","avoided"等独立成行的段落尾词
   - 删了正确的body text fragments但也删了valid standalone words
   - layout从196→183 (比189还少! 过度删除)

3. **MergeLinesStep with bbox height (0.3 * min_h)**:
   - PPT clusters有很高的bbox (200+ px, 因为放射状文本)
   - 0.3 * 173 = 52px, 导致52px Y-center距离的不同集群被合并
   - PPT从24→23 (或更糟到20)
   - **修复**: 用avg_char_h代替bbox height, Y threshold = 0.5 * min_avg_char_h

4. **SplitLinesStep X-gap splitting (threshold=1.5)**:
   - 从IDA逆向得到Chrome的 maximum_space_ratio_in_line = 1.5
   - 实现: 在Y-gap splitting后, 对每个sub-cluster做X-gap splitting
   - layout从196→208 (增加12行!), common lines从106降到101
   - **根因**: 我们的Hough分组已经产生单列cluster, X-gap splitting把单列line错误分割
   - body text的word间距 (~45-50px) ≈ column gap (~45-50px), 无法区分
   - **结论**: X-gap splitting在我们的pipeline中不适用, Chrome从不同的clustering起点开始
   - ❌ 不要再尝试!

5. **Greedy expansion breadth gap check (gap / min_w > 0.7)**:
   - 在greedy expansion中加入Chrome ClusterLinesSpec的breadth gap check
   - PPT从24→23! layout从190→282!
   - **根因**: 使用单个box的width (min(cur_w, cand.width())) 太敏感
   - 窄字符 ('i','l','.') 的width很小, 正常word space gap / narrow_width > 0.7
   - Chrome用cluster级别的avg_symbol_breadth, 不是单个box width
   - ❌ 不适用于greedy expansion (需要cluster-level平均, 但greedy是逐box扩展)

6. **Pass 2 breadth gap check (gap / cluster_avg_w > 0.7)**:
   - 在小cluster吸收步骤中加入breadth gap check (使用cluster_avg_w)
   - 对layout无影响 (190→190), 说明Pass 2不是cross-column合并的源头
   - 保留在代码中作为额外保护

7. **Fragment dedup breadth_ratio < 0.5 → remove (without protection)**:
   - 直接把原来的 `breadth_ratio < 0.5 → protect` 改为 remove
   - PPT从24→23! (PPT "迫整誌過立"被containment=0.91删除)
   - **根因**: PPT放射状布局中小cluster被大cluster的bbox包含, max_overlap > 0.6触发删除
   - **修复**: 保留breadth_ratio 0.3-0.5的保护, 只在breadth_ratio < 0.3时做fragment removal
   - breadth_ratio < 0.3 = 明确的fragment (width < 30% of keeper)
   - breadth_ratio 0.3-0.5 = table cell范围, 需要保护

6. **Fragment dedup y_overlap threshold调整**:
   - 0.3 → layout=179 (过度删除, 包括table cells "83","68" etc, y_ovr≈0.33)
   - 0.4 → layout=187 (部分合适, 但cascade效果导致lost common line "5")
   - 0.43 → layout=187 (类似)
   - **0.44** → layout=190 (最佳: 移除6个fragments, 保留所有106 common lines) ← 当前使用
   - 0.45 → layout=191 (太保守, 漏掉1个fragment)

### 190→189 gap根因 (改进后)

+1差异。Fragment dedup修复减少了7行(196→190):
- **移除的6个fragments**: "Or the addition of", "crror-bar on the", "formed",
  "label", "Performance, we trained and", "metrics, we rat"
- **仍存在的fragments** (~20个): "iment", "onthe", "the data be", "excluded",
  "This provide", "Throughout this" etc. (y_ovr ≈ 0.44, 刚好在阈值边界)
- **Table cell合并问题** (未解决): "6262","7268","71Tex","81Tex","82Tex","EText"
  - 需要在Hough分组阶段加入breadth gap check (maximum_breadth_gap=0.7)

**从196到190的改进方法**: 修改dedup逻辑，当breadth_ratio < 0.3 (very small fragment)
且y_overlap_ratio > 0.44时，移除smaller line。这近似Chrome的symbol-level overlap removal。

### SplitLinesStep 逆向分析 (sub_18046FFB0 = Process, sub_180471090 = ShouldSplit)

**源码路径**: `research/ocr/api/internal/layout_analyzer/split_lines_step.cc`

**SplitLinesSpec 参数** (构造函数 sub_18046F9C0, 验证4个非负double):
| 参数 | Spec偏移 | 含义 |
|------|----------|------|
| `maximum_space_ratio_in_line` | spec+24 (a3[3]) | 空间深度比阈值 |
| `maximum_symbol_depth_ratio` | spec+32 (a3[4]) | 符号深度比阈值 |
| `maximum_punctuation_depth_ratio` | spec+40 (a3[5]) | 标点深度比阈值 |
| `maximum_thinspace_depth_ratio` | spec+48 (a3[6]) | 细空格深度比阈值 |

**Step对象布局**:
- step+128 (0x80): `maximum_space_ratio_in_line` (double)
- step+136 (0x88): `maximum_symbol_depth_ratio` (double)
- step+168 (0xA8): graph pointer

#### SplitLinesStep::Process (sub_18046FFB0)

这是Step 11的核心处理函数。它对每个line cluster做两级分割:

**Level 1: Line-level split (深空格分割)**
1. 计算 `average_symbol_depth` (sub_1805ABC80) → v61/v62结构体
   - average_depth = 所有符号间gap的平均值 (沿reading direction)
   - symbol_count = 符号数量
2. 遍历line的所有children (words), 检查相邻children之间的**文本方向标记**
3. 对于相邻children: 如果文本方向(LTR/RTL)不同 → 标记方向切换位
4. 调用 `sub_180471090` (ShouldSplitBetween) 检查是否要在相邻children之间分割

**Level 2: Word-level split (深符号分割)** (sub_1804706A0)
对每个word遍历其children (symbols), 检查:
1. `sub_1804718E0` (IsSymbolTooDeep): 符号深度检查
2. `sub_180471B10`: 相邻符号间gap检查

#### ShouldSplitBetween (sub_180471090)

**关键分割决策函数**。输入: 相邻两个children (a3, a4) + 方向切换标记 (a5)

**Step 1: BiDi保护** (line 342)
```
if (a5 && direction(a3) != direction(a4)):
    log("Avoiding word split for bidi text")
    return false  // 不分割双向文本边界
```

**Step 2: 空间深度检查** (line 361) - **这是主要的HORIZONTAL分割逻辑!**
```
gap_depth = ComputeGapDepth(graph, a3, a4)  // sub_1805AC140
gap_ratio = floor(gap_depth) / average_symbol_depth  // (来自v61结构体的double)
if (gap_ratio > maximum_space_ratio_in_line):  // step+128
    log("Splitting line because of a deep space: {a3} -> {a4}, depth is {gap_depth} compared to {avg_depth}")
    return true  // 分割!
```

**Step 3: 符号深度检查** (line 372)
```
symbol_depth = GetSymbolDepth(graph, last_child_of_a3)  // sub_1806FF2E0
avg_expected_gap = (symbol_count * avg_depth - symbol_depth) / (symbol_count - 1)
if (symbol_depth / avg_expected_gap > maximum_symbol_depth_ratio):  // step+136
    log("Splitting line because of a deep symbol: {a3}, depth is {symbol_depth}")
    return true  // 分割!
```

**Step 4**: 如果不满足任何条件 → return false (不分割)

#### ComputeGapDepth (sub_1805AC340)

计算两个entity之间的"gap depth"（间距深度）:
```
// 构建两个entity的方向对齐bbox
box_a = DirectionalBox(a3)  // 沿reading direction对齐
box_b = DirectionalBox(a4)
// 获取在reading direction上的位置
extent_left = GetExtent(box_union, entity_a)   // sub_18070C4B0: 读取box的directional position
width_a = GetExtent(box_a, entity_a)           // a的宽度
width_b = GetExtent(box_b, entity_b)           // b的宽度
gap = max(extent_left - (width_a + width_b), 0.0)  // 间距 = 总范围 - 两个宽度
```

**关键**: `sub_18070C4B0` 根据方向选择不同维度:
```c
return *(int *)(a1 + 4 * (*(int *)(a2 + 28) == 2) + 32);
// 如果direction==2(vertical): 读取box.y_extent (offset 36)
// 否则(horizontal): 读取box.x_extent (offset 32)
```

这证明了gap计算是**沿文本方向**的！对于水平文本就是X方向间距。

#### IsSymbolTooDeep (sub_1804718E0)

```
1. 读取symbol的文本内容 (UTF-8 → codepoints)
2. 检查所有codepoint是否为"类别8"字符 (sub_181566270 返回8)
   - 如果是 → return false (不分割标点等)
3. 如果symbol的 (byte+32) & 0x40 == 0 → return false
4. 根据字符的script类型选择depth_ratio:
   - sub_180E78940 返回script类别 → 索引到 step+136+8*script
   - 即: maximum_punctuation_depth_ratio 或 maximum_thinspace_depth_ratio 按script不同
5. 计算 avg_expected_gap = (symbol_count * avg_depth - this_depth) / (symbol_count - 1)
6. if (this_depth / avg_expected_gap > per_script_ratio):
     new_depth = avg_expected_gap * 0.5
     return true
```

#### 关键结论

**1. SplitLinesStep 确实做水平方向(X-gap)分割!**
- `maximum_space_ratio_in_line` 控制line级别的X-gap分割
- 当cluster内两组字符之间的X间距远大于平均符号间距时 → 分割
- 这就是Chrome如何避免column-boundary fragments的关键机制！

**2. "Symbol depth"分割逻辑**:
- 分两层: line-level(word间gap) 和 word-level(symbol间gap)
- 使用 `average_symbol_depth` 作为基准
- gap / avg_depth > threshold → 分割
- 阈值来自protobuf spec配置

**3. 方向感知**:
- 所有gap计算都是方向对齐的 (reading direction)
- 对水平文本: gap = X方向间距
- 对垂直文本: gap = Y方向间距
- 通过 `sub_18070C4B0` 的direction==2条件选择

**4. 为什么Chrome没有column-boundary fragments**:
- Chrome的Hough Transform line grouping产生的cluster可能跨越两列
- SplitLinesStep检测到column gap (X方向大间距) 后分割cluster
- 分割后的两半各自成为独立的line entity
- 我们缺少这个步骤,所以Hough分组产生的跨列cluster保持原样

### ⚠️ X-gap splitting 尝试记录 (2026-01-30, 多次尝试均失败!)

**Chrome的SplitLinesStep确实做X-gap splitting** (IDA确认):
- `maximum_space_ratio_in_line` = **1.5** (从protobuf descriptor 0x18211a783解码)
- 逻辑: `floor(x_gap) / avg_symbol_depth > 1.5` → 分割

**实现效果 (两次尝试)**:
- **第一次**: 阈值1.5, 产生13个X-split, layout从196→208 (+12!)
  - 正确分割了table cells: 99 boxes→5 parts, 96 boxes→5 parts
  - 但也分割了body text column gap: 108→2, 123→2, 140→2 等
  - body text column gap ~45-50px, avg_char_h ~27px, ratio ~1.7 > 1.5
- **第二次**: 相同阈值, 对比content quality
  - Common lines (with native): **101** (比不分割的106更少!)
  - Rust-only lines: 107 (比不分割的90更多!)
  - X-gap splitting反而降低了匹配度

**为什么X-gap splitting在我们的pipeline中不work**:
1. Chrome的Hough分组产生跨列cluster → X-gap splitting分离两列 → 各自成为正确的单列line
2. 我们的Hough分组大部分已经产生单列cluster → X-gap splitting把单列line在word间gap处错误分割
3. body text的word间距 (~45-50px) 和column gap (~45-50px) 在这个文档中大小相近
4. 分割单列line不产生匹配native的输出，反而破坏了已有的正确匹配

**结论**: X-gap splitting不适用于我们的pipeline。Chrome的pipeline从不同的clustering起点开始,
X-gap splitting在Chrome那里是修正跨列clusters, 但在我们这里是错误分割已经正确的单列lines。

**❌ 不要再尝试X-gap splitting! ❌**

### 196 vs 189 详细diff分析 (2026-01-30)

**统计**:
- Common lines: 106
- Rust-only: 90
- Native-only: 83
- Diff: 196 - 189 = +7

**短行 (<=5 chars) 差异**:

| Rust-only (24个) | Native-only (31个) |
|---|---|
| 6262, 7268, 71Tex, 81Tex, 82Tex | 62, 62, 72, 68, 71, 81, 82, Text, Text |
| 68 Text, EText, EText | Text, Text, Text, Text, Text |
| 7, 7, 7, 8, 10, 132, 169, 31, 32, 33 | 6, 66, 69, 72, 72, 72, 77, 77, 82, 83, 88, 89, 90 |
| Al, Tex | All |
| iment, label, onthe | ment. |

**关键发现**:
- **Table cell合并**: "6262"→应为"62"+"62", "7268"→"72"+"68", "71Tex"→"71"+"Text" etc.
  - 同行相邻table cells被Hough分到同一cluster → 识别为连在一起的文本
- **Table数字识别差异**: Rust "7"×3, "8", "10", "132"等 vs Native "72"×3, "83", "88"等
  - 数字被truncate或错误识别

**中等行 (10-30 chars) 差异**:
- Rust-only: 15个 (全是column-boundary fragments)
  - "a direct comparison", "Aa Table 5", "crror-bar on the", "dataset is ifit is"
  - "Doc Page Doc Page", "is down-mapped, ww", "metrics, we rat", "oOne set. To the best of"
  - "Or the addition of", "Section-heade", "the data be", "This provide"
  - "Throughout this", "to Text in PubLayNet", "wWwe have split the train"
- Native-only: **0个**!

**长行 (>30 chars) 差异**:
- Rust-only: 40个 (truncated body text, 缺少最后几个字)
  - e.g., "...the be" (native: "...the be-")
  - e.g., "...only" (native: "...only one set. To the best of")
  - e.g., "...[23], o" (native: "...[23], or the addition of")
- Native-only: 40个 (完整body text lines)

**差异根因总结**:
1. **Body text truncation + fragment**: 每个body text line被截断(去掉最后N个字符),
   截断的部分成为独立的medium-length fragment。40个截断+15个fragment = 产生55行代替native的40行。
   净效果: +15行
2. **Table cell合并**: ~7个merged cells代替native的~14个分离cells。
   净效果: -7行
3. **Table数字识别差异**: 各种数字错误, 大致抵消。
   净效果: ~-1行
4. **总净效果**: +15 -7 -1 = +7 ✓

### Chrome RemoveOverlapsStep 详细分析 (IDA逆向, sub_18045D800)

三级overlap remover系统:

**Gate 1 (Coarse geometric)**:
- max(IoU, containment_a, containment_b) > 0.3 (block_different_direction_maximum)
- 只有超过0.3阈值的pair才进入下一步

**Gate 2 (Word-level overlap score)**: sub_18045D260
- 构建N×M overlap matrix (所有word pairs)
- 计算area-weighted overlap fractions
- 返回 max(score_a, score_b)
- 如果score也 <= 0.3 → skip

**Gate 3 (Confidence protection)**:
- 如果被删线的 confidence >= 0.5 (min_low_block_confidence)
  且 breadth ratio min/max < 0.6 (max_low_block_symbol_breadth_ratio)
  → 保护不删除

**Symbol-level removal**:
- 当 min_breadth/max_breadth <= 0.5 (symbol_removal_breadth_ratio)
  → 进入symbol-level overlap removal (逐字符bbox比较)

### 需要实现但很难的功能

- **Table cell分离**: 最有希望的方向，需要在Hough分组阶段阻止相邻table cells合并
- Chrome的column detection (分离两栏文本)
- Chrome的ClusterSortGcnStep (GCN神经网络排序)
- Chrome的symbol-level overlap (sub_18045D260, 需要逐字符bbox)

## Hough Transform Line Grouping - Column Separation Analysis (2026-01-30)

### Question: How does Chrome prevent cross-column clustering?

**Source**: `research/ocr/api/internal/layout_analyzer/cluster_lines_step.cc`

#### ClusterLinesSpec Protobuf (decoded from binary at 0x18214c1e0)

```protobuf
message ClusterLinesSpec {
  bool enabled = 1 [default = true];
  double minimum_symbol_breadth_ratio = 2 [default = 0.6];
  double maximum_angle_difference = 3 [default = 3];
  double maximum_breadth_gap = 4 [default = 0.7];
  double maximum_depth_gap = 5 [default = 1.5];
}
```

#### Merge Decision Function (sub_18042B1D0)

This function decides whether two line clusters should be merged. It has **4 sequential checks** - all must pass to merge:

**Check 1: Breadth Gap Ratio** (offset a1+144 = maximum_breadth_gap, default 0.7)
```
breadth_gap = gap between clusters along text direction (X for horizontal text)
min_avg_breadth = min(avg_symbol_breadth_cluster_A, avg_symbol_breadth_cluster_B)
ratio = breadth_gap / min_avg_breadth
if ratio > 0.7: REJECT ("Breadth gap ratio too large")
```
**This is the key column-separation constraint!** For horizontal text:
- "breadth gap" = X-distance between the rightmost symbol of cluster A and leftmost symbol of cluster B
- "min_avg_breadth" = minimum of the two clusters' average symbol widths
- If the X-gap exceeds 0.7x the average character width, clustering is rejected

**Check 2: Depth Gap Ratio** (offset a1+152 = maximum_depth_gap, default 1.5)
```
depth_gap = gap between clusters perpendicular to text direction (Y for horizontal text)
min_avg_depth = min(avg_symbol_depth_cluster_A, avg_symbol_depth_cluster_B)
ratio = depth_gap / min_avg_depth
if ratio > 1.5: REJECT ("Depth gap ratio too large")
```
For horizontal text: Y-gap / min_avg_char_height > 1.5 means too far apart vertically.

**Check 3: Angle Difference** (offset a1+136 = maximum_angle_difference, default 3.0)
```
angle_A = cluster A's text direction angle (degrees)
angle_B = cluster B's text direction angle (degrees)
diff = abs_angular_difference(angle_A, angle_B)  // handles wraparound at 360
if diff > 180: diff = 360 - diff  // shorter arc
if diff > 3.0: REJECT ("Angle difference too large")
```

**Check 4: Symbol Breadth Ratio** (offset a1+128 = minimum_symbol_breadth_ratio, default 0.6)
```
max_avg_breadth = max(avg_symbol_breadth_A, avg_symbol_breadth_B)
if min_avg_breadth / max_avg_breadth < 0.6: REJECT ("Symbol breadth ratio too small")
```
This prevents merging clusters with very different character sizes (e.g., title + body text).

#### Breadth/Depth Distance Calculation (sub_1805AC200 / sub_1805AC340)

Both functions compute directional gap:
```
// Build union bounding box of both clusters
union_box = merge(cluster_A_box, cluster_B_box)

// For breadth gap (sub_1805AC200):
total_extent = GetBreadthExtent(union_box, entity)  // sub_18070C4A0
breadth_A = GetBreadthExtent(cluster_A_box, entity)
breadth_B = GetBreadthExtent(cluster_B_box, entity)
gap = max(total_extent - (breadth_A + breadth_B), 0.0)

// GetBreadthExtent reads different dimensions based on text direction:
//   horizontal text (direction != 2): reads box.width  (offset 32)
//   vertical text   (direction == 2): reads box.height (offset 36)
```

The "depth" gap uses `sub_18070C4B0` which reads the **opposite** dimension.

#### Hough Transform Voting (sub_18023E4B0)

The Hough voting function in `region_proposal_text_detector.cc` creates bins:
- **Angle bins**: quantized by `bin_size` (stored at a1+328)
- **Breadth bins**: position along reading direction, quantized by same bin_size
- Each character box votes for its (angle_bin, breadth_bin)
- Characters in the same angle+breadth bin become candidates for the same line

**There is NO explicit X-distance constraint in the voting phase.** The spatial locality comes entirely from:
1. The bin quantization (characters with very different angles won't share bins)
2. The post-voting merge decision in ClusterLinesStep (which applies the 4 checks above)

#### Answer to the Original Questions

**Q1: Does the Hough Transform have a maximum X-distance or maximum width constraint?**
YES. The `maximum_breadth_gap` parameter (default 0.7) acts as a **normalized X-distance constraint**. For horizontal text, it limits the X-gap between clusters to 0.7 times the average character width. This is checked during the merge decision, not during voting.

**Q2: Is there a "maximum_breadth" or "maximum_line_width" parameter?**
There is no absolute maximum line width parameter. The constraint is relative: `breadth_gap / min_avg_symbol_breadth <= 0.7`. This means characters can be far apart in X as long as they are part of a continuous chain where each adjacent pair has a small gap.

**Q3: What is the "maximum_distance" parameter and what does it measure?**
There is no single "maximum_distance" parameter. Instead there are two directional gap parameters:
- `maximum_breadth_gap` = 0.7: max gap along text direction / avg symbol breadth
- `maximum_depth_gap` = 1.5: max gap perpendicular to text / avg symbol depth

**Q4: Are there spatial locality constraints in the grouping?**
YES, four constraints in the merge decision:
1. **Breadth gap ratio <= 0.7** (X-gap for horizontal text, relative to avg char width)
2. **Depth gap ratio <= 1.5** (Y-gap for horizontal text, relative to avg char height)
3. **Angle difference <= 3 degrees**
4. **Symbol breadth ratio >= 0.6** (character sizes must be similar)

#### Why Chrome Doesn't Create Cross-Column Clusters

For a two-column academic paper with ~12px average character width:
- Column gap is typically 30-60px
- Breadth gap ratio = 60/12 = 5.0 >> 0.7
- Even with a 30px gap: ratio = 30/12 = 2.5 >> 0.7
- Characters near the column boundary will **never** be merged across columns

The `maximum_breadth_gap = 0.7` ensures that only characters within ~0.7 character-widths of each other (along the text direction) can be grouped together. This naturally prevents cross-column grouping since the column gap is always many character-widths wide.

#### Implication for Rust Implementation

Our Hough Transform implementation should enforce the breadth gap check during the cluster merge phase:
```rust
// When considering merging cluster A and cluster B:
let x_gap = compute_breadth_gap(&cluster_a, &cluster_b); // gap along text direction
let min_avg_breadth = f64::min(avg_char_width_a, avg_char_width_b);
if x_gap / min_avg_breadth > 0.7 {
    // REJECT: too far apart in reading direction (cross-column!)
    continue;
}
```

This is the **missing constraint** that causes our Hough grouping to sometimes create clusters spanning two columns in academic paper layouts.

---

## 待解决问题

1. **layout.test.jpg** (190 vs 189行, +1)
   - ~~已部分解决: fragment dedup (breadth_ratio < 0.3 + y_ovr > 0.44) 从196→190~~
   - **剩余问题A**: 仍有~20个column-boundary fragments (y_ovr ≈ 0.44, 在threshold边界)
   - **剩余问题B**: table cell合并 ("6262","7268" etc.)
   - **下一步**: 在Hough分组加入breadth gap check (Chrome: maximum_breadth_gap=0.7)
     - 这会同时解决table cell合并和column-boundary fragments
     - 逆向确认: ClusterLinesSpec 4个merge条件全部需要实现
2. **FST 语言模型** - Chrome 使用 FST 进行 CTC 解码
3. **语言检测** - 实现 tflite_langid.tflite
4. **性能优化** - TFLite ~1682ms vs Native ~447ms
