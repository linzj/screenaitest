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

## 当前对比结果 (2026-01-29)

| 图片 | Native | Rust | 状态 |
|------|--------|------|------|
| PPT.png | 24 行 | 24 行 | ✅ 完全匹配 |
| general_ocr_002.png | 38 行 | 38 行 | ✅ 完全匹配 |
| layout.test.jpg | 189 行 | 55 行 | ⚠️ 需 beam search + FST LM |

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

## 相关文件

| 文件 | 功能 |
|------|------|
| `detector.rs` | 检测器，7通道解码，旋转角度提取 |
| `ocr.rs` | 合并逻辑，旋转校正，文本去重，置信度过滤 |
| `recognizer.rs` | 识别器，长行分段合并，CTC解码 |
| `utils.rs` | BBox 结构体（含旋转角度） |

---

## 待解决问题

1. **layout.test.jpg** (55 vs 189 行) - 需要 Beam Search + FST 语言模型
2. **FST 语言模型** - Chrome 使用 FST 进行 CTC 解码
3. **语言检测** - 实现 tflite_langid.tflite
4. **性能优化** - TFLite ~1682ms vs Native ~447ms
