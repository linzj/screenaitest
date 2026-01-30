# Hough Transform 行分组算法逆向分析

## 概述

Chrome 的 `GroupingBoxesHoughTransform` (sub_18049E3B0) 使用 Hough Transform 将字符级检测框分组为文本行。

**源文件**: `ocr/photo/detection/detector_box_merging.cc`

## 算法流程

```
1. 预处理: 排序/过滤、可选 GCN/textflow 聚类
2. 空间哈希: 建立角度/距离网格的 Hough 累加器
3. 投票: 每个字符向 Hough 空间投票
4. 行假设提取: 从累加器中提取最佳行假设
5. 行验证与合并: 验证行假设并使用 Union-Find 合并
6. 后过滤: 过滤并输出最终行聚类
```

## 阶段 1: 预处理 (sub_18049E3B0)

```c
// 参数:
// a1: config (GroupingConfig)
// a2: boxes (vector<BoundingBox>, 56字节/个)
// a3: score 向量 (可选排序索引)
// a4: output cluster 向量
// a5: output result 向量
// a6: image_width
// a7: image_height
```

- 如果有 score 向量且大小匹配: MergeSort 按 score 排序
- 如果 config 指定随机化: MTRandom 随机打乱
- 如果 config+276 == 1 (GCN 模式): 调用 sub_1804A3A00

## 阶段 2: 空间哈希表 (sub_18049AE80)

```c
avg_height = sum(box.height) / num_boxes;
avg_width  = sum(box.width)  / num_boxes;

cell_w = avg_width  * config.cell_horizontal_size_portion;  // config+156
cell_h = avg_height * config.cell_vertical_size_portion;     // config+152

grid_cols = (int)(image_width / cell_w) + 1;
grid_rows = (int)(image_height / cell_h) + 1;

// 空间哈希 (abseil flat_hash_map):
// key: grid_cols * row + col
// value: vector<int> (box 索引列表)
```

中心点计算考虑旋转:
```c
angle_rad = box.angle * 0.017453292;  // deg to rad
cx = box.x + (-height/2 * cos_a + width/2 * sin_a);
cy = box.y + (height/2 * cos_a + width/2 * sin_a);
cell_key = grid_cols * (int)(cy / cell_h) + (int)(cx / cell_w);
```

## 阶段 3: Hough 角度表

```c
num_angle_steps = config.hough_angle_steps;   // config+120, 默认 36
distance_step   = config.hough_distance_step; // config+116, 默认 8

angle_step = PI / num_angle_steps;
angle_one_third = avg_height / 3.0;

for i in 0..num_angle_steps:
    angle = i * angle_step;
    table[i] = (angle, sin(angle) / one_third, cos(angle) / one_third);
```

## 阶段 4: Hough 投票 (sub_1804AC270)

```c
// Hough 变换: r = x*sin(theta)/scale + y*cos(theta)/scale
// scale = avg_height / 3

for angle_idx in 0..num_angle_steps:
    distance = (int)(
        table[actual_angle_idx].sin_ratio * cx +
        table[actual_angle_idx].cos_ratio * cy
    ) + half_distance_step;

    hough_key = num_angle_steps * distance + actual_angle_idx;
    accumulator[hough_key] += vote_weight;  // +1 正向, -1 反向

    if accumulator[hough_key] > best_score:
        update best_score, best_distance, best_angle_idx;
```

`half_distance_step = (distance_step - 1) / 2` (distance_step=8 时为 3)

## 阶段 5: 行假设提取与验证

```
for each unvisited box_i:
    1. Hough 投票 (正向, weight=+1)
    2. 如果 best_score < min_cluster_size: 跳过
    3. 计算方向向量 (cos_a, sin_a), 归一化
    4. 正向搜索邻居 (sub_1804AEA00, direction=+1)
    5. 反向搜索邻居 (sub_1804AEA00, direction=-1)
    6. 收集所有邻居
    7. 如果 len(neighbors) >= min_cluster_size:
       - 创建行聚类
       - 标记所有邻居为已访问
       - 对已投票邻居反向投票抵消
```

## 阶段 6: 邻居搜索 (sub_1804AEA00)

沿文本行方向搜索相邻字符:

**约束参数**:
| 参数 | 配置偏移 | 默认值 | 说明 |
|------|---------|--------|------|
| grouping_max_height_ratio | config+96 | 1.5 | 最大高度比 |
| grouping_max_strict_vertical_distance | config+100 | 0.3 | 最大垂直距离 (× height) |
| grouping_box_overlap | config+128 | 0.1 | 重叠阈值 |
| grouping_max_gap_portion | config+104 | 1.5 | 最大间距 (× avg_height) |
| grouping_max_gap | config+108 | 10 | 最大绝对间距 |

**Gap 处理**: Chrome 在 gap 超过限制时 **中断搜索** (不是跳过继续)。

## 阶段 7: Union-Find 合并

```c
// 行合并检查:
for each box_pair in line_hypothesis:
    // 查找同一网格单元的其他已分配 box
    // 计算边界框 IoU
    if overlap >= grouping_box_overlap (0.1):
        union(head_i, head_j);  // Union-Find 合并
```

角度归一化:
```c
for (; angle <= -180.0; angle += 360.0);
for (; angle > 180.0; angle -= 360.0);
```

30° 阈值是**行间角度差** (不是单框绝对角度过滤):
```c
if (angle_diff <= 30.0)  // 两 box 角度兼容则可合并
```

## 阶段 8: 聚类输出 (sub_1804ACB00)

```c
// 1. 标记 head 节点
for i in 0..num_boxes:
    if parent[i] == i:
        label_map[i] = next_label++;

// 2. 路径压缩
for i in (num_boxes-1)..=0:
    head = find_root(parent, i);
    parent[i] = label_map[head];

// 3. 按标签分组
for i in 0..num_boxes:
    cluster[parent[i]].push(boxes[i]);
```

## 配置参数完整表

| 偏移 | 参数 | 默认值 |
|------|------|--------|
| config+96 | grouping_max_height_ratio | 1.5 |
| config+100 | grouping_max_strict_vertical_distance | 0.3 |
| config+104 | grouping_max_gap_portion | 1.5 |
| config+108 | grouping_max_gap | 10 |
| config+116 | hough_distance_step | 8 |
| config+120 | hough_angle_steps | 36 |
| config+128 | grouping_box_overlap | 0.1 |
| config+132 | random_seed | - |
| config+136 | min_cluster_size | ~135.0 |
| config+152 | cell_vertical_size_portion | 2 |
| config+156 | cell_horizontal_size_portion | 2 |
| config+276 | use_textflow_gcn | boolean |

## 关键地址索引

| 函数 | 地址 | 说明 |
|------|------|------|
| GroupingBoxesHoughTransform | 0x18049E3B0 | 主入口 |
| 空间哈希建立 | 0x18049AE80 | 网格参数计算 |
| Hough 投票 | 0x1804AC270 | 单 box 投票 |
| 邻居搜索 | 0x1804AEA00 | 方向扩展搜索 |
| 聚类输出 | 0x1804ACB00 | Union-Find 结果输出 |
