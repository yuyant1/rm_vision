# 阶段四：卡尔曼滤波追踪与预测

## 运行

已集成在阶段三 pipeline 中，无需额外操作：

```bash
ros2 launch rm_vision_phase3 vision_pipeline.launch.py
```

追踪节点自动启动，发布 /tracked 话题和 /detections/tracked 可视化图像。

## 架构

```
camera → detector → tracker → visualizer
                      ↓
              /tracked (TrackedArray)
              /detections/tracked (标注图像)
```

## 追踪流水线

每帧执行以下步骤：

1. 预测：所有现有追踪器用卡尔曼滤波向前预测一步（等速模型）
2. 匹配：匈牙利算法（IoU 代价矩阵）将当前检测与预测位置关联
3. 更新：匹配成功的追踪器用检测值更新卡尔曼状态
4. 新建：未匹配的检测创建新追踪器
5. 填补：未匹配的追踪器保留并用纯预测位置，超过 10 帧删除
6. 输出：每个目标的追踪 ID、位置、速度、未来 10 帧预测位置

## 卡尔曼滤波器设计

状态向量 [x, y, vx, vy]，等速运动模型。观测值 [x, y] 来自检测中心坐标。

测量噪声协方差 R = 50（检测抖动程度），过程噪声 Q = 1（模型信任度）。初始不确定性 P = 1000 让滤波器快速收敛到检测值。

## 关键参数

max_lost_frames: 10。目标丢失 10 帧后用预测填补，超过则删除。覆盖短暂遮挡。
prediction_steps: 10。预测未来 10 帧位置，用于计算提前量。
iou_threshold: 0.3。匈牙利关联的 IoU 最小阈值。

## 输出说明

TrackedBBox 消息新增字段：
- tracking_id: 目标的唯一追踪 ID（跨帧不变）
- vx/vy: 卡尔曼估计的速度（像素/帧）
- predicted_x/y: 未来位置预测值
