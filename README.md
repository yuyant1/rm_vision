# RoboMaster 视觉组 — 自瞄系统雏形

完整的装甲板检测与追踪系统，从图像采集到射击目标坐标输出的端到端 pipeline。

## 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     RoboMaster Vision Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────┐   ┌─────────────┐   ┌──────────────┐   ┌────────┐ │
│  │ Camera   │──>│  Detector   │──>│   Tracker    │──>│ Output │ │
│  │ Node     │   │  Node       │   │   Node       │   │        │ │
│  │          │   │             │   │              │   │ 射击   │ │
│  │ 读取帧   │   │ YOLOv8 推理 │   │ 卡尔曼滤波   │   │ 目标   │ │
│  │ 发布     │   │ 发布检测框 │   │ 匈牙利关联   │   │ 坐标   │ │
│  │ Image    │   │ Detection-│   │ 速度+预测    │   │        │ │
│  │          │   │ Array      │   │ TrackedArray │   │        │ │
│  └──────────┘   └─────────────┘   └──────────────┘   └────────┘ │
│       │               │                  │               │       │
│       v               v                  v               v       │
│  /camera/       /detections        /tracked        射击决策     │
│  image_raw                                             模块     │
│                                                                   │
│  ┌──────────┐                                                    │
│  │ Launch + Params + RViz │──────────────── 配置层               │
│  └──────────┘                                                    │
└─────────────────────────────────────────────────────────────────┘
```

数据流：Camera Node 从摄像头或视频文件读取帧，发布 sensor_msgs/Image。Detector Node 订阅图像用 YOLOv8 推理，发布 DetectionArray 带类别、位置、置信度。Tracker Node 订阅检测结果，用匈牙利算法关联帧间目标，卡尔曼滤波平滑位置并估计速度，发布 TrackedArray 含追踪 ID、速度向量、未来预测位置。Visualizer Node 订阅各话题在 RViz 实时显示标注图像。

## 各阶段对应关系

| 阶段 | Tag | 内容 | 文件位置 |
|------|-----|------|----------|
| 零 | v0.1-phase0 | ROS2 基础：Pub/Sub、自定义消息、Launch、参数 | rm_vision_demo |
| 一 | v0.2-phase1 | OpenCV 灯条检测：HSV 阈值、形态学、轮廓筛选、灯条配对 | rm_vision_phase1 |
| 二 | v0.3-phase2 | YOLOv8 训练：数据标注、训练流程、模型推理 | rm_vision_phase2 |
| 三 | v0.4-phase3 | ROS2 集成：三节点 pipeline、Launch 一键启动、RViz | rm_vision_phase3 |
| 四 | v0.5-phase4 | 卡尔曼追踪：匈牙利关联、丢失预测、速度估计、提前量 | rm_vision_phase3 |
| 五 | v0.6-phase5 | 综合整合：架构文档、技术总结、系统完善 | 本 README |

## 环境要求

- Ubuntu 26.04
- ROS2 Lyrical
- Python 3.14
- OpenCV 4.10+
- PyTorch 2.13+
- Ultralytics 8.4+
- filterpy, scipy

## 快速开始

```bash
# 1. 克隆仓库
mkdir -p ~/robomaster_ws/src
cd ~/robomaster_ws/src
git clone https://github.com/yuyant1/rm_vision.git .

# 2. 安装 Python 依赖
pip install --break-system-packages filterpy scipy ultralytics torch

# 3. 编译
cd ~/robomaster_ws
source /opt/ros/lyrical/setup.bash
colcon build

# 4. 运行
source install/setup.bash
ros2 launch rm_vision_phase3 vision_pipeline.launch.py

# 5. 可视化（新终端）
source install/setup.bash
rviz2 -d src/rm_vision_phase3/rviz/vision_pipeline.rviz
```

## 运行方式

```bash
# 默认启动（摄像头）
ros2 launch rm_vision_phase3 vision_pipeline.launch.py

# 使用视频文件
ros2 launch rm_vision_phase3 vision_pipeline.launch.py video_path:=/path/to/video.mp4

# 使用 GPU 推理
ros2 launch rm_vision_phase3 vision_pipeline.launch.py device:=cuda

# 调整追踪参数
ros2 launch rm_vision_phase3 vision_pipeline.launch.py max_lost_frames:=15 prediction_steps:=5

# 关闭可视化
ros2 launch rm_vision_phase3 vision_pipeline.launch.py enable_visualization:=false
```

## 参数说明

camera_node：camera_id（摄像头设备号，默认 0）、video_path（视频路径，非空优先）、fps（帧率，默认 30）、image_width/height（缩放，默认 640x480）

detector_node：model_path（YOLO 权重路径）、confidence_threshold（置信度阈值，默认 0.5）、device（cuda/cpu，默认 cuda）、imgsz（推理尺寸，默认 640）

tracker_node：max_lost_frames（丢失删除阈值，默认 10）、prediction_steps（预测步数，默认 10）、iou_threshold（关联 IoU 阈值，默认 0.3）

visualizer_node：enable_visualization（开关）、line_thickness（框线粗细）

## 话题列表

| 话题 | 类型 | 发布者 | 说明 |
|------|------|--------|------|
| /camera/image_raw | sensor_msgs/Image | camera_node | 原始图像 |
| /detections | rm_msg/DetectionArray | detector_node | YOLO 检测结果 |
| /tracked | rm_msg/TrackedArray | tracker_node | 追踪结果（含速度+预测） |
| /detections/debug | sensor_msgs/Image | detector_node | 检测调试图像 |
| /detections/tracked | sensor_msgs/Image | tracker_node | 追踪可视化图像 |
| /detections/image_annotated | sensor_msgs/Image | visualizer_node | 综合标注图像 |

## 自定义消息

rm_msg 包定义四套消息：Target 用于阶段零 Pub/Sub 测试，BBox2D 和 DetectionArray 用于检测节点输出，TrackedBBox 和 TrackedArray 用于追踪节点输出（增加了 tracking_id、vx/vy 速度、predicted_x/y 预测位置）。

## 技术总结

### 遇到的问题与解决方案

**HSV 阈值不稳定。** 不同光照下红色灯条的 H 值变化大，且红色跨越色环两端（0 度和 180 度）。解决：用两段 mask 取并集，并通过 hsv_tuner.py 滑块工具交互调试确定具体阈值。

**形态学操作核大小选择。** 核太小去噪不够，核太大会把邻近目标粘连。解决：5x5 核配合 3 轮闭运算 + 2 轮开运算，在测试图上反复调试确定。

**灯条检测高度不足。** 灯条只是装甲板正面中间一段 LED，面板外框比灯条高出一截。解决：框高设为灯条高度乘 2.25，通过测量实物比例确定。

**YOLO 检测抖动。** 逐帧独立推理导致检测框在连续帧间跳动。解决：引入卡尔曼滤波器做帧间平滑，用等速模型约束位置变化，测量噪声 R=50 让滤波平滑但不迟钝。

**目标短暂丢失。** 装甲板被遮挡或检测漏帧导致追踪中断。解决：匈牙利算法匹配丢失后保留追踪器，最多 10 帧纯预测填补，10 帧后仍未检测到则删除。

**大文件无法推送到 GitHub。** 视频文件和模型权重大于 100MB。解决：.gitignore 排除 mp4 和 pt 文件，在文档中说明文件位置和获取方式。

### 可能的改进方向

检测方面：用 TensorRT 对 YOLO 模型做推理加速，在 Jetson 边缘设备上部署；加入数据增强提高对遮挡和模糊帧的鲁棒性。

追踪方面：将等速模型升级为等加速模型，实现弹道预测；加入 IMU 数据做传感器融合，补偿云台运动造成的图像抖动。

工程方面：单元测试覆盖核心逻辑；CI/CD 自动编译验证；Docker 容器化部署，简化环境配置。

## 目录结构

```
robomaster_ws/
├── src/
│   ├── rm_msg/                          # 自定义消息包
│   │   └── msg/
│   │       ├── Target.msg               #   阶段零用
│   │       ├── BBox2D.msg               #   检测框
│   │       ├── DetectionArray.msg       #   检测结果数组
│   │       ├── TrackedBBox.msg          #   追踪框（含速度+预测）
│   │       └── TrackedArray.msg         #   追踪结果数组
│   ├── rm_vision_demo/                  # 阶段零：ROS2 Pub/Sub
│   ├── rm_vision_phase1/                # 阶段一：HSV 灯条检测
│   ├── rm_vision_phase2/                # 阶段二：YOLOv8 训练
│   └── rm_vision_phase3/                # 阶段三/四/五：ROS2 集成
│       ├── rm_vision_phase3/
│       │   ├── camera_node.py           #   相机/视频采集
│       │   ├── detector_node.py         #   YOLO 推理
│       │   ├── tracker_node.py          #   卡尔曼追踪
│       │   ├── visualizer_node.py       #   可视化
│       │   └── utils.py                 #   绘制工具
│       ├── launch/
│       │   └── vision_pipeline.launch.py #   一键启动
│       ├── config/
│       │   └── params.yaml              #   参数配置
│       └── rviz/
│           └── vision_pipeline.rviz     #   RViz 布局
├── .gitignore
└── README.md
```

## 代码规范

Python 节点统一使用 Google 风格 docstring，类名 PascalCase，函数名 snake_case，参数通过 declare_parameter 声明配合 params.yaml 加载，日志分级使用 info/warn/error，关键路径有异常处理和资源释放（destroy_node 中释放摄像头和模型）。
