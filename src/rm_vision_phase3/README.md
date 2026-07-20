# 阶段三：ROS2 + 目标检测集成

## 运行

```bash
# 一键启动（摄像头或视频文件）
ros2 launch rm_vision_phase3 vision_pipeline.launch.py

# 使用视频文件
ros2 launch rm_vision_phase3 vision_pipeline.launch.py video_path:=/path/to/video.mp4

# 关闭可视化
ros2 launch rm_vision_phase3 vision_pipeline.launch.py enable_visualization:=false

# RViz 可视化
rviz2 -d src/rm_vision_phase3/rviz/vision_pipeline.rviz
```

## 架构

三节点 pipeline：

camera_node --/camera/image_raw--> detector_node --/detections--> visualizer_node --/detections/image_annotated-->

- camera_node：从摄像头或视频文件读取帧，发布 sensor_msgs/Image
- detector_node：订阅图像，运行 YOLOv8 推理，发布 DetectionArray 检测结果
- visualizer_node：同步图像和检测结果，画框发布标注图像

## 节点通信

```
[camera_node] ── image_raw ──> [detector_node] ── detections ──> [visualizer_node] ── image_annotated ──>
```

## 参数

所有参数通过 config/params.yaml 配置，launch 启动时可覆盖。

camera_node：camera_id（摄像头ID）、video_path（视频路径）、fps（帧率）、image_width/height（缩放）
detector_node：model_path（模型权重）、confidence_threshold（置信度）、device（cuda/cpu）、imgsz（推理尺寸）
visualizer_node：enable_visualization（开关）、line_thickness（框线粗细）

## 文件

```
rm_vision_phase3/
├── rm_vision_phase3/
│   ├── camera_node.py            # 相机/视频采集节点
│   ├── detector_node.py          # YOLOv8 推理节点
│   ├── visualizer_node.py        # 可视化节点
│   └── utils.py                  # 绘制工具
├── launch/vision_pipeline.launch.py  # 一键启动
├── config/params.yaml            # 参数配置
├── rviz/vision_pipeline.rviz     # RViz 配置
└── README.md
```
