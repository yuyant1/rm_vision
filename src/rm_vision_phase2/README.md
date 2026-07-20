# 阶段二：YOLOv8 装甲板检测

## 运行

```bash
cd rm_vision_phase2

# 训练
python3 scripts/train.py --config configs/train_baseline.yaml

# 推理（视频）
python3 scripts/predict_video.py --weights weights/yolov8s_baseline_30_best.pt --source <视频路径>

# 检查数据集
python3 scripts/check_dataset.py
```

## 检测思路

使用 YOLOv8s 模型训练 RoboMaster 装甲板检测器。数据集包含约 500 张标注图片，类别包括装甲板编号和哨兵。训练采用默认增强策略，最终模型 mAP@0.5 达标，对视频推理稳定输出检测框和置信度。

## 关键参数

模型 YOLOv8s（速度和精度折中），输入尺寸 640x640，置信度阈值 0.5。训练 30 轮，batch size 根据显存自动调整。设备使用 CUDA GPU。

## 文件

```
rm_vision_phase2/
├── configs/train_baseline.yaml           # 训练配置
├── scripts/train.py                      # 训练脚本
├── scripts/predict_video.py              # 推理脚本
├── weights/yolov8s_baseline_30_best.pt   # 最优权重
├── weights/pretrained/yolov8s.pt         # 预训练权重
├── demo/                                 # 演示视频 + 截图
├── logs/train/yolov8s_baseline_30/       # 训练日志 + 曲线
├── report/training_report.md             # 训练报告
└── results/dataset_check/                # 数据集检查结果
```
