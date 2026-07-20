# 阶段一：装甲板灯条检测

## 运行

```bash
cd rm_vision_phase1
python3 rm_vision_phase1/detector.py          # 蓝色
python3 rm_vision_phase1/detector.py red      # 红色
python3 rm_vision_phase1/hsv_tuner.py         # 调参工具
```

## 检测思路

第一步 HSV 阈值分割。转 HSV 后用 inRange 提颜色，红色需两段 mask 合并（色环两端），蓝色一段。形态学闭运算填洞、开运算去噪。

第二步轮廓筛选。findContours 找白色区域，convexHull 补凹陷，minAreaRect 得旋转矩形。筛面积（50+）、高度（10+）、长宽比（2~10）。

第三步灯条配对。两两匹配，角度差 15 度内、Y 差 30px 内、X 距 10~300px，综合评分取最优。

第四步画框。用平均角度画旋转矩形，框高取灯条高乘 2.25（LED 嵌板内，面板外框更高），输出中心坐标。

## 关键参数

HSV 阈值用 hsv_tuner.py 在测试图上调试得到。灯条筛选参数基于经验值排除背景噪点。配对约束来自装甲板实物几何关系。框高系数 2.25 因灯条只是装甲板正面中间一段，面板上下各有一段不发光的边框。

## 文件

```
rm_vision_phase1/
├── images/OIP-C.webp          # 测试图片
├── output/                    # 检测结果截图
├── rm_vision_phase1/
│   ├── hsv_tuner.py           # HSV 调参工具
│   └── detector.py            # 检测主程序
└── README.md
```
