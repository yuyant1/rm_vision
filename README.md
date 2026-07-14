## robomaster_ws — 视觉组阶段零考核

## 环境

- Ubuntu 26.04
- ROS2 Lyrical
- Python 3.14

## 命令

```bash
# 1. 创建工作空间并克隆仓库
mkdir -p ~/robomaster_ws/src
cd ~/robomaster_ws/src
git clone https://github.com/yuyant1/rm_vision
cd ..

# 2. 编译
source /opt/ros/lyrical/setup.bash
colcon build

# 3. 运行
source install/setup.bash
ros2 launch rm_vision_demo vision_demo.launch.py
```

## 运行方式

```bash
# 默认频率（发布间隔 0.5 秒）
ros2 launch rm_vision_demo vision_demo.launch.py

# 自定义发布频率（0.1 秒一次）
ros2 launch rm_vision_demo vision_demo.launch.py publish_rate:=0.1

# 单独启动节点
ros2 run rm_vision_demo target_publisher
ros2 run rm_vision_demo target_subscriber
```

## 自定义消息

`rm_msg/msg/Target`：


