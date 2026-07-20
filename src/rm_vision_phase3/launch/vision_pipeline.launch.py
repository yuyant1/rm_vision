"""RoboMaster 装甲板检测 Pipeline 一键启动。

用法:
  ros2 launch rm_vision_phase3 vision_pipeline.launch.py
  ros2 launch rm_vision_phase3 vision_pipeline.launch.py video_path:=/path/to/video.mp4
  ros2 launch rm_vision_phase3 vision_pipeline.launch.py confidence_threshold:=0.7
"""

import os
from pathlib import Path

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, LogInfo
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node


def generate_launch_description():
    # ── 包路径 ──────────────────────────────────────────
    pkg_dir = get_package_share_directory("rm_vision_phase3")
    default_params = os.path.join(pkg_dir, "config", "params.yaml")

    # ── Launch 参数声明 ─────────────────────────────────
    camera_id_arg = DeclareLaunchArgument(
        "camera_id", default_value="0",
        description="摄像头设备号"
    )
    video_path_arg = DeclareLaunchArgument(
        "video_path", default_value="",
        description="视频文件路径（优先级高于 camera_id）"
    )
    fps_arg = DeclareLaunchArgument(
        "fps", default_value="30",
        description="发布帧率"
    )
    model_path_arg = DeclareLaunchArgument(
        "model_path", default_value="",
        description="YOLO 权重路径（空=使用默认）"
    )
    conf_arg = DeclareLaunchArgument(
        "confidence_threshold", default_value="0.5",
        description="检测置信度阈值"
    )
    device_arg = DeclareLaunchArgument(
        "device", default_value="cuda",
        description="推理设备 cuda/cpu"
    )
    enable_viz_arg = DeclareLaunchArgument(
        "enable_visualization", default_value="true",
        description="是否启用可视化节点"
    )
    max_lost_arg = DeclareLaunchArgument(
        "max_lost_frames", default_value="10",
        description="目标丢失多少帧后删除"
    )
    pred_steps_arg = DeclareLaunchArgument(
        "prediction_steps", default_value="10",
        description="前向预测步数(提前量)"
    )

    # ── 节点定义 ────────────────────────────────────────
    camera_node = Node(
        package="rm_vision_phase3",
        executable="camera_node",
        name="camera_node",
        output="screen",
        parameters=[default_params, {
            "camera_id": LaunchConfiguration("camera_id"),
            "video_path": LaunchConfiguration("video_path"),
            "fps": LaunchConfiguration("fps"),
        }],
    )

    detector_node = Node(
        package="rm_vision_phase3",
        executable="detector_node",
        name="detector_node",
        output="screen",
        parameters=[default_params, {
            "model_path": LaunchConfiguration("model_path"),
            "confidence_threshold": LaunchConfiguration("confidence_threshold"),
            "device": LaunchConfiguration("device"),
        }],
    )

    tracker_node = Node(
        package="rm_vision_phase3",
        executable="tracker_node",
        name="tracker_node",
        output="screen",
        parameters=[default_params, {
            "max_lost_frames": LaunchConfiguration("max_lost_frames"),
            "prediction_steps": LaunchConfiguration("prediction_steps"),
        }],
    )

    visualizer_node = Node(
        package="rm_vision_phase3",
        executable="visualizer_node",
        name="visualizer_node",
        output="screen",
        parameters=[default_params],
        condition=IfCondition(
            PythonExpression([
                "'", LaunchConfiguration("enable_visualization"), "' == 'true'"
            ])
        ),
    )

    return LaunchDescription([
        camera_id_arg,
        video_path_arg,
        fps_arg,
        model_path_arg,
        conf_arg,
        device_arg,
        enable_viz_arg,
        max_lost_arg,
        pred_steps_arg,
        camera_node,
        detector_node,
        tracker_node,
        visualizer_node,
        LogInfo(msg="Pipeline 启动完毕: camera → detector → tracker → visualizer"),
    ])
