#!/usr/bin/env python3
"""Camera Node — 从摄像头或视频文件读取帧，发布 sensor_msgs/Image。

话题:
  /camera/image_raw  (sensor_msgs/Image)  原始图像

参数（可在 params.yaml 或 launch 中覆盖）:
  camera_id     : 摄像头设备号（默认 0），video_path 非空时忽略
  video_path    : 视频文件路径，不为空时优先使用
  fps           : 发布帧率（默认 30）
  image_width   : 图像缩放宽度（默认 640, -1 保持原始）
  image_height  : 图像缩放高度（默认 480, -1 保持原始）
"""

import time
from pathlib import Path

import cv2
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")

        # ── 声明参数 ──────────────────────────────────────
        self.declare_parameter("camera_id", 0)
        self.declare_parameter("video_path", "")
        self.declare_parameter("fps", 30)
        self.declare_parameter("image_width", 640)
        self.declare_parameter("image_height", 480)

        camera_id = (
            self.get_parameter("camera_id").get_parameter_value().integer_value
        )
        video_path = (
            self.get_parameter("video_path").get_parameter_value().string_value
        )
        fps = self.get_parameter("fps").get_parameter_value().integer_value
        self._width = (
            self.get_parameter("image_width").get_parameter_value().integer_value
        )
        self._height = (
            self.get_parameter("image_height").get_parameter_value().integer_value
        )

        # ── 打开采集源 ────────────────────────────────────
        source = video_path if video_path else camera_id
        self._cap = cv2.VideoCapture(source)
        if not self._cap.isOpened():
            self.get_logger().fatal(f"无法打开采集源: {source}")
            raise RuntimeError(f"无法打开采集源: {source}")

        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        if actual_fps > 0:
            fps = int(actual_fps)
            self.get_logger().info(f"使用采集源原始帧率: {fps} fps")

        self.get_logger().info(f"采集源已打开: {source}")

        # ── 发布者 + 定时器 ───────────────────────────────
        self._bridge = CvBridge()
        self._publisher = self.create_publisher(Image, "/camera/image_raw", 10)
        period = 1.0 / max(fps, 1)
        self._timer = self.create_timer(period, self._publish_frame)
        self.get_logger().info(f"发布周期: {period:.3f}s ({fps} fps)")

    def _publish_frame(self):
        """定时器回调：读取一帧并发布。"""
        t0 = time.monotonic()
        ret, frame = self._cap.read()
        if not ret:
            self.get_logger().warn("读取帧失败，尝试重新打开采集源")
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self._cap.read()
            if not ret:
                self.get_logger().error("采集源已断，跳过本帧")
                return

        # 可选缩放
        if self._width > 0 and self._height > 0:
            frame = cv2.resize(frame, (self._width, self._height))

        msg = self._bridge.cv2_to_imgmsg(frame, encoding="bgr8")
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_frame"
        self._publisher.publish(msg)

        elapsed_ms = (time.monotonic() - t0) * 1000
        self.get_logger().debug(f"发布一帧，耗时 {elapsed_ms:.1f} ms")

    def destroy_node(self):
        self._cap.release()
        super().destroy_node()


def main():
    rclpy.init()
    try:
        node = CameraNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"CameraNode 启动失败: {e}")
    else:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
