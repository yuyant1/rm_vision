#!/usr/bin/env python3
"""Visualizer Node — 订阅图像和检测结果，绘制标注框后发布可视化图像。

订阅:
  /camera/image_raw  (sensor_msgs/Image)       原始图像
  /detections         (rm_msg/DetectionArray)   检测结果

发布:
  /detections/image_annotated  (sensor_msgs/Image)  带检测框的图像

参数:
  enable_visualization : 是否启用（默认 true），关闭时节点空转
  line_thickness       : 框线粗细（默认 2）
  queue_size           : 消息同步队列大小（默认 10）
"""

import numpy as np
import rclpy
from cv_bridge import CvBridge
from message_filters import ApproximateTimeSynchronizer, Subscriber
from rclpy.node import Node
from rm_msg.msg import DetectionArray
from sensor_msgs.msg import Image

from rm_vision_phase3.utils import draw_detections


class VisualizerNode(Node):
    def __init__(self):
        super().__init__("visualizer_node")

        # ── 声明参数 ──────────────────────────────────────
        self.declare_parameter("enable_visualization", True)
        self.declare_parameter("line_thickness", 2)
        self.declare_parameter("queue_size", 10)

        self._enabled = (
            self.get_parameter("enable_visualization")
            .get_parameter_value()
            .bool_value
        )
        line_thickness = (
            self.get_parameter("line_thickness")
            .get_parameter_value()
            .integer_value
        )
        queue_size = (
            self.get_parameter("queue_size")
            .get_parameter_value()
            .integer_value
        )

        if not self._enabled:
            self.get_logger().info("可视化已禁用，节点空转")
            return

        # ── 消息同步订阅 ──────────────────────────────────
        self._bridge = CvBridge()
        self._pub = self.create_publisher(
            Image, "/detections/image_annotated", 10
        )

        img_sub = Subscriber(self, Image, "/camera/image_raw")
        det_sub = Subscriber(self, DetectionArray, "/detections")

        self._sync = ApproximateTimeSynchronizer(
            [img_sub, det_sub],
            queue_size=queue_size,
            slop=0.1,  # 允许 100ms 时间偏差
        )
        self._sync.registerCallback(self._sync_callback)
        self._line_thickness = line_thickness

        self.get_logger().info("VisualizerNode 已就绪")

    def _sync_callback(self, img_msg: Image, det_msg: DetectionArray):
        """同步回调：图像和检测结果对齐后绘制。"""
        if not self._enabled:
            return

        try:
            frame = self._bridge.imgmsg_to_cv2(
                img_msg, desired_encoding="bgr8"
            )
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")
            return

        # 解析检测结果
        class_ids = [d.class_id for d in det_msg.detections]
        boxes_xyxy = np.array(
            [[d.x1, d.y1, d.x2, d.y2] for d in det_msg.detections],
            dtype=np.float32,
        )
        confs = [d.confidence for d in det_msg.detections]

        if len(boxes_xyxy) == 0:
            annotated = frame
        else:
            annotated = draw_detections(
                frame, class_ids, boxes_xyxy, confs, self._line_thickness
            )

        out_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
        out_msg.header = img_msg.header
        self._pub.publish(out_msg)

        self.get_logger().debug(
            f"可视化: {len(boxes_xyxy)} 个目标已绘制"
        )


def main():
    rclpy.init()
    try:
        node = VisualizerNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"VisualizerNode 启动失败: {e}")
    else:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
