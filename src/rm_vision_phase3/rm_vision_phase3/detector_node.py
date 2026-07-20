#!/usr/bin/env python3
"""Detector Node — 订阅图像话题，运行 YOLOv8 推理，发布检测结果。

订阅:
  /camera/image_raw  (sensor_msgs/Image)  输入图像

发布:
  /detections         (rm_msg/DetectionArray)  检测框列表
  /detections/debug   (sensor_msgs/Image)      带标注的调试图像（可选）

参数:
  model_path           : YOLO 权重文件路径
  confidence_threshold : 置信度阈值（默认 0.5）
  device               : 推理设备 cuda / cpu（默认 cuda）
  imgsz                : 推理图像尺寸（默认 640）
  publish_debug_image  : 是否发布带标注的调试图像
"""

import time
from pathlib import Path

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from rm_msg.msg import BBox2D, DetectionArray
from sensor_msgs.msg import Image
from std_msgs.msg import Header

from rm_vision_phase3.utils import draw_detections

# ── 默认模型路径（阶段二训练产出）────────────────────────────
_PHASE2_WEIGHTS = (
    Path(__file__).resolve().parents[2]
    / "rm_vision_phase2"
    / "weights"
    / "yolov8s_baseline_30_best.pt"
)


class DetectorNode(Node):
    def __init__(self):
        super().__init__("detector_node")

        # ── 声明参数 ──────────────────────────────────────
        self.declare_parameter("model_path", str(_PHASE2_WEIGHTS))
        self.declare_parameter("confidence_threshold", 0.5)
        self.declare_parameter("device", "cuda")
        self.declare_parameter("imgsz", 640)
        self.declare_parameter("publish_debug_image", False)

        model_path = (
            self.get_parameter("model_path").get_parameter_value().string_value
        )
        self._conf_thresh = (
            self.get_parameter("confidence_threshold")
            .get_parameter_value()
            .double_value
        )
        device = (
            self.get_parameter("device").get_parameter_value().string_value
        )
        self._imgsz = (
            self.get_parameter("imgsz").get_parameter_value().integer_value
        )
        self._publish_debug = (
            self.get_parameter("publish_debug_image")
            .get_parameter_value()
            .bool_value
        )

        # ── 加载模型 ──────────────────────────────────────
        if not Path(model_path).is_file():
            self.get_logger().fatal(f"模型权重不存在: {model_path}")
            raise FileNotFoundError(f"模型权重不存在: {model_path}")

        self.get_logger().info(f"正在加载模型: {model_path}")
        # 延迟导入，避免未安装 ultralytics 时 import 就报错
        from ultralytics import YOLO

        self._model = YOLO(model_path)
        self._model.to(device)
        self.get_logger().info(f"模型加载完毕，设备: {device}")

        # ── 订阅 + 发布 ───────────────────────────────────
        self._bridge = CvBridge()
        self._sub = self.create_subscription(
            Image, "/camera/image_raw", self._image_callback, 10
        )
        self._pub_det = self.create_publisher(
            DetectionArray, "/detections", 10
        )
        self._pub_debug_img = None
        if self._publish_debug:
            self._pub_debug_img = self.create_publisher(
                Image, "/detections/debug", 10
            )

        self._inference_times: list[float] = []
        self.get_logger().info("DetectorNode 已就绪，等待图像...")

    def _image_callback(self, msg: Image):
        """图像回调：执行推理并发布结果。"""
        t0 = time.monotonic()

        # ROS Image → OpenCV
        try:
            frame = self._bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"图像转换失败: {e}")
            return

        # YOLO 推理
        results = self._model.predict(
            frame,
            imgsz=self._imgsz,
            conf=self._conf_thresh,
            verbose=False,
        )
        result = results[0]

        # 解析检测结果
        if result.boxes is not None:
            boxes_xyxy = result.boxes.xyxy.cpu().numpy()       # (N, 4)
            class_ids = result.boxes.cls.cpu().int().numpy()    # (N,)
            confs = result.boxes.conf.cpu().numpy()             # (N,)
        else:
            boxes_xyxy = np.empty((0, 4))
            class_ids = np.empty((0,), dtype=int)
            confs = np.empty((0,))

        # 封装为 DetectionArray 消息
        det_msg = DetectionArray()
        det_msg.header = Header()
        det_msg.header.stamp = self.get_clock().now().to_msg()
        det_msg.header.frame_id = msg.header.frame_id

        for class_id, box, conf in zip(class_ids, boxes_xyxy, confs):
            bbox = BBox2D()
            bbox.class_id = int(class_id)
            bbox.class_name = ""
            bbox.x1 = float(box[0])
            bbox.y1 = float(box[1])
            bbox.x2 = float(box[2])
            bbox.y2 = float(box[3])
            bbox.confidence = float(conf)
            det_msg.detections.append(bbox)

        self._pub_det.publish(det_msg)

        # 可选调试图像
        if self._pub_debug_img is not None:
            annotated = draw_detections(
                frame.copy(), class_ids.tolist(), boxes_xyxy, confs.tolist()
            )
            debug_msg = self._bridge.cv2_to_imgmsg(annotated, encoding="bgr8")
            debug_msg.header = det_msg.header
            self._pub_debug_img.publish(debug_msg)

        elapsed_ms = (time.monotonic() - t0) * 1000
        self._inference_times.append(elapsed_ms)
        self.get_logger().info(
            f"检测到 {len(class_ids)} 个目标，"
            f"耗时 {elapsed_ms:.1f} ms "
            f"(avg {np.mean(self._inference_times[-30:]):.1f} ms)"
        )

    def destroy_node(self):
        # 打印统计信息
        if self._inference_times:
            self.get_logger().info(
                f"推理统计: avg={np.mean(self._inference_times):.1f}ms, "
                f"min={np.min(self._inference_times):.1f}ms, "
                f"max={np.max(self._inference_times):.1f}ms "
                f"(共 {len(self._inference_times)} 帧)"
            )
        super().destroy_node()


def main():
    rclpy.init()
    try:
        node = DetectorNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"DetectorNode 启动失败: {e}")
    else:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
