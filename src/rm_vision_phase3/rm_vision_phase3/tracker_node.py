#!/usr/bin/env python3
"""
Tracker Node — 卡尔曼滤波追踪 + 目标关联 + 丢失预测。

订阅:
  /detections  (rm_msg/DetectionArray)   YOLO 检测结果

发布:
  /tracked              (rm_msg/TrackedArray)   追踪结果(含速度+预测)
  /detections/tracked   (sensor_msgs/Image)     追踪可视化图像

参数:
  max_lost_frames   : 目标丢失多少帧后删除(默认 10)
  prediction_steps  : 向前预测步数(默认 10, 用于提前量计算)
  iou_threshold     : 匈牙利关联 IoU 阈值(默认 0.3)
"""
import time
import rclpy
import numpy as np
from cv_bridge import CvBridge
from rclpy.node import Node
from rm_msg.msg import BBox2D, DetectionArray, TrackedBBox, TrackedArray
from sensor_msgs.msg import Image
from std_msgs.msg import Header
from filterpy.kalman import KalmanFilter
from scipy.optimize import linear_sum_assignment


# ── 卡尔曼滤波器工厂 ──────────────────────────────────────
def make_kalman():
    """创建等速模型 KalmanFilter，状态 [x, y, vx, vy]，观测 [x, y]"""
    kf = KalmanFilter(dim_x=4, dim_z=2)
    kf.F = np.array([           # 状态转移矩阵 (等速模型)
        [1, 0, 1, 0],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=float)
    kf.H = np.array([           # 观测矩阵 (只观测位置)
        [1, 0, 0, 0],
        [0, 1, 0, 0],
    ], dtype=float)
    kf.P *= 1000.0              # 初始状态不确定性(大=更信观测)
    kf.R = np.eye(2) * 50.0     # 测量噪声协方差 (检测器抖动程度)
    kf.Q = np.eye(4) * 1.0      # 过程噪声协方差 (模型不准确度)
    return kf


# ── 单个追踪目标 ──────────────────────────────────────────
class Tracker:
    def __init__(self, tracking_id, bbox, stamp):
        self.id = tracking_id
        self.kf = make_kalman()
        cx = (bbox.x1 + bbox.x2) / 2.0
        cy = (bbox.y1 + bbox.y2) / 2.0
        self.kf.x = np.array([cx, cy, 0.0, 0.0])  # 初始位置, 速度未知=0
        self.bbox = bbox                             # 当前框
        self.lost_frames = 0                        # 连续丢失帧数
        self.last_seen = stamp                       # 最后看到的时间
        self.age = 1                                 # 存活帧数

    def predict(self):
        """纯预测一步，不引入观测"""
        self.kf.predict()
        self.bbox = self._state_to_bbox()
        self.lost_frames += 1

    def update(self, bbox):
        """用新检测更新"""
        cx = (bbox.x1 + bbox.x2) / 2.0
        cy = (bbox.y1 + bbox.y2) / 2.0
        self.kf.update(np.array([cx, cy]))
        self.bbox = bbox
        self.lost_frames = 0
        self.age += 1

    def predict_future(self, steps=10):
        """预测 steps 帧后的位置，用于提前量计算"""
        x = self.kf.x.copy()
        for _ in range(steps):
            x = self.kf.F @ x
        return x[0], x[1]

    def _state_to_bbox(self):
        """把滤波状态转成 BBox2D"""
        cx, cy = self.kf.x[0], self.kf.x[1]
        w = self.bbox.x2 - self.bbox.x1
        h = self.bbox.y2 - self.bbox.y1
        b = BBox2D()
        b.class_id = self.bbox.class_id
        b.class_name = self.bbox.class_name
        b.x1 = float(cx - w / 2)
        b.y1 = float(cy - h / 2)
        b.x2 = float(cx + w / 2)
        b.y2 = float(cy + h / 2)
        b.confidence = self.bbox.confidence
        return b

    def to_msg(self, prediction_steps=10):
        """输出为 TrackedBBox 消息"""
        t = TrackedBBox()
        t.tracking_id = self.id
        t.class_id = self.bbox.class_id
        t.class_name = self.bbox.class_name
        t.x1 = self.bbox.x1
        t.y1 = self.bbox.y1
        t.x2 = self.bbox.x2
        t.y2 = self.bbox.y2
        t.confidence = self.bbox.confidence
        t.vx = float(self.kf.x[2])
        t.vy = float(self.kf.x[3])
        px, py = self.predict_future(prediction_steps)
        t.predicted_x = float(px)
        t.predicted_y = float(py)
        return t


# ── 工具函数 ──────────────────────────────────────────────
def iou(a: BBox2D, b: BBox2D):
    """计算两个矩形框的 IoU"""
    x1 = max(a.x1, b.x1)
    y1 = max(a.y1, b.y1)
    x2 = min(a.x2, b.x2)
    y2 = min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = (a.x2 - a.x1) * (a.y2 - a.y1)
    area_b = (b.x2 - b.x1) * (b.y2 - b.y1)
    return inter / (area_a + area_b - inter + 1e-8)


def visualize_tracked(frame, tracked_list):
    """在图像上绘制追踪框、ID、速度和预测点"""
    import cv2
    result = frame.copy()
    for t in tracked_list:
        x1, y1 = int(t.x1), int(t.y1)
        x2, y2 = int(t.x2), int(t.y2)
        # 框 + ID
        cv2.rectangle(result, (x1, y1), (x2, y2), (255, 255, 0), 2)
        cv2.putText(result, f"ID:{t.tracking_id}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        # 速度方向箭头
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        vx_int = int(t.vx * 2)
        vy_int = int(t.vy * 2)
        if vx_int != 0 or vy_int != 0:
            cv2.arrowedLine(result, (cx, cy),
                            (cx + vx_int, cy + vy_int),
                            (0, 255, 255), 2, tipLength=0.3)
        # 预测点
        px, py = int(t.predicted_x), int(t.predicted_y)
        if not (x1 <= px <= x2 and y1 <= py <= y2):  # 预测点不在框内才画
            cv2.circle(result, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(result, "pred", (px + 6, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
    return result


# ── ROS2 节点 ─────────────────────────────────────────────
class TrackerNode(Node):
    def __init__(self):
        super().__init__("tracker_node")

        # 参数
        self.declare_parameter("max_lost_frames", 10)
        self.declare_parameter("prediction_steps", 10)
        self.declare_parameter("iou_threshold", 0.3)

        self._max_lost = (self.get_parameter("max_lost_frames")
                          .get_parameter_value().integer_value)
        self._pred_steps = (self.get_parameter("prediction_steps")
                            .get_parameter_value().integer_value)
        self._iou_thresh = (self.get_parameter("iou_threshold")
                            .get_parameter_value().double_value)

        # 内部状态
        self._trackers = []     # 活着的追踪器列表
        self._next_id = 0       # 自增 ID
        self._bridge = CvBridge()

        # 订阅 + 发布
        self._sub = self.create_subscription(
            DetectionArray, "/detections", self._callback, 10)
        self._pub_tracked = self.create_publisher(
            TrackedArray, "/tracked", 10)
        self._pub_viz = self.create_publisher(
            Image, "/detections/tracked", 10)

        # 为了画速度箭头需要原始图像，所以多订阅一个图像话题
        self._img_sub = self.create_subscription(
            Image, "/camera/image_raw", self._image_cb, 10)
        self._last_frame = None

        self.get_logger().info("TrackerNode 已就绪")

    def _image_cb(self, msg):
        """缓存最新帧用于可视化"""
        try:
            self._last_frame = self._bridge.imgmsg_to_cv2(
                msg, desired_encoding="bgr8")
        except Exception:
            pass

    def _callback(self, msg: DetectionArray):
        t0 = time.monotonic()

        dets = msg.detections   # 当前帧所有检测

        # 1. 所有现有追踪器做一次预测
        for trk in self._trackers:
            trk.predict()

        # 2. 匈牙利算法匹配：检测 vs 追踪器
        matched_pairs = self._match(dets, self._trackers)
        matched_det_idx = set()
        matched_trk_idx = set()

        for di, ti in matched_pairs:
            self._trackers[ti].update(dets[di])
            self._trackers[ti].last_seen = msg.header.stamp
            matched_det_idx.add(di)
            matched_trk_idx.add(ti)

        # 3. 未匹配的检测 → 创建新追踪器
        for di in range(len(dets)):
            if di not in matched_det_idx:
                trk = Tracker(self._next_id, dets[di], msg.header.stamp)
                self._trackers.append(trk)
                self._next_id += 1

        # 4. 删除长期丢失的追踪器
        alive = []
        for ti, trk in enumerate(self._trackers):
            if ti in matched_trk_idx:
                alive.append(trk)
            elif trk.lost_frames <= self._max_lost:
                alive.append(trk)   # 保留，用预测位置
        self._trackers = alive

        # 5. 发布追踪结果
        tracked_msg = TrackedArray()
        tracked_msg.header = Header()
        tracked_msg.header.stamp = self.get_clock().now().to_msg()
        tracked_msg.header.frame_id = msg.header.frame_id

        for trk in self._trackers:
            tracked_msg.tracked.append(trk.to_msg(self._pred_steps))

        self._pub_tracked.publish(tracked_msg)

        # 6. 发布可视化
        if self._last_frame is not None:
            viz = visualize_tracked(self._last_frame, tracked_msg.tracked)
            viz_msg = self._bridge.cv2_to_imgmsg(viz, encoding="bgr8")
            viz_msg.header = tracked_msg.header
            self._pub_viz.publish(viz_msg)

        elapsed = (time.monotonic() - t0) * 1000
        matched = len(matched_pairs)
        total = len(self._trackers)
        self.get_logger().info(
            f"追踪: {total} 目标 (匹配 {matched}, 新 {len(dets)-matched}, "
            f"丢失预测 {total-matched}), 耗时 {elapsed:.1f}ms"
        )

    def _match(self, dets, trackers):
        """匈牙利算法匹配，返回 [(det_idx, trk_idx), ...]"""
        nd, nt = len(dets), len(trackers)
        if nd == 0 or nt == 0:
            return []

        # 构建代价矩阵 (1 - IoU)
        cost = np.ones((nd, nt), dtype=float)
        for i, d in enumerate(dets):
            for j, t in enumerate(trackers):
                cost[i, j] = 1.0 - iou(d, t.bbox)

        # 匈牙利算法
        row_ind, col_ind = linear_sum_assignment(cost)

        # 只保留代价 < (1-threshold) 的匹配
        pairs = []
        for i, j in zip(row_ind, col_ind):
            if cost[i, j] < (1.0 - self._iou_thresh):
                pairs.append((i, j))
        return pairs


def main():
    rclpy.init()
    try:
        node = TrackerNode()
        rclpy.spin(node)
    except Exception as e:
        print(f"TrackerNode 启动失败: {e}")
    else:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
