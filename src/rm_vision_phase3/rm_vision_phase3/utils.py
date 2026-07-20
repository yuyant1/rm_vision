"""RoboMaster 装甲板检测可视化工具。

提供类别名称映射、颜色调色板和检测框绘制函数，
被 detector_node 和 visualizer_node 共用。
"""

from typing import Dict, List, Tuple

import cv2
import numpy as np

# ── 类别定义（与阶段二训练数据一致）──────────────────────────────
CLASS_NAMES: Dict[int, str] = {
    0: "red_1",
    1: "blue_1",
    2: "red_3",
    3: "blue_3",
    4: "red_4",
    5: "blue_4",
    6: "red_sentry",
    7: "blue_sentry",
}

# ── BGR 颜色（红方暖色，蓝方冷色，方便人眼区分）──────────────────
CLASS_COLORS: Dict[int, Tuple[int, int, int]] = {
    0: (48, 48, 255),     # red_1     亮红
    1: (255, 128, 0),     # blue_1    橙蓝
    2: (0, 180, 255),     # red_3     暖橙
    3: (0, 200, 0),       # blue_3    绿
    4: (255, 100, 100),   # red_4     粉红
    5: (220, 200, 0),     # blue_4    青金
    6: (180, 80, 255),    # red_sentry  紫红
    7: (255, 0, 180),     # blue_sentry 品红
}


def draw_detections(
    image: np.ndarray,
    class_ids: List[int],
    boxes_xyxy: np.ndarray,
    confidences: List[float],
    line_thickness: int = 2,
) -> np.ndarray:
    """在图片上绘制检测框和标签。

    Parameters
    ----------
    image : np.ndarray
        BGR 格式的输入图片（会被原地修改）。
    class_ids, boxes_xyxy, confidences :
        YOLO 推理输出，一一对应。
    line_thickness : int
        框线粗细，默认 2 px。

    Returns
    -------
    np.ndarray
        绘制完成的图片（与输入是同一个数组引用）。
    """
    for class_id, box, conf in zip(class_ids, boxes_xyxy, confidences):
        x1, y1, x2, y2 = map(int, box)
        color = CLASS_COLORS.get(class_id, (128, 128, 128))
        label = f"{CLASS_NAMES.get(class_id, '?')} {conf:.2f}"

        # 外框
        cv2.rectangle(image, (x1, y1), (x2, y2), color, line_thickness)

        # 标签背景
        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        text_y1 = max(y1 - th - baseline - 6, 0)
        text_y2 = text_y1 + th + baseline + 6
        cv2.rectangle(
            image,
            (x1, text_y1),
            (x1 + tw + 8, text_y2),
            color,
            -1,
        )

        # 标签文字（白色）
        cv2.putText(
            image,
            label,
            (x1 + 4, text_y2 - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return image
