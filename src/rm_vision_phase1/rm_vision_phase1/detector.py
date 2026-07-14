import cv2
import numpy as np
from enum import Enum

#HSV 阈值
RED1_LOWER  = np.array([0,   179, 147])
RED1_UPPER  = np.array([15,  255, 255])
RED2_LOWER  = np.array([158, 179, 147])
RED2_UPPER  = np.array([179, 255, 255])
BLUE_LOWER  = np.array([42,  110, 101])
BLUE_UPPER  = np.array([120, 255, 255])

#筛选
MIN_AREA      = 50      # 最小面积
MIN_HEIGHT    = 10      # 灯条最小高度(像素)
MIN_RATIO     = 2.0     # 长宽比最小值
MAX_RATIO     = 10.0    # 长宽比最大值

#配对参数
MAX_ANGLE_DIFF = 15     # 确保平行
MAX_Y_DIFF     = 30     # 确保同一中心线
MIN_X_DIST     = 10     # X方向最小距离
MAX_X_DIST     = 300    # X方向最大距离


class TargetColor(Enum):
    RED  = "red"
    BLUE = "blue"


def make_mask(frame, color: TargetColor):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    if color == TargetColor.RED:
        mask1 = cv2.inRange(hsv, RED1_LOWER, RED1_UPPER)
        mask2 = cv2.inRange(hsv, RED2_LOWER, RED2_UPPER)
        mask  = cv2.bitwise_or(mask1, mask2)
    else:
        mask = cv2.inRange(hsv, BLUE_LOWER, BLUE_UPPER)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=2)
    return mask


def find_bars(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bars = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA:
            continue

        cnt = cv2.convexHull(cnt)

        rect = cv2.minAreaRect(cnt)
        (cx, cy), (rw, rh), angle = rect

        # 统一: h = 长边, w = 短边
        if rw > rh:
            w, h = rh, rw
            angle += 90
        else:
            w, h = rw, rh

        if h < MIN_HEIGHT:
            continue
        ratio = h / w
        if ratio < MIN_RATIO or ratio > MAX_RATIO:
            continue

        bars.append((cx, cy, w, h, angle))

    return bars


def match_bars(bars):
    armors = []
    n = len(bars)
    used = [False] * n

    for i in range(n):
        if used[i]:
            continue
        cx1, cy1, w1, h1, a1 = bars[i]
        best_j, best_score = -1, float('inf')

        for j in range(i + 1, n):
            if used[j]:
                continue
            cx2, cy2, w2, h2, a2 = bars[j]

            angle_diff = abs(a1 - a2)
            if angle_diff > MAX_ANGLE_DIFF:
                continue

            y_diff = abs(cy1 - cy2)
            if y_diff > MAX_Y_DIFF:
                continue

            x_dist = abs(cx1 - cx2)
            if x_dist < MIN_X_DIST or x_dist > MAX_X_DIST:
                continue

            h_diff = abs(h1 - h2) / max(h1, h2)
            score = angle_diff + y_diff + h_diff * 50

            if score < best_score:
                best_score = score
                best_j = j

        if best_j >= 0:
            used[i] = used[best_j] = True
            cx2, cy2, _, h2, a2 = bars[best_j]
            avg_angle = (a1 + a2) / 2  
            armors.append((cx1, cy1, h1, cx2, cy2, h2, avg_angle))

    return armors


def draw_result(frame, bars, armors):
    result = frame.copy()

    # 画灯条
    for cx, cy, w, h, angle in bars:
        rect = ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(result, [box], 0, (0, 255, 0), 1)

    # 画装甲板
    for cx1, cy1, h1, cx2, cy2, h2, angle in armors:
        center_x = float((cx1 + cx2) / 2)
        center_y = float((cy1 + cy2) / 2)

        # 框的宽和高：灯条间距决定宽，灯条高度决定高
        bar_h = max(h1, h2)
        box_w = abs(cx1 - cx2) + bar_h * 0.3   # 两灯条间距 + 灯条宽度余量
        box_h = bar_h * 2.25                      # 灯条高度 + 上下余量

        # 画旋转矩形
        rect = ((center_x, center_y), (box_w, box_h), angle)
        box = cv2.boxPoints(rect)
        box = np.int32(box)
        cv2.drawContours(result, [box], 0, (0, 0, 255), 2)

        # 中心点 + 坐标文字
        cx = int(center_x)
        cy = int(center_y)
        cv2.circle(result, (cx, cy), 4, (0, 255, 255), -1)
        cv2.putText(result, f"({cx},{cy})",
                    (cx - 40, cy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    return result


def detect(frame, color: TargetColor):
    mask   = make_mask(frame, color)
    bars   = find_bars(mask)
    armors = match_bars(bars)
    result = draw_result(frame, bars, armors)

    centers = []
    for cx1, cy1, h1, cx2, cy2, h2, angle in armors:
        centers.append(((cx1 + cx2) / 2, (cy1 + cy2) / 2))

    return result, centers


def main():
    import sys

    color = TargetColor.BLUE
    if len(sys.argv) > 1 and sys.argv[1] == 'red':
        color = TargetColor.RED

    img = cv2.imread('/home/yyr/robomaster_ws/src/rm_vision_phase1/images/OIP-C.webp')
    if img is None:
        print("图片读取失败！")
        return

    result, centers = detect(img, color)
    print(f"[{color.value}] 检测到 {len(centers)} 个装甲板")
    for i, (cx, cy) in enumerate(centers):
        print(f"  装甲板 {i+1}: 中心=({cx:.1f}, {cy:.1f})")

    mask = make_mask(img, color)
    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    display = np.hstack([img, mask_bgr, result])
    scale = min(1000 / display.shape[1], 0.7)
    display = cv2.resize(display, None, fx=scale, fy=scale)
    cv2.imshow('Armor Detector', display)
    print("按任意键退出")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
