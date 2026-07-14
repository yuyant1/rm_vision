import cv2
import numpy as np

img = cv2.imread('/home/yyr/robomaster_ws/src/rm_vision_phase1/images/OIP-C.webp')
if img is None:
    print("图片读取失败！检查路径")
    exit(1)

cv2.namedWindow('HSV Tuner', cv2.WINDOW_GUI_EXPANDED)

def nop(x): pass

# 通用 H/S/V（蓝色和红色第一段共用）
cv2.createTrackbar('H1_min', 'HSV Tuner', 0, 179, nop)
cv2.createTrackbar('H1_max', 'HSV Tuner', 15, 179, nop)
cv2.createTrackbar('S_min', 'HSV Tuner', 122, 255, nop)
cv2.createTrackbar('S_max', 'HSV Tuner', 255, 255, nop)
cv2.createTrackbar('V_min', 'HSV Tuner', 92, 255, nop)
cv2.createTrackbar('V_max', 'HSV Tuner', 255, 255, nop)

# 红色第二段 H（仅红色模式生效）
cv2.createTrackbar('H2_min', 'HSV Tuner', 160, 179, nop)
cv2.createTrackbar('H2_max', 'HSV Tuner', 179, 179, nop)

# 颜色切换：0=红, 1=蓝
cv2.createTrackbar('Color', 'HSV Tuner', 0, 1, nop)

print("Color: 0=红色(两段H)  1=蓝色(一段H)")
print("H2_min/H2_max 仅红色模式生效")
print("按 q 退出")

while True:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    color = cv2.getTrackbarPos('Color', 'HSV Tuner')
    h1_min = cv2.getTrackbarPos('H1_min', 'HSV Tuner')
    h1_max = cv2.getTrackbarPos('H1_max', 'HSV Tuner')
    s_min  = cv2.getTrackbarPos('S_min', 'HSV Tuner')
    s_max  = cv2.getTrackbarPos('S_max', 'HSV Tuner')
    v_min  = cv2.getTrackbarPos('V_min', 'HSV Tuner')
    v_max  = cv2.getTrackbarPos('V_max', 'HSV Tuner')

    lower1 = np.array([h1_min, s_min, v_min])
    upper1 = np.array([h1_max, s_max, v_max])
    mask = cv2.inRange(hsv, lower1, upper1)

    if color == 0:  # 红色：合并第二段 H
        h2_min = cv2.getTrackbarPos('H2_min', 'HSV Tuner')
        h2_max = cv2.getTrackbarPos('H2_max', 'HSV Tuner')
        lower2 = np.array([h2_min, s_min, v_min])
        upper2 = np.array([h2_max, s_max, v_max])
        mask2 = cv2.inRange(hsv, lower2, upper2)
        mask = cv2.bitwise_or(mask, mask2)

    result = cv2.bitwise_and(img, img, mask=mask)
    display = np.hstack([img, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), result])
    cv2.imshow('HSV Tuner', display)

    key = cv2.waitKey(50) & 0xFF
    if key == ord('q'):
        break

if color == 0:
    h2_min = cv2.getTrackbarPos('H2_min', 'HSV Tuner')
    h2_max = cv2.getTrackbarPos('H2_max', 'HSV Tuner')
    print(f"红色阈值1: lower=[{h1_min}, {s_min}, {v_min}], upper=[{h1_max}, {s_max}, {v_max}]")
    print(f"红色阈值2: lower=[{h2_min}, {s_min}, {v_min}], upper=[{h2_max}, {s_max}, {v_max}]")
else:
    print(f"蓝色阈值: lower=[{h1_min}, {s_min}, {v_min}], upper=[{h1_max}, {s_max}, {v_max}]")

cv2.destroyAllWindows()
