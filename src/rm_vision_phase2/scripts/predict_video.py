#!/usr/bin/env python3

import argparse
import csv
import time
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "yolov8s_baseline_30_best.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "demo"

CLASS_COLORS = {
    0: (40, 40, 255),
    1: (255, 120, 30),
    2: (30, 80, 230),
    3: (255, 180, 20),
    4: (70, 70, 255),
    5: (255, 220, 80),
    6: (20, 20, 180),
    7: (180, 80, 20),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="使用 YOLOv8 对 RoboMaster 测试视频进行装甲板检测"
    )
    parser.add_argument(
        "source",
        help="输入视频路径，或摄像头编号，例如 0",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=DEFAULT_WEIGHTS,
        help="模型权重路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="输出视频路径，默认保存到 demo/<输入名>_detected.mp4",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="置信度阈值",
    )
    parser.add_argument(
        "--iou",
        type=float,
        default=0.7,
        help="NMS IoU 阈值",
    )
    parser.add_argument(
        "--class-aware-nms",
        action="store_true",
        help="只在同类别内做NMS；默认跨类别抑制重叠装甲板框",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="模型输入尺寸",
    )
    parser.add_argument(
        "--device",
        default="0",
        help="推理设备，例如 0 或 cpu",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="实时显示检测画面，按 q 退出",
    )
    parser.add_argument(
        "--realtime",
        action="store_true",
        help="配合 --show 按输入视频原始帧率播放，而不是按最大速度播放",
    )
    parser.add_argument(
        "--display-width",
        type=int,
        default=1280,
        help="预览窗口宽度，默认1280；0表示使用视频原始尺寸",
    )
    parser.add_argument(
        "--screenshot-interval",
        type=int,
        default=150,
        help="有检测结果时每隔多少帧保存截图，0 表示关闭",
    )
    parser.add_argument(
        "--max-screenshots",
        type=int,
        default=20,
        help="最多保存多少张演示截图",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=0,
        help="最多处理多少帧，0 表示处理完整视频",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="OpenCV 输出视频编码，默认 mp4v",
    )
    return parser.parse_args()


def validate_args(args):
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("conf 必须位于 [0, 1] 范围内")
    if not 0.0 <= args.iou <= 1.0:
        raise ValueError("iou 必须位于 [0, 1] 范围内")
    if args.imgsz < 32:
        raise ValueError("imgsz 必须大于等于 32")
    if args.screenshot_interval < 0:
        raise ValueError("screenshot-interval 不能小于 0")
    if args.max_screenshots < 0:
        raise ValueError("max-screenshots 不能小于 0")
    if args.max_frames < 0:
        raise ValueError("max-frames 不能小于 0")
    if args.display_width != 0 and args.display_width < 320:
        raise ValueError("display-width 必须为0或大于等于320")
    if args.realtime and not args.show:
        raise ValueError("--realtime 必须和 --show 一起使用")

    weights = args.weights.expanduser().resolve()
    if not weights.is_file():
        raise FileNotFoundError(f"模型权重不存在: {weights}")
    args.weights = weights

    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "当前进程无法使用 CUDA，请激活 pytorch-env，"
            "或使用 --device cpu。"
        )

    return args


def parse_source(source):
    if source.isdigit():
        return int(source), f"camera_{source}"

    source_path = Path(source).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"输入视频不存在: {source_path}")
    return str(source_path), source_path.stem


def prepare_output_paths(args, source_stem):
    DEFAULT_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if args.output is None:
        output_video = DEFAULT_OUTPUT_DIR / f"{source_stem}_detected.mp4"
    else:
        output_video = args.output.expanduser().resolve()

    output_video.parent.mkdir(parents=True, exist_ok=True)
    output_csv = output_video.with_suffix(".csv")
    output_summary = output_video.with_name(
        f"{output_video.stem}_summary.txt"
    )
    screenshot_dir = output_video.parent / f"{output_video.stem}_screenshots"
    screenshot_dir.mkdir(parents=True, exist_ok=True)

    return output_video, output_csv, output_summary, screenshot_dir


def color_for_class(class_id):
    return CLASS_COLORS.get(class_id, (0, 220, 220))


def draw_detection(frame, box, class_id, class_name, confidence):
    frame_height, frame_width = frame.shape[:2]
    left, top, right, bottom = box.astype(int)

    left = max(0, min(left, frame_width - 1))
    top = max(0, min(top, frame_height - 1))
    right = max(0, min(right, frame_width - 1))
    bottom = max(0, min(bottom, frame_height - 1))

    line_width = max(2, round(min(frame_width, frame_height) / 360))
    font_scale = max(0.5, min(frame_width, frame_height) / 1100)
    font_thickness = max(1, line_width - 1)
    color = color_for_class(class_id)
    label = f"{class_name} {confidence:.2f}"

    cv2.rectangle(
        frame,
        (left, top),
        (right, bottom),
        color,
        line_width,
        cv2.LINE_AA,
    )

    text_size, baseline = cv2.getTextSize(
        label,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        font_thickness,
    )
    text_width, text_height = text_size

    label_top = max(0, top - text_height - baseline - 8)
    label_bottom = label_top + text_height + baseline + 8
    label_right = min(frame_width - 1, left + text_width + 10)

    cv2.rectangle(
        frame,
        (left, label_top),
        (label_right, label_bottom),
        color,
        -1,
    )
    cv2.putText(
        frame,
        label,
        (left + 5, label_bottom - baseline - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA,
    )


def draw_status(frame, processing_fps, detection_count, frame_index):
    status = (
        f"FPS {processing_fps:.1f} | "
        f"Objects {detection_count} | Frame {frame_index}"
    )
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = max(0.55, min(frame.shape[:2]) / 1000)
    thickness = 2
    text_size, baseline = cv2.getTextSize(
        status,
        font,
        font_scale,
        thickness,
    )
    text_width, text_height = text_size

    cv2.rectangle(
        frame,
        (8, 8),
        (text_width + 24, text_height + baseline + 22),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        frame,
        status,
        (16, text_height + 15),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )


def create_video_writer(path, codec, fps, width, height):
    if len(codec) != 4:
        raise ValueError("codec 必须是4个字符，例如 mp4v 或 XVID")

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*codec),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(
            f"无法创建输出视频: {path}，可尝试 --codec XVID "
            "并将输出扩展名改为 .avi"
        )
    return writer


def write_summary(
    path,
    args,
    source,
    output_video,
    frame_count,
    source_fps,
    elapsed_seconds,
    inference_times,
    class_counts,
    screenshot_count,
):
    processing_fps = frame_count / elapsed_seconds if elapsed_seconds else 0.0
    mean_inference_ms = (
        float(np.mean(inference_times)) if inference_times else 0.0
    )

    lines = [
        f"source: {source}",
        f"weights: {args.weights}",
        f"output_video: {output_video}",
        f"frames_processed: {frame_count}",
        f"source_fps: {source_fps:.3f}",
        f"elapsed_seconds: {elapsed_seconds:.3f}",
        f"average_processing_fps: {processing_fps:.3f}",
        f"average_model_inference_ms: {mean_inference_ms:.3f}",
        f"confidence_threshold: {args.conf}",
        f"iou_threshold: {args.iou}",
        f"class_agnostic_nms: {not args.class_aware_nms}",
        f"image_size: {args.imgsz}",
        f"device: {args.device}",
        f"screenshots_saved: {screenshot_count}",
        "detections_by_class:",
    ]

    if class_counts:
        for class_name, count in sorted(class_counts.items()):
            lines.append(f"  {class_name}: {count}")
    else:
        lines.append("  none: 0")

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    args = validate_args(parse_args())
    source, source_stem = parse_source(args.source)
    output_video, output_csv, output_summary, screenshot_dir = (
        prepare_output_paths(args, source_stem)
    )

    model = YOLO(str(args.weights))
    class_names = model.names

    capture = cv2.VideoCapture(source)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频或摄像头: {args.source}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not source_fps or not np.isfinite(source_fps):
        source_fps = 30.0

    frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_width <= 0 or frame_height <= 0:
        capture.release()
        raise RuntimeError("无法读取输入视频的分辨率")

    writer = create_video_writer(
        output_video,
        args.codec,
        source_fps,
        frame_width,
        frame_height,
    )

    print("=" * 72)
    print(f"输入源: {args.source}")
    print(f"模型权重: {args.weights}")
    print(f"输入分辨率: {frame_width}x{frame_height}")
    print(f"输入帧率: {source_fps:.3f}")
    print(f"总帧数: {total_frames if total_frames > 0 else 'unknown'}")
    print(f"置信度阈值: {args.conf}")
    print(f"推理设备: {args.device}")
    print(f"输出视频: {output_video}")
    print("=" * 72)

    frame_index = 0
    screenshot_count = 0
    smoothed_fps = 0.0
    inference_times = []
    class_counts = Counter()
    started_at = time.perf_counter()

    csv_file = output_csv.open("w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        [
            "frame",
            "timestamp_seconds",
            "class_id",
            "class_name",
            "confidence",
            "x1",
            "y1",
            "x2",
            "y2",
        ]
    )

    try:
        while True:
            frame_started_at = time.perf_counter()
            success, frame = capture.read()
            if not success:
                break

            frame_index += 1
            result = model.predict(
                source=frame,
                conf=args.conf,
                iou=args.iou,
                agnostic_nms=not args.class_aware_nms,
                imgsz=args.imgsz,
                device=args.device,
                verbose=False,
            )[0]

            detections = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.detach().cpu().numpy()
                confidences = result.boxes.conf.detach().cpu().numpy()
                class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)

                for box, confidence, class_id in zip(
                    boxes,
                    confidences,
                    class_ids,
                ):
                    class_name = str(class_names[class_id])
                    detections.append(
                        (box, class_id, class_name, float(confidence))
                    )
                    class_counts[class_name] += 1

                    csv_writer.writerow(
                        [
                            frame_index,
                            f"{(frame_index - 1) / source_fps:.6f}",
                            class_id,
                            class_name,
                            f"{float(confidence):.6f}",
                            f"{float(box[0]):.2f}",
                            f"{float(box[1]):.2f}",
                            f"{float(box[2]):.2f}",
                            f"{float(box[3]):.2f}",
                        ]
                    )

            for box, class_id, class_name, confidence in detections:
                draw_detection(
                    frame,
                    box,
                    class_id,
                    class_name,
                    confidence,
                )

            inference_times.append(float(result.speed.get("inference", 0.0)))
            frame_elapsed = time.perf_counter() - frame_started_at
            instantaneous_fps = 1.0 / frame_elapsed if frame_elapsed else 0.0
            smoothed_fps = (
                instantaneous_fps
                if smoothed_fps == 0.0
                else 0.9 * smoothed_fps + 0.1 * instantaneous_fps
            )

            draw_status(
                frame,
                smoothed_fps,
                len(detections),
                frame_index,
            )
            writer.write(frame)

            should_save_screenshot = (
                detections
                and args.screenshot_interval > 0
                and frame_index % args.screenshot_interval == 0
                and screenshot_count < args.max_screenshots
            )
            if should_save_screenshot:
                screenshot_count += 1
                screenshot_path = screenshot_dir / (
                    f"frame_{frame_index:06d}.jpg"
                )
                cv2.imwrite(str(screenshot_path), frame)

            if args.show:
                display_frame = frame
                if args.display_width and frame.shape[1] > args.display_width:
                    display_scale = args.display_width / frame.shape[1]
                    display_height = round(frame.shape[0] * display_scale)
                    display_frame = cv2.resize(
                        frame,
                        (args.display_width, display_height),
                        interpolation=cv2.INTER_AREA,
                    )

                cv2.imshow("RoboMaster Armor Detection", display_frame)
                wait_time_ms = 1
                if args.realtime:
                    frame_interval = 1.0 / source_fps
                    remaining_time = frame_interval - (
                        time.perf_counter() - frame_started_at
                    )
                    wait_time_ms = max(1, round(remaining_time * 1000))

                if cv2.waitKey(wait_time_ms) & 0xFF == ord("q"):
                    print("用户按 q，提前结束推理")
                    break

            if frame_index % 100 == 0:
                if total_frames > 0:
                    progress = 100.0 * frame_index / total_frames
                    print(
                        f"已处理 {frame_index}/{total_frames} 帧 "
                        f"({progress:.1f}%)，处理FPS {smoothed_fps:.1f}"
                    )
                else:
                    print(
                        f"已处理 {frame_index} 帧，"
                        f"处理FPS {smoothed_fps:.1f}"
                    )

            if args.max_frames and frame_index >= args.max_frames:
                print(f"达到 max-frames={args.max_frames}，提前结束")
                break
    finally:
        elapsed_seconds = time.perf_counter() - started_at
        capture.release()
        writer.release()
        csv_file.close()
        if args.show:
            cv2.destroyAllWindows()

    write_summary(
        output_summary,
        args,
        args.source,
        output_video,
        frame_index,
        source_fps,
        elapsed_seconds,
        inference_times,
        class_counts,
        screenshot_count,
    )

    average_fps = frame_index / elapsed_seconds if elapsed_seconds else 0.0
    average_inference_ms = (
        float(np.mean(inference_times)) if inference_times else 0.0
    )

    print("=" * 72)
    print(f"处理完成，帧数: {frame_index}")
    print(f"总耗时: {elapsed_seconds:.3f} 秒")
    print(f"平均处理FPS: {average_fps:.3f}")
    print(f"平均模型推理耗时: {average_inference_ms:.3f} ms")
    print(f"输出视频: {output_video}")
    print(f"检测记录: {output_csv}")
    print(f"统计摘要: {output_summary}")
    print(f"演示截图目录: {screenshot_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
