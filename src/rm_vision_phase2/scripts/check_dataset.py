#!/usr/bin/env python3

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_CONFIG = PROJECT_ROOT / "configs" / "rm_armor.yaml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "dataset_check"

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

COLORS = [
    (48, 48, 255),
    (255, 128, 0),
    (0, 180, 255),
    (0, 200, 0),
    (220, 200, 0),
    (255, 100, 100),
    (180, 80, 255),
    (255, 0, 180),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="检查并可视化 RoboMaster YOLO 数据集"
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=DEFAULT_DATA_CONFIG,
        help="YOLO 数据配置文件路径",
    )
    parser.add_argument(
        "--split",
        choices=("train", "val"),
        default="train",
        help="要检查的数据集划分",
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=2,
        help="每个类别可视化多少张图片",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="检查结果输出目录",
    )
    return parser.parse_args()


def load_config(config_path):
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    required_keys = {"path", "train", "val", "names"}
    missing_keys = required_keys - config.keys()
    if missing_keys:
        raise ValueError(f"配置文件缺少字段: {sorted(missing_keys)}")

    raw_names = config["names"]
    if isinstance(raw_names, dict):
        class_names = {
            int(class_id): str(class_name)
            for class_id, class_name in raw_names.items()
        }
    else:
        class_names = {
            class_id: str(class_name)
            for class_id, class_name in enumerate(raw_names)
        }

    return config, class_names


def read_label(label_path, class_names):
    annotations = []
    errors = []

    with label_path.open("r", encoding="utf-8") as file:
        lines = file.readlines()

    for line_number, line in enumerate(lines, start=1):
        line = line.strip()
        if not line:
            continue

        parts = line.split()
        if len(parts) != 5:
            errors.append(
                f"{label_path}:{line_number}: "
                f"应有5列，实际有{len(parts)}列"
            )
            continue

        try:
            class_id = int(parts[0])
            center_x, center_y, box_width, box_height = map(
                float, parts[1:]
            )
        except ValueError:
            errors.append(
                f"{label_path}:{line_number}: 包含无法解析的数字"
            )
            continue

        if class_id not in class_names:
            errors.append(
                f"{label_path}:{line_number}: 未知类别ID {class_id}"
            )
            continue

        coordinates_valid = (
            0.0 <= center_x <= 1.0
            and 0.0 <= center_y <= 1.0
            and 0.0 < box_width <= 1.0
            and 0.0 < box_height <= 1.0
        )

        if not coordinates_valid:
            errors.append(
                f"{label_path}:{line_number}: 坐标或宽高超出范围"
            )
            continue

        annotations.append(
            (
                class_id,
                center_x,
                center_y,
                box_width,
                box_height,
            )
        )

    return annotations, errors


def draw_annotations(image, annotations, class_names):
    annotated_image = image.copy()
    image_height, image_width = annotated_image.shape[:2]

    for annotation in annotations:
        class_id, center_x, center_y, box_width, box_height = annotation

        left = int((center_x - box_width / 2) * image_width)
        top = int((center_y - box_height / 2) * image_height)
        right = int((center_x + box_width / 2) * image_width)
        bottom = int((center_y + box_height / 2) * image_height)

        left = max(0, min(left, image_width - 1))
        top = max(0, min(top, image_height - 1))
        right = max(0, min(right, image_width - 1))
        bottom = max(0, min(bottom, image_height - 1))

        color = COLORS[class_id % len(COLORS)]
        label = f"{class_id}: {class_names[class_id]}"

        cv2.rectangle(
            annotated_image,
            (left, top),
            (right, bottom),
            color,
            3,
        )

        text_size, baseline = cv2.getTextSize(
            label,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            2,
        )

        text_width, text_height = text_size
        text_top = max(0, top - text_height - baseline - 6)
        text_bottom = text_top + text_height + baseline + 6

        cv2.rectangle(
            annotated_image,
            (left, text_top),
            (left + text_width + 8, text_bottom),
            color,
            -1,
        )

        cv2.putText(
            annotated_image,
            label,
            (left + 4, text_bottom - baseline - 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return annotated_image


def make_tile(image, title, tile_width=640, tile_height=400):
    header_height = 40
    content_height = tile_height - header_height

    tile = np.full(
        (tile_height, tile_width, 3),
        28,
        dtype=np.uint8,
    )

    image_height, image_width = image.shape[:2]
    scale = min(
        tile_width / image_width,
        content_height / image_height,
    )

    resized_width = max(1, int(image_width * scale))
    resized_height = max(1, int(image_height * scale))

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=cv2.INTER_AREA,
    )

    offset_x = (tile_width - resized_width) // 2
    offset_y = header_height + (content_height - resized_height) // 2

    tile[
        offset_y:offset_y + resized_height,
        offset_x:offset_x + resized_width,
    ] = resized

    cv2.putText(
        tile,
        title,
        (10, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return tile


def choose_samples(candidates, samples_per_class):
    if not candidates:
        return []

    ranked = sorted(candidates, key=lambda item: item[0])

    if samples_per_class == 1:
        return [ranked[-1]]

    fractions = np.linspace(
        0.5,
        1.0,
        min(samples_per_class, len(ranked)),
    )

    selected = []
    selected_indexes = set()

    for fraction in fractions:
        index = round(float(fraction) * (len(ranked) - 1))
        if index not in selected_indexes:
            selected.append(ranked[index])
            selected_indexes.add(index)

    return selected


def main():
    args = parse_args()

    config_path = args.data.expanduser().resolve()
    config, class_names = load_config(config_path)

    dataset_root = Path(config["path"]).expanduser().resolve()
    images_dir = dataset_root / config[args.split]
    labels_dir = images_dir.parent / "labels"

    output_dir = args.output.expanduser().resolve() / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    if not images_dir.is_dir():
        raise FileNotFoundError(f"图片目录不存在: {images_dir}")

    if not labels_dir.is_dir():
        raise FileNotFoundError(f"标注目录不存在: {labels_dir}")

    image_paths = sorted(
        path
        for path in images_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    label_paths = sorted(labels_dir.glob("*.txt"))

    images_by_stem = {
        image_path.stem: image_path
        for image_path in image_paths
    }
    labels_by_stem = {
        label_path.stem: label_path
        for label_path in label_paths
    }

    missing_labels = sorted(
        set(images_by_stem) - set(labels_by_stem)
    )
    missing_images = sorted(
        set(labels_by_stem) - set(images_by_stem)
    )

    class_counts = Counter()
    class_candidates = defaultdict(list)
    empty_label_count = 0
    all_errors = []

    for label_path in label_paths:
        annotations, errors = read_label(label_path, class_names)
        all_errors.extend(errors)

        if not annotations:
            empty_label_count += 1
            continue

        image_path = images_by_stem.get(label_path.stem)
        if image_path is None:
            continue

        for annotation in annotations:
            class_id = annotation[0]
            box_area = annotation[3] * annotation[4]

            class_counts[class_id] += 1
            class_candidates[class_id].append(
                (box_area, image_path, annotations)
            )

    print("=" * 60)
    print(f"数据配置: {config_path}")
    print(f"检查划分: {args.split}")
    print(f"图片目录: {images_dir}")
    print(f"标注目录: {labels_dir}")
    print(f"图片数量: {len(image_paths)}")
    print(f"标注数量: {len(label_paths)}")
    print(f"空标注数量: {empty_label_count}")
    print(f"缺少标注的图片: {len(missing_labels)}")
    print(f"缺少图片的标注: {len(missing_images)}")
    print(f"非法标注行数: {len(all_errors)}")
    print("-" * 60)

    for class_id, class_name in sorted(class_names.items()):
        print(
            f"class {class_id:<2} "
            f"{class_name:<15} "
            f"objects={class_counts[class_id]}"
        )

    if missing_labels:
        print("缺少标注的图片示例:", missing_labels[:10])

    if missing_images:
        print("缺少图片的标注示例:", missing_images[:10])

    if all_errors:
        print("非法标注示例:")
        for error in all_errors[:20]:
            print("  ", error)

    tiles = []

    for class_id, class_name in sorted(class_names.items()):
        selected_samples = choose_samples(
            class_candidates[class_id],
            args.samples_per_class,
        )

        if not selected_samples:
            print(f"警告：类别 {class_id} 没有可视化样本")
            continue

        for sample_index, sample in enumerate(
            selected_samples,
            start=1,
        ):
            _, image_path, annotations = sample
            image = cv2.imread(str(image_path))

            if image is None:
                print(f"警告：无法读取图片 {image_path}")
                continue

            annotated_image = draw_annotations(
                image,
                annotations,
                class_names,
            )

            output_name = (
                f"class_{class_id}_{class_name}_"
                f"{sample_index}_{image_path.stem}.jpg"
            )
            output_path = output_dir / output_name

            cv2.imwrite(str(output_path), annotated_image)

            title = (
                f"focus {class_id}: {class_name} | "
                f"{image_path.name}"
            )
            tiles.append(make_tile(annotated_image, title))

    if tiles:
        columns = 4
        rows = (len(tiles) + columns - 1) // columns

        blank_tile = np.full_like(tiles[0], 28)

        while len(tiles) < rows * columns:
            tiles.append(blank_tile.copy())

        contact_rows = []

        for row_index in range(rows):
            row_tiles = tiles[
                row_index * columns:(row_index + 1) * columns
            ]
            contact_rows.append(np.hstack(row_tiles))

        contact_sheet = np.vstack(contact_rows)
        contact_path = output_dir / f"{args.split}_label_samples.jpg"

        cv2.imwrite(str(contact_path), contact_sheet)
        print("-" * 60)
        print(f"标注样本拼图已保存: {contact_path}")

    summary_path = output_dir / f"{args.split}_dataset_summary.txt"

    summary_lines = [
        f"split: {args.split}",
        f"images: {len(image_paths)}",
        f"labels: {len(label_paths)}",
        f"empty_labels: {empty_label_count}",
        f"missing_labels: {len(missing_labels)}",
        f"missing_images: {len(missing_images)}",
        f"invalid_rows: {len(all_errors)}",
    ]

    for class_id, class_name in sorted(class_names.items()):
        summary_lines.append(
            f"class_{class_id}_{class_name}: "
            f"{class_counts[class_id]}"
        )

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print(f"统计结果已保存: {summary_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
