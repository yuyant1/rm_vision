#!/usr/bin/env python3

import argparse
import os
import shutil
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch
import yaml
from ultralytics import YOLO


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "train_baseline.yaml"


def parse_args():
    parser = argparse.ArgumentParser(
        description="训练 RoboMaster 装甲板 YOLOv8 检测模型"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="训练配置文件路径",
    )
    parser.add_argument("--model", help="覆盖预训练模型或模型配置")
    parser.add_argument("--epochs", type=int, help="覆盖训练轮数")
    parser.add_argument("--imgsz", type=int, help="覆盖输入图片尺寸")
    parser.add_argument("--batch", type=int, help="覆盖批大小")
    parser.add_argument("--device", help="覆盖训练设备，例如 0 或 cpu")
    parser.add_argument("--workers", type=int, help="覆盖数据加载进程数")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="启用或禁用自动混合精度，例如 --no-amp",
    )
    parser.add_argument(
        "--fraction",
        type=float,
        help="使用训练集的比例，范围为 (0, 1]",
    )
    parser.add_argument("--name", help="覆盖实验名称")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="允许复用同名实验目录",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        help="从最近一次或指定的 last.pt 继续训练",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只显示最终配置，不启动训练",
    )
    return parser.parse_args()


def load_training_config(config_path):
    with config_path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if not isinstance(config, dict):
        raise ValueError(f"训练配置必须是 YAML 字典: {config_path}")

    required_keys = {"model", "data", "epochs", "imgsz", "batch"}
    missing_keys = required_keys - config.keys()
    if missing_keys:
        raise ValueError(f"训练配置缺少字段: {sorted(missing_keys)}")

    return config


def resolve_project_path(value):
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def resolve_model(value):
    path = Path(value).expanduser()
    if path.is_absolute():
        return str(path)

    project_path = PROJECT_ROOT / path
    if project_path.exists():
        return str(project_path.resolve())

    return value


def apply_overrides(config, args):
    override_keys = (
        "model",
        "epochs",
        "imgsz",
        "batch",
        "device",
        "workers",
        "amp",
        "fraction",
        "name",
    )

    for key in override_keys:
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    if args.exist_ok:
        config["exist_ok"] = True

    if args.resume is not None:
        config["resume"] = args.resume

    return config


def validate_config(config):
    if int(config["epochs"]) < 1:
        raise ValueError("epochs 必须大于等于 1")

    if int(config["imgsz"]) < 32:
        raise ValueError("imgsz 必须大于等于 32")

    if int(config["batch"]) < 1:
        raise ValueError("batch 必须大于等于 1")

    fraction = float(config.get("fraction", 1.0))
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction 必须位于 (0, 1] 范围内")

    data_path = resolve_project_path(config["data"])
    if not data_path.is_file():
        raise FileNotFoundError(f"数据配置不存在: {data_path}")

    config["data"] = str(data_path)
    config["project"] = str(
        resolve_project_path(config.get("project", "logs/train"))
    )
    config["model"] = resolve_model(config["model"])

    resume = config.get("resume")
    if isinstance(resume, str):
        resume_path = resolve_project_path(resume)
        if not resume_path.is_file():
            raise FileNotFoundError(f"续训权重不存在: {resume_path}")
        config["resume"] = str(resume_path)

    device = str(config.get("device", "")).lower()
    if device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError(
            "当前进程无法使用 CUDA。请确认 NVIDIA 驱动正常，"
            "并在真实终端环境运行训练。"
        )

    return config


def print_environment():
    print("=" * 72)
    print(f"项目目录: {PROJECT_ROOT}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"训练显卡: {torch.cuda.get_device_name(0)}")
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"显存容量: {memory_gb:.1f} GB")
    print("=" * 72)


def copy_best_weight(results, experiment_name):
    run_dir = Path(results.save_dir)
    best_weight = run_dir / "weights" / "best.pt"

    if not best_weight.is_file():
        print(f"未找到最优权重: {best_weight}")
        return

    weights_dir = PROJECT_ROOT / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    target_weight = weights_dir / f"{experiment_name}_best.pt"
    shutil.copy2(best_weight, target_weight)
    print(f"最优权重已复制到: {target_weight}")


def main():
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = load_training_config(config_path)
    config = apply_overrides(config, args)

    if args.dry_run:
        config["data"] = str(resolve_project_path(config["data"]))
        config["project"] = str(
            resolve_project_path(config.get("project", "logs/train"))
        )
        config["model"] = resolve_model(config["model"])
        print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
        return

    config = validate_config(config)
    model_source = config.pop("model")
    experiment_name = str(config.get("name", "train"))

    print_environment()
    print("最终训练参数:")
    print(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))

    model = YOLO(model_source)
    results = model.train(**config)

    print(f"训练结果目录: {results.save_dir}")
    copy_best_weight(results, experiment_name)


if __name__ == "__main__":
    main()
