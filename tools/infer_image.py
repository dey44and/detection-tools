#!/usr/bin/env python3
"""Run the trained YOLO model on one image and visualize its detections."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import torch
import yaml
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

from nets import nn as _nn
from utils.util import non_max_suppression


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path, help="image to run inference on")
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPO_ROOT / "weights" / "best.pt",
        help="checkpoint path (default: weights/best.pt)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=REPO_ROOT / "args.yaml",
        help="configuration containing image_size and names",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="confidence threshold")
    parser.add_argument("--iou", type=float, default=0.45, help="NMS IoU threshold")
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="inference device, for example cuda, cuda:0, or cpu",
    )
    parser.add_argument("--output", type=Path, help="optional annotated-image path")
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="do not open a display window (useful on remote machines)",
    )
    return parser.parse_args()


def preprocess(image_rgb, size: int) -> tuple[torch.Tensor, tuple[float, float], tuple[int, int]]:
    """Apply the same resize, square padding, and normalization as validation."""
    original_height, original_width = image_rgb.shape[:2]
    resize_ratio = max(original_height, original_width) / size
    resized_height = max(1, int(original_height / resize_ratio))
    resized_width = max(1, int(original_width / resize_ratio))

    # Compute padding
    pad_width = size - resized_width
    pad_height = size - resized_height
    left, top = pad_width // 2, pad_height // 2
    right, bottom = pad_width - left, pad_height - top

    # Generate the transformations sequence
    transformations = v2.Compose([
        v2.ToImage(),
        v2.Resize(
            (resized_height, resized_width),
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        ),
        v2.Pad([left, top, right, bottom], fill=114),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        )
    ])

    tensor = transformations(image_rgb)
    scale = (resized_width / original_width, resized_height / original_height)
    return tensor.unsqueeze(0), scale, (left, top)


def restore_boxes(
    detections: torch.Tensor,
    scale: tuple[float, float],
    padding: tuple[int, int],
    image_shape: tuple[int, ...],
) -> None:
    """Map boxes from the padded model input back to the original image."""
    scale_x, scale_y = scale
    left, top = padding
    detections[:, 0::2].sub_(left).div_(scale_x).clamp_(0, image_shape[1])
    detections[:, 1::2].sub_(top).div_(scale_y).clamp_(0, image_shape[0])


def draw_detections(image, detections: torch.Tensor, names: dict[int, str]):
    colors = ((255, 56, 56), (56, 255, 56), (56, 56, 255), (255, 180, 56))
    result = image.copy()

    # Scale text and line based on image size
    thickness = max(2, round(min(image.shape[:2]) / 300))
    font_scale = max(0.5, min(image.shape[:2]) / 1000)

    for x1, y1, x2, y2, confidence, class_id in detections.cpu().tolist():
        class_id = int(class_id)
        color = colors[class_id % len(colors)]
        label = f"{names[class_id]} {confidence:.2f}"
        point1, point2 = (round(x1), round(y1)), (round(x2), round(y2))
        cv2.rectangle(result, point1, point2, color, thickness)

        (text_width, text_height), _ = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )

        text_y = max(point1[1], text_height + 4)
        cv2.rectangle(result, (point1[0], text_y - text_height - 4), (point1[0] + text_width + 4, text_y), color, -1)
        cv2.putText(
            result,
            label,
            (point1[0] + 2, text_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return result


def main() -> None:
    args = parse_args()
    if not 0 <= args.conf <= 1 or not 0 <= args.iou <= 1:
        raise ValueError("--conf and --iou must be between 0 and 1")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    names = {int(class_id): name for class_id, name in config["names"].items()}
    size = int(config["image_size"])

    device = torch.device(args.device)
    checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
    model = checkpoint["model"]  # main.py stores the EMA model under this key.
    expected_class_ids = set(range(model.head.nc))
    if set(names) != expected_class_ids:
        raise ValueError(
            f"Checkpoint expects class IDs {sorted(expected_class_ids)}, "
            f"but config defines {sorted(names)}"
        )
    model = model.eval()
    model = model.half() if device.type == "cuda" else model.float()

    image_bgr = cv2.imread(str(args.image))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    tensor, scale, padding = preprocess(image_rgb, size)
    tensor = tensor.to(device)
    tensor = tensor.half() if device.type == "cuda" else tensor.float()

    with torch.inference_mode():
        prediction = model(tensor)
        detections = non_max_suppression(prediction, args.conf, args.iou)[0]
    detections = detections.clone()
    restore_boxes(detections[:, :4], scale, padding, image_bgr.shape)

    annotated = draw_detections(image_bgr, detections, names)
    print(f"Detected {len(detections)} object(s)")
    for *box, confidence, class_id in detections.cpu().tolist():
        print(f"  {names[int(class_id)]:<14} confidence={confidence:.3f} box={box}")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(args.output), annotated):
            raise RuntimeError(f"Could not write output image: {args.output}")
        print(f"Saved annotated image to {args.output}")

    if not args.no_show:
        import matplotlib.pyplot as plt

        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
