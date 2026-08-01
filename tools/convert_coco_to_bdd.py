#!/usr/bin/env python3
"""
Convert a Roboflow-style COCO object-detection export to the BDD-style JSON
consumed by utils/bdd_dataset.py.

The conversion is non-destructive. It builds a new dataset in a temporary
sibling directory, validates it, and atomically renames it to the requested
destination when complete.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable


ANNOTATION_FILENAME = "_annotations.coco.json"
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert COCO detection annotations to BDD-style JSON.",
    )
    parser.add_argument("source", type=Path, help="COCO dataset root")
    parser.add_argument("destination", type=Path, help="New BDD-format dataset root")
    parser.add_argument(
        "--image-mode",
        choices=("copy", "hardlink", "symlink"),
        default="copy",
        help="How images are placed in the converted dataset (default: copy)",
    )
    parser.add_argument(
        "--no-verify-images",
        action="store_false",
        dest="verify_images",
        help="Skip decoding every source image and checking its dimensions",
    )
    return parser.parse_args()


def require_unique(values: list[Any], description: str) -> None:
    if len(values) != len(set(values)):
        duplicates = [value for value, count in Counter(values).items() if count > 1]
        raise ValueError(f"Duplicate {description}: {duplicates[:10]}")


def safe_basename(file_name: str) -> str:
    path = Path(file_name)
    if path.is_absolute() or path.name != file_name or file_name in ("", ".", ".."):
        raise ValueError(f"Image file_name must be a safe basename: {file_name!r}")
    return file_name


def normalized_source_stem(file_name: str) -> str:
    """
    Remove Roboflow's augmentation hash to identify variants of one source image.
    """
    return file_name.split(".rf.", maxsplit=1)[0]


def load_coco(annotation_path: Path) -> dict[str, Any]:
    with annotation_path.open("r", encoding="utf-8") as stream:
        data = json.load(stream)

    required = {"images", "annotations", "categories"}
    missing = required.difference(data)
    if missing:
        raise ValueError(f"{annotation_path} is missing COCO keys: {sorted(missing)}")
    return data


def category_table(data: dict[str, Any], split: str) -> dict[int, str]:
    category_ids = [category["id"] for category in data["categories"]]
    category_names = [category["name"] for category in data["categories"]]
    require_unique(category_ids, f"category IDs in {split}")
    require_unique(category_names, f"category names in {split}")

    table: dict[int, str] = {}
    for category in data["categories"]:
        category_id = category["id"]
        name = category["name"]
        if not isinstance(category_id, int):
            raise TypeError(f"Category ID must be an integer in {split}: {category_id!r}")
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"Invalid category name in {split}: {name!r}")
        table[category_id] = name
    return table


def verify_image(path: Path, declared_width: int, declared_height: int) -> None:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Pillow is required for image verification. "
            "Install it or pass --no-verify-images."
        ) from error

    with Image.open(path) as image:
        actual_width, actual_height = image.size
        image.verify()

    if (actual_width, actual_height) != (declared_width, declared_height):
        raise ValueError(
            f"Image dimension mismatch for {path}: "
            f"COCO={(declared_width, declared_height)}, "
            f"actual={(actual_width, actual_height)}"
        )


def convert_split(
    source_root: Path,
    source_split: str,
    output_split: str,
    verify_images: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any], list[tuple[Path, str]]]:
    split_dir = source_root / source_split
    annotation_path = split_dir / ANNOTATION_FILENAME
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Missing COCO annotations: {annotation_path}")

    data = load_coco(annotation_path)
    categories = category_table(data, source_split)
    images = data["images"]
    annotations = data["annotations"]

    image_ids = [image["id"] for image in images]
    annotation_ids = [annotation["id"] for annotation in annotations]
    require_unique(image_ids, f"image IDs in {source_split}")
    require_unique(annotation_ids, f"annotation IDs in {source_split}")

    image_by_id: dict[int, dict[str, Any]] = {}
    output_names: list[str] = []
    source_images: list[tuple[Path, str]] = []
    for image in images:
        image_id = image["id"]
        if not isinstance(image_id, int):
            raise TypeError(f"Image ID must be an integer in {source_split}: {image_id!r}")

        output_name = safe_basename(image["file_name"])
        width = image["width"]
        height = image["height"]
        if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
            raise ValueError(
                f"Invalid dimensions for {output_name}: {(width, height)}"
            )

        source_image = split_dir / output_name
        if not source_image.is_file():
            raise FileNotFoundError(f"Missing image declared by COCO: {source_image}")
        if verify_images:
            verify_image(source_image, width, height)

        image_by_id[image_id] = image
        output_names.append(output_name)
        source_images.append((source_image, output_name))

    require_unique(output_names, f"image filenames in {source_split}")
    casefold_names = [name.casefold() for name in output_names]
    require_unique(casefold_names, f"case-insensitive image filenames in {source_split}")

    declared_files = set(output_names)
    actual_files = {
        path.name
        for path in split_dir.iterdir()
        if path.is_file() and path.name != ANNOTATION_FILENAME
    }
    missing_files = declared_files.difference(actual_files)
    extra_files = actual_files.difference(declared_files)
    if missing_files or extra_files:
        raise ValueError(
            f"Image inventory mismatch in {source_split}: "
            f"missing={sorted(missing_files)[:10]}, extra={sorted(extra_files)[:10]}"
        )

    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    category_counts: Counter[str] = Counter()
    duplicate_keys: Counter[tuple[Any, ...]] = Counter()
    clipped_boxes = 0

    for annotation in annotations:
        image_id = annotation["image_id"]
        category_id = annotation["category_id"]
        if image_id not in image_by_id:
            raise ValueError(
                f"Annotation {annotation['id']} references unknown image {image_id}"
            )
        if category_id not in categories:
            raise ValueError(
                f"Annotation {annotation['id']} references unknown category {category_id}"
            )
        if annotation.get("iscrowd", 0):
            raise ValueError(
                f"Crowd annotation {annotation['id']} cannot be represented faithfully"
            )

        bbox = annotation.get("bbox")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"Invalid bbox for annotation {annotation['id']}: {bbox!r}")
        if not all(isinstance(value, (int, float)) and math.isfinite(value) for value in bbox):
            raise ValueError(f"Non-finite bbox for annotation {annotation['id']}: {bbox!r}")

        x, y, width, height = (float(value) for value in bbox)
        if width <= 0 or height <= 0:
            raise ValueError(f"Non-positive bbox for annotation {annotation['id']}: {bbox!r}")

        image = image_by_id[image_id]
        x1 = min(max(x, 0.0), float(image["width"]))
        y1 = min(max(y, 0.0), float(image["height"]))
        x2 = min(max(x + width, 0.0), float(image["width"]))
        y2 = min(max(y + height, 0.0), float(image["height"]))
        if (x1, y1, x2, y2) != (x, y, x + width, y + height):
            clipped_boxes += 1
        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Degenerate bbox after clipping for annotation {annotation['id']}: {bbox!r}"
            )

        category = categories[category_id]
        converted_annotation = {
            "category": category,
            "box2d": {
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            },
            "_source_annotation_id": annotation["id"],
        }
        annotations_by_image[image_id].append(converted_annotation)
        category_counts[category] += 1
        duplicate_keys[(image_id, category_id, x, y, width, height)] += 1

    converted_images: list[dict[str, Any]] = []
    empty_images = 0
    for image in sorted(images, key=lambda value: value["file_name"]):
        converted_labels = []
        source_labels = sorted(
            annotations_by_image.get(image["id"], []),
            key=lambda value: value["_source_annotation_id"],
        )
        for label_id, source_label in enumerate(source_labels):
            converted_labels.append(
                {
                    "category": source_label["category"],
                    "attributes": {},
                    "manualShape": True,
                    "manualAttributes": True,
                    "box2d": source_label["box2d"],
                    "id": label_id,
                }
            )

        if not converted_labels:
            empty_images += 1
        converted_images.append(
            {
                "name": image["file_name"],
                "attributes": {},
                "timestamp": 0,
                "labels": converted_labels,
            }
        )

    duplicate_annotations = sum(count - 1 for count in duplicate_keys.values() if count > 1)
    report = {
        "source_split": source_split,
        "output_split": output_split,
        "images": len(images),
        "annotations": len(annotations),
        "empty_images": empty_images,
        "clipped_boxes": clipped_boxes,
        "duplicate_annotations_preserved": duplicate_annotations,
        "category_counts": dict(sorted(category_counts.items())),
    }
    return converted_images, report, source_images


def image_writer(mode: str) -> Callable[[Path, Path], None]:
    if mode == "copy":
        return lambda source, destination: shutil.copy2(source, destination)
    if mode == "hardlink":
        return lambda source, destination: os.link(source, destination)
    if mode == "symlink":
        return lambda source, destination: destination.symlink_to(
            os.path.relpath(source, start=destination.parent)
        )
    raise ValueError(f"Unsupported image mode: {mode}")


def write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def write_dataset_yaml(
    path: Path,
    destination: Path,
    class_names: list[str],
) -> None:
    image_root = destination / "images" / "100k"
    label_root = destination / "labels"
    lines = [
        f"train_images_path: {json.dumps(str(image_root / 'train'))}",
        f"train_labels_path: {json.dumps(str(label_root / 'football_labels_images_train.json'))}",
        f"val_images_path: {json.dumps(str(image_root / 'val'))}",
        f"val_labels_path: {json.dumps(str(label_root / 'football_labels_images_val.json'))}",
        f"test_images_path: {json.dumps(str(image_root / 'test'))}",
        f"test_labels_path: {json.dumps(str(label_root / 'football_labels_images_test.json'))}",
        "",
        "names:",
    ]
    lines.extend(
        f"  {class_id}: {json.dumps(name, ensure_ascii=False)}"
        for class_id, name in enumerate(class_names)
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def leakage_report(
    split_images: dict[str, list[tuple[Path, str]]],
) -> dict[str, Any]:
    stem_splits: dict[str, set[str]] = defaultdict(set)
    stem_images: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for split, images in split_images.items():
        for _, name in images:
            stem = normalized_source_stem(name)
            stem_splits[stem].add(split)
            stem_images[stem].append((split, name))

    leaking_stems = {
        stem: sorted(splits)
        for stem, splits in stem_splits.items()
        if len(splits) > 1
    }
    affected_by_split: Counter[str] = Counter()
    affected_images = 0
    for stem in leaking_stems:
        for split, _ in stem_images[stem]:
            affected_by_split[split] += 1
            affected_images += 1

    combinations = Counter("+".join(splits) for splits in leaking_stems.values())
    return {
        "normalization": "filename prefix before '.rf.'",
        "unique_source_stems": len(stem_splits),
        "cross_split_source_stems": len(leaking_stems),
        "affected_images": affected_images,
        "affected_images_by_split": dict(sorted(affected_by_split.items())),
        "split_combinations": dict(sorted(combinations.items())),
        "sample_stems": [
            {
                "stem": stem,
                "splits": splits,
                "images": [
                    {"split": split, "name": name}
                    for split, name in stem_images[stem]
                ],
            }
            for stem, splits in sorted(leaking_stems.items())[:20]
        ],
    }


def validate_output(
    root: Path,
    converted: dict[str, list[dict[str, Any]]],
    reports: dict[str, dict[str, Any]],
) -> None:
    for split, records in converted.items():
        image_dir = root / "images" / "100k" / split
        output_files = {path.name for path in image_dir.iterdir() if path.is_file()}
        record_names = {record["name"] for record in records}
        if output_files != record_names:
            raise RuntimeError(
                f"Output inventory mismatch for {split}: "
                f"files={len(output_files)}, records={len(record_names)}"
            )

        annotation_count = 0
        for record in records:
            if set(record).difference({"name", "attributes", "timestamp", "labels"}):
                raise RuntimeError(f"Unexpected BDD image fields in {record['name']}")
            for label in record["labels"]:
                box = label["box2d"]
                if not (box["x2"] > box["x1"] and box["y2"] > box["y1"]):
                    raise RuntimeError(f"Invalid output box in {record['name']}: {box}")
                annotation_count += 1

        expected_annotations = reports[split]["annotations"]
        if annotation_count != expected_annotations:
            raise RuntimeError(
                f"Output annotation mismatch for {split}: "
                f"{annotation_count} != {expected_annotations}"
            )


def main() -> None:
    args = parse_args()
    source = args.source.expanduser().resolve()
    destination = args.destination.expanduser().resolve()

    if not source.is_dir():
        raise FileNotFoundError(f"Source dataset does not exist: {source}")
    if destination.exists():
        raise FileExistsError(
            f"Destination already exists; refusing to overwrite it: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)

    converted: dict[str, list[dict[str, Any]]] = {}
    reports: dict[str, dict[str, Any]] = {}
    split_images: dict[str, list[tuple[Path, str]]] = {}
    category_tables: list[dict[int, str]] = []
    used_category_ids: set[int] = set()

    for source_split, output_split in SPLIT_MAP.items():
        data = load_coco(source / source_split / ANNOTATION_FILENAME)
        categories = category_table(data, source_split)
        category_tables.append(categories)
        used_category_ids.update(annotation["category_id"] for annotation in data["annotations"])

        records, report, images = convert_split(
            source,
            source_split,
            output_split,
            args.verify_images,
        )
        converted[output_split] = records
        reports[output_split] = report
        split_images[output_split] = images

    reference_categories = category_tables[0]
    if any(table != reference_categories for table in category_tables[1:]):
        raise ValueError("COCO category tables differ between splits")
    unknown_used_categories = used_category_ids.difference(reference_categories)
    if unknown_used_categories:
        raise ValueError(f"Unknown used categories: {sorted(unknown_used_categories)}")

    ordered_category_ids = sorted(used_category_ids)
    class_names = [reference_categories[category_id] for category_id in ordered_category_ids]
    require_unique(class_names, "used category names")
    unused_categories = [
        {"coco_id": category_id, "name": name}
        for category_id, name in sorted(reference_categories.items())
        if category_id not in used_category_ids
    ]

    temporary_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.tmp-",
            dir=destination.parent,
        )
    )
    try:
        labels_dir = temporary_root / "labels"
        image_root = temporary_root / "images" / "100k"
        labels_dir.mkdir(parents=True)
        image_root.mkdir(parents=True)

        place_image = image_writer(args.image_mode)
        for split in ("train", "val", "test"):
            image_dir = image_root / split
            image_dir.mkdir()
            for source_image, output_name in split_images[split]:
                place_image(source_image, image_dir / output_name)

            label_path = labels_dir / f"football_labels_images_{split}.json"
            write_json(label_path, converted[split])

        leakage = leakage_report(split_images)
        report = {
            "source": str(source),
            "destination": str(destination),
            "source_format": "COCO object detection",
            "output_format": "BDD100K-style image annotation JSON",
            "image_mode": args.image_mode,
            "images_verified": args.verify_images,
            "class_mapping": [
                {
                    "model_id": model_id,
                    "source_coco_id": category_id,
                    "name": reference_categories[category_id],
                }
                for model_id, category_id in enumerate(ordered_category_ids)
            ],
            "unused_categories": unused_categories,
            "splits": reports,
            "potential_split_leakage": leakage,
            "notes": [
                "COCO xywh boxes were converted to absolute pixel-space BDD xyxy boxes.",
                "Negative images were preserved with an empty labels list.",
                "Exact duplicate annotations were preserved for a faithful format conversion.",
                "The original train/valid/test split assignment was preserved.",
            ],
        }
        write_json(temporary_root / "conversion_report.json", report)
        write_dataset_yaml(
            temporary_root / "dataset.yaml",
            destination,
            class_names,
        )

        for readme_name in ("README.dataset.txt", "README.roboflow.txt"):
            source_readme = source / readme_name
            if source_readme.is_file():
                shutil.copy2(source_readme, temporary_root / readme_name)

        converted_readme = "\n".join(
            [
                "# Football dataset converted to BDD-style JSON",
                "",
                f"Source: `{source}`",
                "",
                "The original COCO dataset was preserved. Images are under",
                "`images/100k/{train,val,test}` and annotations are under `labels/`.",
                "",
                "Model classes:",
                *[
                    f"- `{model_id}`: `{name}`"
                    for model_id, name in enumerate(class_names)
                ],
                "",
                "See `conversion_report.json` for counts, validation results,",
                "preserved duplicate annotations, and the potential split-leakage warning.",
                "",
            ]
        )
        (temporary_root / "README.md").write_text(converted_readme, encoding="utf-8")

        validate_output(temporary_root, converted, reports)
        temporary_root.rename(destination)
    except Exception:
        shutil.rmtree(temporary_root, ignore_errors=True)
        raise

    total_images = sum(report["images"] for report in reports.values())
    total_annotations = sum(report["annotations"] for report in reports.values())
    print(f"Converted dataset: {destination}")
    print(f"Images: {total_images}")
    print(f"Annotations: {total_annotations}")
    print(f"Classes: {class_names}")
    print(
        "Potential split leakage: "
        f"{leakage['cross_split_source_stems']} source stems, "
        f"{leakage['affected_images']} images"
    )


if __name__ == "__main__":
    main()
