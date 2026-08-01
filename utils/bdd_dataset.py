import json
import os
import cv2
import numpy as np
import torch
import random

from torch.utils.data import Dataset
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode
from glob import glob
from utils.mosaic import Mosaic
from typing import Tuple, Dict, List, Any


class PadToSquare:
    def __init__(self, fill_color=0.0):
        self.fill_color = fill_color

    def __call__(self, inputs):
        # Extract dimensions (handles both PIL Images and Tensors)
        img = inputs["image"]
        if isinstance(img, torch.Tensor):
            # Tensor shape is [..., C, H, W]
            h, w = img.shape[-2], img.shape[-1]
        else:
            w, h = img.size

        original_shape = inputs["shapes"]
        ratios = (h / inputs["shapes"][0], w / inputs["shapes"][1])

        # Calculate difference and padding amounts
        if h > w:
            difference = h - w

            left = difference // 2
            right = difference - left

            padding = [left, 0, right, 0]
            pad_meta = (left, 0)
        elif h < w:
            difference = w - h

            top = difference // 2
            bottom = difference - top
            padding = [0, top, 0, bottom]
            pad_meta = (0, top)
        else:
            # Already square
            pad_meta = (0, 0)

        # Update shapes for the expected format
        inputs["shapes"] = (original_shape, (ratios, pad_meta))

        if h != w:
            pad = v2.Pad(padding=padding, fill=self.fill_color, padding_mode="constant")
            return pad(inputs)

        return inputs


def collate_fn(batch: List[Dict[str, torch.Tensor | List[torch.Tensor]]]) -> Dict[str, torch.Tensor | List[torch.Tensor]]:
    images: List = []
    labels: List = []
    boxes: List = []
    for item in batch:
        images.append(item["image"])
        labels.append(item["labels"])
        boxes.append(item["boxes"])

    return {
        "image": torch.stack(images, dim=0),
        "labels": labels,
        "boxes": boxes
    }


def collate_fn_yolo(batch: List[Dict[str, torch.Tensor | List[torch.Tensor]]]) -> Dict[str, torch.Tensor | Tuple]:
    images: List = []
    labels: List = []
    shapes: List = []
    for i, item in enumerate(batch):
        images.append(item["image"])

        batch_id = torch.ones(item["labels"].shape) * i

        boxes = item["boxes"].clone()
        boxes[:, 0] = (item["boxes"][:, 0] + item["boxes"][:, 2]) / 2
        boxes[:, 1] = (item["boxes"][:, 1] + item["boxes"][:, 3]) / 2
        boxes[:, 2] = (item["boxes"][:, 2] - item["boxes"][:, 0])
        boxes[:, 3] = (item["boxes"][:, 3] - item["boxes"][:, 1])
        boxes[:, 0::2] /= item["image"].shape[2]
        boxes[:, 1::2] /= item["image"].shape[1]

        label = torch.cat((batch_id, item["labels"], boxes), dim=1)
        labels.append(label)

        shapes.append(item["shapes"])

    return {
        "image": torch.stack(images, dim=0),
        "labels": torch.cat(labels, dim=0),
        "shapes": tuple(shapes)
    }


def load_image(key: str) -> np.ndarray:
    """
    Loads an image from its index.
    :param key: The key of the image.
    :return: Returns the image as a numpy array.
    """
    img: np.ndarray | any = cv2.imread(key)
    if img is None:
        raise FileNotFoundError(F"Could not read image {key}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_mosaic(
        images: List[torch.Tensor],
        boxes: List[torch.Tensor],
        labels: List[torch.Tensor],
        keys: List[int]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Generates mosaic image which consists of the initial image and three other random images.
    :param images:
    :param boxes:
    :param labels:
    :param keys:
    :return:
    """
    # Convert labels into the required format of Mosaic
    targets: List[Dict] = []
    for t_boxes, t_labels, key in zip(boxes, labels, keys):
        area = (t_boxes[:, 3] - t_boxes[:, 1]) * (t_boxes[:, 2] - t_boxes[:, 0])
        target = {
            'boxes': t_boxes,
            'labels': t_labels,
            'area': area,
            'iscrowd': torch.zeros(len(t_labels), dtype=torch.int64),
            'image_id': torch.tensor(key, dtype=torch.int64),
        }
        targets.append(target)

    mosaic = Mosaic(
        output_size=1280,
        min_possible_image_area=0.25,
        centre_pad=False,
        crop=True,
        min_visibility=0.01,
        intelligent_resize=True,
        max_aspect_ratio=1.2,
        crop_to_square=False,
    )

    img, targets = mosaic(tuple(images), tuple(targets))
    img, t_boxes, t_labels = img[0], targets[0]['boxes'], targets[0]['labels']
    # Convert image back to uint8
    img = tv_tensors.Image(
        img.clamp(0, 255).to(torch.uint8)
    )
    # Reshape labels for concatenation
    t_labels = t_labels.reshape(-1, 1)

    return img, t_boxes, t_labels


class BDDDataset(Dataset):
    def __init__(self, config: dict, transform: bool=False, train: bool=True):
        super().__init__()
        self.config = config
        self.transform = transform
        self.train = train

        # Load images
        self.images_dir = config["train_images_path"] if train else config["val_images_path"]
        self.images: List[str] = glob(os.path.join(self.images_dir, "*"))

        # Load JSON and create a dictionary for fast access
        self.classes: Dict = config["names"]
        self.inverse_classes: Dict = {v: k for k, v in self.classes.items()}
        json_labels = json.load(open(config["train_labels_path"] if train else config["val_labels_path"], "r"))
        keys = [label['name'] for label in json_labels]
        for label in json_labels:
            del label['name']

        self.json_labels_dict = dict(zip(keys, json_labels))

    def __getitem__(self, idx: int):
        # Check if Mosaic augmentation can be applied (only during training)
        if self.train and random.random() < self.config["mosaic"]:
            # Get the three other images
            keys: List[int] = random.choices(range(len(self.images)), k=3)
            keys.append(idx)
            images, boxes, labels = [], [], []
            for idx in keys:
                # Load the image
                key: str = self.images[idx]
                img = load_image(key)
                img = v2.ToImage()(img)
                # Get the labels and filter them based on the provided classes
                raw_labels_filtered = self.load_labels(key)
                # Convert the raw labels into separate tensors for boxes and classes
                t_boxes, t_labels = self.get_boxes_and_labels(raw_labels_filtered)
                # Append the new image and labels
                images.append(img)
                boxes.append(t_boxes)
                labels.append(t_labels)
            img, t_boxes, t_labels = get_mosaic(images, boxes, labels, keys)
        else:
            # Load the image
            key: str = self.images[idx]
            img = load_image(key)
            img = v2.ToImage()(img)
            # Get the labels and filter them based on the provided classes
            raw_labels_filtered = self.load_labels(key)
            # Convert the raw labels into separate tensors for boxes and classes
            t_boxes, t_labels = self.get_boxes_and_labels(raw_labels_filtered)

        # Check for transforms
        if self.transform:
            # Convert to bounding boxes format accepted by torchvision
            height, width = img.shape[1], img.shape[2]
            t_boxes = tv_tensors.BoundingBoxes(t_boxes, format="xyxy", canvas_size=(height, width))

            # Resize the larger dimension to the fixed value
            ratio = height / self.config["image_size"] if height > width else width / self.config["image_size"]
            shapes = (height, width)
            height, width = int(height / ratio), int(width / ratio)

            # Get the transformations function
            transforms = self.get_transforms_array(height, width)

            transformed_dict = transforms({
                "image": img,
                "labels": t_labels,
                "boxes": t_boxes,
                "shapes": shapes
            })
            return transformed_dict
        else:
            height, width = img.shape[1], img.shape[2]
            shapes = ((height, width), ((1, 1), (0, 0)))
            return {
                "image": img,
                "labels": t_labels,
                "boxes": t_boxes,
                "shapes": shapes
            }

    def __len__(self):
        """
        :return: Returns the length of the dataset.
        """
        return len(self.images)

    def get_boxes_and_labels(self, raw_labels: List[Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Converts the raw labels into separate bounding boxes and labels
        :param raw_labels: A dictionary or labels stored as a JSON
        :return: Returns a tuple formed by boxes and labels tensors.
        """
        # Get raw boxes and labels
        t_boxes = torch.Tensor([
            [label['box2d']['x1'], label['box2d']['y1'], label['box2d']['x2'], label['box2d']['y2']]
            for label in raw_labels
        ]).reshape(-1, 4)
        t_labels = torch.Tensor([
            self.inverse_classes.get(label['category'])
            for label in raw_labels
        ]).reshape(-1, 1)
        return t_boxes, t_labels

    def get_transforms_array(self, height: int, width: int) -> Any:
        """
        Generates the sequence of transformations which will be applied to the image and labels
        :param height: The new height of the image.
        :param width: The new width of the image.
        :return: Returns the transformations function generated by 'v2.Compose'.
        """
        transforms_list: List[Any] = [
            # Initial transform
            v2.Resize((height, width))
        ]

        if self.train:
            augmentations_list = [
                # Flip
                v2.RandomHorizontalFlip(p=self.config["flip_lr"]),
                # Affine transforms
                v2.RandomAffine(
                    degrees=self.config["degrees"],
                    translate=(self.config["translate"], self.config["translate"]),
                    scale=(1 - self.config["scale"], 1 + self.config["scale"]),
                    shear=self.config["shear"],
                    interpolation=InterpolationMode.BILINEAR,
                    fill=114
                ),
                # Color transforms
                v2.ColorJitter(
                    brightness=self.config["hsv_v"],
                    contrast=self.config["hsv_v"],
                    saturation=self.config["hsv_s"],
                    hue=self.config["hsv_h"],
                ),
            ]
            transforms_list.extend(augmentations_list)

        transforms_list.extend([
            # Final pad
            PadToSquare(fill_color=114),
            # Clamp bounding boxes
            v2.ClampBoundingBoxes()
        ])

        # Filter small bounding boxes only during training
        if self.train:
            transforms_list.append(
                v2.SanitizeBoundingBoxes(min_size=5.0, min_area=20.0)
            )

        transforms_list.extend([
            v2.ToDtype(torch.float32, scale=True),
            # Normalization
            v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        return v2.Compose(transforms_list)

    def load_labels(self, key: str) -> List[Any]:
        """
        Loads the labels stored as a JSON file.
        :param key: The path to the image.
        :return: Returns a list of labels.
        """
        raw_labels = self.json_labels_dict[os.path.basename(key)]['labels'] if os.path.basename(key) in self.json_labels_dict else []
        raw_labels_filtered = [label for label in raw_labels if label['category'] in self.classes.values()]
        return raw_labels_filtered
