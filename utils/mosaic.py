"""
The entire implementation is almost identical with: https://github.com/prymeka/mosaic-augmentation-pytorch
"""
from typing import Any
from abc import ABCMeta
from abc import abstractmethod

import torch
import numpy as np
from more_itertools import grouper
from torchvision import tv_tensors
from torchvision.transforms import v2
from torchvision.transforms.v2 import functional as F

SMALL_NUMBER = 1e-3
ImagesType = tuple[tv_tensors.Image]
TargetsType = tuple[dict[str, tv_tensors.BoundingBoxes | Any]]


class MosaicSubTransform(torch.nn.Module, metaclass=ABCMeta):
    """
    Batch transform meant to operate on batches of size of multiple of 4.
    """

    @abstractmethod
    def forward(self, images: ImagesType, targets: TargetsType) -> tuple[ImagesType, TargetsType]:
        pass


class SquarePad(MosaicSubTransform):

    def __init__(self, centre_pad: bool = False, fill_value: int = 114) -> None:
        """
        Pad images to square. If batch size is not divisible by 4, the remainder of images
        will not be transformed.

        Parameters
        ----------
        centre_pad: bool
            If `centre_pad` is `False`, images will be padded "on the outside", i.e.,
            such that no padding will be around the centre of mosaic. Otherwise, padding
            will be equal on all sides of individual images. Defaults to `False`.
        fill_value: int
            Value to pad with. Defaults to `114` following:
            https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py#L843
        """
        super().__init__()
        self.centre_pad = centre_pad
        self.fill_value = fill_value

    def forward(self, images: ImagesType, targets: TargetsType) -> tuple[ImagesType, TargetsType]:
        images, targets = list(images), list(targets)
        for batch_idx in grouper(range(len(images)), n=4, incomplete='ignore'):
            for i, pos in zip(batch_idx, ['tl', 'tr', 'bl', 'br']):
                _, h, w = images[i].shape
                if h == w:
                    continue
                max_wh = np.max([w, h])
                p_top, p_left = [(max_wh - s) // 2 for s in (h, w)]
                p_bottom, p_right = [max_wh - (s + pad) for s, pad in zip((h, w), (p_top, p_left))]
                # padding values: (left, top, right, bottom)
                if self.centre_pad:
                    padding = (p_left, p_top, p_right, p_bottom)
                    targets[i]['boxes'][:, [0, 2]] += padding[0]
                    targets[i]['boxes'][:, [1, 3]] += padding[1]
                else:
                    if pos == 'tl':
                        padding = (p_left + p_right, p_top + p_bottom, 0, 0)
                        targets[i]['boxes'][:, [0, 2]] += padding[0]
                        targets[i]['boxes'][:, [1, 3]] += padding[1]
                    elif pos == 'tr':
                        padding = (0, p_top + p_bottom, p_left + p_right, 0)
                        targets[i]['boxes'][:, [1, 3]] += padding[1]
                    elif pos == 'bl':
                        padding = (p_left + p_right, 0, 0, p_top + p_bottom)
                        targets[i]['boxes'][:, [0, 2]] += padding[0]
                    elif pos == 'br':
                        padding = (0, 0, p_left + p_right, p_top + p_bottom)
                images[i] = F.pad(images[i], padding, self.fill_value, 'constant')

        return tuple(images), tuple(targets)


class Resize(MosaicSubTransform):

    def __init__(self, max_size: int) -> None:
        """
        Rescale an image so that maximum side is equal to `max_size`, keeping the aspect ratio of the initial image.
        """
        super().__init__()
        self.max_size = max_size

    def forward(self, images: ImagesType, targets: TargetsType) -> tuple[ImagesType, TargetsType]:
        images, targets = list(images), list(targets)
        for i, image in enumerate(images):
            _, h, w = image.shape
            aspect_ratio = np.min([w, h]) / np.max([w, h])
            small_side_size = int(self.max_size * aspect_ratio)
            # if int is passed to F.resize, then the small side will be matched to it
            images[i] = F.resize(images[i], small_side_size, antialias=True)
            # adjust bounding boxes and area
            resize_factor = self.max_size / np.max([w, h])
            targets[i]['boxes'] = targets[i]['boxes'].to(torch.float32)
            targets[i]['boxes'] *= resize_factor
            targets[i]['area'] = targets[i]['area'].to(torch.float32)
            targets[i]['area'] *= resize_factor ** 2

        return tuple(images), tuple(targets)


class RandomCrop(MosaicSubTransform):

    def __init__(
            self,
            output_size: int | tuple[int, int] | None,
            min_visibility: float | None = None,
    ) -> None:
        """
        Randomly crop the image to size.

        Parameters
        ----------
        output_size: int | tuple[int, int] | None
            The width and height of the cropped image. If an `int` is passed
            the output will be a square. The `output_size` can be `None`, but then
            the output size must be passed as the 3rd argument in the `forward` call.
        min_visibility: float, optional
            If `min_visibility` is a `float`, then bounding boxes with
            `[area after crop] / [area before crop] < min_visibility` will be removed.
            If `min_visibility <= 0.0` or `None`, then no bounding boxes will be
            removed save for those completely outside of the cropped image.
        """
        super().__init__()
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size
        self.min_visibility = min_visibility

    def _resolve_output_size(
        self,
        output_size: int | tuple[int, int] | None,
    ) -> tuple[int, int]:
        resolved_size = (
            self.output_size
            if output_size is None
            else output_size
        )

        if resolved_size is None:
            raise ValueError("An output size must be provided.")

        if isinstance(resolved_size, int):
            resolved_size = (resolved_size, resolved_size)

        output_width, output_height = resolved_size

        if output_width <= 0 or output_height <= 0:
            raise ValueError(
                f"Invalid crop size: {resolved_size}."
            )

        return int(output_width), int(output_height)

    @staticmethod
    def _sample_crop_start(max_offset: int) -> int:
        """
        Sample an integer in [0, max_offset], including max_offset.
        """
        if max_offset <= 0:
            return 0

        return int(np.random.randint(0, max_offset + 1))


    def forward(
        self,
        images: ImagesType,
        targets: TargetsType,
        output_size: int | tuple[int, int] | None = None,
    ) -> tuple[ImagesType, TargetsType]:
        output_width, output_height = self._resolve_output_size(output_size)

        images = list(images)
        targets = list(targets)

        for index, (image, target) in enumerate(zip(images, targets)):
            _, image_height, image_width = image.shape

            if image_width < output_width or image_height < output_height:
                raise ValueError(
                    "Crop size cannot exceed image size. "
                    f"Image: {(image_width, image_height)}, "
                    f"crop: {(output_width, output_height)}."
                )

            if image_width == output_width and image_height == output_height:
                continue

            max_x = image_width - output_width
            max_y = image_height - output_height

            crop_x = self._sample_crop_start(max_x)
            crop_y = self._sample_crop_start(max_y)

            images[index] = image[
                :,
                crop_y:crop_y + output_height,
                crop_x:crop_x + output_width,
            ]

            targets[index] = self._adjust_target(
                target=target,
                output_size=(output_width, output_height),
                x=crop_x,
                y=crop_y,
            )

        return tuple(images), tuple(targets)

    def _adjust_target(
        self,
        target: dict[str, tv_tensors.BoundingBoxes | Any],
        output_size: tuple[int, int],
        x: int,
        y: int,
    ) -> dict[str, tv_tensors.BoundingBoxes | Any]:
        """
        Crop and filter target boxes while preserving empty-target shapes
        and all target fields.
        """
        output_width, output_height = output_size

        raw_boxes = target["boxes"]

        # Avoid retaining BoundingBoxes metadata with outdated canvas size.
        if isinstance(raw_boxes, tv_tensors.BoundingBoxes):
            raw_boxes = raw_boxes.as_subclass(torch.Tensor)

        boxes = raw_boxes.to(dtype=torch.float32).reshape(-1, 4).clone()

        num_boxes = boxes.shape[0]

        # Original areas before moving or clipping the boxes.
        old_widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(0)
        old_heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
        old_areas = old_widths * old_heights

        # Move boxes into the crop coordinate system.
        boxes[:, [0, 2]] -= float(x)
        boxes[:, [1, 3]] -= float(y)

        # Clip them to the crop.
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clamp(
            min=0.0,
            max=float(output_width),
        )
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clamp(
            min=0.0,
            max=float(output_height),
        )

        new_widths = (boxes[:, 2] - boxes[:, 0]).clamp_min(0)
        new_heights = (boxes[:, 3] - boxes[:, 1]).clamp_min(0)
        new_areas = new_widths * new_heights

        valid_geometry = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])

        minimum_visibility = (
            SMALL_NUMBER
            if self.min_visibility is None
            else max(0.0, float(self.min_visibility))
        )

        visibility = (
            new_areas
            / old_areas.clamp_min(SMALL_NUMBER)
        )

        keep = (
            valid_geometry
            & (visibility >= minimum_visibility)
        )

        new_target: dict[
            str,
            tv_tensors.BoundingBoxes | Any,
        ] = {}

        for key, value in target.items():
            if key == "boxes":
                new_target[key] = boxes[keep]

            elif key == "area":
                # Area must correspond to the clipped boxes.
                new_target[key] = new_areas[keep]

            elif (
                isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == num_boxes
            ):
                # Filters labels, iscrowd and other per-box fields.
                new_target[key] = value[keep]

            else:
                # Preserve image_id and other image-level metadata.
                new_target[key] = value

        if "area" not in new_target:
            new_target["area"] = new_areas[keep]

        # Important guarantees:
        # boxes:  [N, 4], including [0, 4]
        # labels: original trailing shape, e.g. [N] or [N, 1]
        assert new_target["boxes"].ndim == 2
        assert new_target["boxes"].shape[1] == 4

        return new_target


class SafeRandomCrop(RandomCrop):
    """
    Random crop that attempts to preserve at least one box.

    If the target contains no boxes, it falls back to an ordinary
    random crop and returns a correctly shaped empty target.
    """

    def forward(
        self,
        images: ImagesType,
        targets: TargetsType,
        output_size: int | tuple[int, int] | None = None,
    ) -> tuple[ImagesType, TargetsType]:
        output_width, output_height = self._resolve_output_size(
            output_size
        )

        images = list(images)
        targets = list(targets)

        for index, (image, target) in enumerate(
            zip(images, targets)
        ):
            _, image_height, image_width = image.shape

            if (
                image_width < output_width
                or image_height < output_height
            ):
                raise ValueError(
                    "Crop size cannot exceed image size. "
                    f"Image: {(image_width, image_height)}, "
                    f"crop: {(output_width, output_height)}."
                )

            if (
                image_width == output_width
                and image_height == output_height
            ):
                continue

            max_x = image_width - output_width
            max_y = image_height - output_height

            raw_boxes = target["boxes"]

            if isinstance(raw_boxes, tv_tensors.BoundingBoxes):
                raw_boxes = raw_boxes.as_subclass(torch.Tensor)

            boxes = (
                raw_boxes
                .to(dtype=torch.float32)
                .reshape(-1, 4)
            )

            if boxes.shape[0] == 0:
                # Negative image: crop normally.
                crop_x = self._sample_crop_start(max_x)
                crop_y = self._sample_crop_start(max_y)

            else:
                box_index = int(
                    np.random.randint(0, boxes.shape[0])
                )

                x1, y1, x2, y2 = (
                    float(value)
                    for value in boxes[box_index].tolist()
                )

                crop_x = self._sample_start_preserving_box(
                    box_start=x1,
                    box_end=x2,
                    crop_size=output_width,
                    max_offset=max_x,
                )

                crop_y = self._sample_start_preserving_box(
                    box_start=y1,
                    box_end=y2,
                    crop_size=output_height,
                    max_offset=max_y,
                )

            cropped_image = image[
                :,
                crop_y:crop_y + output_height,
                crop_x:crop_x + output_width,
            ]

            _, cropped_height, cropped_width = cropped_image.shape

            if (
                cropped_width != output_width
                or cropped_height != output_height
            ):
                raise RuntimeError(
                    "Unexpected crop shape. "
                    f"Expected {(output_width, output_height)}, "
                    f"got {(cropped_width, cropped_height)}."
                )

            images[index] = cropped_image

            targets[index] = self._adjust_target(
                target=target,
                output_size=(output_width, output_height),
                x=crop_x,
                y=crop_y,
            )

        return tuple(images), tuple(targets)

    @staticmethod
    def _sample_start_preserving_box(
        box_start: float,
        box_end: float,
        crop_size: int,
        max_offset: int,
    ) -> int:
        """
        Select a crop origin so that the chosen box is contained whenever
        full containment is geometrically possible.
        """
        # For full containment:
        #
        # crop_start <= box_start
        # crop_start + crop_size >= box_end
        #
        # Therefore:
        # box_end - crop_size <= crop_start <= box_start

        lower_bound = max(
            0,
            int(np.ceil(box_end - crop_size)),
        )

        upper_bound = min(
            max_offset,
            int(np.floor(box_start)),
        )

        if lower_bound <= upper_bound:
            return int(
                np.random.randint(
                    lower_bound,
                    upper_bound + 1,
                )
            )

        # Full containment is impossible, for example when the box is
        # wider than the crop. Centre the crop over the box instead.
        box_center = (box_start + box_end) / 2.0
        crop_start = round(box_center - crop_size / 2.0)

        return int(
            np.clip(
                crop_start,
                0,
                max_offset,
            )
        )


class IntelligentSquareResize(MosaicSubTransform):

    def __init__(
            self,
            output_size: int,
            max_aspect_ratio: float | None = None,
            crop_to_square: bool = False,
            centre_pad: bool = False,
            min_visibility: float | None = None,
    ) -> None:
        """
        Resize padded and/or cropped images to minimise the amount of padding
        in the final mosaic.

        Parameters
        ----------
        output_size: int
            The width and height of each individual image in the mosaic
            (it must be a square, hence only `int` is allowed).
        max_aspect_ratio: float, optional
            If a `float`, then images with aspect ratio (defined as the greater side
            divided by the lesser side, thus, always greater than unity) greater than
            `max_aspect_ratio` will be cropped to square before any other transform
            is applied.
        crop_to_square: bool
            If `True`, all images will be cropped to square before any other transform
            is applied. If `True`, `max_aspect_ratio` is ignored. Defaults to `False`.
        centre_pad: bool
            See `SquarePad`.
        min_visibility: float, optional
            See `RandomCrop`.
        """
        super().__init__()
        self.output_size = output_size
        self.max_aspect_ratio = max_aspect_ratio
        self.crop_to_square = crop_to_square
        self.centre_pad = centre_pad
        self.min_visibility = min_visibility

        if not self.crop_to_square and self.max_aspect_ratio is None:
            raise ValueError(f'Both crop_to_square and max_aspect_ratio can not be None.')

        self.pad_and_resize = v2.Compose([SquarePad(self.centre_pad), Resize(self.output_size)])
        self.random_crop = SafeRandomCrop(-1, self.min_visibility)

    def forward(self, images: ImagesType, targets: TargetsType) -> tuple[ImagesType, TargetsType]:
        images, targets = list(images), list(targets)
        for i, (image, target) in enumerate(zip(images, targets)):
            _, h, w = image.shape
            if self.crop_to_square:
                if h == w:
                    continue
                output_size = (min(h, w), min(h, w))
                image, target = self.random_crop((image,), (target,), output_size)
                images[i], targets[i] = image[0], target[0]
                continue
            aspect_ratio = max(h, w) / min(h, w)
            if self.max_aspect_ratio and aspect_ratio > self.max_aspect_ratio:
                greater_size = int(self.max_aspect_ratio * min(h, w))
                output_size = (greater_size, h) if w > h else (w, greater_size)
                image, target = self.random_crop((image,), (target,), output_size)
                images[i], targets[i] = image[0], target[0]
        images, targets = self.pad_and_resize(images, targets)

        return tuple(images), tuple(targets)


class TypeConverter(torch.nn.Module):
    """
    Converts images to `torch.float32`, as well as, target bounding boxes
    to `tv_tensors.BoundingBoxes` and target area to `torch.int32`.
    """

    def forward(self, images: ImagesType, targets: TargetsType) -> tuple[ImagesType, TargetsType]:
        images, targets = list(images), list(targets)
        for i in range(len(images)):
            images[i] = images[i].to(torch.float32)
            targets[i]['boxes'] = tv_tensors.BoundingBoxes(
                targets[i]['boxes'],
                format='XYXY',
                canvas_size=tuple(images[i].shape[1:]),
            )
            targets[i]['area'] = targets[i]['area'].to(torch.int32)

        return tuple(images), tuple(targets)


class Mosaic(torch.nn.Module):

    def __init__(
            self,
            output_size: int = 1_000,
            min_possible_image_area: float = 0.25,
            centre_pad: bool = False,
            crop: bool = True,
            min_visibility: bool | float = 0.05,
            intelligent_resize: bool = True,
            max_aspect_ratio: float | None = None,
            crop_to_square: bool = False,
    ) -> None:
        """
        Mosaic data augmentation.

        Parameters
        ----------
        output_size: int
            The size of the final image. Defaults to `1000`.
        min_possible_image_area: float
            The minimal fraction of any of the four images to be visiable in the cropped mosaic.
            This is added in order to avoid excessive cropping of any of the images.
            It is used to calculate pre-cropping size of the grid. Deafults to `0.25`.

            For example, if `output_size = 1000` and `min_possible_image_area = 0.25`, then, first,
            a `1500x1500` grid will be constructed (each image will be resized to `750x750`) and
            then it will be randomly cropped to `1000x1000`.

            If `crop = False`, then a `1000x1000` grid will be constructed straight away.
        centre_pad: bool
            If `True`, the four images will be padded with equal padding on each side as needed
            to create a square. Otherwise, the padding will be added such that no image has
            padding separating it from the centre of the mosaic. Defaults to `False`.
            See `SquarePad`.
        crop: bool
            If `True`, a larger grid will be constructed first before randomly cropping
            to the required size. Defaults to `True`.
        min_visibility: bool | float
            Parameter passed to `RandomCrop`. If not `False` and greater than `0.0`, the bounding
            boxes that after cropping have less than `min_visibility` fraction of area
            visiable in the cropped image will be removed. This is done to avoid edge cases where
            almost all of the bounding box resides outside of the cropped image. Defaults to `0.05`.
            See `RandomCrop`.
        intelligent_resize: bool
            If `False`, all images will be padded to square and resized. This may result in large amounts
            of padding visiable in the final mosaic if the dataset has images of various aspect ratios.
            If `True`, images with aspect ratio (defined as the greater side divided by the lesser side,
            thus, always greater than unity) greater than `max_aspect_ratio` will be cropped to square
            before any other transforms are applied. Defaults to `True`.
        max_aspect_ratio: float, optional
            If a `float`, then images with aspect ratio (defined as the greater side divided by the lesser
            side, thus, always greater than unity) greater than `max_aspect_ratio` will be cropped to
            square before any other transform is applied.
        crop_to_square: bool
            If `True`, all images will be cropped to square before any other transform is applied.
            If `True`, `max_aspect_ratio` is ignored. Defaults to `False`.
        """
        super().__init__()
        self.output_size = output_size
        self.min_possible_image_area = min_possible_image_area
        self.centre_pad = centre_pad
        self.crop = crop
        self.min_visibility = min_visibility
        self.intelligent_resize = intelligent_resize
        self.max_aspect_ratio = max_aspect_ratio
        self.crop_to_square = crop_to_square
        # the total size of the grid of 4 images
        self.grid_size = self.output_size * (
                    2 - np.sqrt(self.min_possible_image_area)) if self.crop else self.output_size
        # the target size of an image
        self.images_size = int(self.grid_size / 2)
        # preprocessing transforms
        if self.intelligent_resize:
            self.preprocess = v2.Compose([IntelligentSquareResize(
                output_size=self.images_size,
                max_aspect_ratio=self.max_aspect_ratio,
                crop_to_square=self.crop_to_square,
                min_visibility=self.min_visibility,
            )])
        else:
            self.preprocess = v2.Compose([SquarePad(self.centre_pad), Resize(self.images_size)])
        # postprocessing transforms
        if crop:
            self.postprocess = v2.Compose([SafeRandomCrop(self.output_size, self.min_visibility), TypeConverter()])
        else:
            self.postprocess = v2.Compose([TypeConverter()])

    def forward(self, images: ImagesType, targets: TargetsType) -> tuple[ImagesType, TargetsType]:
        images, targets = list(images), list(targets)
        # resize the images to fit their quarters
        # odd images will be ignored
        n = (len(images) // 4) * 4
        images[:n], targets[:n] = self.preprocess(images[:n], targets[:n])
        # join the images into a grid
        new_images, new_targets = [], []
        for batch_idx in grouper(range(len(images)), n=4, incomplete='ignore'):
            # join images
            img_batch = [images[i] for i in batch_idx]
            top = torch.cat(img_batch[:2], dim=2)
            bottom = torch.cat(img_batch[2:], dim=2)
            grid_images = torch.cat([top, bottom], dim=1)
            new_images.append(grid_images)
            # correct targets to fit the grid
            tar_batch = [targets[i] for i in batch_idx]
            assert set([len(target['boxes'].shape) for target in tar_batch]) == {2}, tar_batch
            for i, pos in enumerate(['tl', 'tr', 'bl', 'br']):
                if pos == 'tl':
                    continue
                elif pos == 'tr':
                    tar_batch[i]['boxes'][:, [0, 2]] += self.images_size
                elif pos == 'bl':
                    tar_batch[i]['boxes'][:, [1, 3]] += self.images_size
                elif pos == 'br':
                    tar_batch[i]['boxes'][:, [0, 2]] += self.images_size
                    tar_batch[i]['boxes'][:, [1, 3]] += self.images_size
            grid_targets = {}
            for key in tar_batch[0].keys():
                grid_targets[key] = torch.cat([
                    target[key].reshape(1) if len(target[key].shape) == 0 else target[key]
                    for target in tar_batch
                ])
            new_targets.append(grid_targets)
        # crop the images to output size and convert to proper dtypes
        new_images, new_targets = self.postprocess(new_images, new_targets)
        new_images, new_targets = list(new_images), list(new_targets)

        if (rem := divmod(len(images), 4)[1]) != 0:
            new_images += images[-rem:]
            new_targets += targets[-rem:]

        return tuple(new_images), tuple(new_targets)
