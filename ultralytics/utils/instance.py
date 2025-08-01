# Ultralytics YOLO 🚀, AGPL-3.0 license

from collections import abc
from itertools import repeat
from numbers import Number
from typing import List

import numpy as np
import torch

from .ops import ltwh2xywh, ltwh2xyxy, xywh2ltwh, xywh2xyxy, xyxy2ltwh, xyxy2xywh


def _ntuple(n):
    """From PyTorch internals."""

    def parse(x):
        """Parse bounding boxes format between XYWH and LTWH."""
        return x if isinstance(x, abc.Iterable) else tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)
to_4tuple = _ntuple(4)

# `xyxy` means left top and right bottom
# `xywh` means center x, center y and width, height(YOLO format)
# `ltwh` means left top and width, height(COCO format)
_formats = ["xyxy", "xywh", "ltwh"]

__all__ = ("Bboxes",)  # tuple or list


class Bboxes:
    """
    A class for handling bounding boxes.

    The class supports various bounding box formats like 'xyxy', 'xywh', and 'ltwh'.
    Bounding box data should be provided in numpy arrays.

    Attributes:
        bboxes (numpy.ndarray): The bounding boxes stored in a 2D numpy array.
        format (str): The format of the bounding boxes ('xyxy', 'xywh', or 'ltwh').

    Note:
        This class does not handle normalization or denormalization of bounding boxes.
    """

    def __init__(self, bboxes, format="xyxy") -> None:
        """Initializes the Bboxes class with bounding box data in a specified format."""
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        bboxes = bboxes[None, :] if bboxes.ndim == 1 else bboxes
        assert bboxes.ndim == 2
        assert bboxes.shape[1] == 4
        self.bboxes = bboxes
        self.format = format
        # self.normalized = normalized

    def convert(self, format):
        """Converts bounding box format from one type to another."""
        assert format in _formats, f"Invalid bounding box format: {format}, format must be one of {_formats}"
        if self.format == format:
            return
        elif self.format == "xyxy":
            func = xyxy2xywh if format == "xywh" else xyxy2ltwh
        elif self.format == "xywh":
            func = xywh2xyxy if format == "xyxy" else xywh2ltwh
        else:
            func = ltwh2xyxy if format == "xyxy" else ltwh2xywh
        self.bboxes = func(self.bboxes)
        self.format = format

    def areas(self):
        """Return box areas."""
        self.convert("xyxy")
        return (self.bboxes[:, 2] - self.bboxes[:, 0]) * (self.bboxes[:, 3] - self.bboxes[:, 1])

    # def denormalize(self, w, h):
    #    if not self.normalized:
    #         return
    #     assert (self.bboxes <= 1.0).all()
    #     self.bboxes[:, 0::2] *= w
    #     self.bboxes[:, 1::2] *= h
    #     self.normalized = False
    #
    # def normalize(self, w, h):
    #     if self.normalized:
    #         return
    #     assert (self.bboxes > 1.0).any()
    #     self.bboxes[:, 0::2] /= w
    #     self.bboxes[:, 1::2] /= h
    #     self.normalized = True

    def mul(self, scale):
        """
        Args:
            scale (tuple | list | int): the scale for four coords.
        """
        if isinstance(scale, Number):
            scale = to_4tuple(scale)
        assert isinstance(scale, (tuple, list))
        assert len(scale) == 4
        self.bboxes[:, 0] *= scale[0]
        self.bboxes[:, 1] *= scale[1]
        self.bboxes[:, 2] *= scale[2]
        self.bboxes[:, 3] *= scale[3]

    def add(self, offset):
        """
        Args:
            offset (tuple | list | int): the offset for four coords.
        """
        if isinstance(offset, Number):
            offset = to_4tuple(offset)
        assert isinstance(offset, (tuple, list))
        assert len(offset) == 4
        self.bboxes[:, 0] += offset[0]
        self.bboxes[:, 1] += offset[1]
        self.bboxes[:, 2] += offset[2]
        self.bboxes[:, 3] += offset[3]

    def __len__(self):
        """Return the number of boxes."""
        return len(self.bboxes)

    @classmethod
    def concatenate(cls, boxes_list: List["Bboxes"], axis=0) -> "Bboxes":
        """
        Concatenate a list of Bboxes objects into a single Bboxes object.

        Args:
            boxes_list (List[Bboxes]): A list of Bboxes objects to concatenate.
            axis (int, optional): The axis along which to concatenate the bounding boxes.
                                   Defaults to 0.

        Returns:
            Bboxes: A new Bboxes object containing the concatenated bounding boxes.

        Note:
            The input should be a list or tuple of Bboxes objects.
        """
        assert isinstance(boxes_list, (list, tuple))
        if not boxes_list:
            return cls(np.empty(0))
        assert all(isinstance(box, Bboxes) for box in boxes_list)

        if len(boxes_list) == 1:
            return boxes_list[0]
        return cls(np.concatenate([b.bboxes for b in boxes_list], axis=axis))

    def __getitem__(self, index) -> "Bboxes":
        """
        Retrieve a specific bounding box or a set of bounding boxes using indexing.

        Args:
            index (int, slice, or np.ndarray): The index, slice, or boolean array to select
                                               the desired bounding boxes.

        Returns:
            Bboxes: A new Bboxes object containing the selected bounding boxes.

        Raises:
            AssertionError: If the indexed bounding boxes do not form a 2-dimensional matrix.

        Note:
            When using boolean indexing, make sure to provide a boolean array with the same
            length as the number of bounding boxes.
        """
        if isinstance(index, int):
            return Bboxes(self.bboxes[index].view(1, -1))
        b = self.bboxes[index]
        assert b.ndim == 2, f"Indexing on Bboxes with {index} failed to return a matrix!"
        return Bboxes(b)


class Instances:
    """
    Container for bounding boxes, locations, segments, and keypoints of detected objects in an image.

    Attributes:
        _bboxes (Bboxes): Internal object for handling bounding box operations. Default is None. 
        locations (ndarray): Locations array with shape [N, 2]. Default is None. 
        keypoints (ndarray): keypoints(x, y, visible) with shape [N, 17, 3]. Default is None.
        normalized (bool): Flag indicating whether the bounding box coordinates are normalized.
        segments (ndarray): Segments array with shape [N, 1000, 2] after resampling.

    Args:
        bboxes (ndarray): An array of bounding boxes with shape [N, 4]. Can be None if not available. 
        locations (ndarray): An array of locations with shape [N, 2]. Default is None. 
        segments (list | ndarray, optional): A list or array of object segments. Default is None.
        keypoints (ndarray, optional): An array of keypoints with shape [N, 17, 3]. Default is None.
        bbox_format (str, optional): The format of bounding boxes ('xywh' or 'xyxy'). Default is 'xywh'.
        normalized (bool, optional): Whether the bounding box / locations coordinates are normalized. Default is True.

    Examples:
        ```python
        # Create an Instances object
        instances = Instances(
            bboxes=np.array([[10, 10, 30, 30], [20, 20, 40, 40]]),
            segments=[np.array([[5, 5], [10, 10]]), np.array([[15, 15], [20, 20]])],
            keypoints=np.array([[[5, 5, 1], [10, 10, 1]], [[15, 15, 1], [20, 20, 1]]])
        )
        ```

    Note:
        The bounding box format is either 'xywh' or 'xyxy', and is determined by the `bbox_format` argument.
        This class does not perform input validation, and it assumes the inputs are well-formed.
    """

    def __init__(self, bboxes=None, locations=None, segments=None, keypoints=None, bbox_format="xywh", normalized=True) -> None:
        """
        Args:
            bboxes (ndarray): bboxes with shape [N, 4].
            segments (list | ndarray): segments.
            keypoints (ndarray): keypoints(x, y, visible) with shape [N, 17, 3].
        """
        self._bboxes = Bboxes(bboxes=bboxes, format=bbox_format) if bboxes is not None else None
        self.locations = locations
        self.keypoints = keypoints
        self.normalized = normalized
        self.segments = segments

    def convert_bbox(self, format):
        """Convert bounding box format."""
        self._bboxes.convert(format=format)

    @property
    def bbox_areas(self):
        """Calculate the area of bounding boxes."""
        return self._bboxes.areas()

    def scale(self, scale_w, scale_h, bbox_only=False):
        """This might be similar with denormalize func but without normalized sign."""
        if self._bboxes is not None:       
            self._bboxes.mul(scale=(scale_w, scale_h, scale_w, scale_h))
            if bbox_only:
                return
        if self.locations is not None:
            self.locations[:, 0] *= scale_w
            self.locations[:, 1] *= scale_h
        if self.segments is not None:
            self.segments[..., 0] *= scale_w
            self.segments[..., 1] *= scale_h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= scale_w
            self.keypoints[..., 1] *= scale_h

    def denormalize(self, w, h):
        """Denormalizes boxes, segments, and keypoints from normalized coordinates."""
        if not self.normalized:
            return
        if self._bboxes is not None:
            self._bboxes.mul(scale=(w, h, w, h))
        if self.locations is not None:
            self.locations[:, 0] *= w
            self.locations[:, 1] *= h
        if self.segments is not None:
            self.segments[..., 0] *= w
            self.segments[..., 1] *= h
        if self.keypoints is not None:
            self.keypoints[..., 0] *= w
            self.keypoints[..., 1] *= h
        self.normalized = False

    def normalize(self, w, h):
        """Normalize bounding boxes, segments, and keypoints to image dimensions."""
        if self.normalized:
            return
        if self._bboxes is not None:
            self._bboxes.mul(scale=(1 / w, 1 / h, 1 / w, 1 / h))
        if self.locations is not None:
            self.locations[:, 0] /= w
            self.locations[:, 1] /= h
        if self.segments is not None:
            self.segments[..., 0] /= w
            self.segments[..., 1] /= h
        if self.keypoints is not None:
            self.keypoints[..., 0] /= w
            self.keypoints[..., 1] /= h
        self.normalized = True

    def add_padding(self, padw, padh):
        """Handle rect and mosaic situation."""
        assert not self.normalized, "you should add padding with absolute coordinates."
        if self._bboxes is not None:
            self._bboxes.add(offset=(padw, padh, padw, padh))
        if self.locations is not None:
            self.locations[:, 0] += padw
            self.locations[:, 1] += padh
        if self.segments is not None:
            self.segments[..., 0] += padw
            self.segments[..., 1] += padh
        if self.keypoints is not None:
            self.keypoints[..., 0] += padw
            self.keypoints[..., 1] += padh

    def __getitem__(self, index) -> "Instances":
        """
        Retrieve a specific instance or a set of instances using indexing.

        Args:
            index (int, slice, or np.ndarray): The index, slice, or boolean array to select
                                               the desired instances.

        Returns:
            Instances: A new Instances object containing the selected bounding boxes,
                       segments, and keypoints if present.

        Note:
            When using boolean indexing, make sure to provide a boolean array with the same
            length as the number of instances.
        """
        locations = self.locations[index] if self.locations is not None else None
        segments = self.segments[index] if len(self.segments) else self.segments
        keypoints = self.keypoints[index] if self.keypoints is not None else None
        bboxes = self.bboxes[index] if self._bboxes is not None else None
        bbox_format = self._bboxes.format if self._bboxes is not None else None
        return Instances(
            bboxes=bboxes,
            locations=locations,
            segments=segments,
            keypoints=keypoints,
            bbox_format=bbox_format,
            normalized=self.normalized,
        )

    def flipud(self, h):
        """Flips the coordinates of bounding boxes, segments, and keypoints vertically."""
        if self._bboxes is not None:
            if self._bboxes.format == "xyxy":
                y1 = self.bboxes[:, 1].copy()
                y2 = self.bboxes[:, 3].copy()
                self.bboxes[:, 1] = h - y2
                self.bboxes[:, 3] = h - y1
            else:
                self.bboxes[:, 1] = h - self.bboxes[:, 1]
        if self.locations is not None:
            self.locations[:, 1] = h - self.locations[:, 1]
        if self.segments is not None:
            self.segments[..., 1] = h - self.segments[..., 1]
        if self.keypoints is not None:
            self.keypoints[..., 1] = h - self.keypoints[..., 1]

    def fliplr(self, w):
        """Reverses the order of the bounding boxes and segments horizontally."""
        if self._bboxes is not None:
            if self._bboxes.format == "xyxy":
                x1 = self.bboxes[:, 0].copy()
                x2 = self.bboxes[:, 2].copy()
                self.bboxes[:, 0] = w - x2
                self.bboxes[:, 2] = w - x1
            else:
                self.bboxes[:, 0] = w - self.bboxes[:, 0]
        if self.locations is not None:
            self.locations[:, 0] = w - self.locations[:, 0]
        if self.segments is not None:
            self.segments[..., 0] = w - self.segments[..., 0]
        if self.keypoints is not None:
            self.keypoints[..., 0] = w - self.keypoints[..., 0]

    def clip(self, w, h):
        """Clips bounding boxes, segments, and keypoints values to stay within image boundaries.
        In the case of locations, annotations outside of the image broders are removed."""
        good = None
        
        if self._bboxes is not None:
            ori_format = self._bboxes.format
            self.convert_bbox(format="xyxy")
            self.bboxes[:, [0, 2]] = self.bboxes[:, [0, 2]].clip(0, w)
            self.bboxes[:, [1, 3]] = self.bboxes[:, [1, 3]].clip(0, h)
            if ori_format != "xyxy":
                self.convert_bbox(format=ori_format)
        if self.locations is not None:
            good = (self.locations[:, 0] > 0) & (self.locations[:, 0] < w) & (self.locations[:, 1] > 0) & (self.locations[:, 1] < h)
            self.locations = self.locations[good]
        if self.segments is not None:
            self.segments[..., 0] = self.segments[..., 0].clip(0, w)
            self.segments[..., 1] = self.segments[..., 1].clip(0, h)
        if self.keypoints is not None:
            self.keypoints[..., 0] = self.keypoints[..., 0].clip(0, w)
            self.keypoints[..., 1] = self.keypoints[..., 1].clip(0, h)

        return good

    def remove_zero_area_boxes(self):
        """
        Remove zero-area boxes, i.e. after clipping some boxes may have zero width or height.

        This removes them.
        """
        good = self.bbox_areas > 0
        if not all(good):
            self._bboxes = self._bboxes[good]
            if len(self.segments):
                self.segments = self.segments[good]
            if self.keypoints is not None:
                self.keypoints = self.keypoints[good]
        return good

    def update(self, bboxes=None, locations=None, segments=None, keypoints=None):
        """Updates instance variables."""
        if bboxes is not None:
            self._bboxes = Bboxes(bboxes, format=self._bboxes.format)
        if locations is not None:
            self.locations = locations
        if segments is not None:
            self.segments = segments
        if keypoints is not None:
            self.keypoints = keypoints

    def __len__(self):
        """Return the length of the instance list."""
        return len(self.bboxes) if self._bboxes is not None else len(self.locations)

    @classmethod
    def concatenate(cls, instances_list: List["Instances"], axis=0) -> "Instances":
        """
        Concatenates a list of Instances objects into a single Instances object.

        Args:
            instances_list (List[Instances]): A list of Instances objects to concatenate.
            axis (int, optional): The axis along which the arrays will be concatenated. Defaults to 0.

        Returns:
            Instances: A new Instances object containing the concatenated bounding boxes,
                       segments, and keypoints if present.

        Note:
            The `Instances` objects in the list should have the same properties, such as
            the format of the bounding boxes, whether keypoints are present, and if the
            coordinates are normalized.
        """
        assert isinstance(instances_list, (list, tuple))
        if not instances_list:
            return cls(np.empty(0))
        assert all(isinstance(instance, Instances) for instance in instances_list)

        if len(instances_list) == 1:
            return instances_list[0]

        use_locations = instances_list[0].locations is not None
        use_keypoint = instances_list[0].keypoints is not None
        bbox_format = instances_list[0]._bboxes.format if instances_list[0].bboxes is not None else None
        normalized = instances_list[0].normalized

        cat_boxes = np.concatenate([ins.bboxes for ins in instances_list], axis=axis) if not use_locations else None
        cat_locations = np.concatenate([ins.locations for ins in instances_list], axis=axis) if use_locations else None
        cat_segments = np.concatenate([b.segments for b in instances_list], axis=axis) if not use_locations else None
        cat_keypoints = np.concatenate([b.keypoints for b in instances_list], axis=axis) if use_keypoint else None
        return cls(cat_boxes, cat_locations, cat_segments, cat_keypoints, bbox_format, normalized)

    @property
    def bboxes(self):
        """Return bounding boxes."""
        return self._bboxes.bboxes if self._bboxes is not None else None
