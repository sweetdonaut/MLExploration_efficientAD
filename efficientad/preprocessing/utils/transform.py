# Copyright (C) 2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Utility functions for transforms.

This module provides utility functions for managing transforms in the pre-processing
pipeline.
"""

import copy

from torchvision.transforms.v2 import CenterCrop, Compose, Resize, Transform

from efficientad.data.transforms import ExportableCenterCrop


def get_exportable_transform(transform: Transform | None) -> Transform | None:
    """Get an exportable version of a transform.

    This function converts a torchvision transform into a format that is compatible with
    ONNX and OpenVINO export. It handles two main compatibility issues:

    1. Disables antialiasing in ``Resize`` transforms
    2. Converts ``CenterCrop`` to ``ExportableCenterCrop``

    Args:
        transform (Transform | None): The transform to convert. If ``None``, returns
            ``None``.

    Returns:
        Transform | None: The converted transform that is compatible with ONNX/OpenVINO
            export. Returns ``None`` if input transform is ``None``.
    """
    if transform is None:
        return None
    transform = copy.deepcopy(transform)
    transform = disable_antialiasing(transform)
    return convert_center_crop_transform(transform)


def disable_antialiasing(transform: Transform) -> Transform:
    """Disable antialiasing in Resize transforms.

    This function recursively disables antialiasing in any ``Resize`` transforms found
    within the provided transform or transform composition. This is necessary because
    antialiasing is not supported during ONNX export.

    Args:
        transform (Transform): Transform or composition of transforms to process.

    Returns:
        Transform: The processed transform with antialiasing disabled in any
            ``Resize`` transforms.
    """
    if isinstance(transform, Resize):
        transform.antialias = False
    if isinstance(transform, Compose):
        for tr in transform.transforms:
            disable_antialiasing(tr)
    return transform


def convert_center_crop_transform(transform: Transform) -> Transform:
    """Convert torchvision's CenterCrop to ExportableCenterCrop.

    This function recursively converts any ``CenterCrop`` transforms found within the
    provided transform or transform composition to ``ExportableCenterCrop``. This is
    necessary because torchvision's ``CenterCrop`` is not supported during ONNX
    export.

    Args:
        transform (Transform): Transform or composition of transforms to process.

    Returns:
        Transform: The processed transform with all ``CenterCrop`` transforms
            converted to ``ExportableCenterCrop``.
    """
    if isinstance(transform, CenterCrop):
        transform = ExportableCenterCrop(size=transform.size)
    if isinstance(transform, Compose):
        for index in range(len(transform.transforms)):
            tr = transform.transforms[index]
            transform.transforms[index] = convert_center_crop_transform(tr)
    return transform
