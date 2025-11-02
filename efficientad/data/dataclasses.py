# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Data structures for EfficientAD.

This module provides PyTorch-based dataclasses used in EfficientAD for handling
input data and model predictions.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, fields, is_dataclass, replace
from types import NoneType
from typing import Any, Generic, NamedTuple, TypeVar, get_args, get_type_hints

import numpy as np
import torch
from torchvision.tv_tensors import Image, Mask, Video

# Type variables
ImageT = TypeVar("ImageT", Image, Video, np.ndarray)
T = TypeVar("T", torch.Tensor, np.ndarray)
MaskT = TypeVar("MaskT", Mask, np.ndarray)
PathT = TypeVar("PathT", list[str], str)
Instance = TypeVar("Instance")
Value = TypeVar("Value")


class InferenceBatch(NamedTuple):
    """Batch for use in torch and inference models.

    Args:
        pred_score (torch.Tensor | None): Predicted anomaly scores.
            Defaults to ``None``.
        pred_label (torch.Tensor | None): Predicted anomaly labels.
            Defaults to ``None``.
        anomaly_map (torch.Tensor | None): Generated anomaly maps.
            Defaults to ``None``.
        pred_mask (torch.Tensor | None): Predicted anomaly masks.
            Defaults to ``None``.
    """

    pred_score: torch.Tensor | None = None
    pred_label: torch.Tensor | None = None
    anomaly_map: torch.Tensor | None = None
    pred_mask: torch.Tensor | None = None


class FieldDescriptor(Generic[Value]):
    """Descriptor for Anomalib's dataclass fields.

    Using a descriptor ensures that the values of dataclass fields can be
    validated before being set. This allows validation of the input data not
    only when it is first set, but also when it is updated.

    Args:
        validator_name: Name of the validator method to call when setting value.
            Defaults to ``None``.
        default: Default value for the field. Defaults to ``None``.
    """

    def __init__(self, validator_name: str | None = None, default: Value | None = None) -> None:
        self.validator_name = validator_name
        self.default = default

    def __set_name__(self, owner: type[Instance], name: str) -> None:
        """Set the name of the descriptor."""
        self.name = name

    def __get__(self, instance: Instance | None, owner: type[Instance]) -> Value | None:
        """Get the value of the descriptor."""
        if instance is None:
            if self.default is not None or self.is_optional(owner):
                return self.default
            msg = f"No default attribute value specified for field '{self.name}'."
            raise AttributeError(msg)
        return instance.__dict__[self.name]

    def __set__(self, instance: object, value: Value) -> None:
        """Set the value of the descriptor."""
        if self.validator_name is not None:
            validator = getattr(instance, self.validator_name)
            value = validator(value)
        instance.__dict__[self.name] = value

    def get_types(self, owner: type[Instance]) -> tuple[type, ...]:
        """Get the types of the descriptor."""
        try:
            types = get_args(get_type_hints(owner)[self.name])
            return get_args(types[0]) if hasattr(types[0], "__args__") else (types[0],)
        except (KeyError, TypeError, AttributeError) as e:
            msg = f"Unable to determine types for {self.name} in {owner}"
            raise TypeError(msg) from e

    def is_optional(self, owner: type[Instance]) -> bool:
        """Check if the descriptor is optional."""
        return NoneType in self.get_types(owner)


@dataclass
class _InputFields(Generic[T, ImageT, MaskT, PathT], ABC):
    """Generic dataclass that defines the standard input fields for Anomalib.

    Attributes:
        image: Input image or video
        gt_label: Ground truth label
        gt_mask: Ground truth segmentation mask
        mask_path: Path to mask file
    """

    image: FieldDescriptor[ImageT] = FieldDescriptor(validator_name="validate_image")
    gt_label: FieldDescriptor[T | None] = FieldDescriptor(validator_name="validate_gt_label")
    gt_mask: FieldDescriptor[MaskT | None] = FieldDescriptor(validator_name="validate_gt_mask")
    mask_path: FieldDescriptor[PathT | None] = FieldDescriptor(validator_name="validate_mask_path")

    @staticmethod
    @abstractmethod
    def validate_image(image: ImageT) -> ImageT:
        """Validate the image."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_gt_mask(gt_mask: MaskT | None) -> MaskT | None:
        """Validate the ground truth mask."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_mask_path(mask_path: PathT | None) -> PathT | None:
        """Validate the mask path."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_gt_label(gt_label: T) -> T | None:
        """Validate the ground truth label."""
        raise NotImplementedError


@dataclass
class _OutputFields(Generic[T, MaskT, PathT], ABC):
    """Generic dataclass that defines the standard output fields for Anomalib.

    Attributes:
        anomaly_map: Predicted anomaly heatmap
        pred_score: Predicted anomaly score
        pred_mask: Predicted segmentation mask
        pred_label: Predicted label
        explanation: Path to explanation visualization
    """

    anomaly_map: FieldDescriptor[MaskT | None] = FieldDescriptor(validator_name="validate_anomaly_map")
    pred_score: FieldDescriptor[T | None] = FieldDescriptor(validator_name="validate_pred_score")
    pred_mask: FieldDescriptor[MaskT | None] = FieldDescriptor(validator_name="validate_pred_mask")
    pred_label: FieldDescriptor[T | None] = FieldDescriptor(validator_name="validate_pred_label")
    explanation: FieldDescriptor[PathT | None] = FieldDescriptor(validator_name="validate_explanation")

    @staticmethod
    @abstractmethod
    def validate_anomaly_map(anomaly_map: MaskT) -> MaskT | None:
        """Validate the anomaly map."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_pred_score(pred_score: T) -> T | None:
        """Validate the predicted score."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_pred_mask(pred_mask: MaskT) -> MaskT | None:
        """Validate the predicted mask."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_pred_label(pred_label: T) -> T | None:
        """Validate the predicted label."""
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def validate_explanation(explanation: PathT) -> PathT | None:
        """Validate the explanation."""
        raise NotImplementedError


@dataclass
class UpdateMixin:
    """Mixin class for dataclasses that allows for in-place replacement of attrs."""

    def update(self, in_place: bool = True, **changes) -> Any:  # noqa: ANN401
        """Replace fields in place and call __post_init__ to reinitialize.

        Args:
            in_place: Whether to modify in place or return new instance
            **changes: Field names and new values to update

        Returns:
            Updated instance (self if in_place=True, new instance otherwise)
        """
        if not is_dataclass(self):
            msg = "replace can only be used with dataclass instances"
            raise TypeError(msg)

        if in_place:
            for field in fields(self):
                if field.init and field.name in changes:
                    setattr(self, field.name, changes[field.name])
            if hasattr(self, "__post_init__"):
                self.__post_init__()
            return self
        return replace(self, **changes)


@dataclass
class _GenericBatch(
    UpdateMixin,
    Generic[T, ImageT, MaskT, PathT],
    _OutputFields[T, MaskT, PathT],
    _InputFields[T, ImageT, MaskT, PathT],
):
    """Generic dataclass for a batch of items in Anomalib datasets.

    This class represents a batch of data items, combining both input and output
    fields for anomaly detection tasks.
    """


@dataclass
class Batch(Generic[ImageT], _GenericBatch[torch.Tensor, ImageT, Mask, list[str]]):
    """Base dataclass for batches of items in Anomalib datasets using PyTorch.

    This class extends the generic ``_GenericBatch`` class to provide a
    PyTorch-specific implementation for batches of data in Anomalib datasets.
    """

    def keys(self, include_none: bool = True) -> list[str]:
        """Return a list of field names in the Batch.

        Args:
            include_none: If True, returns all possible field names including those
                that are None. If False, returns only field names that have non-None values.

        Returns:
            List of field names that can be accessed on this Batch instance.
        """
        if include_none:
            return [field.name for field in fields(self)]
        return [field.name for field in fields(self) if getattr(self, field.name) is not None]

    def __getitem__(self, key: str) -> Any:  # noqa: ANN401
        """Get a field value using dictionary-like syntax.

        Args:
            key: Field name to access.

        Returns:
            The value of the specified field.

        Raises:
            KeyError: If the field name is not found.
        """
        if not hasattr(self, key):
            msg = f"Field '{key}' not found in {self.__class__.__name__}"
            raise KeyError(msg)
        return getattr(self, key)

    @staticmethod
    def validate_image(image: ImageT) -> ImageT:
        """Validate the image (pass-through for PyTorch tensors)."""
        return image

    @staticmethod
    def validate_gt_label(gt_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the ground truth label."""
        return gt_label

    @staticmethod
    def validate_gt_mask(gt_mask: Mask | None) -> Mask | None:
        """Validate the ground truth mask."""
        return gt_mask

    @staticmethod
    def validate_mask_path(mask_path: list[str] | None) -> list[str] | None:
        """Validate the mask path."""
        return mask_path

    @staticmethod
    def validate_anomaly_map(anomaly_map: Mask | None) -> Mask | None:
        """Validate the anomaly map."""
        return anomaly_map

    @staticmethod
    def validate_pred_score(pred_score: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the predicted score."""
        return pred_score

    @staticmethod
    def validate_pred_mask(pred_mask: Mask | None) -> Mask | None:
        """Validate the predicted mask."""
        return pred_mask

    @staticmethod
    def validate_pred_label(pred_label: torch.Tensor | None) -> torch.Tensor | None:
        """Validate the predicted label."""
        return pred_label

    @staticmethod
    def validate_explanation(explanation: list[str] | None) -> list[str] | None:
        """Validate the explanation."""
        return explanation
