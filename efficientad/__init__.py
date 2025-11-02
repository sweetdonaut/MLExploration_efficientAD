# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""EfficientAD - Standalone implementation extracted from Anomalib.

This module provides a standalone implementation of the EfficientAD model
for anomaly detection, extracted from the Anomalib library.
"""

from enum import Enum

__version__ = "1.0.0"


class LearningType(str, Enum):
    """Learning type defining how the model learns from the dataset samples.

    This enum defines the different learning paradigms supported by anomalib models:

        - ``ONE_CLASS``: Model learns from a single class of normal samples
        - ``ZERO_SHOT``: Model learns without any task-specific training samples
        - ``FEW_SHOT``: Model learns from a small number of training samples

    Example:
        >>> from efficientad import LearningType
        >>> learning_type = LearningType.ONE_CLASS
        >>> print(learning_type)
        'one_class'

    Note:
        The learning type affects how the model is trained and what kind of data
        it expects during training.
    """

    ONE_CLASS = "one_class"
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"


class TaskType(str, Enum):
    """Task type defining the model's prediction output format.

    This enum defines the different task types supported by anomalib models:

        - ``CLASSIFICATION``: Model predicts anomaly scores at the image level
        - ``SEGMENTATION``: Model predicts pixel-wise anomaly scores and masks

    Example:
        >>> from efficientad import TaskType
        >>> task_type = TaskType.CLASSIFICATION
        >>> print(task_type)
        'classification'

    Note:
        The task type determines:
            - The model architecture and output format
            - Required ground truth annotation format
            - Evaluation metrics used
            - Visualization methods available
    """

    CLASSIFICATION = "classification"
    SEGMENTATION = "segmentation"


__all__ = ["LearningType", "TaskType"]
