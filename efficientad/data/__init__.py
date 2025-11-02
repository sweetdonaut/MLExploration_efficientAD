# Copyright (C) 2024-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Data structures and utilities for EfficientAD."""

from .dataclasses import Batch, InferenceBatch
from .utils import DownloadInfo, download_and_extract

__all__ = ["Batch", "InferenceBatch", "DownloadInfo", "download_and_extract"]
