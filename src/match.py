from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

from .encoder import PATCH_SIZE, patch_features
from .types import Match


def _crop(
    image: NDArray[np.uint8],
    bbox: tuple[int, int, int, int],
) -> NDArray[np.uint8]:
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


def _correlation_map(
    image_features: torch.Tensor,
    template_features: torch.Tensor,
) -> torch.Tensor | None:
    template_grid_height = template_features.shape[1]
    template_grid_width = template_features.shape[2]
    image_grid_height = image_features.shape[1]
    image_grid_width = image_features.shape[2]

    if (
        image_grid_height < template_grid_height
        or image_grid_width < template_grid_width
    ):
        return None

    image_channels = image_features.permute(0, 3, 1, 2)
    template_kernel = template_features.permute(0, 3, 1, 2)
    kernel_height, kernel_width = template_kernel.shape[-2:]
    scores = F.conv2d(image_channels, template_kernel)
    return scores.squeeze(0).squeeze(0) / (kernel_height * kernel_width)


def match_template(
    image: NDArray[np.uint8],
    template: NDArray[np.uint8],
    threshold: float = 0.42,
    *,
    roi: tuple[int, int, int, int] | None = None,
) -> Match | None:
    offset_x, offset_y = 0, 0
    search_image = image

    if roi is not None:
        offset_x, offset_y, _, _ = roi
        search_image = _crop(image, roi)

    template_height, template_width = template.shape[:2]

    if (
        search_image.shape[0] < template_height
        or search_image.shape[1] < template_width
    ):
        return None

    scores = _correlation_map(patch_features(search_image), patch_features(template))

    if scores is None:
        return None

    flat_index = int(scores.argmax().item())
    grid_width = scores.shape[1]
    grid_y, grid_x = divmod(flat_index, grid_width)
    confidence = float(scores[grid_y, grid_x].item())

    if confidence < threshold:
        return None

    return Match(
        confidence=confidence,
        x=offset_x + grid_x * PATCH_SIZE,
        y=offset_y + grid_y * PATCH_SIZE,
        w=template_width,
        h=template_height,
    )
