from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from numpy.typing import NDArray

PATCH_SIZE = 14
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

_encoder: torch.nn.Module | None = None
_device: torch.device | None = None


def _resolve_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _load_encoder() -> tuple[torch.nn.Module, torch.device]:
    global _encoder, _device

    if _encoder is None:
        _device = _resolve_device()
        _encoder = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
        _encoder.eval()
        _encoder.to(_device)

    return _encoder, _device  # type: ignore[return-value]


def to_rgb(image: NDArray[np.uint8]) -> NDArray[np.uint8]:
    if image.ndim == 2:
        return np.stack([image, image, image], axis=-1)
    if image.shape[2] == 4:
        return image[:, :, :3]
    return image


def pad_to_patch(image: NDArray[np.uint8]) -> tuple[NDArray[np.uint8], int, int]:
    height, width = image.shape[:2]
    pad_height = (PATCH_SIZE - height % PATCH_SIZE) % PATCH_SIZE
    pad_width = (PATCH_SIZE - width % PATCH_SIZE) % PATCH_SIZE

    if pad_height == 0 and pad_width == 0:
        return image, 0, 0

    pad_width_spec = ((0, pad_height), (0, pad_width))
    if image.ndim == 3:
        pad_width_spec = (*pad_width_spec, (0, 0))

    padded = np.pad(image, pad_width_spec, mode="edge")
    return padded, pad_height, pad_width


def _to_tensor(image: NDArray[np.uint8], device: torch.device) -> torch.Tensor:
    rgb = to_rgb(image)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).to(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(3, 1, 1)
    return ((tensor - mean) / std).unsqueeze(0)


def patch_features(image: NDArray[np.uint8]) -> torch.Tensor:
    encoder, device = _load_encoder()
    rgb = to_rgb(image)
    padded, _, _ = pad_to_patch(rgb)
    tensor = _to_tensor(padded, device)
    grid_height = padded.shape[0] // PATCH_SIZE
    grid_width = padded.shape[1] // PATCH_SIZE

    with torch.inference_mode():
        tokens = encoder.forward_features(tensor)["x_norm_patchtokens"]
        features = tokens.reshape(1, grid_height, grid_width, -1)
        return F.normalize(features, dim=-1)
