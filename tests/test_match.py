from unittest.mock import patch

import numpy as np
import torch
from dino.encoder import pad_to_patch, to_rgb
from dino.match import match_template


def test_pad_to_patch():
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    padded, pad_height, pad_width = pad_to_patch(image)

    assert padded.shape[0] % 14 == 0
    assert padded.shape[1] % 14 == 0
    assert pad_height == 12
    assert pad_width == 12


def test_pad_to_patch_grayscale():
    image = np.zeros((100, 100), dtype=np.uint8)
    padded, pad_height, pad_width = pad_to_patch(image)

    assert padded.shape == (112, 112)
    assert pad_height == 12
    assert pad_width == 12


def test_grayscale_to_rgb():
    gray = np.zeros((28, 28), dtype=np.uint8)
    rgb = to_rgb(gray)

    assert rgb.shape == (28, 28, 3)


def test_match_template_finds_injected_pattern():
    template = np.full((28, 28), 200, dtype=np.uint8)
    scene = np.full((140, 140), 80, dtype=np.uint8)
    scene[56:84, 70:98] = template

    def fake_patch_features(image: np.ndarray) -> torch.Tensor:
        grid_height = (image.shape[0] + 13) // 14
        grid_width = (image.shape[1] + 13) // 14
        features = torch.zeros(1, grid_height, grid_width, 4)

        if image.shape == template.shape:
            features[:, :, :, 0] = 1.0
            return features

        patch_y = 56 // 14
        patch_x = 70 // 14
        features[:, patch_y : patch_y + 2, patch_x : patch_x + 2, 0] = 1.0
        return features

    with patch("dino.match.patch_features", side_effect=fake_patch_features):
        result = match_template(scene, template, threshold=0.9)

    assert result is not None
    assert result.x == 70
    assert result.y == 56
