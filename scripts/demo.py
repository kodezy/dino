from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
from dino.encoder import PATCH_SIZE
from dino.match import match_template


def build_template(rng: np.random.Generator) -> np.ndarray:
    template = np.zeros((42, 84), dtype=np.uint8)
    template[0::2, :] = 210
    template[1::2, :] = 120
    template[:, 0::3] = rng.integers(80, 180, template[:, 0::3].shape, dtype=np.uint8)
    return template


def build_scene(template: np.ndarray, position: tuple[int, int]) -> np.ndarray:
    height, width = 420, 560
    scene = np.random.default_rng(0).integers(40, 180, (height, width), dtype=np.uint8)
    x, y = position
    template_height, template_width = template.shape[:2]
    scene[y : y + template_height, x : x + template_width] = template
    return scene


def opencv_match(
    image: np.ndarray,
    template: np.ndarray,
    threshold: float,
) -> tuple[float, int, int, int, int] | None:
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < threshold:
        return None

    template_height, template_width = template.shape[:2]
    return max_val, max_loc[0], max_loc[1], template_width, template_height


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic DINOv2 vs OpenCV template matching demo",
    )
    parser.add_argument("--threshold-opencv", type=float, default=0.8)
    parser.add_argument("--threshold-dino", type=float, default=0.42)
    args = parser.parse_args()

    rng = np.random.default_rng(7)
    template = build_template(rng)
    expected = (180, 95)
    scene = build_scene(template, expected)

    print("Loading DINOv2 (first run downloads weights)...")

    started = time.perf_counter()
    dino_result = match_template(scene, template, threshold=args.threshold_dino)
    dino_ms = (time.perf_counter() - started) * 1000

    started = time.perf_counter()
    opencv_result = opencv_match(scene, template, args.threshold_opencv)
    opencv_ms = (time.perf_counter() - started) * 1000

    print(f"Expected position: x={expected[0]}, y={expected[1]}")
    print()

    if opencv_result:
        confidence, x, y, _, _ = opencv_result
        print(f"OpenCV: conf={confidence:.3f} x={x} y={y} {opencv_ms:.1f}ms")
    else:
        print(f"OpenCV: no match {opencv_ms:.1f}ms")

    if dino_result:
        offset_x = abs(dino_result.x - expected[0])
        offset_y = abs(dino_result.y - expected[1])
        is_hit = offset_x <= PATCH_SIZE and offset_y <= PATCH_SIZE
        print(
            f"DINOv2: conf={dino_result.confidence:.3f} "
            f"x={dino_result.x} y={dino_result.y} {dino_ms:.1f}ms "
            f"{'HIT' if is_hit else 'MISS'}",
        )
    else:
        print(f"DINOv2: no match {dino_ms:.1f}ms")


if __name__ == "__main__":
    main()
