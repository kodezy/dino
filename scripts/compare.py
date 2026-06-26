from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np
from dino.encoder import PATCH_SIZE
from dino.match import match_template

OpenCvMatch = tuple[float, int, int, int, int]


def load_grayscale(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(path)
    return image


def opencv_match(
    image: np.ndarray,
    template: np.ndarray,
    threshold: float,
) -> OpenCvMatch | None:
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)

    if max_val < threshold:
        return None

    template_height, template_width = template.shape[:2]
    return max_val, max_loc[0], max_loc[1], template_width, template_height


def iter_templates(root: Path) -> Iterator[tuple[str, Path]]:
    for path in sorted(root.rglob("*.png")):
        yield path.relative_to(root).as_posix(), path


def positions_agree(opencv: OpenCvMatch, dino_x: int, dino_y: int) -> bool:
    return (
        abs(opencv[1] - dino_x) <= PATCH_SIZE and abs(opencv[2] - dino_y) <= PATCH_SIZE
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare DINOv2 vs OpenCV template matching",
    )
    parser.add_argument(
        "--frame",
        type=Path,
        required=True,
        help="Grayscale frame to search in",
    )
    parser.add_argument(
        "--templates",
        type=Path,
        required=True,
        help="Directory of template PNG files",
    )
    parser.add_argument("--threshold-opencv", type=float, default=0.8)
    parser.add_argument("--threshold-dino", type=float, default=0.42)
    parser.add_argument("--limit", type=int, default=10)
    args = parser.parse_args()

    if not args.frame.is_file():
        print(f"Frame not found: {args.frame}")
        sys.exit(1)

    if not args.templates.is_dir():
        print(f"Templates directory not found: {args.templates}")
        sys.exit(1)

    frame = load_grayscale(args.frame)
    templates = list(iter_templates(args.templates))

    if not templates:
        print(f"No PNG templates found under: {args.templates}")
        sys.exit(1)

    print(f"Frame: {args.frame} ({frame.shape[1]}x{frame.shape[0]})")
    print(f"Templates: {args.templates}")
    print()

    for index, (label, template_path) in enumerate(templates):
        if index >= args.limit:
            break

        template = load_grayscale(template_path)

        started = time.perf_counter()
        opencv = opencv_match(frame, template, args.threshold_opencv)
        opencv_ms = (time.perf_counter() - started) * 1000

        started = time.perf_counter()
        dino = match_template(frame, template, threshold=args.threshold_dino)
        dino_ms = (time.perf_counter() - started) * 1000

        opencv_pos = (
            f"({opencv[1]}, {opencv[2]}) conf={opencv[0]:.3f}" if opencv else "miss"
        )
        dino_pos = (
            f"({dino.x}, {dino.y}) conf={dino.confidence:.3f}" if dino else "miss"
        )
        has_both = opencv is not None and dino is not None
        flag = (
            "OK"
            if has_both and positions_agree(opencv, dino.x, dino.y)
            else ("partial" if has_both else "-")
        )

        print(label)
        print(f"  OpenCV:  {opencv_pos}  {opencv_ms:.0f}ms")
        print(f"  DINOv2:  {dino_pos}  {dino_ms:.0f}ms  [{flag}]")
        print()


if __name__ == "__main__":
    main()
