# dino

Zero-shot template matching with DINOv2 patch features.

## Features

- Match a template image inside a larger frame
- Accept grayscale or RGB `uint8` images
- Search the full frame or a rectangular region of interest
- Return match position, size, and confidence score
- Run without training or fine-tuning
- Compare results against OpenCV template matching
- Run a synthetic demo for a quick sanity check

## Requirements

**Required**

- Python 3.13+
- PyTorch 2.5+
- NumPy 2.0+
- OpenCV (headless) 4.10+
- Network access on first run to download DINOv2 weights (~85 MB for `dinov2_vits14`)

**Optional**

- CUDA or Apple MPS for faster inference

## Installation

```bash
git clone https://github.com/kodezy/dino
cd dino
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

The first DINOv2 run downloads model weights (~85 MB).

## Configuration

DINOv2 confidence scores are **not** comparable to OpenCV normalized cross-correlation.

| Matcher | Typical threshold range |
| --- | --- |
| OpenCV `TM_CCOEFF_NORMED` | `0.80` – `0.95` |
| dino | `0.40` – `0.55` |

Default threshold in `match_template` is `0.42`. Adjust it per template and dataset.

## Usage

### Python API

```python
import cv2
from dino import Match, match_template

frame = cv2.imread("screenshot.png", cv2.IMREAD_GRAYSCALE)
template = cv2.imread("template.png", cv2.IMREAD_GRAYSCALE)

match = match_template(frame, template, threshold=0.42)
if match:
    print(match.x, match.y, match.w, match.h, match.confidence)
```

Search inside a region of interest:

```python
match = match_template(
    frame,
    template,
    threshold=0.42,
    roi=(100, 200, 500, 600),
)
```

`roi` uses `(x1, y1, x2, y2)` pixel coordinates.

### Scripts

Synthetic demo:

```bash
python scripts/demo.py
```

Compare dino against OpenCV:

```bash
python scripts/compare.py \
  --frame /path/to/frame.png \
  --templates /path/to/templates \
  --limit 20
```

Script options:

| Flag | Default | Description |
| --- | --- | --- |
| `--threshold-opencv` | `0.8` | OpenCV match threshold |
| `--threshold-dino` | `0.42` | dino match threshold |
| `--limit` | `10` | Max templates to compare (`compare.py` only) |

```bash
python scripts/demo.py --help
python scripts/compare.py --help
```

## Architecture

```
src/
  __init__.py   # public exports
  encoder.py    # DINOv2 feature extraction
  match.py      # template matching API
  types.py      # Match result type
scripts/
  compare.py    # benchmark against OpenCV
  demo.py       # synthetic sanity check
tests/
  test_match.py
```

Public API:

- `match_template(image, template, threshold=0.42, roi=None) -> Match | None`
- `Match(confidence, x, y, w, h)`

Position accuracy is tied to DINOv2 patch size (`14` pixels).
