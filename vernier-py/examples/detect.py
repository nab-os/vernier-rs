"""
detect.py — vernier-py quick-start example.

Loads the reference 9 µm / 12-bit Megarena image and runs both periodic
and absolute detection.  Expected output (matches vernier-cli):

    megarena  x=-4452.12  y=-26759.51  theta=1.429510

Usage
-----
Build and install the extension first, then run from any directory:

    cd vernier-rs
    maturin build -p vernier-py --release
    pip install target/wheels/vernier_py-*.whl
    python vernier-py/examples/detect.py

Dependencies: numpy, Pillow  (pip install numpy pillow)
"""

from pathlib import Path

import numpy as np
from PIL import Image

import vernier_py

# ── Image path ────────────────────────────────────────────────────────────────

HERE    = Path(__file__).resolve().parent
IMG_DIR = HERE / ".." / ".." / "resources" / "images"
IMG     = IMG_DIR / "megarenaPatternImage_12bits_9um.jpg"


def load_gray_f32(path: Path) -> "np.ndarray[np.float32]":
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def main() -> None:
    det = vernier_py.Detector()
    print(f"backend : {det.backend_name()}")

    img = load_gray_f32(IMG)
    print(f"image   : {img.shape[1]}×{img.shape[0]}  ({IMG.name})")

    # ── Periodic detection ────────────────────────────────────────────────────
    pose = det.detect_periodic(
        img, period=9.0, min_frequency=20, max_frequency=500
    )
    print(f"periodic  x={pose.x:.4f}  y={pose.y:.4f}  theta={pose.theta:.6f}")

    # ── Megarena absolute detection ───────────────────────────────────────────
    pose = det.detect_megarena(
        img, physical_period=9.0, code_size=12,
        min_frequency=20, max_frequency=500
    )
    print(f"megarena  x={pose.x:.2f}  y={pose.y:.2f}  theta={pose.theta:.6f}")
    print(f"expected  x≈-4452      y≈-26759  (matches vernier-cli)")

    # ── CUDA (graceful fallback) ──────────────────────────────────────────────
    try:
        gpu  = vernier_py.Detector.cuda()
        pg   = gpu.detect_megarena(
            img, physical_period=9.0, code_size=12,
            min_frequency=20, max_frequency=500
        )
        print(f"cuda      x={pg.x:.2f}  y={pg.y:.2f}")
    except RuntimeError as e:
        print(f"cuda      not available: {e}")


if __name__ == "__main__":
    main()
