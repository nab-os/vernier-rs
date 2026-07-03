"""
bench.py — vernier-py binding benchmark.

Generates a synthetic sinusoidal image in NumPy and times detect_periodic()
across N iterations.  Prints one line to stdout:

    binding=python backend=cpu size=512x512 iters=N mean=X.XXms best=X.XXms

When the requested backend is unavailable the script exits with code 2 and
prints a message to stderr (no stdout line); the orchestrator treats this as
"n/a".

Usage
-----
Build and install the extension first:

    cd vernier-rs
    maturin build -p vernier-py --release
    pip install target/wheels/vernier_py-*.whl

Then run:

    python vernier-py/examples/bench.py [--size 512] [--iters 50] [--backend cpu|cuda]
"""

import argparse
import math
import sys
import time

import numpy as np

import vernier_py


def synthetic_image(size: int) -> "np.ndarray[np.float32]":
    period = size / 16.0
    theta = 0.1
    cx, cy = size / 2.0, size / 2.0
    c, s = math.cos(-theta), math.sin(-theta)
    cols = np.arange(size, dtype=np.float32)
    rows = np.arange(size, dtype=np.float32)
    dx = cols[np.newaxis, :] - cx
    dy = rows[:, np.newaxis] - cy
    xp = c * dx - s * dy
    return (0.5 + 0.5 * np.cos(2.0 * math.pi * xp / period)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size",    type=int, default=512)
    ap.add_argument("--iters",   type=int, default=50)
    ap.add_argument("--backend", choices=["cpu", "cuda"], default="cpu")
    args = ap.parse_args()

    if args.backend == "cuda":
        try:
            det = vernier_py.Detector.cuda()
        except RuntimeError as e:
            print(f"cuda unavailable: {e}", file=sys.stderr)
            sys.exit(2)
    else:
        det = vernier_py.Detector()

    img = synthetic_image(args.size)
    period = args.size / 16.0

    # Warm-up: prime FFT/GPU pipeline cache.
    det.detect_periodic(img, period=period, min_frequency=5, max_frequency=0)

    best = float("inf")
    total = 0.0
    for _ in range(args.iters):
        t0 = time.perf_counter()
        det.detect_periodic(img, period=period, min_frequency=5, max_frequency=0)
        ms = (time.perf_counter() - t0) * 1e3
        total += ms
        if ms < best:
            best = ms

    mean = total / args.iters
    print(
        f"binding=python backend={args.backend} size={args.size}x{args.size}"
        f" iters={args.iters} mean={mean:.2f}ms best={best:.2f}ms"
    )


if __name__ == "__main__":
    main()
