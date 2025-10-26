import io
import time
import numpy as np
from PIL import Image
from pathlib import Path

import torch
import pytest

from app.model_utils import stream_to_tensor
from app.cnn_model import ViolenceCNN


def _make_jpeg_bytes(size):
    """Return JPEG bytes for a randomly generated RGB image of given (width, height)."""
    w, h = size
    arr = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    return buf.getvalue()


@pytest.mark.parametrize("size", [(1280, 720), (640, 360)])
def test_benchmark_image_pipeline(size):
    """Generate an image at given size, convert to model tensor and run a forward pass while measuring time.

    This is a lightweight benchmark for the conversion + inference pipeline. It does NOT load any external
    checkpoint; it instantiates the model class directly so the test stays fast and self-contained.

    The test appends timing results to `tests/benchmark_results.txt` so results are available even when
    pytest captures stdout.
    """
    img_bytes = _make_jpeg_bytes(size)

    # Measure conversion from bytes -> tensor batch
    t0 = time.perf_counter()
    tensor = stream_to_tensor(img_bytes, num_frames=15)
    t_conv = time.perf_counter() - t0

    assert isinstance(tensor, torch.Tensor)
    # expected shape: [num_frames, 3, 224, 224]
    assert tensor.dim() == 4
    assert tensor.shape[1:] == (3, 224, 224)

    # Instantiate model (no checkpoint loaded) and run forward pass
    model = ViolenceCNN(num_classes=2)
    model.to("cpu")

    with torch.no_grad():
        t0 = time.perf_counter()
        out = model(tensor)
        t_infer = time.perf_counter() - t0

    # Basic sanity checks and timings
    assert isinstance(out, torch.Tensor)
    assert t_conv > 0
    assert t_infer > 0

    # Record results to a file so they are visible after the test run
    tests_dir = Path(__file__).parent
    results_file = tests_dir / "benchmark_results.txt"
    with results_file.open("a", encoding="utf-8") as f:
        f.write(f"size={size[0]}x{size[1]}, convert={t_conv:.6f}s, infer={t_infer:.6f}s\n")

    # Also print (pytest -s will show this); file contains canonical record
    print(f"size={size[0]}x{size[1]}: convert={t_conv:.4f}s, infer={t_infer:.4f}s")
