"""The engine-wide utilities: ImageProcessor."""
import numpy as np

import pyvai


def test_image_processor_output_layout_and_normalize():
    """Pixels are RGB, HWC and 0..255 unless a step says otherwise; chw(), bgr()
    and normalize() describe the output, and each step returns the processor."""
    image = np.zeros((2, 3, 3), dtype=np.uint8)
    image[...] = (10, 20, 30)
    proc = pyvai.ImageProcessor(image)
    assert proc.chw() is proc
    hwc = pyvai.ImageProcessor(image).to_numpy()
    assert hwc.shape == (2, 3, 3) and hwc.dtype == np.float32
    np.testing.assert_array_equal(hwc, image.astype(np.float32))

    chw = pyvai.ImageProcessor(image).chw().bgr().normalize(-1.0, 1.0).to_numpy()
    assert chw.shape == (3, 2, 3)
    np.testing.assert_allclose(chw[:, 0, 0], np.array([30, 20, 10]) / 127.5 - 1.0, atol=1e-6)

    gray = (pyvai.ImageProcessor(np.full((4, 5), 255, dtype=np.uint8))
            .normalize(0.0, 1.0).chw().to_numpy())
    assert gray.shape == (1, 4, 5) and (gray == 1.0).all()
