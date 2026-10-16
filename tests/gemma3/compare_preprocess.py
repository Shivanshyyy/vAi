"""Compare ImageProcessor's Gemma3 preprocessing with the local Gemma3 processor.

Run in the vai environment after examples/gemma3/demo.py:
    python tests/gemma3/compare_preprocess.py --dump tests/gemma3/dumps/preprocess-reference/pixel_values.npy

Compiles only vai/utils/image-processor.cpp in a temporary folder; no model
weights or vAi runtime are needed.
"""
import argparse
import ctypes
from pathlib import Path
import subprocess
import tempfile

import numpy as np
from PIL import Image
from transformers import AutoProcessor

ROOT = Path(__file__).resolve().parents[2]


def metrics(label, actual, expected):
    diff = np.abs(actual.astype(np.float64) - expected.astype(np.float64))
    print(f"{label}: MAE={diff.mean():.10g}, max={diff.max():.10g}, "
          f"different={np.count_nonzero(diff)}/{diff.size}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dump', type=Path)
    parser.add_argument('--assert-match', action='store_true')
    args = parser.parse_args()
    processor = AutoProcessor.from_pretrained(ROOT / 'data/gemma3/4B', local_files_only=True).image_processor
    print(f'processor: {type(processor).__module__}.{type(processor).__name__}')
    wrapper = '''
#include "image-processor.h"
#include "stb_image.h"
#include <algorithm>
#include <cstddef>
#include <exception>
extern "C" int preprocess(const uint8_t* p, size_t n, uint32_t w, uint32_t h, uint32_t s, float* out) {
    try {
        if (n != size_t(w) * h * 3) return -1;
        std::vector<float> result = ImageProcessor(p, w, h, 3)
            .resize(s, Resize::BilinearAntialiasInt16)
            .normalize(-1.0f, 1.0f)
            .chw();
        std::copy(result.begin(), result.end(), out);
        return 0;
    } catch (const std::exception&) { return -1; }
}
extern "C" void decode(const char* path, uint8_t* out) {
    int w, h, c;
    uint8_t* rgb = stbi_load(path, &w, &h, &c, 3);
    std::copy(rgb, rgb + size_t(w) * h * 3, out);
    stbi_image_free(rgb);
}
'''
    with tempfile.TemporaryDirectory(prefix='gemma-preprocess-') as tmp:
        cpp = Path(tmp) / 'compare.cpp'
        so = Path(tmp) / 'compare.so'
        cpp.write_text(wrapper)
        subprocess.run(['g++', '-std=c++20', '-O2', '-shared', '-fPIC',
                        '-I', str(ROOT / 'vai/utils'), str(cpp),
                        str(ROOT / 'vai/utils/image-processor.cpp'), '-o', str(so)], check=True)
        lib = ctypes.CDLL(str(so))
        lib.preprocess.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint32,
                                  ctypes.c_uint32, ctypes.c_uint32, ctypes.c_void_p]
        lib.preprocess.restype = ctypes.c_int
        lib.decode.argtypes = [ctypes.c_char_p, ctypes.c_void_p]
        lib.decode.restype = None

        def run(rgb, size):
            rgb = np.ascontiguousarray(rgb)
            out = np.empty((3, size, size), dtype=np.float32)
            status = lib.preprocess(rgb.ctypes.data, rgb.size, rgb.shape[1], rgb.shape[0], size, out.ctypes.data)
            assert status == 0
            return out

        def reference(rgb, size):
            return processor(images=Image.fromarray(rgb), size={'height': size, 'width': size},
                             input_data_format='channels_first', return_tensors='pt')['pixel_values'][0].numpy()

        image_path = ROOT / 'data/gemma3/image.jpg'
        rgb = np.asarray(Image.open(image_path).convert('RGB'))
        print(f'image: {rgb.shape}, dtype={rgb.dtype}, range=[{rgb.min()}, {rgb.max()}]')
        expected = reference(rgb, 896)
        actual = run(rgb, 896)
        metrics('same RGB / processor', actual, expected)
        if args.dump:
            dump = np.load(args.dump)
            print(f'dump: {dump.shape}, dtype={dump.dtype}, range=[{dump.min()}, {dump.max()}]')
            metrics('processor / saved dump', expected, dump)
            metrics('C++ same RGB / saved dump', actual, dump)
            np.testing.assert_array_equal(expected, dump)
        stb = np.empty_like(rgb)
        lib.decode(str(image_path).encode(), stb.ctypes.data)
        metrics('STB / Pillow decoded RGB (0..255)', stb, rgb)
        metrics('C++ STB end-to-end / processor Pillow', run(stb, 896), expected)
        if args.assert_match:
            np.testing.assert_array_equal(actual, expected)
        # An already resized image must preserve every pixel before normalization.
        square = np.asarray(Image.fromarray(rgb).resize((896, 896), Image.Resampling.BILINEAR))
        square_actual = run(square, 896)
        square_reference = reference(square, 896)
        identity = (square.transpose(2, 0, 1).astype(np.float32) - np.float32(127.5)) / np.float32(127.5)
        metrics('project image 896x896->896 / processor', square_actual, square_reference)
        metrics('project image 896x896->896 / normalization only', square_actual, identity)
        if args.assert_match:
            np.testing.assert_array_equal(square_actual, square_reference)
            np.testing.assert_array_equal(square_actual, identity)
        rng = np.random.default_rng(42)
        for h, w, size in [(17,23,41), (83,97,29), (43,19,29), (1,7,13),
                           (7,1,3), (1,1,1), (19,19,19), (19,29,29), (29,19,29), (13,17,1),
                           (896,896,896), (448,448,896), (1792,1792,896),
                           (897,895,896)]:
            test = rng.integers(0,256,(h,w,3),dtype=np.uint8)
            a, b = run(test, size), reference(test, size)
            metrics(f'random {w}x{h}->{size}', a, b)
            if args.assert_match:
                np.testing.assert_array_equal(a, b)
        # Check invalid dimensions and a mismatched input buffer without allocating huge arrays.
        dummy = np.zeros(3, dtype=np.uint8)
        out = np.zeros(3, dtype=np.float32)
        for n,w,h,s in [(3,0,1,1),(3,1,0,1),(3,1,1,0),(2,1,1,1)]:
            assert lib.preprocess(dummy.ctypes.data,n,w,h,s,out.ctypes.data) == -1


if __name__ == '__main__':
    main()
