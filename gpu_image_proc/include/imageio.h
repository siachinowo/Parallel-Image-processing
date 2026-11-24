// imageio.h - image loading/saving (stb_image), folder scan, synthetic images, comparison
#pragma once
#include <string>
#include <vector>
#include "common.h"

struct Image {
    int w = 0, h = 0, c = 0;
    std::vector<uchar> data;   // interleaved: R,G,B,R,G,B,... row after row
};

// Loads PNG/JPG/BMP etc. and forces `channels` (3 = RGB, 1 = gray). Throws on failure.
Image load_image(const std::string& path, int channels = 3);

// Saves by file extension (.png, .jpg/.jpeg, .bmp). Returns false on failure.
bool save_image(const std::string& path, const uchar* data, int w, int h, int channels);

// Sorted list of image files (png/jpg/jpeg/bmp) in a folder.
std::vector<std::string> list_images(const std::string& folder);

// Deterministic test image with gradients, shapes and noise (so Sobel finds edges).
// Lets the benchmark run at any size without needing image files.
Image make_synthetic(int w, int h, int channels, unsigned seed = 1);

// Pixel-by-pixel comparison of two buffers.
struct DiffResult {
    int maxDiff = 0;
    double avgDiff = 0.0;
    long long mismatches = 0;   // pixels differing by more than `tol`
};
DiffResult compare(const uchar* a, const uchar* b, size_t n, int tol = 0);
