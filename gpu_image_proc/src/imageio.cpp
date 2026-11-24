// imageio.cpp - image I/O via stb, folder scanning, synthetic images, comparison
#define STB_IMAGE_IMPLEMENTATION          // define these in exactly ONE .cpp file,
#include "stb_image.h"                    // otherwise you get duplicate-symbol link errors
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include "imageio.h"
#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

static std::string lower(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return std::tolower(c); });
    return s;
}

Image load_image(const std::string& path, int channels) {
    int w, h, orig_c;
    uchar* px = stbi_load(path.c_str(), &w, &h, &orig_c, channels);   // force channel count
    if (!px) throw std::runtime_error("cannot load '" + path + "': " + stbi_failure_reason());
    Image img;
    img.w = w; img.h = h; img.c = channels;
    img.data.assign(px, px + (size_t)w * h * channels);
    stbi_image_free(px);
    return img;
}

bool save_image(const std::string& path, const uchar* data, int w, int h, int channels) {
    std::string ext = lower(fs::path(path).extension().string());
    if (ext == ".png")                  return stbi_write_png(path.c_str(), w, h, channels, data, w * channels) != 0;
    if (ext == ".jpg" || ext == ".jpeg") return stbi_write_jpg(path.c_str(), w, h, channels, data, 95) != 0;
    if (ext == ".bmp")                  return stbi_write_bmp(path.c_str(), w, h, channels, data) != 0;
    return false;
}

std::vector<std::string> list_images(const std::string& folder) {
    std::vector<std::string> files;
    if (!fs::is_directory(folder)) throw std::runtime_error("not a folder: " + folder);
    for (const auto& e : fs::directory_iterator(folder)) {
        if (!e.is_regular_file()) continue;
        std::string ext = lower(e.path().extension().string());
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp")
            files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());   // same order on every run
    return files;
}

Image make_synthetic(int w, int h, int channels, unsigned seed) {
    Image img;
    img.w = w; img.h = h; img.c = channels;
    img.data.resize((size_t)w * h * channels);
    unsigned s = seed * 2654435761u + 12345u;
    auto rnd = [&s]() { s = s * 1664525u + 1013904223u; return (int)((s >> 24) & 0xFF); };
    const double cx = w / 2.0, cy = h / 2.0, rad = std::min(w, h) / 4.0;
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int r = (int)(255.0 * x / std::max(1, w - 1));            // horizontal gradient
            int g = (int)(255.0 * y / std::max(1, h - 1));            // vertical gradient
            int b = (((x / 32) + (y / 32)) & 1) ? 200 : 40;           // checkerboard -> sharp edges
            double d = std::hypot(x - cx, y - cy);
            if (d < rad) { r = 240; g = 240; b = 60; }                // bright disc
            int noise = rnd() % 16 - 8;                               // small noise
            int v[3] = {r + noise, g + noise, b + noise};
            uchar* p = &img.data[((size_t)y * w + x) * channels];
            for (int k = 0; k < channels; k++) p[k] = (uchar)clampi(v[k % 3], 0, 255);
        }
    return img;
}

DiffResult compare(const uchar* a, const uchar* b, size_t n, int tol) {
    DiffResult r;
    double sum = 0;
    for (size_t i = 0; i < n; i++) {
        int d = std::abs((int)a[i] - (int)b[i]);
        sum += d;
        if (d > r.maxDiff) r.maxDiff = d;
        if (d > tol) r.mismatches++;
    }
    r.avgDiff = n ? sum / n : 0.0;
    return r;
}
