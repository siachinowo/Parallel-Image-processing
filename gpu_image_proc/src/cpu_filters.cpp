// cpu_filters.cpp - sequential reference implementations (the speed baseline AND the
// correctness reference). Same math, rounding and border rule as the GPU kernels.
#include "filters.h"
#include <cmath>

// Fixed-point version of gray = 0.299R + 0.587G + 0.114B using weights scaled by 65536
// (19595 + 38470 + 7471 = 65536). Integer math => identical result on CPU and GPU.
void grayscale_cpu(const uchar* rgb, uchar* out, int w, int h) {
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            const uchar* p = &rgb[((size_t)y * w + x) * 3];
            out[(size_t)y * w + x] =
                (uchar)((19595 * p[0] + 38470 * p[1] + 7471 * p[2] + 32768) >> 16);
        }
}

void blur_cpu(const uchar* gray, uchar* out, int w, int h, int ksize) {
    const int r = ksize / 2, total = gauss_total(ksize);
    const int* k = gauss1d(ksize);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int sum = 0;
            for (int dy = -r; dy <= r; dy++) {
                int yy = clampi(y + dy, 0, h - 1);              // clamp at the borders
                for (int dx = -r; dx <= r; dx++) {
                    int xx = clampi(x + dx, 0, w - 1);
                    sum += k[dy + r] * k[dx + r] * gray[(size_t)yy * w + xx];
                }
            }
            out[(size_t)y * w + x] = (uchar)((sum + total / 2) / total);   // rounded average
        }
}

void sobel_cpu(const uchar* gray, uchar* out, int w, int h) {
    auto px = [&](int x, int y) -> int {
        return gray[(size_t)clampi(y, 0, h - 1) * w + clampi(x, 0, w - 1)];
    };
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) {
            int gx = (px(x + 1, y - 1) + 2 * px(x + 1, y) + px(x + 1, y + 1))
                   - (px(x - 1, y - 1) + 2 * px(x - 1, y) + px(x - 1, y + 1));
            int gy = (px(x - 1, y + 1) + 2 * px(x, y + 1) + px(x + 1, y + 1))
                   - (px(x - 1, y - 1) + 2 * px(x, y - 1) + px(x + 1, y - 1));
            int v = (int)(std::sqrt((float)(gx * gx + gy * gy)) + 0.5f);
            out[(size_t)y * w + x] = (uchar)(v > 255 ? 255 : v);
        }
}

void run_filter_cpu(Filter f, const uchar* in, uchar* out, int w, int h, int ksize) {
    switch (f) {
        case Filter::Gray:  grayscale_cpu(in, out, w, h);       break;
        case Filter::Blur:  blur_cpu(in, out, w, h, ksize);     break;
        case Filter::Sobel: sobel_cpu(in, out, w, h);           break;
    }
}
