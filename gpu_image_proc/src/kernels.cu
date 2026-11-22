// kernels.cu - the three CUDA kernels and their launchers.
// One thread computes one output pixel, using a 2D grid of 2D blocks.
#include "filters.h"

// 1D Gaussian mask in constant memory (cached; every thread reads the same few values).
__constant__ int c_w1d[5];

void gpu_set_blur_weights(int ksize) {
    CUDA_CHECK(cudaMemcpyToSymbol(c_w1d, gauss1d(ksize), ksize * sizeof(int)));
}

// ---------------- Grayscale: 3 channels in, 1 channel out ----------------
__global__ void grayscaleKernel(const uchar* in, uchar* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;      // column
    int y = blockIdx.y * blockDim.y + threadIdx.y;      // row
    if (x >= w || y >= h) return;                       // grid is rounded up: extra threads exit

    const uchar* p = &in[((size_t)y * w + x) * 3];
    out[(size_t)y * w + x] = (uchar)((19595 * p[0] + 38470 * p[1] + 7471 * p[2] + 32768) >> 16);
}

// ---------------- Blur: (2r+1)x(2r+1) Gaussian, clamped borders ----------------
__global__ void blurKernel(const uchar* in, uchar* out, int w, int h, int r, int total) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int sum = 0;
    for (int dy = -r; dy <= r; dy++) {
        int yy = clampi(y + dy, 0, h - 1);
        for (int dx = -r; dx <= r; dx++) {
            int xx = clampi(x + dx, 0, w - 1);
            sum += c_w1d[dy + r] * c_w1d[dx + r] * in[(size_t)yy * w + xx];
        }
    }
    out[(size_t)y * w + x] = (uchar)((sum + total / 2) / total);
}

// ---------------- Sobel: gradient magnitude ----------------
__global__ void sobelKernel(const uchar* in, uchar* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;

    int x0 = clampi(x - 1, 0, w - 1), x1 = x, x2 = clampi(x + 1, 0, w - 1);
    int y0 = clampi(y - 1, 0, h - 1), y1 = y, y2 = clampi(y + 1, 0, h - 1);

    int a = in[(size_t)y0 * w + x0], b = in[(size_t)y0 * w + x1], c = in[(size_t)y0 * w + x2];
    int d = in[(size_t)y1 * w + x0],                               f = in[(size_t)y1 * w + x2];
    int g = in[(size_t)y2 * w + x0], hh = in[(size_t)y2 * w + x1], i = in[(size_t)y2 * w + x2];

    int gx = (c + 2 * f + i) - (a + 2 * d + g);
    int gy = (g + 2 * hh + i) - (a + 2 * b + c);
    int v = (int)(sqrtf((float)(gx * gx + gy * gy)) + 0.5f);
    out[(size_t)y * w + x] = (uchar)(v > 255 ? 255 : v);
}

// ---------------- Launcher ----------------
void launch_filter(Filter f, const uchar* d_in, uchar* d_out, int w, int h,
                   int ksize, dim3 block) {
    // Ceiling division so the grid always covers the whole image.
    dim3 grid((w + block.x - 1) / block.x, (h + block.y - 1) / block.y);
    switch (f) {
        case Filter::Gray:
            grayscaleKernel<<<grid, block>>>(d_in, d_out, w, h);
            break;
        case Filter::Blur:
            blurKernel<<<grid, block>>>(d_in, d_out, w, h, ksize / 2, gauss_total(ksize));
            break;
        case Filter::Sobel:
            sobelKernel<<<grid, block>>>(d_in, d_out, w, h);
            break;
    }
    CUDA_CHECK(cudaGetLastError());   // catches bad launch configurations immediately
}
