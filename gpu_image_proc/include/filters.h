// filters.h - filter list, CPU reference filters, GPU kernel launchers
#pragma once
#include "common.h"

enum class Filter { Gray = 0, Blur = 1, Sobel = 2 };
const int NUM_FILTERS = 3;

inline const char* filter_name(Filter f) {
    return f == Filter::Gray ? "gray" : (f == Filter::Blur ? "blur" : "sobel");
}
// Grayscale reads RGB (3 channels). Blur and Sobel read a grayscale (1 channel) image.
inline int filter_in_channels(Filter f) { return f == Filter::Gray ? 3 : 1; }

// ---- CPU reference (cpu_filters.cpp): plain sequential loops ----
void grayscale_cpu(const uchar* rgb, uchar* out, int w, int h);
void blur_cpu(const uchar* gray, uchar* out, int w, int h, int ksize);
void sobel_cpu(const uchar* gray, uchar* out, int w, int h);
void run_filter_cpu(Filter f, const uchar* in, uchar* out, int w, int h, int ksize);

// ---- GPU (kernels.cu): launchers for the CUDA kernels ----
void gpu_set_blur_weights(int ksize);   // copies the 1D mask into constant memory
void launch_filter(Filter f, const uchar* d_in, uchar* d_out, int w, int h,
                   int ksize, dim3 block = dim3(16, 16));
