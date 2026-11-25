#pragma once
#include "timing.h"

// ---------------------------------------------------------------------------
// Part D: Sobel edge detection
//
// Input : 1-channel gray image, width*height bytes
//         (for example the output of grayscaleCPU / grayscaleGPU from Part B)
// Output: 1-channel edge image, width*height bytes
//
// Rules (identical to cpu_sobel / gpu_sobel in filters_cpu.cpp / filters_gpu.cu
// so that all results are directly comparable):
//   Gx, Gy  = standard 3x3 Sobel kernels
//   edge    = min(|Gx| + |Gy|, 255)      (integer math, so CPU == GPU exactly)
//   borders = coordinates are clamped to the image edge
// ---------------------------------------------------------------------------

// Sequential CPU reference.
void sobelCPU(const unsigned char* in, unsigned char* out,
              int width, int height);

// Self-contained GPU versions, same style as grayscaleGPU:
// HOST pointers in and out, they do cudaMalloc / copies / free themselves
// and return the H2D, kernel and D2H times.
GpuTimes sobelGPU(const unsigned char* h_in, unsigned char* h_out,
                  int width, int height);          // plain (global memory only)

GpuTimes sobelSharedGPU(const unsigned char* h_in, unsigned char* h_out,
                        int width, int height);    // shared-memory tiling

// Device-pointer entry point, same style as gpu_sobel in filters.h:
// DEVICE pointers, no cudaMalloc / cudaMemcpy / sync inside.
// Lets the batch benchmark time the shared-memory kernel like the other filters.
void gpu_sobel_shared(const unsigned char* d_in, unsigned char* d_out,
                      int width, int height);
