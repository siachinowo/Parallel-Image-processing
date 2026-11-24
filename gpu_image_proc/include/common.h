// common.h - shared types, error checking, timers, filter constants
#pragma once
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <cuda_runtime.h>

typedef unsigned char uchar;

// Wrap EVERY CUDA runtime call: prints file/line and the error text, then exits.
#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t e_ = (call);                                                  \
        if (e_ != cudaSuccess) {                                                  \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(e_), \
                    __FILE__, __LINE__, cudaGetErrorString(e_));                  \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

// Border rule used by BOTH the CPU and GPU code: clamp (replicate the edge pixel).
__host__ __device__ inline int clampi(int v, int lo, int hi) {
    return v < lo ? lo : (v > hi ? hi : v);
}

// Gaussian blur masks are the outer product of a 1D integer mask with itself.
//   3x3: [1 2 1]       -> total weight 4^2  = 16
//   5x5: [1 4 6 4 1]   -> total weight 16^2 = 256
// Integer weights make CPU and GPU results identical (no floating-point rounding).
inline const int* gauss1d(int ksize) {
    static const int g3[3] = {1, 2, 1};
    static const int g5[5] = {1, 4, 6, 4, 1};
    return ksize == 5 ? g5 : g3;
}
inline int gauss_total(int ksize) { return ksize == 5 ? 256 : 16; }

// Wall-clock timer for CPU code (std::chrono).
class CpuTimer {
    std::chrono::steady_clock::time_point t0_;
public:
    void tic() { t0_ = std::chrono::steady_clock::now(); }
    double toc_ms() const {
        return std::chrono::duration<double, std::milli>(
                   std::chrono::steady_clock::now() - t0_).count();
    }
};
