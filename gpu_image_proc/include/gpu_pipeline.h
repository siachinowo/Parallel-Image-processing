// gpu_pipeline.h - GPU processor: owns device buffers and times every stage
#pragma once
#include "filters.h"

struct GpuTimes {
    double h2d = 0, kernel = 0, d2h = 0;   // milliseconds
    double total() const { return h2d + kernel + d2h; }
    GpuTimes& operator+=(const GpuTimes& o) {
        h2d += o.h2d; kernel += o.kernel; d2h += o.d2h; return *this;
    }
};

class GpuProcessor {
public:
    explicit GpuProcessor(int blur_ksize = 3, int block_x = 16, int block_y = 16);
    ~GpuProcessor();
    GpuProcessor(const GpuProcessor&) = delete;
    GpuProcessor& operator=(const GpuProcessor&) = delete;

    void set_blur_ksize(int k);
    int  blur_ksize() const { return ksize_; }
    void set_block(int bx, int by) { block_ = dim3(bx, by); }

    // One image: H2D copy -> kernel -> D2H copy, each stage timed with CUDA events.
    // Device buffers are allocated once and reused (grown only if a bigger image arrives),
    // so cudaMalloc cost does not pollute batch timings.
    GpuTimes run(Filter f, const uchar* h_in, uchar* h_out, int w, int h);

private:
    void ensure(size_t in_bytes, size_t out_bytes);
    uchar* d_in_ = nullptr;
    uchar* d_out_ = nullptr;
    size_t cap_in_ = 0, cap_out_ = 0;
    int ksize_ = 3;
    dim3 block_;
    cudaEvent_t ev_[4];
};
