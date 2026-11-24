// gpu_pipeline.cpp - GpuProcessor: device buffer management + per-stage CUDA-event timing
#include "gpu_pipeline.h"
#include <stdexcept>

GpuProcessor::GpuProcessor(int blur_ksize, int block_x, int block_y)
    : block_(block_x, block_y) {
    CUDA_CHECK(cudaFree(0));                    // force CUDA context creation up front
    for (int i = 0; i < 4; i++) CUDA_CHECK(cudaEventCreate(&ev_[i]));
    set_blur_ksize(blur_ksize);
}

GpuProcessor::~GpuProcessor() {
    if (d_in_)  cudaFree(d_in_);
    if (d_out_) cudaFree(d_out_);
    for (int i = 0; i < 4; i++) cudaEventDestroy(ev_[i]);
}

void GpuProcessor::set_blur_ksize(int k) {
    if (k != 3 && k != 5) throw std::invalid_argument("blur mask size must be 3 or 5");
    ksize_ = k;
    gpu_set_blur_weights(k);
}

void GpuProcessor::ensure(size_t in_bytes, size_t out_bytes) {
    if (in_bytes > cap_in_) {
        if (d_in_) CUDA_CHECK(cudaFree(d_in_));
        CUDA_CHECK(cudaMalloc(&d_in_, in_bytes));
        cap_in_ = in_bytes;
    }
    if (out_bytes > cap_out_) {
        if (d_out_) CUDA_CHECK(cudaFree(d_out_));
        CUDA_CHECK(cudaMalloc(&d_out_, out_bytes));
        cap_out_ = out_bytes;
    }
}

GpuTimes GpuProcessor::run(Filter f, const uchar* h_in, uchar* h_out, int w, int h) {
    const size_t in_bytes  = (size_t)w * h * filter_in_channels(f);
    const size_t out_bytes = (size_t)w * h;
    ensure(in_bytes, out_bytes);

    // Events are timestamps recorded on the GPU stream, so they time the real GPU work
    // (a CPU timer around an asynchronous kernel launch would measure almost nothing).
    CUDA_CHECK(cudaEventRecord(ev_[0]));
    CUDA_CHECK(cudaMemcpy(d_in_, h_in, in_bytes, cudaMemcpyHostToDevice));     // H2D
    CUDA_CHECK(cudaEventRecord(ev_[1]));
    launch_filter(f, d_in_, d_out_, w, h, ksize_, block_);                      // kernel
    CUDA_CHECK(cudaEventRecord(ev_[2]));
    CUDA_CHECK(cudaMemcpy(h_out, d_out_, out_bytes, cudaMemcpyDeviceToHost));   // D2H
    CUDA_CHECK(cudaEventRecord(ev_[3]));
    CUDA_CHECK(cudaEventSynchronize(ev_[3]));

    float a, b, c;
    CUDA_CHECK(cudaEventElapsedTime(&a, ev_[0], ev_[1]));
    CUDA_CHECK(cudaEventElapsedTime(&b, ev_[1], ev_[2]));
    CUDA_CHECK(cudaEventElapsedTime(&c, ev_[2], ev_[3]));
    GpuTimes t;
    t.h2d = a; t.kernel = b; t.d2h = c;
    return t;
}
