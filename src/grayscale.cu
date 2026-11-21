#include "grayscale.h"
#include "cuda_utils.h"

__global__ void grayscaleKernel(const unsigned char* in, unsigned char* out,
                                int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int idx = y * width + x;
    unsigned char r = in[idx * 3];
    unsigned char g = in[idx * 3 + 1];
    unsigned char b = in[idx * 3 + 2];

    out[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
}

GpuTimes grayscaleGPU(const unsigned char* h_in, unsigned char* h_out,
                      int width, int height)
{
    size_t inBytes  = (size_t)width * height * 3;
    size_t outBytes = (size_t)width * height;

    unsigned char* d_in = NULL;
    unsigned char* d_out = NULL;
    CUDA_CHECK(cudaMalloc(&d_in, inBytes));
    CUDA_CHECK(cudaMalloc(&d_out, outBytes));

    cudaEvent_t e0, e1, e2, e3;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));
    CUDA_CHECK(cudaEventCreate(&e3));

    dim3 block(16, 16);
    dim3 grid((width + 15) / 16, (height + 15) / 16);

    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, inBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));

    grayscaleKernel<<<grid, block>>>(d_in, d_out, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e2));

    CUDA_CHECK(cudaMemcpy(h_out, d_out, outBytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaEventRecord(e3));

    CUDA_CHECK(cudaEventSynchronize(e3));

    GpuTimes t;
    CUDA_CHECK(cudaEventElapsedTime(&t.h2d_ms, e0, e1));
    CUDA_CHECK(cudaEventElapsedTime(&t.kernel_ms, e1, e2));
    CUDA_CHECK(cudaEventElapsedTime(&t.d2h_ms, e2, e3));

    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    cudaEventDestroy(e2);
    cudaEventDestroy(e3);
    cudaFree(d_in);
    cudaFree(d_out);

    return t;
}
