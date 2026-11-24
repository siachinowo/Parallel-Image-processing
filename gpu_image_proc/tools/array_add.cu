// array_add.cu - Week-1 smoke test: if this passes, nvcc + driver + GPU all work.
#include <cstdio>
#include <vector>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); return 1; } } while (0)

__global__ void add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int n = 1 << 20;
    std::vector<float> a(n), b(n), c(n);
    for (int i = 0; i < n; i++) { a[i] = (float)i; b[i] = 2.0f * i; }
    float *da, *db, *dc;
    CUDA_CHECK(cudaMalloc(&da, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&db, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dc, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(da, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(db, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    add<<<(n + 255) / 256, 256>>>(da, db, dc, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(c.data(), dc, n * sizeof(float), cudaMemcpyDeviceToHost));
    int bad = 0;
    for (int i = 0; i < n; i++) if (c[i] != 3.0f * i) bad++;
    printf(bad == 0 ? "array_add: PASS (%d elements)\n" : "array_add: FAIL (%d wrong)\n", bad ? bad : n);
    cudaFree(da); cudaFree(db); cudaFree(dc);
    return bad != 0;
}
