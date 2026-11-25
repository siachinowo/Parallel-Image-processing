#include "sobel.h"
#include "cuda_utils.h"

// Block size is 16 x 16 (same as grayscaleKernel). The shared-memory kernel
// REQUIRES a block of exactly SOBEL_TILE x SOBEL_TILE threads.
#define SOBEL_TILE 16

static __device__ __forceinline__ int sobelClamp(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

// Combine the 8 neighbours of a pixel into one edge value.
// (The centre pixel has weight 0 in both Sobel kernels.)
static __device__ __forceinline__ unsigned char sobelValue(
    int tl, int t, int tr, int l, int r, int bl, int b, int br)
{
    int gx = -tl + tr - 2 * l + 2 * r - bl + br;
    int gy = -tl - 2 * t - tr + bl + 2 * b + br;
    int m = abs(gx) + abs(gy);
    return (unsigned char)(m > 255 ? 255 : m);
}

// ---------------------------------------------------------------------------
// Version 1: plain kernel. Every thread reads its 8 neighbours straight from
// global memory, so each input pixel is read up to 8 times by different threads.
// ---------------------------------------------------------------------------
__global__ void sobelGlobalKernel(const unsigned char* in, unsigned char* out,
                                  int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x >= width || y >= height) {
        return;
    }

    int xm = sobelClamp(x - 1, 0, width - 1);
    int xp = sobelClamp(x + 1, 0, width - 1);
    int ym = sobelClamp(y - 1, 0, height - 1);
    int yp = sobelClamp(y + 1, 0, height - 1);

    out[y * width + x] = sobelValue(
        in[ym * width + xm], in[ym * width + x], in[ym * width + xp],
        in[y  * width + xm],                     in[y  * width + xp],
        in[yp * width + xm], in[yp * width + x], in[yp * width + xp]);
}

// ---------------------------------------------------------------------------
// Version 2: shared-memory tiling.
// A block computes a 16x16 patch of output pixels, which needs an 18x18 patch
// of input pixels (the 16x16 patch plus a 1-pixel "halo" on every side).
// The block first copies that 18x18 patch into fast shared memory ONCE,
// then every thread reads its neighbours from there.
// ---------------------------------------------------------------------------
__global__ void sobelSharedKernel(const unsigned char* in, unsigned char* out,
                                  int width, int height)
{
    __shared__ unsigned char tile[SOBEL_TILE + 2][SOBEL_TILE + 2];

    int x = blockIdx.x * SOBEL_TILE + threadIdx.x;
    int y = blockIdx.y * SOBEL_TILE + threadIdx.y;

    // Top-left corner (in image coordinates) of the 18x18 patch.
    int originX = blockIdx.x * SOBEL_TILE - 1;
    int originY = blockIdx.y * SOBEL_TILE - 1;

    // 256 threads must fill 18*18 = 324 slots, so some threads load two.
    int tid = threadIdx.y * SOBEL_TILE + threadIdx.x;
    for (int i = tid; i < (SOBEL_TILE + 2) * (SOBEL_TILE + 2);
         i += SOBEL_TILE * SOBEL_TILE) {
        int ty = i / (SOBEL_TILE + 2);
        int tx = i % (SOBEL_TILE + 2);
        int gx = sobelClamp(originX + tx, 0, width - 1);   // clamp = border rule
        int gy = sobelClamp(originY + ty, 0, height - 1);
        tile[ty][tx] = in[gy * width + gx];
    }

    // EVERY thread of the block must reach this line, so the out-of-range
    // check comes AFTER it (not before).
    __syncthreads();

    if (x >= width || y >= height) {
        return;
    }

    int cx = threadIdx.x + 1;   // this thread's pixel inside the tile
    int cy = threadIdx.y + 1;

    out[y * width + x] = sobelValue(
        tile[cy - 1][cx - 1], tile[cy - 1][cx], tile[cy - 1][cx + 1],
        tile[cy][cx - 1],                       tile[cy][cx + 1],
        tile[cy + 1][cx - 1], tile[cy + 1][cx], tile[cy + 1][cx + 1]);
}

// ---------------------------------------------------------------------------
// Launch helpers (device pointers, no sync) and the timed host wrappers
// ---------------------------------------------------------------------------
static void launchGlobal(const unsigned char* d_in, unsigned char* d_out,
                         int width, int height)
{
    dim3 block(SOBEL_TILE, SOBEL_TILE);
    dim3 grid((width + SOBEL_TILE - 1) / SOBEL_TILE,
              (height + SOBEL_TILE - 1) / SOBEL_TILE);
    sobelGlobalKernel<<<grid, block>>>(d_in, d_out, width, height);
}

static void launchShared(const unsigned char* d_in, unsigned char* d_out,
                         int width, int height)
{
    dim3 block(SOBEL_TILE, SOBEL_TILE);
    dim3 grid((width + SOBEL_TILE - 1) / SOBEL_TILE,
              (height + SOBEL_TILE - 1) / SOBEL_TILE);
    sobelSharedKernel<<<grid, block>>>(d_in, d_out, width, height);
}

void gpu_sobel_shared(const unsigned char* d_in, unsigned char* d_out,
                      int width, int height)
{
    launchShared(d_in, d_out, width, height);
}

typedef void (*SobelLauncher)(const unsigned char*, unsigned char*, int, int);

// Same structure as grayscaleGPU: malloc, copy in, kernel, copy out, 3 timings.
// (Input and output are both 1 byte per pixel here.)
static GpuTimes runSobel(SobelLauncher launch,
                         const unsigned char* h_in, unsigned char* h_out,
                         int width, int height)
{
    size_t bytes = (size_t)width * height;

    unsigned char* d_in = NULL;
    unsigned char* d_out = NULL;
    CUDA_CHECK(cudaMalloc(&d_in, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    cudaEvent_t e0, e1, e2, e3;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));
    CUDA_CHECK(cudaEventCreate(&e2));
    CUDA_CHECK(cudaEventCreate(&e3));

    CUDA_CHECK(cudaEventRecord(e0));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaEventRecord(e1));

    launch(d_in, d_out, width, height);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventRecord(e2));

    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));
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

GpuTimes sobelGPU(const unsigned char* h_in, unsigned char* h_out,
                  int width, int height)
{
    return runSobel(launchGlobal, h_in, h_out, width, height);
}

GpuTimes sobelSharedGPU(const unsigned char* h_in, unsigned char* h_out,
                        int width, int height)
{
    return runSobel(launchShared, h_in, h_out, width, height);
}
