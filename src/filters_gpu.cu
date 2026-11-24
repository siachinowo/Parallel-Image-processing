#include "filters.h"

__device__ __forceinline__ int dcl(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

__global__ void k_gray(const u8* in, u8* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    int i = (y * w + x) * 3;
    float g = 0.299f * in[i] + 0.587f * in[i+1] + 0.114f * in[i+2];
    out[y*w + x] = (u8)(g + 0.5f);
}

__global__ void k_blur(const u8* in, u8* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int k[9] = {1,2,1, 2,4,2, 1,2,1};
    int s = 0;
    for (int dy = -1; dy <= 1; ++dy)
        for (int dx = -1; dx <= 1; ++dx)
            s += k[(dy+1)*3 + (dx+1)] * in[dcl(y+dy,0,h-1)*w + dcl(x+dx,0,w-1)];
    out[y*w + x] = (u8)(s / 16);
}

__global__ void k_sobel(const u8* in, u8* out, int w, int h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    #define P(dx,dy) ((int)in[dcl(y+(dy),0,h-1)*w + dcl(x+(dx),0,w-1)])
    int gx = -P(-1,-1) + P(1,-1) - 2*P(-1,0) + 2*P(1,0) - P(-1,1) + P(1,1);
    int gy = -P(-1,-1) - 2*P(0,-1) - P(1,-1) + P(-1,1) + 2*P(0,1) + P(1,1);
    #undef P
    int m = abs(gx) + abs(gy);
    out[y*w + x] = (u8)(m > 255 ? 255 : m);
}

static dim3 grid_for(int w, int h, dim3 b) { return dim3((w + b.x - 1) / b.x, (h + b.y - 1) / b.y); }

void gpu_grayscale(const u8* d_in, u8* d_out, int w, int h) { dim3 b(16,16); k_gray <<<grid_for(w,h,b), b>>>(d_in, d_out, w, h); }
void gpu_blur     (const u8* d_in, u8* d_out, int w, int h) { dim3 b(16,16); k_blur <<<grid_for(w,h,b), b>>>(d_in, d_out, w, h); }
void gpu_sobel    (const u8* d_in, u8* d_out, int w, int h) { dim3 b(16,16); k_sobel<<<grid_for(w,h,b), b>>>(d_in, d_out, w, h); }
