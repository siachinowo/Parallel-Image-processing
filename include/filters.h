#pragma once
using u8 = unsigned char;

// CPU versions: HOST pointers
void cpu_grayscale(const u8* rgb,  u8* gray, int w, int h);
void cpu_blur     (const u8* gray, u8* out,  int w, int h);
void cpu_sobel    (const u8* gray, u8* out,  int w, int h);

// GPU versions: DEVICE pointers, no cudaMalloc/cudaMemcpy/sync inside
void gpu_grayscale(const u8* d_rgb,  u8* d_gray, int w, int h);
void gpu_blur     (const u8* d_gray, u8* d_out,  int w, int h);
void gpu_sobel    (const u8* d_gray, u8* d_out,  int w, int h);
