# Part D: Sobel edge detection + shared-memory optimization

## Files

| File | What it is |
|---|---|
| `include/sobel.h` | Declarations: `sobelCPU`, `sobelGPU`, `sobelSharedGPU`, `gpu_sobel_shared` |
| `src/sobel_cpu.cpp` | Sequential CPU reference |
| `src/sobel.cu` | Plain kernel, shared-memory (tiled) kernel, timed host wrappers |
| `tests/test_sobel.cu` | Correctness on odd sizes + CPU vs GPU benchmark table |
| `tests/sobel_image.cu` | Demo: photo -> gray (Part B) -> Sobel (Part D) -> PNG |

## Build and run (from the repo root)

```
nvcc -O2 -Iinclude tests/test_sobel.cu src/sobel.cu src/sobel_cpu.cpp -o test_sobel
./test_sobel check        # correctness only, fast. Expect: "Correctness: PASS"
./test_sobel              # correctness + benchmark table (CSV)

nvcc -O2 -Iinclude -Ithird_party tests/sobel_image.cu src/sobel.cu \
     src/sobel_cpu.cpp src/grayscale.cu -o sobel_image
./sobel_image samples/sample_input.jpeg samples/sample_sobel_output.png
```
If you get "no kernel image is available", add `-arch=native` (or `-arch=sm_XX` for your GPU).
`third_party/` (stb headers) exists once Part A (Fatima's PR) is merged.

## How it fits the rest of the code

- Same style as Part B: `sobelCPU` / `sobelGPU(h_in, h_out, w, h)` returning `GpuTimes`,
  checked with `verifyImages`, error handling with `CUDA_CHECK` from `cuda_utils.h`,
  16x16 blocks, `(width + 15) / 16` grid.
- Input is the gray image Part B produces (1 byte per pixel).
- Same rules as Part A's `cpu_sobel` / `gpu_sobel`: edge = min(|Gx| + |Gy|, 255),
  clamp-to-edge borders. Integer math, so CPU and GPU match **exactly** (tolerance 0).
- `gpu_sobel_shared(d_in, d_out, w, h)` has the same signature as Part A's `gpu_*`
  functions, so the batch benchmark can time it as one more filter:
  apply `benchmark_add_sobel_shared.patch` (2 added lines) AFTER Part A is merged, and add
  `src/sobel.cu` to the benchmark's nvcc command. The existing plot script handles the new
  `sobel_shared` filter automatically.
- All helper names are `static` or prefixed with `sobel`, so nothing clashes with other parts.

## The optimization (for the report)

Plain kernel: every thread reads 8 neighbours from global memory, so each input pixel is
fetched up to 8 times. Shared-memory kernel: each 16x16 block loads one 18x18 tile (block + 1-pixel
halo) into `__shared__` memory once, calls `__syncthreads()`, then reads neighbours from the tile.
Compare `kernel_plain_ms` vs `kernel_shared_ms` in the `./test_sobel` table. On newer GPUs the
L1/L2 caches already help the plain kernel, so the gain may be small, and that is a valid finding.

## Notes

- Verified with a CUDA emulator (real threads + barrier for `__syncthreads`) on 8 sizes
  including 1x1, 17x17 and 1001x777: both kernels match the CPU exactly. Not yet run on a real GPU,
  so run `./test_sobel check` on your GPU machine first.
