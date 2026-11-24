# GPU-Accelerated Image Processing using CUDA

Grayscale, Gaussian blur (3x3 / 5x5) and Sobel edge detection, implemented twice:
sequential C++ (CPU baseline) and CUDA kernels (GPU), with batch processing and a benchmark.

## Setup
```bash
bash scripts/setup_check.sh     # checks nvidia-smi / nvcc, downloads stb headers
make test_cuda                  # array-add smoke test, must print PASS
make                            # builds ./imgproc and ./benchmark   (make ARCH=sm_XX if needed)
```

## Process a folder of images
```bash
./imgproc --input images --output results/output --blur 3 --verify
```
Writes `<name>_gray.png`, `<name>_blur.png`, `<name>_sobel.png` for each input image and prints
per-filter H2D / kernel / D2H times. `--verify` also runs the CPU versions and checks that the
GPU output is identical (max difference 0).

## Benchmark and plots
```bash
./benchmark --quick                         # smoke test
./benchmark                                 # full run -> results/benchmark.csv
./benchmark --cpu-max-batch 5               # faster: time max 5 images on the CPU, scale up
python3 scripts/plot.py results/benchmark.csv
```
Plots go to `results/plots/`. Needs `pip install pandas matplotlib`.

## Layout
| File | Purpose |
|---|---|
| `include/common.h` | `CUDA_CHECK`, `CpuTimer`, clamp, blur masks |
| `src/imageio.cpp` | stb load/save, folder scan, synthetic images, `compare()` |
| `src/cpu_filters.cpp` | sequential reference filters |
| `src/kernels.cu` | CUDA kernels + launcher |
| `src/gpu_pipeline.cpp` | `GpuProcessor`: buffers + CUDA-event timing of H2D / kernel / D2H |
| `src/batch.cpp`, `src/main.cpp` | folder pipeline and CLI |
| `src/benchmark.cpp` | CPU-vs-GPU benchmark -> CSV |
| `scripts/plot.py` | figures from the CSV |

## Design notes
* **CPU/GPU parity:** all filters use integer arithmetic (fixed-point grayscale weights, integer
  blur masks), so GPU output must match the CPU exactly. `max_diff` in the CSV should be 0.
* **Border rule:** clamp (replicate edge pixels) in both versions.
* **Blur and Sobel take a grayscale input;** the pipeline runs grayscale first.
* **Benchmark batches** repeat one synthetic image N times; each GPU call includes both copies.
