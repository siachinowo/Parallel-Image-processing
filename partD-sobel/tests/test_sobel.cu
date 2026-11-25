// Part D test: correctness on awkward sizes + CPU vs GPU benchmark.
//
// Build (from the repo root):
//   nvcc -O2 -Iinclude tests/test_sobel.cu src/sobel.cu src/sobel_cpu.cpp -o test_sobel
// Run:
//   ./test_sobel          correctness checks + benchmark table (CSV)
//   ./test_sobel check    correctness checks only (fast)
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <chrono>
#include "sobel.h"
#include "verify.h"

// Test picture: smooth gradient + sharp 32x32 squares + a little noise.
// Pure random noise would be a bad test: almost every pixel would saturate
// at 255 and hide bugs. This mix has weak edges, strong edges and flat areas.
static void fillTestImage(std::vector<unsigned char>& img, int w, int h)
{
    srand(42);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            int v = (x * 255 / w + y * 255 / h) / 2;
            if (((x / 32) + (y / 32)) % 2 == 1) {
                v += 70;
            }
            v += rand() % 8;
            if (v > 255) v = 255;
            img[(size_t)y * w + x] = (unsigned char)v;
        }
    }
}

// Both GPU versions must match the CPU exactly (tolerance 0).
static bool checkSize(int w, int h)
{
    std::vector<unsigned char> in((size_t)w * h);
    std::vector<unsigned char> cpuOut((size_t)w * h);
    std::vector<unsigned char> gpuOut((size_t)w * h);
    fillTestImage(in, w, h);

    sobelCPU(in.data(), cpuOut.data(), w, h);

    printf("Size %dx%d\n", w, h);

    sobelGPU(in.data(), gpuOut.data(), w, h);
    printf(" plain : ");
    bool ok1 = verifyImages(cpuOut.data(), gpuOut.data(), w * h, 0);

    memset(gpuOut.data(), 0, gpuOut.size());
    sobelSharedGPU(in.data(), gpuOut.data(), w, h);
    printf(" shared: ");
    bool ok2 = verifyImages(cpuOut.data(), gpuOut.data(), w * h, 0);

    return ok1 && ok2;
}

int main(int argc, char** argv)
{
    bool checkOnly = (argc > 1 && strcmp(argv[1], "check") == 0);
    bool allOk = true;

    // ---- Part 1: correctness, including sizes that are NOT multiples of 16 ----
    int cw[] = {1, 3, 16, 17, 33, 100, 257, 1001};
    int ch[] = {1, 3, 16, 17, 65, 75, 131, 777};
    printf("== Correctness ==\n");
    for (int i = 0; i < 8; i++) {
        if (!checkSize(cw[i], ch[i])) {
            allOk = false;
        }
    }
    printf("Correctness: %s\n", allOk ? "PASS" : "FAIL");

    if (checkOnly) {
        return allOk ? 0 : 1;
    }

    // ---- Part 2: benchmark (same columns idea as test_grayscale) ----
    int sizes[] = {256, 512, 1024, 2048, 4096};
    int numSizes = 5;
    int trials = 10;

    printf("\n== Benchmark ==\n");
    printf("size,cpu_ms,h2d_ms,d2h_ms,kernel_plain_ms,kernel_shared_ms,"
           "shared_vs_plain,kspeedup_plain,kspeedup_shared,"
           "full_speedup_plain,full_speedup_shared,correct\n");

    for (int s = 0; s < numSizes; s++) {
        int w = sizes[s];
        int h = sizes[s];

        std::vector<unsigned char> in((size_t)w * h);
        std::vector<unsigned char> cpuOut((size_t)w * h);
        std::vector<unsigned char> plainOut((size_t)w * h);
        std::vector<unsigned char> sharedOut((size_t)w * h);
        fillTestImage(in, w, h);

        // Warm-up (results thrown away)
        sobelGPU(in.data(), plainOut.data(), w, h);
        sobelSharedGPU(in.data(), sharedOut.data(), w, h);

        // CPU timing
        double cpuMs = 0.0;
        for (int t = 0; t < trials; t++) {
            auto start = std::chrono::high_resolution_clock::now();
            sobelCPU(in.data(), cpuOut.data(), w, h);
            auto end = std::chrono::high_resolution_clock::now();
            cpuMs += std::chrono::duration<double, std::milli>(end - start).count();
        }
        cpuMs /= trials;

        // GPU timing (both versions)
        double h2d = 0.0, d2h = 0.0, kPlain = 0.0, kShared = 0.0;
        for (int t = 0; t < trials; t++) {
            GpuTimes p = sobelGPU(in.data(), plainOut.data(), w, h);
            GpuTimes q = sobelSharedGPU(in.data(), sharedOut.data(), w, h);
            h2d += p.h2d_ms + q.h2d_ms;      // averaged over both runs below
            d2h += p.d2h_ms + q.d2h_ms;
            kPlain += p.kernel_ms;
            kShared += q.kernel_ms;
        }
        h2d /= (2.0 * trials);
        d2h /= (2.0 * trials);
        kPlain /= trials;
        kShared /= trials;

        bool ok = verifyImages(cpuOut.data(), plainOut.data(), w * h, 0) &&
                  verifyImages(cpuOut.data(), sharedOut.data(), w * h, 0);
        if (!ok) {
            allOk = false;
        }

        printf("%dx%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%.2f,%.2f,%.2f,%s\n",
               w, h, cpuMs, h2d, d2h, kPlain, kShared,
               kPlain / kShared,
               cpuMs / kPlain, cpuMs / kShared,
               cpuMs / (h2d + kPlain + d2h),
               cpuMs / (h2d + kShared + d2h),
               ok ? "PASS" : "FAIL");
    }

    return allOk ? 0 : 1;
}
