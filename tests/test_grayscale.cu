#include <cstdio>
#include <cstdlib>
#include <vector>
#include <chrono>
#include "grayscale.h"
#include "verify.h"

int main()
{
    int sizes[] = {256, 512, 1024, 2048, 4096};
    int numSizes = 5;
    int trials = 10;

    printf("size,cpu_ms,h2d_ms,kernel_ms,d2h_ms,gpu_total_ms,"
           "kernel_speedup,full_speedup,correct\n");

    for (int s = 0; s < numSizes; s++) {
        int w = sizes[s];
        int h = sizes[s];

        std::vector<unsigned char> in((size_t)w * h * 3);
        std::vector<unsigned char> cpuOut((size_t)w * h);
        std::vector<unsigned char> gpuOut((size_t)w * h);

        srand(42);
        for (size_t i = 0; i < in.size(); i++) {
            in[i] = (unsigned char)(rand() % 256);
        }

        // Warm-up (result thrown away)
        grayscaleGPU(in.data(), gpuOut.data(), w, h);

        // CPU timing
        double cpuMs = 0.0;
        for (int t = 0; t < trials; t++) {
            auto start = std::chrono::high_resolution_clock::now();
            grayscaleCPU(in.data(), cpuOut.data(), w, h);
            auto end = std::chrono::high_resolution_clock::now();
            cpuMs += std::chrono::duration<double, std::milli>(end - start).count();
        }
        cpuMs /= trials;

        // GPU timing
        double h2d = 0.0;
        double kernel = 0.0;
        double d2h = 0.0;
        for (int t = 0; t < trials; t++) {
            GpuTimes g = grayscaleGPU(in.data(), gpuOut.data(), w, h);
            h2d += g.h2d_ms;
            kernel += g.kernel_ms;
            d2h += g.d2h_ms;
        }
        h2d /= trials;
        kernel /= trials;
        d2h /= trials;
        double total = h2d + kernel + d2h;

        bool ok = verifyImages(cpuOut.data(), gpuOut.data(), w * h, 0);
        const char* status = "FAIL";
        if (ok) {
            status = "PASS";
        }

        printf("%dx%d,%.3f,%.3f,%.3f,%.3f,%.3f,%.2f,%.2f,%s\n",
               w, h, cpuMs, h2d, kernel, d2h, total,
               cpuMs / kernel, cpuMs / total, status);
    }
    return 0;
}
