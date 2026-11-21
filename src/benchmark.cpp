#include "common.h"
#include "filters.h"
#include "image_io.h"
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <filesystem>

namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;
static double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}

struct GpuTimer {
    cudaEvent_t a, b;
    GpuTimer()  { CUDA_CHECK(cudaEventCreate(&a)); CUDA_CHECK(cudaEventCreate(&b)); }
    ~GpuTimer() { cudaEventDestroy(a); cudaEventDestroy(b); }
    void start() { CUDA_CHECK(cudaEventRecord(a)); }
    float stop() {
        CUDA_CHECK(cudaEventRecord(b));
        CUDA_CHECK(cudaEventSynchronize(b));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, a, b));
        return ms;
    }
};

using FilterFn = void (*)(const u8*, u8*, int, int);
struct FilterDef { const char* name; int inCh; FilterFn cpu; FilterFn gpu; };

static void diff_stats(const std::vector<u8>& a, const std::vector<u8>& b, int& maxd, double& meand) {
    maxd = 0; double sum = 0;
    for (size_t i = 0; i < a.size(); ++i) {
        int d = std::abs((int)a[i] - (int)b[i]);
        maxd = std::max(maxd, d); sum += d;
    }
    meand = sum / a.size();
}

int main(int argc, char** argv) {
    std::string dir = "data/1024", csv = "results/results.csv", saveDir;
    int batch = 10, trials = 10, warmup = 2;
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&]() { return std::string(argv[++i]); };
        if      (k == "--dir")    dir = next();
        else if (k == "--batch")  batch = std::stoi(next());
        else if (k == "--trials") trials = std::stoi(next());
        else if (k == "--warmup") warmup = std::stoi(next());
        else if (k == "--csv")    csv = next();
        else if (k == "--save")   saveDir = next();
    }

    // ---- load images (untimed) ----
    auto files = list_images(dir);
    if (files.empty()) { std::cerr << "No images in " << dir << "\n"; return 1; }
    int U = std::min<int>(batch, (int)files.size());
    std::vector<Image> imgs(U);
    for (int i = 0; i < U; ++i)
        if (!load_rgb(files[i], imgs[i])) return 1;
    const int w = imgs[0].w, h = imgs[0].h;
    for (auto& im : imgs) if (im.w != w || im.h != h) { std::cerr << "Mixed sizes in folder\n"; return 1; }
    const size_t npix = (size_t)w * h;

    // ---- GPU init + one-time buffers ----
    CUDA_CHECK(cudaFree(0));
    u8 *d_in, *d_out;
    CUDA_CHECK(cudaMalloc(&d_in,  npix * 3));
    CUDA_CHECK(cudaMalloc(&d_out, npix));

    // precompute gray inputs (untimed)
    std::vector<std::vector<u8>> gray(U, std::vector<u8>(npix));
    for (int u = 0; u < U; ++u) cpu_grayscale(imgs[u].rgb.data(), gray[u].data(), w, h);

    FilterDef filters[] = {
        {"grayscale", 3, cpu_grayscale, gpu_grayscale},
        {"blur",      1, cpu_blur,      gpu_blur},
        {"sobel",     1, cpu_sobel,     gpu_sobel},
    };

    // ---- CSV ----
    fs::path parent = fs::path(csv).parent_path();
    if (!parent.empty()) fs::create_directories(parent);
    bool newFile = !fs::exists(csv) || fs::file_size(csv) == 0;
    std::ofstream out(csv, std::ios::app);
    if (newFile)
        out << "filter,width,height,batch,trial,cpu_ms,h2d_ms,kernel_ms,d2h_ms,gpu_total_ms,max_diff,mean_diff\n";

    GpuTimer timer;
    for (auto& f : filters) {
        std::vector<std::vector<u8>> cpuOut(U, std::vector<u8>(npix)), gpuOut(U, std::vector<u8>(npix));
        auto input = [&](int u) -> const u8* { return f.inCh == 3 ? imgs[u].rgb.data() : gray[u].data(); };
        size_t inBytes = npix * f.inCh;
        int maxd = 0; double meand = 0;

        for (int trial = -warmup; trial < trials; ++trial) {
            // CPU
            auto t0 = Clock::now();
            for (int i = 0; i < batch; ++i) { int u = i % U; f.cpu(input(u), cpuOut[u].data(), w, h); }
            double cpu_ms = ms_since(t0);

            // GPU phases
            double h2d = 0, ker = 0, d2h = 0;
            for (int i = 0; i < batch; ++i) {
                int u = i % U;
                timer.start();
                CUDA_CHECK(cudaMemcpy(d_in, input(u), inBytes, cudaMemcpyHostToDevice));
                h2d += timer.stop();

                timer.start();
                f.gpu(d_in, d_out, w, h);
                CUDA_CHECK(cudaGetLastError());
                ker += timer.stop();

                timer.start();
                CUDA_CHECK(cudaMemcpy(gpuOut[u].data(), d_out, npix, cudaMemcpyDeviceToHost));
                d2h += timer.stop();
            }

            if (trial == 0) {
                maxd = 0; meand = 0;
                for (int u = 0; u < U; ++u) {
                    int m; double a; diff_stats(cpuOut[u], gpuOut[u], m, a);
                    maxd = std::max(maxd, m); meand += a / U;
                }
                if (!saveDir.empty()) {
                    save_png(saveDir + "/" + f.name + "_cpu_" + std::to_string(w) + ".png", w, h, 1, cpuOut[0].data());
                    save_png(saveDir + "/" + f.name + "_gpu_" + std::to_string(w) + ".png", w, h, 1, gpuOut[0].data());
                }
            }
            if (trial >= 0) {
                out << f.name << ',' << w << ',' << h << ',' << batch << ',' << trial << ','
                    << cpu_ms << ',' << h2d << ',' << ker << ',' << d2h << ','
                    << (h2d + ker + d2h) << ',' << maxd << ',' << meand << '\n';
            }
        }
        std::printf("%-9s %dx%d batch=%d done (max diff vs CPU = %d, mean = %.4f)\n",
                    f.name, w, h, batch, maxd, meand);
    }
    cudaFree(d_in); cudaFree(d_out);
    return 0;
}
