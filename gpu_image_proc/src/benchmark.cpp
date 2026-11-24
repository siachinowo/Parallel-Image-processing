// benchmark.cpp - CPU vs GPU benchmark over image sizes and batch sizes -> CSV
//
//   ./benchmark                         full run (sizes 256..4096, batches 1,10,50,100)
//   ./benchmark --quick                 small run for a smoke test
//   ./benchmark --sizes 512,1024 --batches 1,10 --trials 5 --cpu-max-batch 5
//
// A "batch" of N means the same image processed N times in a row (synthetic test image),
// so memory use does not grow with N. Each GPU call does H2D copy -> kernel -> D2H copy.
#include "gpu_pipeline.h"
#include "imageio.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

struct Args {
    std::vector<int> sizes   = {256, 512, 1024, 2048, 4096};
    std::vector<int> batches = {1, 10, 50, 100};
    bool f[NUM_FILTERS]      = {true, true, true};
    int trials = 5, warmup = 1, ksize = 3, bx = 16, by = 16;
    int cpu_max_batch = 0;      // 0 = time every image on the CPU. >0 = time at most this many
                                // images and scale up (CPU time is linear in batch size).
    std::string out = "results/benchmark.csv";
};

static std::vector<int> parse_list(const std::string& s) {
    std::vector<int> v; std::stringstream ss(s); std::string t;
    while (std::getline(ss, t, ',')) if (!t.empty()) v.push_back(std::stoi(t));
    return v;
}
static double mean(const std::vector<double>& v) {
    double s = 0; for (double x : v) s += x; return v.empty() ? 0 : s / v.size();
}
static double stdev(const std::vector<double>& v) {
    if (v.size() < 2) return 0;
    double m = mean(v), s = 0; for (double x : v) s += (x - m) * (x - m);
    return std::sqrt(s / (v.size() - 1));
}

int main(int argc, char** argv) {
    Args a;
    for (int i = 1; i < argc; i++) {
        std::string k = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (k == "--sizes")   a.sizes = parse_list(next());
        else if (k == "--batches") a.batches = parse_list(next());
        else if (k == "--trials")  a.trials = std::stoi(next());
        else if (k == "--warmup")  a.warmup = std::stoi(next());
        else if (k == "--blur")    a.ksize = std::stoi(next());
        else if (k == "--out")     a.out = next();
        else if (k == "--cpu-max-batch") a.cpu_max_batch = std::stoi(next());
        else if (k == "--block")   { std::string b = next(); sscanf(b.c_str(), "%dx%d", &a.bx, &a.by); }
        else if (k == "--filters") {
            std::string s = next();
            a.f[0] = s.find("gray") != std::string::npos;
            a.f[1] = s.find("blur") != std::string::npos;
            a.f[2] = s.find("sobel") != std::string::npos;
        } else if (k == "--quick") { a.sizes = {256, 1024}; a.batches = {1, 10}; a.trials = 3; }
        else { printf("unknown option %s\n", k.c_str()); return 1; }
    }

    GpuProcessor gpu(a.ksize, a.bx, a.by);
    cudaDeviceProp prop; CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int drv = 0, rt = 0; cudaDriverGetVersion(&drv); cudaRuntimeGetVersion(&rt);
    printf("GPU: %s | CUDA driver %d, runtime %d | block %dx%d | blur %dx%d | trials %d (+%d warm-up)\n",
           prop.name, drv, rt, a.bx, a.by, a.ksize, a.ksize, a.trials, a.warmup);

    std::filesystem::path outp(a.out);
    if (outp.has_parent_path()) std::filesystem::create_directories(outp.parent_path());
    std::ofstream csv(a.out);
    csv << "filter,width,height,batch,trials,cpu_reps,cpu_ms,cpu_std_ms,h2d_ms,kernel_ms,d2h_ms,"
           "gpu_total_ms,gpu_std_ms,gpu_wall_ms,speedup_kernel,speedup_total,max_diff\n";
    std::ofstream info(a.out + ".info.txt");
    info << prop.name << "\nCUDA driver " << drv << ", runtime " << rt
         << "\nblock " << a.bx << "x" << a.by << ", blur " << a.ksize << "x" << a.ksize
         << ", trials " << a.trials << ", cpu_max_batch " << a.cpu_max_batch << "\n";
    info.close();

    for (int size : a.sizes) {
        Image rgb = make_synthetic(size, size, 3, 1);
        const size_t npix = (size_t)size * size;
        std::vector<uchar> gray(npix), out_gpu(npix), out_cpu(npix);
        grayscale_cpu(rgb.data.data(), gray.data(), size, size);   // input for blur / sobel

        for (int fi = 0; fi < NUM_FILTERS; fi++) {
            if (!a.f[fi]) continue;
            const Filter f = (Filter)fi;
            const uchar* in = (f == Filter::Gray) ? rgb.data.data() : gray.data();

            for (int batch : a.batches) {
                const int cpu_reps = a.cpu_max_batch > 0 ? std::min(batch, a.cpu_max_batch) : batch;

                // Warm-up (not recorded): CUDA initialisation, cold caches, first-touch page faults.
                for (int w = 0; w < a.warmup; w++) gpu.run(f, in, out_gpu.data(), size, size);
                run_filter_cpu(f, in, out_cpu.data(), size, size, a.ksize);

                std::vector<double> cpu_t, h2d, ker, d2h, tot, wall;
                for (int t = 0; t < a.trials; t++) {
                    CpuTimer c; c.tic();
                    for (int i = 0; i < cpu_reps; i++)
                        run_filter_cpu(f, in, out_cpu.data(), size, size, a.ksize);
                    cpu_t.push_back(c.toc_ms() * batch / cpu_reps);

                    GpuTimes acc; CpuTimer w; w.tic();
                    for (int i = 0; i < batch; i++) acc += gpu.run(f, in, out_gpu.data(), size, size);
                    wall.push_back(w.toc_ms());
                    h2d.push_back(acc.h2d); ker.push_back(acc.kernel);
                    d2h.push_back(acc.d2h); tot.push_back(acc.total());
                }
                DiffResult d = compare(out_gpu.data(), out_cpu.data(), npix, 0);

                const double cpu_m = mean(cpu_t), ker_m = mean(ker), tot_m = mean(tot);
                char row[512];
                snprintf(row, sizeof row,
                    "%s,%d,%d,%d,%d,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.3f,%.3f,%d\n",
                    filter_name(f), size, size, batch, a.trials, cpu_reps, cpu_m, stdev(cpu_t),
                    mean(h2d), ker_m, mean(d2h), tot_m, stdev(tot), mean(wall),
                    ker_m > 0 ? cpu_m / ker_m : 0.0, tot_m > 0 ? cpu_m / tot_m : 0.0, d.maxDiff);
                csv << row; csv.flush();
                printf("%-6s %4dx%-4d batch %-3d | CPU %10.2f ms | GPU %9.2f ms (kernel %8.2f) | "
                       "speedup %6.2fx kernel, %6.2fx total | maxdiff %d\n",
                       filter_name(f), size, size, batch, cpu_m, tot_m, ker_m,
                       ker_m > 0 ? cpu_m / ker_m : 0.0, tot_m > 0 ? cpu_m / tot_m : 0.0, d.maxDiff);
            }
        }
    }
    printf("\nResults written to %s\n", a.out.c_str());
    return 0;
}
