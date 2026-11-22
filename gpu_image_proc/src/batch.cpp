// batch.cpp - folder pipeline: for every image: load -> gray -> blur -> sobel -> save
#include "batch.h"
#include "imageio.h"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <stdexcept>

namespace fs = std::filesystem;

FolderStats process_folder(const FolderOptions& o, GpuProcessor& gpu) {
    FolderStats st;
    std::vector<std::string> files = list_images(o.in_dir);
    if (o.max_images > 0 && (int)files.size() > o.max_images) files.resize(o.max_images);
    if (files.empty()) throw std::runtime_error("no images found in " + o.in_dir);
    fs::create_directories(o.out_dir);
    gpu.set_blur_ksize(o.ksize);

    const bool enabled[NUM_FILTERS] = {o.gray, o.blur, o.sobel};
    std::vector<uchar> gray, out, ref;
    CpuTimer wall; wall.tic();

    for (const std::string& path : files) {
        // 1) load (images are processed one at a time, so memory use stays small)
        CpuTimer t; t.tic();
        Image img;
        try { img = load_image(path, 3); }
        catch (const std::exception& e) { fprintf(stderr, "skip: %s\n", e.what()); continue; }
        st.load_ms += t.toc_ms();

        const size_t npix = (size_t)img.w * img.h;
        gray.resize(npix); out.resize(npix); ref.resize(npix);
        const std::string stem = fs::path(path).stem().string();

        // 2) run one filter on the GPU, optionally verify on the CPU, then save
        auto step = [&](Filter f, const uchar* in, uchar* dst) {
            const int k = (int)f;
            GpuTimes gt = gpu.run(f, in, dst, img.w, img.h);
            st.gpu[k] += gt;
            if (o.verify) {
                CpuTimer c; c.tic();
                run_filter_cpu(f, in, ref.data(), img.w, img.h, o.ksize);
                st.cpu_ms[k] += c.toc_ms();
                DiffResult d = compare(dst, ref.data(), npix, 0);
                st.max_diff[k] = std::max(st.max_diff[k], d.maxDiff);
                st.mismatches[k] += d.mismatches;
            }
            CpuTimer s; s.tic();
            std::string name = o.out_dir + "/" + stem + "_" + filter_name(f) + ".png";
            if (!save_image(name, dst, img.w, img.h, 1))
                fprintf(stderr, "could not save %s\n", name.c_str());
            st.save_ms += s.toc_ms();
        };

        // Grayscale is always needed (it is the input of blur and sobel).
        if (enabled[0]) step(Filter::Gray, img.data.data(), gray.data());
        else            gpu.run(Filter::Gray, img.data.data(), gray.data(), img.w, img.h);
        if (enabled[1]) step(Filter::Blur,  gray.data(), out.data());
        if (enabled[2]) step(Filter::Sobel, gray.data(), out.data());
        st.images++;
    }
    st.wall_ms = wall.toc_ms();
    return st;
}

void print_summary(const FolderStats& st, const FolderOptions& o) {
    printf("\nProcessed %d image(s) in %.1f ms wall time (load %.1f ms, save %.1f ms)\n",
           st.images, st.wall_ms, st.load_ms, st.save_ms);
    printf("\n%-7s %10s %10s %10s %10s", "filter", "H2D(ms)", "kernel(ms)", "D2H(ms)", "GPU(ms)");
    if (o.verify) printf(" %10s %9s %8s", "CPU(ms)", "speedup", "maxdiff");
    printf("\n");
    const bool enabled[NUM_FILTERS] = {o.gray, o.blur, o.sobel};
    for (int k = 0; k < NUM_FILTERS; k++) {
        if (!enabled[k]) continue;
        const GpuTimes& g = st.gpu[k];
        printf("%-7s %10.3f %10.3f %10.3f %10.3f", filter_name((Filter)k),
               g.h2d, g.kernel, g.d2h, g.total());
        if (o.verify)
            printf(" %10.3f %8.2fx %8d", st.cpu_ms[k],
                   g.total() > 0 ? st.cpu_ms[k] / g.total() : 0.0, st.max_diff[k]);
        printf("\n");
    }
    if (o.verify) {
        long long bad = 0;
        for (int k = 0; k < NUM_FILTERS; k++) bad += st.mismatches[k];
        printf("\nVerification: %s\n", bad == 0 ? "PASS (GPU output identical to CPU)"
                                               : "FAIL (GPU and CPU differ, see maxdiff)");
    }
}
