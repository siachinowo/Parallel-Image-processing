// main.cpp - batch image processor.  Usage:
//   ./imgproc --input images --output results/output [--blur 3|5] [--filters gray,blur,sobel]
//             [--verify] [--max N]
#include "batch.h"
#include <cstdio>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

static void usage() {
    printf("Usage: imgproc --input <folder> [--output <folder>] [--blur 3|5]\n"
           "               [--filters gray,blur,sobel] [--verify] [--max N]\n"
           "  --verify   also run the CPU versions, time them and compare with the GPU output\n");
}

int main(int argc, char** argv) {
    FolderOptions o;
    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        auto next = [&]() -> std::string { return (i + 1 < argc) ? argv[++i] : ""; };
        if      (a == "--input")   o.in_dir = next();
        else if (a == "--output")  o.out_dir = next();
        else if (a == "--blur")    o.ksize = std::stoi(next());
        else if (a == "--max")     o.max_images = std::stoi(next());
        else if (a == "--verify")  o.verify = true;
        else if (a == "--filters") {
            std::string f = next();
            o.gray  = f.find("gray")  != std::string::npos;
            o.blur  = f.find("blur")  != std::string::npos;
            o.sobel = f.find("sobel") != std::string::npos;
        } else { usage(); return a == "--help" ? 0 : 1; }
    }
    if (o.in_dir.empty()) { usage(); return 1; }

    try {
        GpuProcessor gpu(o.ksize);
        // Untimed warm-up on a tiny image so CUDA start-up cost is not counted.
        { std::vector<uchar> tiny(64 * 64 * 3, 0), tout(64 * 64);
          gpu.run(Filter::Gray, tiny.data(), tout.data(), 64, 64); }
        FolderStats st = process_folder(o, gpu);
        print_summary(st, o);
    } catch (const std::exception& e) {
        fprintf(stderr, "error: %s\n", e.what());
        return 1;
    }
    return 0;
}
