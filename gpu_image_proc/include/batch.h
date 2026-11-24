// batch.h - folder (batch) pipeline: load -> GPU filters -> save, with optional CPU verification
#pragma once
#include <string>
#include <vector>
#include "gpu_pipeline.h"

struct FolderOptions {
    std::string in_dir, out_dir = "results/output";
    int  ksize = 3;                 // blur mask size: 3 or 5
    bool gray = true, blur = true, sobel = true;
    bool verify = false;            // also run CPU versions, time them and compare
    int  max_images = 0;            // 0 = all
};

struct FolderStats {
    int images = 0;
    double load_ms = 0, save_ms = 0, wall_ms = 0;
    GpuTimes gpu[NUM_FILTERS];
    double cpu_ms[NUM_FILTERS] = {0, 0, 0};
    int    max_diff[NUM_FILTERS] = {0, 0, 0};
    long long mismatches[NUM_FILTERS] = {0, 0, 0};
};

FolderStats process_folder(const FolderOptions& opt, GpuProcessor& gpu);
void print_summary(const FolderStats& st, const FolderOptions& opt);
