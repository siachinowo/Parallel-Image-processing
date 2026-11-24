#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include "image_io.h"
#include <filesystem>
#include <algorithm>
#include <cstdio>
namespace fs = std::filesystem;

std::vector<std::string> list_images(const std::string& dir) {
    std::vector<std::string> files;
    if (!fs::is_directory(dir)) return files;
    for (auto& e : fs::directory_iterator(dir)) {
        if (!e.is_regular_file()) continue;
        std::string ext = e.path().extension().string();
        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
        if (ext == ".png" || ext == ".jpg" || ext == ".jpeg" || ext == ".bmp")
            files.push_back(e.path().string());
    }
    std::sort(files.begin(), files.end());
    return files;
}

bool load_rgb(const std::string& path, Image& img) {
    int w, h, ch;
    unsigned char* p = stbi_load(path.c_str(), &w, &h, &ch, 3);
    if (!p) { fprintf(stderr, "Cannot load %s: %s\n", path.c_str(), stbi_failure_reason()); return false; }
    img.w = w; img.h = h; img.name = fs::path(path).filename().string();
    img.rgb.assign(p, p + (size_t)w * h * 3);
    stbi_image_free(p);
    return true;
}

bool save_png(const std::string& path, int w, int h, int ch, const u8* data) {
    fs::path parent = fs::path(path).parent_path();
    if (!parent.empty()) fs::create_directories(parent);
    return stbi_write_png(path.c_str(), w, h, ch, data, w * ch) != 0;
}
