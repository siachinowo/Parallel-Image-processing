#pragma once
#include <string>
#include <vector>
#include "filters.h"

struct Image { int w = 0, h = 0; std::string name; std::vector<u8> rgb; };

std::vector<std::string> list_images(const std::string& dir);
bool load_rgb(const std::string& path, Image& img);
bool save_png(const std::string& path, int w, int h, int channels, const u8* data);
