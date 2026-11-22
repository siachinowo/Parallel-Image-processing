#pragma once
#include "timing.h"

void grayscaleCPU(const unsigned char* in, unsigned char* out,
                  int width, int height);

GpuTimes grayscaleGPU(const unsigned char* h_in, unsigned char* h_out,
                      int width, int height);
