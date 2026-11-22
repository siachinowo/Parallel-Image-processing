#include "grayscale.h"

void grayscaleCPU(const unsigned char* in, unsigned char* out,
                  int width, int height)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            int idx = y * width + x;
            unsigned char r = in[idx * 3];
            unsigned char g = in[idx * 3 + 1];
            unsigned char b = in[idx * 3 + 2];
            out[idx] = (unsigned char)(0.299f * r + 0.587f * g + 0.114f * b);
        }
    }
}
