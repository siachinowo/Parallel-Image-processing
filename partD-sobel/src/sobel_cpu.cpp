#include "sobel.h"

static int clampInt(int v, int lo, int hi)
{
    if (v < lo) return lo;
    if (v > hi) return hi;
    return v;
}

void sobelCPU(const unsigned char* in, unsigned char* out,
              int width, int height)
{
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            // Neighbour coordinates, clamped so the border pixels
            // simply repeat the edge pixel.
            int xm = clampInt(x - 1, 0, width - 1);
            int xp = clampInt(x + 1, 0, width - 1);
            int ym = clampInt(y - 1, 0, height - 1);
            int yp = clampInt(y + 1, 0, height - 1);

            int tl = in[ym * width + xm];
            int t  = in[ym * width + x];
            int tr = in[ym * width + xp];
            int l  = in[y  * width + xm];
            int r  = in[y  * width + xp];
            int bl = in[yp * width + xm];
            int b  = in[yp * width + x];
            int br = in[yp * width + xp];

            int gx = -tl + tr - 2 * l + 2 * r - bl + br;
            int gy = -tl - 2 * t - tr + bl + 2 * b + br;

            int m = (gx < 0 ? -gx : gx) + (gy < 0 ? -gy : gy);
            if (m > 255) m = 255;
            out[y * width + x] = (unsigned char)m;
        }
    }
}
