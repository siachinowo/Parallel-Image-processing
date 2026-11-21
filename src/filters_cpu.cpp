#include "filters.h"

static inline int cl(int v, int lo, int hi) { return v < lo ? lo : (v > hi ? hi : v); }

void cpu_grayscale(const u8* rgb, u8* gray, int w, int h) {
    for (int i = 0; i < w * h; ++i) {
        float g = 0.299f * rgb[3*i] + 0.587f * rgb[3*i+1] + 0.114f * rgb[3*i+2];
        gray[i] = (u8)(g + 0.5f);
    }
}

void cpu_blur(const u8* in, u8* out, int w, int h) {
    const int k[9] = {1,2,1, 2,4,2, 1,2,1};
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            int s = 0;
            for (int dy = -1; dy <= 1; ++dy)
                for (int dx = -1; dx <= 1; ++dx)
                    s += k[(dy+1)*3 + (dx+1)] * in[cl(y+dy,0,h-1)*w + cl(x+dx,0,w-1)];
            out[y*w + x] = (u8)(s / 16);
        }
}

void cpu_sobel(const u8* in, u8* out, int w, int h) {
    for (int y = 0; y < h; ++y)
        for (int x = 0; x < w; ++x) {
            #define P(dx,dy) ((int)in[cl(y+(dy),0,h-1)*w + cl(x+(dx),0,w-1)])
            int gx = -P(-1,-1) + P(1,-1) - 2*P(-1,0) + 2*P(1,0) - P(-1,1) + P(1,1);
            int gy = -P(-1,-1) - 2*P(0,-1) - P(1,-1) + P(-1,1) + 2*P(0,1) + P(1,1);
            #undef P
            int m = (gx < 0 ? -gx : gx) + (gy < 0 ? -gy : gy);
            out[y*w + x] = (u8)(m > 255 ? 255 : m);
        }
}
