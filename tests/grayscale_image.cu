#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdio>
#include <vector>
#include "grayscale.h"
#include "verify.h"

int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("Usage: ./test_image input.jpg output.png\n");
        return 1;
    }

    int w = 0;
    int h = 0;
    int channels = 0;

    // The last argument 3 forces RGB (3 bytes per pixel)
    unsigned char* img = stbi_load(argv[1], &w, &h, &channels, 3);
    if (img == NULL) {
        printf("Could not load image: %s\n", argv[1]);
        return 1;
    }
    printf("Loaded %s: %d x %d\n", argv[1], w, h);

    std::vector<unsigned char> cpuOut((size_t)w * h);
    std::vector<unsigned char> gpuOut((size_t)w * h);

    grayscaleCPU(img, cpuOut.data(), w, h);

    // Warm-up run, result thrown away
    grayscaleGPU(img, gpuOut.data(), w, h);

    GpuTimes t = grayscaleGPU(img, gpuOut.data(), w, h);
    printf("H2D: %.3f ms, kernel: %.3f ms, D2H: %.3f ms\n",
           t.h2d_ms, t.kernel_ms, t.d2h_ms);

    bool ok = verifyImages(cpuOut.data(), gpuOut.data(), w * h, 1);
    if (ok) {
        printf("Result: PASS\n");
    } else {
        printf("Result: FAIL\n");
    }

    // Save the GPU result as a 1-channel (gray) PNG
    stbi_write_png(argv[2], w, h, 1, gpuOut.data(), w);
    printf("Saved %s\n", argv[2]);

    stbi_image_free(img);
    return 0;
}
