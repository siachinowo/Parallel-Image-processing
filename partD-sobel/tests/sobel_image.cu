// Part D demo on a real picture: RGB -> gray (Part B) -> Sobel edges (Part D).
//
// Build (from the repo root; stb headers are in third_party/ after Part A is merged):
//   nvcc -O2 -Iinclude -Ithird_party tests/sobel_image.cu src/sobel.cu \
//        src/sobel_cpu.cpp src/grayscale.cu -o sobel_image
// Run:
//   ./sobel_image samples/sample_input.jpeg samples/sample_sobel_output.png
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#include <cstdio>
#include <vector>
#include "grayscale.h"
#include "sobel.h"
#include "verify.h"

int main(int argc, char** argv)
{
    if (argc < 3) {
        printf("Usage: ./sobel_image input.jpg output.png\n");
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

    std::vector<unsigned char> gray((size_t)w * h);
    std::vector<unsigned char> cpuOut((size_t)w * h);
    std::vector<unsigned char> plainOut((size_t)w * h);
    std::vector<unsigned char> sharedOut((size_t)w * h);

    // Part B's grayscale feeds Part D's Sobel
    grayscaleGPU(img, gray.data(), w, h);

    sobelCPU(gray.data(), cpuOut.data(), w, h);

    // Warm-up runs, results thrown away
    sobelGPU(gray.data(), plainOut.data(), w, h);
    sobelSharedGPU(gray.data(), sharedOut.data(), w, h);

    GpuTimes a = sobelGPU(gray.data(), plainOut.data(), w, h);
    GpuTimes b = sobelSharedGPU(gray.data(), sharedOut.data(), w, h);
    printf("plain : H2D %.3f ms, kernel %.3f ms, D2H %.3f ms\n",
           a.h2d_ms, a.kernel_ms, a.d2h_ms);
    printf("shared: H2D %.3f ms, kernel %.3f ms, D2H %.3f ms\n",
           b.h2d_ms, b.kernel_ms, b.d2h_ms);

    printf("plain : ");
    bool ok1 = verifyImages(cpuOut.data(), plainOut.data(), w * h, 0);
    printf("shared: ");
    bool ok2 = verifyImages(cpuOut.data(), sharedOut.data(), w * h, 0);
    printf("Result: %s\n", (ok1 && ok2) ? "PASS" : "FAIL");

    // Save the shared-memory GPU result as a 1-channel (gray) PNG
    stbi_write_png(argv[2], w, h, 1, sharedOut.data(), w);
    printf("Saved %s\n", argv[2]);

    stbi_image_free(img);
    return (ok1 && ok2) ? 0 : 1;
}
