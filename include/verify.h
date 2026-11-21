#pragma once
#include <cstdio>
#include <cstdlib>

inline bool verifyImages(const unsigned char* cpu, const unsigned char* gpu,
                         int numValues, int tolerance)
{
    int mismatches = 0;
    int maxDiff = 0;

    for (int i = 0; i < numValues; i++) {
        int diff = abs((int)cpu[i] - (int)gpu[i]);
        if (diff > maxDiff) {
            maxDiff = diff;
        }
        if (diff > tolerance) {
            mismatches++;
        }
    }

    printf("  verify: max diff = %d, mismatches = %d of %d\n",
           maxDiff, mismatches, numValues);

    if (mismatches == 0) {
        return true;
    } else {
        return false;
    }
}
