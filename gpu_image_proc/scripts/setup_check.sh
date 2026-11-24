#!/usr/bin/env bash
# Run this first. Checks driver + compiler, and downloads the stb headers if missing.
cd "$(dirname "$0")/.." || exit 1

echo "== GPU / driver =="
if ! nvidia-smi; then
  echo "nvidia-smi failed: install/repair the NVIDIA driver before continuing."; exit 1
fi
echo; echo "== CUDA compiler =="
if ! nvcc --version; then
  echo "nvcc not found: install the CUDA toolkit (and add its bin folder to PATH)."; exit 1
fi
echo "(The CUDA version shown by nvidia-smi is the MAXIMUM the driver supports;"
echo " the nvcc version above must not be newer than that.)"

cap=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1)
if [ -n "$cap" ]; then
  echo; echo "GPU compute capability: $cap"
  echo "If 'make' complains about -arch=native, use:  make ARCH=sm_${cap/./}"
fi

echo; echo "== stb headers =="
mkdir -p include
for f in stb_image.h stb_image_write.h; do
  if [ -f "include/$f" ]; then echo "$f: present"; continue; fi
  url="https://raw.githubusercontent.com/nothings/stb/master/$f"
  echo "downloading $f ..."
  curl -fsSL "$url" -o "include/$f" 2>/dev/null || wget -q "$url" -O "include/$f" \
    || echo "could not download $f; get it from https://github.com/nothings/stb and copy to include/"
done
echo; echo "Setup check finished. Next: make test_cuda && make"
