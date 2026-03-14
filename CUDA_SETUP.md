# CUDA Setup Notes — GTX 1070 on Fedora 43

## System

- **CPU:** Intel Xeon E5-2667 v4 @ 3.20GHz
- **GPU:** NVIDIA GeForce GTX 1070 (Pascal, sm_61, 8GB VRAM)
- **OS:** Fedora 43, glibc 2.42
- **Default GCC:** 15.2.1

## Problems and Solutions

### 1. CUDA 13.1 dropped Pascal (sm_61) support

CUDA 13.1 (installed at `/usr/local/cuda`) removed code generation for `compute_61`. The GTX 1070 requires sm_61.

**Fix:** Installed CUDA 12.6 from the NVIDIA fedora39 repo — the last CUDA 12.x release, which still supports Pascal.

```bash
sudo dnf config-manager addrepo --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/cuda-fedora39.repo
sudo dnf install --repo=cuda-fedora39-x86_64 -y cuda-toolkit-12-6
```

### 2. CUDA 12.6 incompatible with GCC 14/15

CUDA 12.6's internal `cudafe` preprocessor only supports up to GCC 13. GCC 14 and 15 use builtins (`__type_pack_element`) that `cudafe` doesn't understand. GCC 13 from Homebrew was compiled against older glibc and incompatible with glibc 2.42's pthreads headers.

**Fix:** Use **clang++-21** (system package) as the CUDA compiler instead of nvcc. Clang has its own CUDA frontend that doesn't use `cudafe`.

### 3. glibc 2.42 conflicts with CUDA 12.6 math headers

glibc 2.42 added ISO C23 math functions (`cospi`, `sinpi`, `rsqrt`, etc.) to `mathcalls.h` with `noexcept(true)`. CUDA 12.6's `crt/math_functions.h` declares the same functions without `noexcept`, causing a conflict.

**Fix:** Patched `/usr/local/cuda-12.6/targets/x86_64-linux/include/crt/math_functions.h` to add `noexcept` to the six conflicting declarations:

```
rsqrt(double x)   rsqrtf(float x)
sinpi(double x)   sinpif(float x)
cospi(double x)   cospif(float x)
```

Backup saved at `math_functions.h.bak`.

### 4. nvcc-only flags broke clang compilation

`ggml/src/ggml-cuda/CMakeLists.txt` unconditionally set `-use_fast_math -extended-lambda`, which are nvcc-only flags not understood by clang.

**Fix:** Added a compiler ID check in `CMakeLists.txt`:

```cmake
if (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    set(CUDA_FLAGS -ffast-math)
else()
    set(CUDA_FLAGS -use_fast_math -extended-lambda)
endif()
```

## Working Build Command

Uses the **system cmake** (`/usr/bin/cmake`) — the Homebrew cmake lacks the `Clang-CUDA.cmake` module required for clang as CUDA compiler.

```bash
rm -rf build && /usr/bin/cmake -B build \
  -DGGML_CUDA=ON \
  -DCMAKE_CUDA_COMPILER=/usr/bin/clang++-21 \
  -DCMAKE_C_COMPILER=/usr/bin/clang-21 \
  -DCMAKE_CXX_COMPILER=/usr/bin/clang++-21 \
  -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 \
  "-DCMAKE_CUDA_FLAGS=--cuda-path=/usr/local/cuda-12.6 --gcc-install-dir=/usr/lib/gcc/x86_64-redhat-linux/14 -Wno-unknown-cuda-version" \
  "-DCMAKE_CXX_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-redhat-linux/14" \
  "-DCMAKE_C_FLAGS=--gcc-install-dir=/usr/lib/gcc/x86_64-redhat-linux/14" \
  -S . && \
/usr/bin/cmake --build build --parallel $(nproc)
```

## Systemd Services

The installed `/usr/local/bin/llama-server` was a broken ROCm/HIP binary (`libggml-hip.so.0`) from a previous build. Fixed by:

1. Copying the new CUDA binary to `/usr/local/bin/llama-server`
2. Installing shared libs to `/usr/local/lib/` and running `ldconfig`
3. Removing `HSA_OVERRIDE_GFX_VERSION=8.0.3` (AMD/ROCm env var) from both service files
4. Fixing `llama-server.service` description ("Vulkan GPU" → "CUDA GPU") and `WantedBy=default.target` → `multi-user.target`
5. Updating `llama-reason.service` `--n-gpu-layers 33` → `36` (DeepSeek-R1-8B has 36 layers total)

```bash
sudo cp build/bin/llama-server /usr/local/bin/llama-server
sudo cp build/bin/libggml*.so* build/bin/libllama*.so* /usr/local/lib/
sudo ldconfig
sudo systemctl daemon-reload
```

## Performance

**DeepSeek-R1-0528-Qwen3-8B Q4_K_M** (all 36 layers on GPU, ctx 8192):

| Metric | Result |
|--------|--------|
| Prompt processing | ~20–65 t/s |
| Token generation | ~17–20 t/s |
| Model size | 4.68 GiB |
| VRAM headroom | ~2.5 GB remaining |
