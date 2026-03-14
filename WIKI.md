# llama.cpp CUDA Build — GTX 1070 on Fedora 43

## Project Overview

This project documents the setup, configuration, and operation of a local LLM inference server using [llama.cpp](https://github.com/ggml-org/llama.cpp) with CUDA GPU acceleration on a consumer desktop running Fedora 43. The server runs continuously as a systemd service and exposes an OpenAI-compatible HTTP API for use by local tooling.

The entire setup — from diagnosing build failures to patching CUDA headers and benchmarking models — was carried out collaboratively with **[Claude Code](https://claude.ai/claude-code)** (Anthropic's AI-assisted development CLI), which is documented throughout this wiki.

---

## Hardware

| Component | Detail |
|---|---|
| CPU | Intel Xeon E5-2667 v4 @ 3.20GHz (16 threads) |
| GPU | NVIDIA GeForce GTX 1070 (Pascal, sm_61, 8GB VRAM) |
| OS | Fedora 43 |
| glibc | 2.42 |

### VRAM Budget

The GTX 1070 has 8GB VRAM, but it is shared with the desktop environment:

| Consumer | VRAM |
|---|---|
| GNOME Shell / Wayland | ~274 MiB |
| Xwayland | ~7 MiB |
| GNOME Remote Desktop | ~86 MiB |
| Docker Desktop | ~14 MiB |
| Other desktop processes | ~230 MiB |
| **Available for models** | **~7.2 GB** |

A Q4_K_M quantised 7-8B model uses ~4.4–4.7 GB, leaving ~2.5 GB headroom for the KV cache at ctx=8192.

---

## Why llama.cpp?

llama.cpp is a C/C++ inference engine for GGUF-format LLMs. It supports CUDA, Vulkan, Metal, and CPU backends, is actively maintained, and exposes an OpenAI-compatible REST API via `llama-server`. It was chosen for:

- Low overhead compared to Python-based inference stacks
- Direct CUDA support without requiring a full ML framework
- Compatibility with GGUF models from HuggingFace / LM Studio
- A stable HTTP API usable by any OpenAI-compatible client

---

## The Build Challenge

Building llama.cpp with CUDA on this machine was non-trivial due to a chain of compatibility problems. Each was diagnosed and resolved collaboratively with Claude Code.

### Problem 1 — CUDA 13.x Dropped Pascal Support

The system default CUDA installation (`/usr/local/cuda`) was version 13.1, which removed code generation for `compute_61` (Pascal architecture). The GTX 1070 requires `sm_61`.

**Resolution:** Installed CUDA 12.6 from the NVIDIA fedora39 repository — the last CUDA 12.x release, which still supports Pascal:

```bash
sudo dnf config-manager addrepo \
  --from-repofile=https://developer.download.nvidia.com/compute/cuda/repos/fedora39/x86_64/cuda-fedora39.repo
sudo dnf install --repo=cuda-fedora39-x86_64 -y cuda-toolkit-12-6
```

CUDA 12.6 is installed at `/usr/local/cuda-12.6`. The default symlink at `/usr/local/cuda` still points to 13.x — the 12.6 path must be specified explicitly in cmake flags.

### Problem 2 — GCC 14/15 Incompatible with CUDA 12.6

CUDA 12.6's internal `cudafe` preprocessor supports up to GCC 13. The system default GCC is 15.2.1; GCC 14 is also available. Both fail at the `cudafe` stage with errors about unknown builtins (`__type_pack_element`), not just version warnings.

GCC 13 from Homebrew was investigated but rejected — Homebrew GCC is compiled against an older glibc and fails with pthread initialisation errors on glibc 2.42.

**Resolution:** Use **clang++-21** (system Clang, packaged by Fedora) as the CUDA compiler. Clang has its own CUDA frontend that bypasses `cudafe` entirely and handles modern C++ and glibc headers cleanly.

### Problem 3 — glibc 2.42 Conflicts with CUDA 12.6 Math Headers

glibc 2.42 added ISO C23 math functions (`cospi`, `sinpi`, `rsqrt`, etc.) to `mathcalls.h` with `noexcept(true)`. CUDA 12.6's `crt/math_functions.h` declares the same functions without `noexcept`, causing a conflict:

```
error: exception specification is incompatible with that of declaration
```

**Resolution:** Patched `/usr/local/cuda-12.6/targets/x86_64-linux/include/crt/math_functions.h` to add `noexcept` to the six conflicting declarations (`rsqrt`, `rsqrtf`, `sinpi`, `sinpif`, `cospi`, `cospif`). A backup was saved alongside the patched file. This patch must be re-applied after any CUDA 12.6 toolkit update.

### Problem 4 — nvcc-Only Flags in CMakeLists.txt

`ggml/src/ggml-cuda/CMakeLists.txt` unconditionally passed `-use_fast_math -extended-lambda` as CUDA compiler flags. These are nvcc-only flags; clang does not recognise them and fails at compile time.

**Resolution:** Patched `CMakeLists.txt` to check the compiler ID:

```cmake
if (CMAKE_CUDA_COMPILER_ID STREQUAL "Clang")
    set(CUDA_FLAGS -ffast-math)
else()
    set(CUDA_FLAGS -use_fast_math -extended-lambda)
endif()
```

This patch is committed to the local working tree (visible in `git diff`).

### Problem 5 — Homebrew cmake Lacks Clang-CUDA Support

The Homebrew cmake installation does not include the `Clang-CUDA.cmake` module required to use clang as a CUDA compiler. Using it produces a configure error with no obvious cause.

**Resolution:** Always use the system cmake at `/usr/bin/cmake`.

---

## Working Build Command

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
  -S /home/brendan/llama.cpp && \
/usr/bin/cmake --build build --parallel $(nproc)
```

cmake detects `61-real` as the CUDA architecture and builds specifically for Pascal.

---

## Deployment

After building, deploy the server binary and shared libraries:

```bash
sudo cp build/bin/llama-server /usr/local/bin/llama-server
sudo cp build/bin/libggml*.so* build/bin/libllama*.so* /usr/local/lib/
sudo ldconfig
sudo systemctl daemon-reload
```

---

## Systemd Service

The server runs as a systemd service under the `ollama` user on port 11434 (OpenAI-compatible API).

**`/etc/systemd/system/llama-server.service`:**

```ini
[Unit]
Description=llama.cpp Server (CUDA GPU)
After=network-online.target
Wants=network-online.target

[Service]
ExecStart=/usr/local/bin/llama-server \
            --model /home/brendan/.lmstudio/models/Qwen/Qwen2.5-Coder-7B-Instruct-GGUF/qwen2.5-coder-7b-instruct-q4_k_m.gguf \
            --host 0.0.0.0 \
            --port 11434 \
            --n-gpu-layers 33 \
            --ctx-size 8192 \
            --parallel 1 \
            --threads 16 \
            --alias qwen2.5-coder-7b
User=ollama
Group=ollama
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
```

> **Note:** Port 11434 is also Ollama's default port. If Ollama is running, `llama-server` will fail to bind. Run `sudo systemctl stop ollama` first.

---

## Models

| Model | Size | VRAM | Use Case |
|---|---|---|---|
| Qwen2.5-Coder-7B-Instruct Q4_K_M | 4.36 GiB | ~4.4 GB | Primary server model — code generation, chat |
| DeepSeek-R1-0528-Qwen3-8B Q4_K_M | 4.68 GiB | ~4.7 GB | Reasoning tasks — runs on demand on port 11436 |

Models are stored under `/home/brendan/.lmstudio/models/` and sourced from LM Studio / HuggingFace.

---

## Performance

Benchmarked with `llama-bench` on the Qwen2.5-Coder-7B Q4_K_M model:

| Test | Speed |
|---|---|
| Prompt processing (pp512) | ~489 t/s |
| Token generation (tg128) | ~32.5 t/s |

### Tuning Findings

| Parameter | Finding |
|---|---|
| Flash attention (`-fa 1`) | **Slower** on Pascal (~473 vs ~489 t/s). Disabled. |
| Thread count (4 vs 16) | No effect — fully GPU-bound |
| `--n-gpu-layers` above actual layer count | No effect — Qwen2.5-7B has 28 layers; ngl=33 already fully offloads |

The ~32.5 t/s generation speed is near the GTX 1070's memory bandwidth ceiling for this model size.

---

## Known Issues & Gotchas

### Port conflict with Ollama
Ollama uses port 11434 by default. Check with `sudo ss -tlnp | grep 11434` and stop Ollama before starting llama-server.

### Stray llama-bench processes
`llama-bench` can leave a running process holding VRAM after being killed or backgrounded. This prevents llama-server from loading. Always check `nvidia-smi` for lingering compute processes before diagnosing a server startup failure.

### CUDA path must be explicit
`/usr/local/cuda` symlinks to CUDA 13.x which drops Pascal support. Always pass `-DCUDAToolkit_ROOT=/usr/local/cuda-12.6` explicitly to cmake.

### math_functions.h patch is not persistent
The patch to `/usr/local/cuda-12.6/.../crt/math_functions.h` will be overwritten by a CUDA toolkit update. Re-apply from the `.bak` file if the build breaks after a system update.

---

## LLM Capability Comparison

As part of this project, Claude Code and the locally-served models were compared on an infrastructure code generation task: *"Create Terraform configuration for an AKS cluster with 8 CPU node pools for a dev lab environment."*

| | Claude Sonnet 4.6 | Qwen2.5-Coder-7B | DeepSeek-R1-8B |
|---|---|---|---|
| Valid, deployable Terraform | **Yes** | No | No |
| Correct resource structure | **Yes** | No — used `node_pool {}` blocks inside the cluster resource (invalid) | Never reached main.tf |
| Provider version pinning | **Yes** | No | No |
| VNet / subnet provisioned | **Yes** | No | No |
| Node pools as reusable variable | **Yes** (list of objects) | No (copy-pasted blocks) | N/A |
| Outputs file included | **Yes** | No | No |
| Repetition / hallucination | None | Minor | Severe — hit token limit in a loop |
| Generation speed | — | ~31 t/s | ~25 t/s |

**Conclusion:** At the 7-8B parameter scale, open models can generate plausible-looking IaC but require careful human review before use. Neither model produced code that would pass `terraform validate`. Reliable IaC generation from local models requires 32B+ parameters, which exceeds the VRAM capacity of the GTX 1070.

For tasks like code completion, explanation, and debugging, the 7B Qwen2.5-Coder model is useful. For tasks requiring strict structural correctness (IaC, complex API integrations), Claude remains the better tool.

---

## Role of Claude Code

This entire project was built interactively with **Claude Code** — Anthropic's CLI tool that gives Claude direct access to the terminal, filesystem, and running processes.

Claude Code was used to:

- **Diagnose build failures** — reading compiler error output and identifying root causes across the CUDA/glibc/GCC compatibility chain
- **Research and apply fixes** — patching `math_functions.h`, modifying `CMakeLists.txt`, selecting the correct compiler flags
- **Run and interpret benchmarks** — executing `llama-bench`, querying the HTTP API, and analysing speed results
- **Manage system services** — starting, stopping, and diagnosing systemd services; identifying the Ollama port conflict; killing stray processes
- **Write documentation** — generating `CUDA_SETUP.md`, `LESSONS_LEARNT.md`, `setup.json`, and this wiki
- **Run model comparisons** — sending identical prompts to local models and Claude, evaluating output quality

The workflow was conversational: problems were described or encountered live, and Claude Code proposed and executed solutions in the terminal, with the user approving individual tool calls. This made it practical to work through a technically dense setup that involved multiple interacting compatibility problems simultaneously.

---

## File Reference

| File | Purpose |
|---|---|
| `CUDA_SETUP.md` | Step-by-step CUDA setup notes with commands |
| `LESSONS_LEARNT.md` | Numbered list of lessons from the build and operation experience |
| `setup.json` | Machine-readable project configuration for sharing/recreation |
| `WIKI.md` | This file — full project documentation |
| `ggml/src/ggml-cuda/CMakeLists.txt` | Locally patched to support clang as CUDA compiler |
