# Lessons Learnt — llama.cpp CUDA Setup on Fedora 43

## 1. CUDA drops old GPU architectures aggressively

CUDA 13.x dropped Pascal (sm_61) entirely. If your GPU is older than Turing (GTX 10xx series or earlier), check the CUDA release notes before upgrading. The last CUDA version to support Pascal is **CUDA 12.x**.

Always verify GPU compute capability against CUDA architecture support before installing a new toolkit version.

## 2. The system compiler and CUDA toolkit must be compatible — check both ends

CUDA 12.6 supports up to GCC 13. GCC 14/15 will fail at the `cudafe` preprocessing stage with errors about unknown builtins, not just version check warnings. The version check error (`--allow-unsupported-compiler`) can be suppressed, but the actual compilation will still fail.

Check `/usr/local/cuda-X.Y/include/crt/host_config.h` for the `#if __GNUC__ > N` limit before trying to build.

## 3. Non-system compilers (Homebrew GCC) may be compiled against a different glibc

Homebrew GCC is built on Ubuntu and linked against an older glibc. On Fedora 43 with glibc 2.42, Homebrew GCC 13 failed with pthread initialisation errors (`__GTHREAD_COND_INIT` incompatible). Always prefer distro-packaged compilers when glibc compatibility matters.

## 4. Bleeding-edge glibc adds functions that old CUDA headers also declare

glibc 2.42 added ISO C23 math functions (`cospi`, `sinpi`, `rsqrt`, etc.) that CUDA 12.6 already declares in its own headers — with different exception specifications. This causes `cudafe` to fail with "exception specification is incompatible" errors.

The fix is to patch the CUDA headers to match glibc's `noexcept` declarations. Keep a backup and re-apply after CUDA toolkit updates.

## 5. Clang as CUDA compiler sidesteps most of these issues

Using clang as the CUDA compiler (rather than nvcc) avoids the `cudafe` preprocessor entirely. Clang has its own CUDA frontend that handles modern C++ headers cleanly. When nvcc proves incompatible with the system toolchain, clang is often the path of least resistance.

Use the **system cmake**, not Homebrew cmake — Homebrew cmake may lack the `Clang-CUDA.cmake` module needed for clang as a CUDA compiler.

## 6. Check what GPU backend a binary was actually built for before assuming it works

The installed `llama-server` binary was a ROCm/HIP build masquerading as a working binary. It failed silently at runtime with a missing shared library. Always run `ldd <binary> | grep "not found"` to verify all dependencies resolve before putting a binary into production use.

## 7. Systemd service files can silently carry over wrong configuration

Both service files had `HSA_OVERRIDE_GFX_VERSION=8.0.3` — an AMD ROCm tuning variable — left over from a previous GPU. It caused no visible error but indicated the services had never been properly updated for the current hardware. When changing GPU hardware, audit all environment variables in service files.

## 8. GPU VRAM is shared with the desktop — account for it

On a desktop system, GNOME, Wayland compositing, and other GPU processes consume VRAM (600–700 MiB on this machine). Model layer counts and context sizes must be tuned with this overhead in mind, not just the card's total VRAM.

## 9. Stray processes hold VRAM and cause false OOM errors

Interactive llama.cpp sessions (especially ones killed mid-run or left in background) hold VRAM until the process exits. Always check `nvidia-smi` for lingering compute processes before diagnosing an out-of-memory error as a configuration problem.

## 10. CUDA toolkit version and driver version are independent

The NVIDIA driver reported CUDA 13.0 support, but the installed toolkit was 13.1 with a separate 12.6 toolkit also present. The driver version sets the maximum CUDA version supported at runtime; the toolkit version is what you compile against. They don't need to match exactly but the driver must be >= the toolkit's required driver version.

## 11. llama-bench leaves a stray process that holds VRAM

`llama-bench` does not always clean up after itself when killed mid-run or backgrounded. The process persists and holds its full VRAM allocation, preventing `llama-server` from loading. Before diagnosing a server startup failure, check `nvidia-smi` for lingering `llama-bench` processes and kill them explicitly.

## 12. Port conflicts with Ollama — check before starting llama-server

Ollama runs on port 11434 by default — the same port used by `llama-server`. If Ollama is running, `llama-server` will fail at bind with "couldn't bind HTTP server socket". Run `sudo ss -tlnp | grep 11434` to check, then `sudo systemctl stop ollama` before starting `llama-server`.

## 13. Flash attention is slower on Pascal (sm_61)

On the GTX 1070 (sm_61), enabling flash attention (`-fa 1`) reduced prompt processing speed by ~3% (473 t/s vs 488 t/s) with no improvement to token generation. Flash attention is optimised for Turing (sm_75+) tensor cores. Leave it disabled on Pascal hardware.

## 14. Thread count has no effect when fully GPU-bound

With all model layers offloaded to GPU (`-ngl 99`), varying CPU thread count from 4 to 16 made no measurable difference to throughput. The bottleneck is GPU memory bandwidth. Don't waste time tuning `--threads` on a fully GPU-offloaded model.

## 15. `ngl` only needs to cover actual layer count

Setting `--n-gpu-layers 33` vs `99` made no difference for Qwen2.5-7B, which has 28 transformer layers. Any value above the actual layer count puts the whole model on GPU. There is no benefit to setting an arbitrarily large value, but it also causes no harm.

## 16. Small LLMs (7-8B) cannot generate valid IaC reliably

Tested Qwen2.5-Coder-7B and DeepSeek-R1-0528-Qwen3-8B on an AKS Terraform task:
- Qwen2.5-Coder-7B produced plausible but invalid Terraform — used `node_pool {}` blocks inside `azurerm_kubernetes_cluster` (not a real azurerm resource) and deprecated `addon_profile` syntax.
- DeepSeek-R1-8B hit the token limit in a repetition loop and never produced `main.tf`.
For IaC generation that needs to be deployable, 7-8B models require careful human review. 32B+ models are needed to approach reliability.
