<div align="center">

# Haojie Guo · 郭浩杰

**GPU Kernels · HPC · AI Infra**

MSc in High Performance Computing @ Moscow State University<br/>
AI Infra Intern @ Huawei 2012 Labs

[![Email](https://img.shields.io/badge/ghaojie@yandex.ru-D14836?style=flat-square&logo=maildotru&logoColor=white)](mailto:ghaojie@yandex.ru)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=flat-square&logo=nvidia&logoColor=white)
![Triton](https://img.shields.io/badge/Triton-EE4C2C?style=flat-square)
![MPI](https://img.shields.io/badge/MPI%20%2B%20OpenMP-0B1F3A?style=flat-square)

</div>

---

I write kernels that chase cuBLAS, and inference paths that cut tail latency.

### Work

**[SGEMM_CUDA_P100](https://github.com/GuoHaojiee/SGEMM_CUDA_P100)** — SGEMM from scratch, three paths to cuBLAS parity<br/>
`Triton Split-K` **106.6%** of cuBLAS · `CUDA Core` **91.6%** · `Tensor Core` **86–90%** (M=N=K=4096)

**[FlashAttention_CUDA](https://github.com/GuoHaojiee/FlashAttention_CUDA)** — tiling + online softmax, O(N²) → O(N)<br/>
**7.2×** end-to-end speedup, profiled kernel by kernel with Nsight Compute

**[nano-vllm](https://github.com/GuoHaojiee/nano-vllm)** — chunked prefill scheduling for LLM inference<br/>
TBT p99 **−53.7%** (189.0 → 87.5 ms) · CUDA Graph alone **+44.3%** throughput — Qwen3-0.6B on RTX 3050

**[NavierStokes-3D](https://github.com/GuoHaojiee/NavierStokes-3D)** — 3D incompressible pseudo-spectral solver<br/>
**8.46×** over MPI heFFTe with NVLink cuFFTXt (512³, 4×GPU) · all variants verified to ~1e-14

### Experience

**Huawei 2012 Labs** — Ascend operator modeling & optimization<br/>
Cycle-level performance modeling (TileSim, DSL→IR→Cost Model→Evaluation) · `re_glu` kernel time **−38.1%**, end-to-end **+7%** · modeling accuracy to **91%**

**Russian Academy of Sciences** — HPC scientific computing<br/>
Ported a Navier-Stokes solver across 7 distributed FFT backends on MSU-270 (CPU/GPU, MPI/OpenMP/CUDA)

### Stack

`C++` `CUDA` `Triton` `Python` `MPI` `OpenMP` `PyTorch` `Nsight Compute` `Ascend CANN` `CMake` `Linux`

<div align="center">

<img height="140" src="https://github-readme-stats.vercel.app/api?username=GuoHaojiee&show_icons=true&hide_title=true&hide=issues&theme=github_dark&hide_border=true&icon_color=76B900" alt="stats" />

</div>
