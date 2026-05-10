# Blackwell Microbenchmarks

A collection of microbenchmarks for NVIDIA Blackwell (SM 100) GPUs, covering
memory throughput, latency, tensor core (UMMA) performance, and HBM-resident
elementwise throughput.

https://newsletter.semianalysis.com/p/dissecting-nvidia-blackwell-tensor

<img width="1456" height="1231" alt="image" src="https://github.com/user-attachments/assets/104eabab-7c77-403f-b669-3402cc7a4b86" />

## Benchmarks

| Path | Purpose |
|---|---|
| `ldgsts_throughput/` | LDGSTS HBM throughput |
| `tma2d_throughput/` | TMA 2D HBM throughput |
| `ldgsts_latency/` | LDGSTS latency |
| `tma2d_latency/` | TMA 2D latency |
| `umma_throughput/` | UMMA tensor-core throughput |
| `umma_latency/` | UMMA tensor-core latency |
| `elementwise_throughput/` | fp32 HBM-resident activation/elementwise throughput |

## Acknowledgements

Compute for this project is generously sponsored by **Nebius** and **Verda**.

<p align="center">
  <img src="assets/nebius_logo.jpeg" alt="Nebius" height="100">
  &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
  <img src="assets/verda_cl_logo.jpeg" alt="Verda" height="100">
</p>
