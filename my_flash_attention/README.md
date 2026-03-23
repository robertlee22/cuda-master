# my_flash_attention

这个目录提供一个最小可运行版本：

- 使用 CUTLASS 做两次 GEMM：`Q @ K^T` 和 `softmax(QK^T) @ V`
- 使用 CUDA kernel 实现 row-wise softmax
- 使用 Python（PyTorch）调用

## 目录结构

- `flash_attention_kernel.cu`：CUDA/CUTLASS 内核 + PyBind 导出
- `attention.py`：Python 封装，按需 JIT 编译扩展
- `setup.py`：可选的安装构建脚本
- `demo.py`：最小调用示例

## 环境要求

- CUDA + nvcc
- PyTorch（CUDA 版本）
- CUTLASS（建议放在 `~/cuda/cutlass`，或设置 `CUTLASS_PATH`）

## 朴素 PyTorch vs CUTLASS 吞吐量对比

```bash
export CUTLASS_PATH=~/cuda/cutlass
python -m my_flash_attention.benchmark_attention --batch 4 --heads 8 --seq 256 --dim 64 --warmup 20 --iters 100
```

输出包含：

- `Latency(ms)`：平均每次前向耗时
- `TFLOPS`：按 attention 两次 GEMM 的近似 FLOPs 估算
- `Token/s`：每秒处理 token 数（`B*S/latency`）
- `speedup`：`naive / cutlass`

## 测试结果（示例）

以下为本地一次实际运行结果（用于确认编译和调用链路正常）：

```bash
cd /root/cuda
export CUTLASS_PATH=/root/cuda/cutlass
export OMP_NUM_THREADS=1
python -m my_flash_attention.benchmark_attention --batch 1 --heads 2 --seq 64 --dim 64 --warmup 1 --iters 3
```

关键输出：

```text
=== Attention Benchmark ===
shape: B=1, H=2, S=64, D=64, dtype=float16
warmup=1, iters=3
max_abs_diff (naive vs cutlass): 0.000610

Impl           Latency(ms)      TFLOPS       Token/s
----------------------------------------------------
Naive                0.103       0.020      618693.1
CUTLASS              0.072       0.029      885351.1
----------------------------------------------------
speedup (naive / cutlass): 1.431x
```

说明：

- 首次运行会触发 JIT 编译，耗时会明显更长；再次运行会复用缓存。
- `TORCH_CUDA_ARCH_LIST` 未设置时会出现告警，但不影响功能正确性。

