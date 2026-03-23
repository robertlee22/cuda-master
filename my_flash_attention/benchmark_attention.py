import argparse
import time

import torch

try:
    from my_flash_attention import attention as cutlass_attention
except ImportError:
    # 兼容从包目录直接执行: python benchmark_attention.py
    from attention import attention as cutlass_attention


def naive_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    d = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (d ** -0.5)
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v)


def benchmark_one(fn, q, k, v, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        _ = fn(q, k, v)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        _ = fn(q, k, v)
    torch.cuda.synchronize()
    end = time.perf_counter()

    return (end - start) / iters


def estimate_tflops(b: int, h: int, s: int, d: int, sec_per_iter: float) -> float:
    # 近似 FLOPs:
    # QK^T: 2 * B * H * S * S * D
    # P@V : 2 * B * H * S * S * D
    # 合计: 4 * B * H * S^2 * D
    flops = 4.0 * b * h * s * s * d
    return flops / sec_per_iter / 1e12


def token_throughput(b: int, s: int, sec_per_iter: float) -> float:
    return (b * s) / sec_per_iter


def main():
    parser = argparse.ArgumentParser(description="Benchmark naive PyTorch attention vs CUTLASS attention")
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--seq", type=int, default=256)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=100)
    parser.add_argument("--dtype", type=str, default="float16", choices=["float16"])
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 设备来运行 benchmark")

    dtype = torch.float16

    device = "cuda"
    b, h, s, d = args.batch, args.heads, args.seq, args.dim
    torch.manual_seed(0)

    q = torch.randn(b, h, s, d, device=device, dtype=dtype)
    k = torch.randn(b, h, s, d, device=device, dtype=dtype)
    v = torch.randn(b, h, s, d, device=device, dtype=dtype)

    with torch.no_grad():
        out_naive = naive_attention(q, k, v)
        out_cutlass = cutlass_attention(q, k, v)
        max_diff = (out_naive - out_cutlass).abs().max().item()

    naive_sec = benchmark_one(naive_attention, q, k, v, args.warmup, args.iters)
    cutlass_sec = benchmark_one(cutlass_attention, q, k, v, args.warmup, args.iters)

    naive_tflops = estimate_tflops(b, h, s, d, naive_sec)
    cutlass_tflops = estimate_tflops(b, h, s, d, cutlass_sec)
    naive_tok_s = token_throughput(b, s, naive_sec)
    cutlass_tok_s = token_throughput(b, s, cutlass_sec)

    speedup = naive_sec / cutlass_sec if cutlass_sec > 0 else float("inf")

    print("=== Attention Benchmark ===")
    print(f"shape: B={b}, H={h}, S={s}, D={d}, dtype={args.dtype}")
    print(f"warmup={args.warmup}, iters={args.iters}")
    print(f"max_abs_diff (naive vs cutlass): {max_diff:.6f}")
    print("")
    print(f"{'Impl':<12}{'Latency(ms)':>14}{'TFLOPS':>12}{'Token/s':>14}")
    print("-" * 52)
    print(f"{'Naive':<12}{naive_sec * 1e3:>14.3f}{naive_tflops:>12.3f}{naive_tok_s:>14.1f}")
    print(f"{'CUTLASS':<12}{cutlass_sec * 1e3:>14.3f}{cutlass_tflops:>12.3f}{cutlass_tok_s:>14.1f}")
    print("-" * 52)
    print(f"speedup (naive / cutlass): {speedup:.3f}x")


if __name__ == "__main__":
    main()
