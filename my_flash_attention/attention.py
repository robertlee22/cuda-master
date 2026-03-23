import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

_EXT = None


def _resolve_cutlass_include() -> str:
    cutlass_path = os.environ.get("CUTLASS_PATH", str(Path("~/cuda/cutlass").expanduser()))
    include_dir = Path(cutlass_path).expanduser() / "include"
    if not include_dir.exists():
        raise FileNotFoundError(
            f"未找到 CUTLASS include 目录: {include_dir}. "
            "请设置环境变量 CUTLASS_PATH，例如 export CUTLASS_PATH=~/cuda/cutlass"
        )
    return str(include_dir)


def _load_ext():
    global _EXT
    if _EXT is not None:
        return _EXT

    # 优先复用 setup.py 构建出来的扩展
    try:
        from my_flash_attention import _C as prebuilt_ext  # type: ignore

        _EXT = prebuilt_ext
        return _EXT
    except ImportError:
        pass

    this_dir = Path(__file__).resolve().parent
    include_dir = _resolve_cutlass_include()

    _EXT = load(
        name="my_flash_attention_ext",
        sources=[str(this_dir / "flash_attention_kernel.cu")],
        extra_include_paths=[include_dir],
        extra_cflags=["-O3", "-std=c++17"],
        extra_cuda_cflags=["-O3", "-std=c++17", "--use_fast_math"],
        verbose=True,
    )
    return _EXT


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if not torch.cuda.is_available():
        raise RuntimeError("需要 CUDA 环境")
    if q.dtype != torch.float16 or k.dtype != torch.float16 or v.dtype != torch.float16:
        raise TypeError("q/k/v 必须是 float16")
    if q.device.type != "cuda" or k.device.type != "cuda" or v.device.type != "cuda":
        raise TypeError("q/k/v 必须在 CUDA 上")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v shape 必须一致，预期 [B, H, S, D]")
    if q.dim() != 4:
        raise ValueError("q/k/v 必须是 4 维 [B, H, S, D]")

    ext = _load_ext()
    return ext.forward(q.contiguous(), k.contiguous(), v.contiguous())
