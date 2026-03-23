import os
from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def resolve_cutlass_include():
    env_path = os.environ.get("CUTLASS_PATH")
    if env_path:
        candidate = Path(env_path).expanduser() / "include"
        if candidate.exists():
            return str(candidate)

    default_path = Path("~/cuda/cutlass/include").expanduser()
    if default_path.exists():
        return str(default_path)

    raise RuntimeError(
        "找不到 CUTLASS 头文件。请设置 CUTLASS_PATH，例如: export CUTLASS_PATH=~/cuda/cutlass"
    )


cutlass_include = resolve_cutlass_include()

setup(
    name="my_flash_attention",
    version="0.1.0",
    packages=["my_flash_attention"],
    package_dir={"my_flash_attention": "."},
    ext_modules=[
        CUDAExtension(
            name="my_flash_attention._C",
            sources=["flash_attention_kernel.cu"],
            include_dirs=[cutlass_include],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": ["-O3", "-std=c++17", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
