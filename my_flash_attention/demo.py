import torch

try:
    from my_flash_attention import attention
except ImportError:
    # 兼容从包目录直接执行: python demo.py
    from attention import attention


def main():
    torch.manual_seed(0)
    device = "cuda"
    b, h, s, d = 1, 4, 128, 64

    q = torch.randn(b, h, s, d, device=device, dtype=torch.float16)
    k = torch.randn(b, h, s, d, device=device, dtype=torch.float16)
    v = torch.randn(b, h, s, d, device=device, dtype=torch.float16)

    out = attention(q, k, v)
    print("output shape:", tuple(out.shape))
    print("output dtype:", out.dtype)
    print("output[0,0,0,:8]:", out[0, 0, 0, :8])


if __name__ == "__main__":
    main()
