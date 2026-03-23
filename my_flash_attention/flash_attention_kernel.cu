#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <cutlass/cutlass.h>
#include <cutlass/gemm/device/gemm.h>
#include <cutlass/half.h>

#include <cfloat>
#include <cmath>
#include <stdexcept>
#include <vector>

#define CHECK_CUDA(x) TORCH_CHECK((x).is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")
#define CHECK_DTYPE_HALF(x) TORCH_CHECK((x).scalar_type() == at::kHalf, #x " must be torch.float16")

namespace {

constexpr int kSoftmaxThreads = 256;

__global__ void rowwise_softmax_kernel(const float* scores, float* probs, int rows, int cols) {
  int row = blockIdx.x;
  if (row >= rows) return;

  const float* in_row = scores + static_cast<long long>(row) * cols;
  float* out_row = probs + static_cast<long long>(row) * cols;

  float thread_max = -FLT_MAX;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    thread_max = fmaxf(thread_max, in_row[col]);
  }

  __shared__ float shared_max;
  __shared__ float shared_sum;
  __shared__ float shared_buffer[kSoftmaxThreads];

  shared_buffer[threadIdx.x] = thread_max;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_buffer[threadIdx.x] = fmaxf(shared_buffer[threadIdx.x], shared_buffer[threadIdx.x + stride]);
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    shared_max = shared_buffer[0];
  }
  __syncthreads();

  float thread_sum = 0.f;
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    float v = __expf(in_row[col] - shared_max);
    out_row[col] = v;
    thread_sum += v;
  }

  shared_buffer[threadIdx.x] = thread_sum;
  __syncthreads();

  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      shared_buffer[threadIdx.x] += shared_buffer[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    shared_sum = shared_buffer[0];
  }
  __syncthreads();

  float inv_sum = 1.f / (shared_sum + 1e-6f);
  for (int col = threadIdx.x; col < cols; col += blockDim.x) {
    out_row[col] *= inv_sum;
  }
}

inline void launch_rowwise_softmax(const at::Tensor& scores, at::Tensor& probs) {
  int rows = static_cast<int>(scores.size(0) * scores.size(1));
  int cols = static_cast<int>(scores.size(2));
  rowwise_softmax_kernel<<<rows, kSoftmaxThreads, 0, at::cuda::getDefaultCUDAStream()>>>(
      scores.data_ptr<float>(),
      probs.data_ptr<float>(),
      rows,
      cols);
}

}  // namespace

at::Tensor cutlass_attention_forward(at::Tensor q, at::Tensor k, at::Tensor v) {
  CHECK_CUDA(q);
  CHECK_CUDA(k);
  CHECK_CUDA(v);
  CHECK_CONTIGUOUS(q);
  CHECK_CONTIGUOUS(k);
  CHECK_CONTIGUOUS(v);
  CHECK_DTYPE_HALF(q);
  CHECK_DTYPE_HALF(k);
  CHECK_DTYPE_HALF(v);

  TORCH_CHECK(q.dim() == 4, "q must be [B, H, S, D]");
  TORCH_CHECK(k.dim() == 4, "k must be [B, H, S, D]");
  TORCH_CHECK(v.dim() == 4, "v must be [B, H, S, D]");
  TORCH_CHECK(q.sizes() == k.sizes(), "q and k must have same shape");
  TORCH_CHECK(q.sizes() == v.sizes(), "q and v must have same shape");

  int64_t B = q.size(0);
  int64_t H = q.size(1);
  int64_t S = q.size(2);
  int64_t D = q.size(3);
  int64_t BH = B * H;

  TORCH_CHECK(S > 0 && D > 0, "S and D must be > 0");
  TORCH_CHECK(S <= INT_MAX && D <= INT_MAX && BH <= INT_MAX, "shape too large for this demo kernel");

  auto scores = torch::zeros({BH, S, S}, q.options().dtype(torch::kFloat32));
  auto probs = torch::zeros({BH, S, S}, q.options().dtype(torch::kFloat32));
  auto out = torch::zeros_like(q);

  using GemmQK = cutlass::gemm::device::Gemm<
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::ColumnMajor,
      float,
      cutlass::layout::RowMajor,
      float>;

  using GemmPV = cutlass::gemm::device::Gemm<
      float,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      cutlass::half_t,
      cutlass::layout::RowMajor,
      float>;

  GemmQK gemm_qk;
  GemmPV gemm_pv;
  float alpha_qk = 1.0f / std::sqrt(static_cast<float>(D));
  float beta = 0.0f;

  const auto* q_ptr = reinterpret_cast<const cutlass::half_t*>(q.data_ptr<at::Half>());
  const auto* k_ptr = reinterpret_cast<const cutlass::half_t*>(k.data_ptr<at::Half>());
  auto* scores_ptr = scores.data_ptr<float>();

  int M = static_cast<int>(S);
  int N = static_cast<int>(S);
  int K = static_cast<int>(D);

  for (int64_t i = 0; i < BH; ++i) {
    const cutlass::half_t* qi = q_ptr + i * S * D;
    const cutlass::half_t* ki = k_ptr + i * S * D;
    float* si = scores_ptr + i * S * S;

    cutlass::Status status = gemm_qk({
        {M, N, K},
        {qi, K},
        {ki, K},
        {si, N},
        {si, N},
        {alpha_qk, beta}});
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS QK^T GEMM failed");
  }

  launch_rowwise_softmax(scores, probs);
  C10_CUDA_KERNEL_LAUNCH_CHECK();

  const auto* probs_ptr = probs.data_ptr<float>();
  const auto* v_ptr = reinterpret_cast<const cutlass::half_t*>(v.data_ptr<at::Half>());
  auto* out_ptr = reinterpret_cast<cutlass::half_t*>(out.data_ptr<at::Half>());

  M = static_cast<int>(S);
  N = static_cast<int>(D);
  K = static_cast<int>(S);
  float alpha_pv = 1.0f;

  for (int64_t i = 0; i < BH; ++i) {
    const float* pi = probs_ptr + i * S * S;
    const cutlass::half_t* vi = v_ptr + i * S * D;
    cutlass::half_t* oi = out_ptr + i * S * D;

    cutlass::Status status = gemm_pv({
        {M, N, K},
        {pi, K},
        {vi, N},
        {oi, N},
        {oi, N},
        {alpha_pv, beta}});
    TORCH_CHECK(status == cutlass::Status::kSuccess, "CUTLASS P*V GEMM failed");
  }

  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &cutlass_attention_forward, "CUTLASS Attention forward (CUDA)");
}
