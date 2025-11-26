from typing import Tuple

import math
import random
import torch
import ctypes
import triton
import triton.language as tl
from triton import Config



@triton.jit
def act_quant_kernel(x_ptr, y_ptr, s_ptr, BLOCK_SIZE: tl.constexpr):
    """
    Quantizes the input tensor `x_ptr` and stores the result in `y_ptr` and the scaling factor in `s_ptr`.

    Args:
        x_ptr (triton.Pointer): Pointer to the input tensor.
        y_ptr (triton.Pointer): Pointer to the output tensor where quantized values will be stored.
        s_ptr (triton.Pointer): Pointer to the output tensor where scaling factors will be stored.
        BLOCK_SIZE (tl.constexpr): The size of the block to be processed by each program instance.

    Returns:
        None
    """
    pid = tl.program_id(axis=0)
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    x = tl.load(x_ptr + offs).to(tl.float32)
    s = tl.max(tl.abs(x)) / 448.
    y = x / s
    y = y.to(y_ptr.dtype.element_ty)
    tl.store(y_ptr + offs, y)
    tl.store(s_ptr + pid, s)

# 把 张量 x 进行 量化
def act_quant(x: torch.Tensor, block_size: int = 128) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantizes the input tensor `x` using block-wise quantization.

    Args:
        x (torch.Tensor): The input tensor to be quantized. Must be contiguous and its last dimension size must be divisible by `block_size`.
        block_size (int, optional): The size of the blocks to be used for quantization. Default is 128.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - The quantized tensor with dtype `torch.float8_e4m3fn`.
            - A tensor of scaling factors with dtype `torch.float32`.
    """
    assert x.is_contiguous(), 'Input tensor must be contiguous'
    assert x.size(-1) % block_size == 0, f'Last dimension size must be divisible by block_size (block_size={block_size})'
    # 创建两个张量：一个形状与x 一致且dtype为FP8的张量y；一个是专门储存scale因子的张量s，依旧是每128维储存一个scale因子
    # （按照上述代码来看，s的张量形状为(2, 3, 7168 // 128)=(2, 3, 56)，数据类型为FP32）。
    y = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    s = x.new_empty(*x.size()[:-1], x.size(-1) // block_size, dtype=torch.float32)
    # 之后的两行代码，便涉及到了Triton Kernel的调度计算。Triton是一个专门用于优化GPU计算的编程框架。内核调度（Kernel Scheduling）指的是
    # 如何将计算任务分配给GPU上的计算单元（SMs-Streaming Multiprocessors）。内核（kernel）指的是要求在 GPU 上并行执行的那段代码（也可以说是计算任务）。
    # 众所周知，GPU并不像CPU那样串行计算，而是同时运行多个计算块（blocks），每个 block又包含多个线程，它们并行执行任务，以提高计算效率。
    # grid 决定多少个计算block被调度到 GPU 上。这里调用了triton.cdiv(x.numel(), meta['BLOCK_SIZE']) 来计算需要多少个 blocks。
    # x.numel()是输入x张量里元素的个数，在本例中为2×3×7168个。 triton.cdiv()负责作向上取整的除法，以确保整个张量都能被块覆盖。
    # meta['BLOCK_SIZE']=128 ，于是可知grid为(2×3×7168/128, )=(336, ) ，即最终会划分为336块blocks进行并行计算。
    grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']), )
    act_quant_kernel[grid](x, y, s, BLOCK_SIZE=block_size)
    return y, s


@triton.jit
def weight_dequant_kernel(x_ptr, s_ptr, y_ptr, M, N, BLOCK_SIZE: tl.constexpr):
    """
    Dequantizes weights using the provided scaling factors and stores the result.

    Args:
        x_ptr (tl.pointer): Pointer to the quantized weights.
        s_ptr (tl.pointer): Pointer to the scaling factors.
        y_ptr (tl.pointer): Pointer to the output buffer for dequantized weights.
        M (int): Number of rows in the weight matrix.
        N (int): Number of columns in the weight matrix.
        BLOCK_SIZE (tl.constexpr): Size of the block for tiling.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    n = tl.cdiv(N, BLOCK_SIZE)
    offs_m = pid_m * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs_n = pid_n * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    offs = offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    x = tl.load(x_ptr + offs, mask=mask).to(tl.float32)
    s = tl.load(s_ptr + pid_m * n + pid_n)
    y = x * s
    tl.store(y_ptr + offs, y, mask=mask)


def weight_dequant(x: torch.Tensor, s: torch.Tensor, block_size: int = 128) -> torch.Tensor:
    """
    Dequantizes the given weight tensor using the provided scale tensor.

    Args:
        x (torch.Tensor): The quantized weight tensor of shape (M, N).
        s (torch.Tensor): The scale tensor of shape (M, N).
        block_size (int, optional): The block size to use for dequantization. Defaults to 128.

    Returns:
        torch.Tensor: The dequantized weight tensor of the same shape as `x`.

    Raises:
        AssertionError: If `x` or `s` are not contiguous or if their dimensions are not 2.
    """
    assert x.is_contiguous() and s.is_contiguous(), 'Input tensors must be contiguous'
    assert x.dim() == 2 and s.dim() == 2, 'Input tensors must have 2 dimensions'
    M, N = x.size()
    y = torch.empty_like(x, dtype=torch.get_default_dtype())
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']), triton.cdiv(N, meta['BLOCK_SIZE']))
    weight_dequant_kernel[grid](x, s, y, M, N, BLOCK_SIZE=block_size)
    return y


fp8_gemm_configs = [
    Config({'BLOCK_SIZE_M': block_m, 'BLOCK_SIZE_N': block_n, 'BLOCK_SIZE_K': 128}, num_stages=num_stages, num_warps=8)
    for block_m in [16, 32, 64] for block_n in [32, 64, 128] for num_stages in [3, 4, 5, 6]
]

@triton.autotune(configs=fp8_gemm_configs, key=['N', 'K'])
@triton.jit
def fp8_gemm_kernel(a_ptr, b_ptr, c_ptr,
                    a_s_ptr, b_s_ptr,
                    M, N: tl.constexpr, K: tl.constexpr,
                    BLOCK_SIZE_M: tl.constexpr,
                    BLOCK_SIZE_N: tl.constexpr,
                    BLOCK_SIZE_K: tl.constexpr):
    """
    Performs a matrix multiplication operation on FP8 matrices with scaling factors.

    Args:
        a_ptr (tl.tensor): Pointer to the first input matrix A.
        b_ptr (tl.tensor): Pointer to the second input matrix B.
        c_ptr (tl.tensor): Pointer to the output matrix C.
        a_s_ptr (tl.tensor): Pointer to the scaling factors for matrix A.
        b_s_ptr (tl.tensor): Pointer to the scaling factors for matrix B.
        M (int): Number of rows in matrix A and C.
        N (tl.constexpr): Number of columns in matrix B and C.
        K (tl.constexpr): Number of columns in matrix A and rows in matrix B.
        BLOCK_SIZE_M (tl.constexpr): Block size for the M dimension.
        BLOCK_SIZE_N (tl.constexpr): Block size for the N dimension.
        BLOCK_SIZE_K (tl.constexpr): Block size for the K dimension.

    Returns:
        None
    """
    pid_m = tl.program_id(axis=0)
    pid_n = tl.program_id(axis=1)
    k = tl.cdiv(K, BLOCK_SIZE_K)
    offs_m = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_n = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + offs_m[:, None] * K + offs_k[None, :]
    b_ptrs = b_ptr + offs_n[None, :] * K + offs_k[:, None]
    a_s_ptrs = a_s_ptr + offs_m * k
    b_s_ptrs = b_s_ptr + (offs_n // BLOCK_SIZE_K) * k

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for i in range(k):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - i * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - i * BLOCK_SIZE_K, other=0.0)
        a_s = tl.load(a_s_ptrs)
        b_s = tl.load(b_s_ptrs)
        accumulator += tl.dot(a, b) * a_s[:, None] * b_s[None, :]
        a_ptrs += BLOCK_SIZE_K
        b_ptrs += BLOCK_SIZE_K
        a_s_ptrs += 1
        b_s_ptrs += 1
    c = accumulator.to(c_ptr.dtype.element_ty)
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + offs_m[:, None] * N + offs_n[None, :]
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, c, mask=mask)

# FP8通用矩阵乘法
def fp8_gemm(a: torch.Tensor, a_s: torch.Tensor, b: torch.Tensor, b_s: torch.Tensor):
    """
    Perform a matrix multiplication using FP8 precision.

    Args:
        a (torch.Tensor): The first input matrix, must be contiguous.
        a_s (torch.Tensor): The scaling factor for the first input matrix, must be contiguous.
        b (torch.Tensor): The second input matrix, must be contiguous.
        b_s (torch.Tensor): The scaling factor for the second input matrix, must be contiguous.

    Returns:
        torch.Tensor: The result of the matrix multiplication.
    """
    assert a.is_contiguous() and b.is_contiguous(), 'Input tensors must be contiguous'
    assert a_s.is_contiguous() and b_s.is_contiguous(), 'Scaling factor tensors must be contiguous'
    K = a.size(-1)
    M = a.numel() // K
    N = b.size(0)
    c = a.new_empty(*a.size()[:-1], N, dtype=torch.get_default_dtype())
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']), triton.cdiv(N, META['BLOCK_SIZE_N']))
    fp8_gemm_kernel[grid](a, b, c, a_s, b_s, M, N, K)
    return c

# 加载 CUDA 动态库
lib = ctypes.CDLL("./libint64gemm.so")

# 定义参数类型
lib.int64_64_bmm_broadcast_launcher.argtypes = [
    ctypes.c_void_p,  # A
    ctypes.c_void_p,  # B
    ctypes.c_void_p,  # C
    ctypes.c_void_p,  # R
    ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]

lib.int64_32_bmm_broadcast_launcher.argtypes = [
    ctypes.c_void_p,  # A
    ctypes.c_void_p,  # B
    ctypes.c_void_p,  # C
    ctypes.c_void_p,  # R
    ctypes.c_longlong, ctypes.c_longlong, ctypes.c_longlong,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]

lib.complex_int64_mul.argtypes = [
    ctypes.c_void_p,  # A
    ctypes.c_void_p,  # B
    ctypes.c_void_p,  # C
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]

lib.rms_norm_32.argtypes = [
    ctypes.c_void_p,  # A
    ctypes.c_void_p,  # W
    ctypes.c_void_p,  # rms
    ctypes.c_void_p,  # C
    ctypes.c_int, ctypes.c_int
]

lib.rms_norm_64.argtypes = [
    ctypes.c_void_p,  # A
    ctypes.c_void_p,  # W
    ctypes.c_void_p,  # rms
    ctypes.c_void_p,  # C
    ctypes.c_int, ctypes.c_int
]

lib.einsum_bshd_hdc_bshc.argtypes = [
    ctypes.c_void_p,  # A
    ctypes.c_void_p,  # B
    ctypes.c_void_p,  # C
    ctypes.c_longlong,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]

lib.einsum_bshc_btc_bsht.argtypes = [
    ctypes.c_void_p,  # A
    ctypes.c_void_p,  # B
    ctypes.c_void_p,  # C
    ctypes.c_longlong,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]

lib.einsum_bsht_btc_bshc.argtypes = [
    ctypes.c_void_p,  # A
    ctypes.c_void_p,  # B
    ctypes.c_void_p,  # C
    ctypes.c_longlong,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]

lib.einsum_bshc_hdc_bshd.argtypes = [
    ctypes.c_void_p,  # A
    ctypes.c_void_p,  # B
    ctypes.c_void_p,  # C
    ctypes.c_longlong,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]

lib.softmax_q21.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]

lib.softmax_q19.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int
]

lib.softmax_init_q21.argtypes = [
    ctypes.c_void_p
]

lib.softmax_init_q19.argtypes = [
    ctypes.c_void_p
]

lib.silu_q25.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

lib.sigmoid_q25.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

lib.silu_init_q25.argtypes = [
    ctypes.c_void_p
]

lib.silu_q23.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

lib.sigmoid_q23.argtypes = [
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_void_p,
    ctypes.c_int, ctypes.c_int, ctypes.c_int
]

lib.silu_init_q23.argtypes = [
    ctypes.c_void_p
]


def int64_bmm_broadcast(A: torch.Tensor, B: torch.Tensor, a_rescale, b_rescale, c_rescale) -> tuple[torch.Tensor]:
    """
    int64 批量矩阵乘法: (B, M, K) x (N, K) -> (B, M, N)
    """
    global lib

    assert A.dtype == torch.int64
    # and B.dtype == torch.int64
    assert A.is_cuda and B.is_cuda
    Bdim, M, K = A.shape
    N, K2 = B.shape
    assert K2 == K

    C = torch.empty((Bdim, M, N), dtype=torch.int64, device="cuda")
    R = torch.empty((Bdim, M, N), dtype=torch.int64, device="cuda")

    if B.dtype == torch.int64:
        lib.int64_64_bmm_broadcast_launcher(
            A.data_ptr(), B.data_ptr(), C.data_ptr(), R.data_ptr(),
            a_rescale, b_rescale, c_rescale,
            Bdim, M, K, N
        )
    elif B.dtype == torch.int32:
        lib.int64_32_bmm_broadcast_launcher(
            A.data_ptr(), B.data_ptr(), C.data_ptr(), R.data_ptr(),
            a_rescale, b_rescale, c_rescale,
            Bdim, M, K, N
        )
    else:
        print(f'Unsupported B type: {B.dtype}')
    return (C, R)

def complex_int64_mul_broadcast(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    """
    int64 复数逐元素乘法
    """
    global lib

    # print(f'A type: {A.dtype}, B type: {B.dtype}')
    assert A.dtype == torch.int64 and B.dtype == torch.int64
    assert A.is_cuda and B.is_cuda

    batch = A.shape[0]
    seqLen = A.shape[1]
    head = A.shape[2]
    headDim = A.shape[3]

    C = torch.zeros(A.shape, dtype=torch.int64, device=A.device)

    lib.complex_int64_mul(
            A.data_ptr(), B.data_ptr(), C.data_ptr(),
            # high_rescale, row_rescale,
            batch, seqLen, head, headDim)

    return C

def einsum_bshd_hdc_bshc(A: torch.Tensor, B: torch.Tensor, rescale) -> torch.Tensor:
    global lib

    assert A.shape[2] == B.shape[0] and A.shape[3] == B.shape[1]
    assert A.is_cuda and B.is_cuda

    Batch = A.shape[0]
    S = A.shape[1]
    H = A.shape[2]
    D = A.shape[3]
    Cp = B.shape[2]

    C = torch.zeros([Batch, S, H, Cp], dtype=torch.int64, device=A.device)

    lib.einsum_bshd_hdc_bshc(A.data_ptr(), B.data_ptr(), C.data_ptr(),
        # (1 << rescale), Batch, S, H, D, Cp)
        rescale, Batch, S, H, D, Cp)

    return C

def einsum_bshc_btc_bsht(A: torch.Tensor, B: torch.Tensor, rescale) -> torch.Tensor:
    global lib

    Bsz = A.shape[0]
    S = A.shape[1]
    H = A.shape[2]
    Cdim = A.shape[3]
    T = B.shape[1]

    assert Bsz == B.shape[0] and Cdim == B.shape[2]
    assert A.is_cuda and B.is_cuda

    C = torch.zeros([Bsz, S, H, T], dtype=torch.int64, device=A.device)

    lib.einsum_bshc_btc_bsht(A.data_ptr(), B.data_ptr(), C.data_ptr(),
        # (1 << rescale), Bsz, S, H, T, Cdim)
        rescale, Bsz, S, H, T, Cdim)

    return C

def einsum_bsht_btc_bshc(A: torch.Tensor, B: torch.Tensor, rescale) -> torch.Tensor:
    global lib

    Bsz = A.shape[0]
    S = A.shape[1]
    H = A.shape[2]
    T = A.shape[3]
    Cdim = B.shape[2]

    assert Bsz == B.shape[0] and T == B.shape[1]
    assert A.is_cuda and B.is_cuda

    C = torch.zeros([Bsz, S, H, Cdim], dtype=torch.int64, device=A.device)

    lib.einsum_bsht_btc_bshc(A.data_ptr(), B.data_ptr(), C.data_ptr(),
        # (1 << rescale), Bsz, S, H, T, Cdim)
        rescale, Bsz, S, H, T, Cdim)

    return C

def einsum_bshc_hdc_bshd(A: torch.Tensor, B: torch.Tensor, rescale) -> torch.Tensor:
    global lib

    Bsz = A.shape[0]
    S = A.shape[1]
    H = A.shape[2]
    D = B.shape[1]
    Cdim = A.shape[3]

    assert H == B.shape[0] and Cdim == B.shape[2]
    assert A.is_cuda and B.is_cuda

    C = torch.zeros([Bsz, S, H, D], dtype=torch.int64, device=A.device)

    lib.einsum_bshc_hdc_bshd(A.data_ptr(), B.data_ptr(), C.data_ptr(),
        # (1 << rescale), Bsz, S, H, D, Cdim)
        rescale, Bsz, S, H, D, Cdim)

    return C

def int64_RMS0(A: torch.Tensor, eps: int, dim: int) -> torch.Tensor:
    assert A.dtype == torch.int64
    assert A.ndim == 1

    N = A.shape[0]

    # 初始化累加器
    acc = eps

    for i in range(0, N):
        a = A[i].item()
        acc += a * a

    acc = acc // dim

    res1 = math.isqrt(acc)

    return res1

# x 的 scale 为 2 ** 31，范围为 0 - 2^31
# weight的scale 为 2 ** 21, 范围为 2^5 - 2^20
# rms 的 scale 为 2 ** 31
# 返回的结果 scale 为 2 ** 21，31 + 21 - 31 = 21
@triton.jit
def int64_rms_norm_kernel(
    A_ptr, W_ptr, C_ptr, RMS_ptr,
    N,
    batch_stride_a, batch_stride_c,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid_m = tl.program_id(0)

    for i in range(0, N):
        a_ptrs = A_ptr + pid_m * batch_stride_a + i
        w_ptrs = W_ptr + i
        rms_ptrs = RMS_ptr + pid_m
        a = tl.load(a_ptrs, mask=None)
        w = tl.load(w_ptrs, mask=None)
        rms = tl.load(rms_ptrs, mask=None)

        res = a * w // rms

        prod = a * w
        tl.device_assert(prod > -2 ** 62 and prod < 2 ** 62, "Integer overflow risk!!!")

        c_ptrs = C_ptr + pid_m * batch_stride_c + i
        tl.store(c_ptrs, res, mask=None)

rms = torch.empty((500, ), dtype=torch.int64, device='cpu')
rms_gpu = torch.empty((500, ), dtype=torch.int64, device='cuda')

def RMS_Norm_int64(A: torch.Tensor, W: torch.Tensor, eps, dim) -> torch.Tensor:
    global lib
    global rms
    global rms_gpu

    assert A.dtype == torch.int64
    assert A.is_cuda and W.is_cuda
    assert A.ndim == 2

    M, N = A.shape

    for i in range(M):
        rms[i] = int64_RMS0(A[i], eps, dim)

    rms_gpu.copy_(rms)
    C = torch.empty((M, N), dtype=torch.int64, device=A.device)

    if W.dtype == torch.int32:
        lib.rms_norm_32(A.data_ptr(), W.data_ptr(), rms_gpu.data_ptr(), C.data_ptr(), M, N)
    else:
        lib.rms_norm_64(A.data_ptr(), W.data_ptr(), rms_gpu.data_ptr(), C.data_ptr(), M, N)

    return (C, rms)


def saveTensor(fileName, t):
    with open(fileName, "w", encoding="utf-8") as f:
        # for row in tensor:
        #     vs = [str(v.item()) for v in row]
        #     ss = ' '.join(vs) + '\n'
        #     f.write(ss)
        t = t.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        t = t.contiguous()
        with open(fileName, "wb") as f:
            # .numpy() -> bytes（C-order）
            f.write(t.numpy().tobytes(order="C"))

EXP2_FRAC_LUT_Q21 = None
# LOG_TABLE_SIZE = 10
LOG_TABLE_SIZE = 8

def softmax_init_q21():
    global lib
    global EXP2_FRAC_LUT_Q21

    EXP2_FRAC_LUT0 = torch.zeros((2 ** LOG_TABLE_SIZE, ), dtype=torch.int64, device="cpu")
    lib.softmax_init_q21(EXP2_FRAC_LUT0.data_ptr())
    # print(EXP2_FRAC_LUT0[619])

    EXP2_FRAC_LUT_Q21 = EXP2_FRAC_LUT0.cuda()

EXP2_FRAC_LUT_Q19 = None
def softmax_init_q19():
    global lib
    global EXP2_FRAC_LUT_Q19

    EXP2_FRAC_LUT0 = torch.zeros((2 ** LOG_TABLE_SIZE, ), dtype=torch.int64, device="cpu")
    lib.softmax_init_q19(EXP2_FRAC_LUT0.data_ptr())
    # print(EXP2_FRAC_LUT0[619])

    EXP2_FRAC_LUT_Q19 = EXP2_FRAC_LUT0.cuda()
    # saveTensor(f'zkdata/softmax_q19_table.bin', EXP2_FRAC_LUT0.cpu())



def softmax_q21(R: torch.Tensor, C: torch.Tensor):
    global lib
    global EXP2_FRAC_LUT_Q21

    assert R.is_cuda and C.is_cuda

    # print(EXP2_FRAC_LUT_Q21)
    Bsz = R.shape[0]
    S = R.shape[1]
    H = R.shape[2]
    T = R.shape[3]
    lib.softmax_q21(R.data_ptr(), C.data_ptr(), EXP2_FRAC_LUT_Q21.data_ptr(), Bsz, S, H, T)

def softmax_q19(R: torch.Tensor, C: torch.Tensor):
    global lib
    global EXP2_FRAC_LUT_Q19

    assert R.is_cuda and C.is_cuda

    # print(EXP2_FRAC_LUT_Q19)
    Bsz = R.shape[0]
    S = R.shape[1]
    H = R.shape[2]
    T = R.shape[3]
    lib.softmax_q19(R.data_ptr(), C.data_ptr(), EXP2_FRAC_LUT_Q19.data_ptr(), Bsz, S, H, T)


# start of silu_q25 ---------------------------------
EXP2_FRAC_LUT_Q25 = None

def silu_init_q25():
    global lib
    global EXP2_FRAC_LUT_Q25

    EXP2_FRAC_LUT0 = torch.zeros((2 ** LOG_TABLE_SIZE, ), dtype=torch.int64, device="cpu")
    lib.silu_init_q25(EXP2_FRAC_LUT0.data_ptr())
    # print(EXP2_FRAC_LUT0[619])

    EXP2_FRAC_LUT_Q25 = EXP2_FRAC_LUT0.cuda()

def silu_q25(R: torch.Tensor, C: torch.Tensor):
    global lib
    global EXP2_FRAC_LUT_Q25

    # print(EXP2_FRAC_LUT_Q25)
    Bsz = R.shape[0]
    S = R.shape[1]
    Dim = R.shape[2]
    lib.silu_q25(R.data_ptr(), C.data_ptr(), EXP2_FRAC_LUT_Q25.data_ptr(), Bsz, S, Dim)

def sigmoid_q25(R: torch.Tensor, C: torch.Tensor):
    global lib
    global EXP2_FRAC_LUT_Q25

    Bsz = R.shape[0]
    S = R.shape[1]
    Dim = R.shape[2]
    lib.sigmoid_q25(R.data_ptr(), C.data_ptr(), EXP2_FRAC_LUT_Q25.data_ptr(), Bsz, S, Dim)
# end of silu_q25 ---------------------------------

# start of silu_q23 ---------------------------------
EXP2_FRAC_LUT_Q23 = None

def silu_init_q23():
    global lib
    global EXP2_FRAC_LUT_Q23

    EXP2_FRAC_LUT0 = torch.zeros((2 ** LOG_TABLE_SIZE, ), dtype=torch.int64, device="cpu")
    lib.silu_init_q23(EXP2_FRAC_LUT0.data_ptr())
    # print(EXP2_FRAC_LUT0[619])

    EXP2_FRAC_LUT_Q23 = EXP2_FRAC_LUT0.cuda()

    # saveTensor(f'zkdata/silu_q23_table.bin', EXP2_FRAC_LUT0.cpu())

def silu_q23(R: torch.Tensor, C: torch.Tensor):
    global lib
    global EXP2_FRAC_LUT_Q23

    # print(EXP2_FRAC_LUT_Q23)
    Bsz = R.shape[0]
    S = R.shape[1]
    Dim = R.shape[2]
    lib.silu_q23(R.data_ptr(), C.data_ptr(), EXP2_FRAC_LUT_Q23.data_ptr(), Bsz, S, Dim)

def sigmoid_q23(R: torch.Tensor, C: torch.Tensor):
    global lib
    global EXP2_FRAC_LUT_Q23

    Bsz = R.shape[0]
    S = R.shape[1]
    Dim = R.shape[2]
    lib.sigmoid_q23(R.data_ptr(), C.data_ptr(), EXP2_FRAC_LUT_Q23.data_ptr(), Bsz, S, Dim)
# end of silu_q23 ---------------------------------


if __name__ == "__main__":
    softmax_init_q21()

    torch.manual_seed(0)
    device = "cuda"

    Bsz = 1
    S = 1
    H = 2
    T = 10

    A = torch.rand([Bsz, S, H, T], dtype=torch.bfloat16, device=device)
    a = (A.to(torch.float32) * (2 ** 21)).to(torch.int64)
    # a = (A * (2 ** 21)).to(torch.int64)

    print('A: ' + str(A))
    print('a: ' + str(a))

    c = torch.zeros([Bsz, S, H, T], dtype=torch.int64, device=device)

    softmax_q21(a, c)

    r0 = A.softmax(dim=-1, dtype=torch.float32).type_as(A)
    print('r0: ' + str(r0))

    r1 = (c.to(torch.float32) * (2 ** -21)).to(torch.bfloat16)
    print('r1: ' + str(r1))


    R0 = (r0.to(torch.float32) * (2 ** 21)).to(torch.int64)
    # R0 = (r0 * (2 ** 21)).to(torch.int64)
    print('R0: ' + str(R0))

    print('R1: ' + str(c))
