import os
import math
import datetime
from dataclasses import dataclass
from typing import Tuple, Optional, Literal

import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from safetensors.torch import load_model

from kernel import act_quant, weight_dequant, fp8_gemm, int64_bmm_broadcast, \
    complex_int64_mul_broadcast, einsum_bshd_hdc_bshc, einsum_bshc_btc_bsht, softmax_init_q21, softmax_q21, einsum_bsht_btc_bshc, einsum_bshc_hdc_bshd, \
    silu_init_q25, silu_q25, sigmoid_q25, softmax_init_q19, softmax_q19, silu_init_q23, silu_q23, sigmoid_q23, RMS_Norm_int64


world_size = 1
rank = 0
block_size = 128
gemm_impl: Literal["bf16", "fp8"] = "bf16"
attn_impl: Literal["naive", "absorb"] = "absorb"

snark = True
zkDataDir = '../zkdata'

@dataclass
class ModelArgs:
    """
    Data class for defining model arguments and hyperparameters.

    Attributes:
        max_batch_size (int): Maximum batch size.
        max_seq_len (int): Maximum sequence length.
        dtype (Literal["bf16", "fp8"]): Data type for computations.
        vocab_size (int): Vocabulary size.
        dim (int): Model dimension.
        inter_dim (int): Intermediate dimension for MLP layers.
        moe_inter_dim (int): Intermediate dimension for MoE layers.
        n_layers (int): Number of transformer layers.
        n_dense_layers (int): Number of dense layers in the model.
        n_heads (int): Number of attention heads.
        n_routed_experts (int): Number of routed experts for MoE layers.
        n_shared_experts (int): Number of shared experts for MoE layers.
        n_activated_experts (int): Number of activated experts in MoE layers.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited groups for MoE routing.
        score_func (Literal["softmax", "sigmoid"]): Scoring function for MoE routing.
        route_scale (float): Scaling factor for routing scores.
        q_lora_rank (int): LoRA rank for query projections.
        kv_lora_rank (int): LoRA rank for key-value projections.
        qk_nope_head_dim (int): Dimension for query-key projections without positional embeddings.
        qk_rope_head_dim (int): Dimension for query-key projections with rotary embeddings.
        v_head_dim (int): Dimension for value projections.
        original_seq_len (int): Original sequence length.
        rope_theta (float): Base for rotary positional encoding.
        rope_factor (float): Scaling factor for extended sequence lengths.
        beta_fast (int): Fast beta correction factor.
        beta_slow (int): Slow beta correction factor.
        mscale (float): Scaling factor for extended attention.
    """
    max_batch_size: int = 8
    max_seq_len: int = 4096 * 4
    dtype: Literal["bf16", "fp8"] = "bf16"
    vocab_size: int = 102400
    dim: int = 2048
    inter_dim: int = 10944
    moe_inter_dim: int = 1408
    n_layers: int = 27
    n_dense_layers: int = 1
    n_heads: int = 16
    # moe
    n_routed_experts: int = 64
    n_shared_experts: int = 2
    n_activated_experts: int = 6
    n_expert_groups: int = 1
    n_limited_groups: int = 1
    score_func: Literal["softmax", "sigmoid"] = "softmax"
    route_scale: float = 1.
    # mla
    q_lora_rank: int = 0
    kv_lora_rank: int = 512
    qk_nope_head_dim: int = 128
    qk_rope_head_dim: int = 64
    v_head_dim: int = 128
    # yarn
    original_seq_len: int = 4096
    rope_theta: float = 10000.0
    rope_factor: float = 40
    beta_fast: int = 32
    beta_slow: int = 1
    mscale: float = 1.

def saveTensor(fileName, t):
    with open(fileName, "w", encoding="utf-8") as f:
        t = t.detach()
        if t.device.type != "cpu":
            t = t.cpu()
        t = t.contiguous()
        with open(fileName, "wb") as f:
            # .numpy() -> bytes（C-order）
            f.write(t.numpy().tobytes(order="C"))

class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        # weight 的 shape: [129280, 7168]
        self.register_buffer("weight", torch.empty(self.part_vocab_size, self.dim, dtype=torch.int64))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
        # print('aaab ' + str(self.weight[0][0].type()))
        if world_size > 1:
            # 找出 x 中 的值不在 [vocab_start_idx, vocab_end_idx) 范围内的下标
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            # x 中所有的值都减去 vocab_start_idx
            x = x - self.vocab_start_idx
            # 之前找出的标记为 mask 下标的值设置为0
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)

        # print(f'ParallelEmbedding x: {x}', flush=True)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies a linear transformation to the incoming data: y = xA^T + b.
    This function supports specialized implementations based on quantization
    and tensor formats.

    Args:
        x (torch.Tensor): The input tensor.
        weight (torch.Tensor): The weight tensor. It may be quantized and 
            requires dequantization for certain cases.
        bias (Optional[torch.Tensor]): The bias tensor to be added. Default is None.

    Returns:
        torch.Tensor: The result of the linear transformation, which may involve 
        quantization-aware computations depending on the input parameters.

    Notes:
        - If `weight` is quantized (e.g., `element_size() == 1`), a dequantized version 
          is used for computation.
        - If `gemm_impl == "bf16"`, dequantization and a `bf16` GEMM operation are applied.
        - For other cases, the function applies quantization to `x` and uses `fp8_gemm` for computation.
    """

    element_size = weight.element_size()
    typ = weight.type()
    # print(f'linear weight element_size {element_size}, type: {typ}', flush=True)
    if weight.element_size() > 1:
        # print('linear weight.element_size > 1, element_size=' + str(weight.element_size()), flush=True)
        return F.linear(x, weight, bias)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)
        return F.linear(x, weight, bias)
    else:
        # print('linear act_quant', flush=True)
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return y

def linear_int(x: torch.Tensor, weight: torch.Tensor, x_rescale, weight_rescale, res_rescale, bias: Optional[torch.Tensor] = None) -> tuple[torch.Tensor]:
    if weight.element_size() > 1:
        (q, r) = int64_bmm_broadcast(x, weight, x_rescale, weight_rescale, res_rescale)
        return (q, r)
    elif gemm_impl == "bf16":
        weight = weight_dequant(weight, weight.scale)

        return (F.linear(x, weight, bias), torch.tensor(0, dtype=torch.int64))
    else:
        print('linear act_quant', flush=True)
        x, scale = act_quant(x, block_size)
        y = fp8_gemm(x, scale, weight, weight.scale)
        if bias is not None:
            y += bias
        return (y, torch.tensor(0, dtype=torch.int64))

class Linear_int(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.int64

    def __init__(self, layer_id, in_features: int, out_features: int, x_rescale, weight_rescale, res_rescale, dtype, bias: bool = False):
        super().__init__()
        self.layer_id = layer_id
        self.in_features = in_features
        self.out_features = out_features

        self.x_rescale = x_rescale
        self.weight_rescale = weight_rescale
        self.res_rescale = res_rescale

        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=dtype))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor]:
        q, r = linear_int(x, self.weight, self.x_rescale, self.weight_rescale, self.res_rescale, self.bias)
        return q, r

class Linear_rescale_int(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.int64

    def __init__(self, layer_id, in_features: int, out_features: int, x_rescale, weight_rescale, dtype, bias: bool = False):
        super().__init__()
        self.layer_id = layer_id
        self.in_features = in_features
        self.out_features = out_features

        self.x_rescale = x_rescale
        self.weight_rescale = weight_rescale

        self.register_buffer("weight", torch.empty(out_features, in_features, dtype=dtype))
        self.register_buffer("scale", torch.tensor(0, dtype=torch.int32))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rescale = self.scale.item()
        y, _r = linear_int(x, self.weight, self.x_rescale, self.weight_rescale, rescale, self.bias)
        return y

class Linear(nn.Module):
    """
    Custom linear layer with support for quantized weights and optional bias.

    Args:
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    dtype = torch.bfloat16

    def __init__(self, layer_id, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.layer_id = layer_id
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype))

        # print('Linear.weight.element_size: ' + str(self.weight.element_size()))

        # nn.Parameter.element_size() 返回的是 每个元素在内存中占用的字节数
        # torch.float32 -> 4 字节
        # torch.float64 -> 8 字节
        # torch.int64 -> 8 字节
        # torch.bfloat16 -> 2 字节
        # torch.float8_e4m3fn -> 1 字节
        if self.weight.element_size() == 1:
            scale_out_features = (out_features + block_size - 1) // block_size
            scale_in_features = (in_features + block_size - 1) // block_size

            self.weight.scale = self.scale = nn.Parameter(torch.empty(scale_out_features, scale_in_features, dtype=torch.float32))
        else:
            self.register_parameter("scale", None)

        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor after linear computation.
        """
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    """
    Linear layer with column parallelism, splitting output features across distributed processes.

    Args:
        in_features (int): Number of input features.
        out_features (int): Total number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, layer_id, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(layer_id, in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for column parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with column-parallel computation.
        """
        y = linear(x, self.weight, self.bias)
        return y

class ColumnParallelLinear_int(Linear_int):
    def __init__(self, layer_id, in_features: int, out_features: int, x_rescale, weight_rescale, res_rescale, dtype, bias: bool = False):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(layer_id, in_features, self.part_out_features, x_rescale, weight_rescale, res_rescale, dtype, bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _r = linear_int(x, self.weight, self.x_rescale, self.weight_rescale, self.res_rescale, self.bias)
        return y
    
class ColumnParallelLinear_rescale_int(Linear_int):
    def __init__(self, layer_id, in_features: int, out_features: int, x_rescale, weight_rescale, dtype, bias: bool = False):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(layer_id, in_features, self.part_out_features, x_rescale, weight_rescale, 1, dtype, bias)
        self.register_buffer("scale", torch.tensor(0, dtype=torch.int32))
        # self.res_rescale = self.scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rescale = self.scale.item()
        y, _r = linear_int(x, self.weight, self.x_rescale, self.weight_rescale, rescale, self.bias)
        return y


class RowParallelLinear(Linear):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, layer_id, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(layer_id, self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y

class RowParallelLinear_rescale_int(Linear_int):
    """
    Linear layer with row parallelism, splitting input features across distributed processes.

    Args:
        in_features (int): Total number of input features.
        out_features (int): Number of output features.
        bias (bool): Whether to include a bias term. Defaults to False.
        dtype (optional): Data type for the layer. Defaults to `torch.bfloat16`.
    """
    def __init__(self, layer_id, in_features: int, out_features: int, x_rescale, weight_rescale, res_rescale, dtype, bias: bool = False):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(layer_id, self.part_in_features, out_features, x_rescale, weight_rescale, res_rescale, dtype, bias)
        self.register_buffer("scale", torch.tensor(0, dtype=torch.int32))
        self.res_rescale = self.scale # useless

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for row parallel linear layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Transformed tensor with row-parallel computation.
        """
        # rescale = 2 ** self.scale.item()
        rescale = self.scale.item()
        # print(f'RowParallelLinear_rescale_int forward scale: {self.scale} ' + str(rescale), flush=True)
        y, _ = linear_int(x, self.weight, self.x_rescale, self.weight_rescale, rescale, self.bias)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization (RMSNorm).

    Args:
        dim (int): Dimension of the input tensor.
        eps (float): Epsilon value for numerical stability. Defaults to 1e-6.
    """
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor):
        """
        Forward pass for RMSNorm.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input.
        """
        return F.rms_norm(x, (self.dim,), self.weight, self.eps)

class RMSNorm_int(nn.Module):
    def __init__(self, dim: int, dtype, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.register_buffer(
            "weight",
            torch.ones(dim, dtype=dtype))

    def forward(self, x: torch.Tensor):
        # x 的 scale 为 2 ** 31
        # weight的scale 为 2 ** 15, 范围为 2^7 - 2^14
        # rms 的 scale 为 2 ** 28
        # 返回的结果 scale 为 2 ** 16,因为中间计算的时候 除以了 (1 << 15)，44 + 15 - 28 - 15 = 16
        (c, rms) = RMS_Norm_int64(x[0], self.weight, 1, self.dim)

        return (c[None, :], rms)


def precompute_freqs_cis(args: ModelArgs) -> torch.Tensor:
    """
    Precomputes frequency-based complex exponential values for rotary positional embeddings.

    Args:
        args (ModelArgs): Model arguments containing positional embedding parameters.

    Returns:
        torch.Tensor: Precomputed complex exponential values for positional embeddings.
    """
    # dim = 64
    dim = args.qk_rope_head_dim
    seqlen = args.max_seq_len
    beta_fast = args.beta_fast
    beta_slow = args.beta_slow
    base = args.rope_theta
    factor = args.rope_factor

    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        """
        Computes the correction dimension for a given number of rotations in the rotary positional embedding.

        Args:
            num_rotations (float): Number of rotations to compute the correction for.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            float: The correction dimension based on the input parameters.
        """
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        """
        Computes the range of correction dimensions for rotary positional embeddings.

        Args:
            low_rot (float): Lower bound for the number of rotations.
            high_rot (float): Upper bound for the number of rotations.
            dim (int): Dimensionality of the embedding space.
            base (float): Base value for the exponential computation.
            max_seq_len (int): Maximum sequence length.

        Returns:
            Tuple[int, int]: The range of correction dimensions (low, high), clamped to valid indices.
        """
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim-1)

    def linear_ramp_factor(min, max, dim):
        """
        Computes a linear ramp function used to smooth values between a minimum and maximum range.

        Args:
            min (float): Minimum value for the ramp function.
            max (float): Maximum value for the ramp function.
            dim (int): Dimensionality of the ramp tensor.

        Returns:
            torch.Tensor: A tensor of shape (dim,) with values linearly interpolated between 0 and 1,
                clamped to the range [0, 1].
        """
        if min == max:
            max += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
        ramp_func = torch.clamp(linear_func, 0, 1)
        return ramp_func

    # torch.arange(0, dim, 2, dtype=torch.float32) 的作用是： 生成从 0 开始、步长为 2、到 dim 之前（不含 dim）的一维张量，数据类型为 float32
    # 1/10000^(2k/d_model)
    # freqs shape: 一维向量，长度为 dim /2
    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    # original_seq_len=4096
    if seqlen > args.original_seq_len:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, args.original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    # torch.outer 的作用是计算两个向量的 外积 (outer product)，比如：
    # t = torch.tensor([1, 2, 3])       # shape = [3]
    # freqs = torch.tensor([10, 20])    # shape = [2]
    # out = torch.outer(t, freqs)
    # tensor([[10, 20],
    #         [20, 40],
    #         [30, 60]])
    # freqs shape为 [seqlen, dim/2]
    freqs = torch.outer(t, freqs)
    # torch.polar(abs, angle) 的作用: 把 极坐标 (r, θ) 转换成 复数 (x + iy) 的函数
    # freqs_cis_0 shape为 [seqlen, dim/2]
    freqs_cis_0 = torch.polar(torch.ones_like(freqs), freqs)

    # return freqs_cis_0

    # 复数转换成实数, freqs_cis_1 shape为 [seqlen, dim]
    freqs_cis_1 = torch.view_as_real(freqs_cis_0)

    # freqs_cis = torch.empty_like(freqs_cis_1, dtype=torch.int64, device='cuda')

    # cols 为 2 * freqs_cis_1.shape[1] 是因为 复数的实部 和 虚部
    # rescale 参数为 19 = 42 - 23, ex 部分加 +19，总的rescale为 2^42
    freqs_cis = (freqs_cis_1 * (2 ** 42)).round().to(torch.int64)

    freqs_cis_abs = freqs_cis.abs()
    min1 = freqs_cis_abs.min()
    max1 = freqs_cis_abs.max()
    print(f'freqs_cis min {min1}, max: {max1}', flush=True)

    # print(f'freqs_cis: {freqs_cis}')
    # freqs_cis  的 rescale 为 2^42
    return freqs_cis

#  x(q_pe) 的维度 [batch, seqLen, 128, 64]
def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
    """
    Applies rotary positional embeddings to the input tensor.

    Args:
        x (torch.Tensor): Input tensor with positional embeddings to be applied.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for positional embeddings.

    Returns:
        torch.Tensor: Tensor with rotary embeddings applied.
    """

    # if x.dtype == torch.int64:
    # x 的维度 变为 [batch, seqLen, 128, 32, 2]
    ### important!!! 调用 so lib库之前，必须确保内存连续
    x = x.contiguous().view(*x.shape[:-1], -1, 2)
    # freqs_cis 的维度为 [1, seqLen, 1, 32, 2]
    freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-2), 2)
    # freqs_cis = freqs_cis.view(1, x.size(1), 1, x.size(-1))
    # 4194304 = 1 << (64 - 42), 42是 rescale, int64 * int64 结果的高 64位 乘以 4194304
    # 4398046511104 = 1 << 42
    # print(x)
    # print(f'x shape: {x.shape}, freqs_cis shape: {freqs_cis.shape}')
    # y = complex_int64_mul_broadcast(x, freqs_cis, 4194304, 4398046511104)
    y = complex_int64_mul_broadcast(x, freqs_cis)
    y2 = y.flatten(3)
    return y2


def getBF16PrintStr(ele):
    v = int(ele.cpu().view(torch.uint16).item())
    ex = v >> 7 & 0xFF
    r = '(1+' + str(v & 0x7F) + '/128)'
    rraw = v & 0x7F

    if v & 0x8000:
        vstr = '-' + r + '*2^' + str(ex - 127)
    else:
        vstr = r + '*2^' + str(ex - 127)
    return vstr

class MLA(nn.Module):
    """
    Multi-Headed Attention Layer (MLA).

    Attributes:
        dim (int): Dimensionality of the input features.
        n_heads (int): Number of attention heads.
        n_local_heads (int): Number of local attention heads for distributed systems.
        q_lora_rank (int): Rank for low-rank query projection.
        kv_lora_rank (int): Rank for low-rank key/value projection.
        qk_nope_head_dim (int): Dimensionality of non-positional query/key projections.
        qk_rope_head_dim (int): Dimensionality of rotary-positional query/key projections.
        qk_head_dim (int): Total dimensionality of query/key projections.
        v_head_dim (int): Dimensionality of value projections.
        softmax_scale (float): Scaling factor for softmax in attention computation.
    """
    def __init__(self, layer_id, args: ModelArgs):
        super().__init__()

        # RowParallelLinear和ColumnParallelLinear是将Linear层按照行和列划分为多个子线性层并分配到各个设备上，每个设备维护一个子线性层，
        # 如线性层的shape为[in_features, out_features]，RowParallelLinear的shape为[in_features/world_size, out_features]，
        # ColumnParallelLinear的shape为[in_features，out_features/world_size]，world_size是设备数

        self.layer_id = layer_id

        # 7168
        self.dim = args.dim
        # 128
        self.n_heads = args.n_heads
        # 当前进程跑的header数目
        self.n_local_heads = args.n_heads // world_size
        # query向下投影矩阵维度，默认为0表示不压缩，实际使用过程为 1536
        self.q_lora_rank = args.q_lora_rank
        # key和value向下投影矩阵维度，实际使用过程为 512;
        self.kv_lora_rank = args.kv_lora_rank
        # query/key不包含位置信息的隐藏层维度, 实际使用过程为 128
        self.qk_nope_head_dim = args.qk_nope_head_dim
        # query/key包含rope位置信息的隐藏层维度, 实际使用过程为 64
        self.qk_rope_head_dim = args.qk_rope_head_dim

        # 192
        self.qk_head_dim = args.qk_nope_head_dim + args.qk_rope_head_dim
        # value隐藏层维度, 实际使用过程为 128
        self.v_head_dim = args.v_head_dim

        # query向下投影矩阵维度，默认为0表示不压缩，实际使用过程为 1536
        if self.q_lora_rank == 0:
            self.wq = ColumnParallelLinear(layer_id, self.dim, self.n_heads * self.qk_head_dim)
        else:
            # query向下投影矩阵, shape [7168, 1536], Float8_e4m3fnTensor
            self.wq_a = Linear_int(layer_id, self.dim, self.q_lora_rank, 1, 1, 30, torch.int32)
            self.q_norm = RMSNorm_int(self.q_lora_rank, torch.int32)
            # query向上投影矩阵的列并行线性层, shape [1536, 24576(128 * 192)], Float8_e4m3fnTensor
            # self.wq_b = ColumnParallelLinear_int(layer_id, self.q_lora_rank, self.n_heads * self.qk_head_dim, 1, 1, (1 << 30), torch.int32)
            self.wq_b1 = ColumnParallelLinear_int(layer_id, self.q_lora_rank, self.n_heads * args.qk_nope_head_dim, 1, 1, 30, torch.int32)
            self.wq_b2 = ColumnParallelLinear_int(layer_id, self.q_lora_rank, self.n_heads * args.qk_rope_head_dim, 1, 1, 30, torch.int32)

        # key和value的向下投影矩阵, shape [576, 7168], Float8_e4m3fnTensor, kv_lora_rank=512, qk_rope_head_dim=64
        # self.wkv_a = Linear_int(layer_id, self.dim, self.kv_lora_rank + self.qk_rope_head_dim, 1, 1, (1 << 29), torch.int32)
        self.wkv_a1 = Linear_int(layer_id, self.dim, self.kv_lora_rank, 1, 1, 29, torch.int32)
        self.wkv_a2 = Linear_int(layer_id, self.dim, self.qk_rope_head_dim, 1, 1, 29, torch.int32)
        # self.kv_norm = RMSNorm(self.kv_lora_rank)
        self.kv_norm = RMSNorm_int(self.kv_lora_rank, torch.int32)
        # key和value向上投影矩阵的列并行线性层, shape [32768, 512], Float8_e4m3fnTensor
        # kv_lora_rank=512, n_heads = 128, qk_nope_head_dim = 128, v_head_dim = 128
        # self.wkv_b = ColumnParallelLinear(layer_id, self.kv_lora_rank, self.n_heads * (self.qk_nope_head_dim + self.v_head_dim))
        self.wkv_b_1 = ColumnParallelLinear_rescale_int(layer_id, self.kv_lora_rank, self.n_heads * self.qk_nope_head_dim, 1, 1, torch.int32)
        self.wkv_b_2 = ColumnParallelLinear_rescale_int(layer_id, self.kv_lora_rank, self.n_heads * self.v_head_dim, 1, 1, torch.int32)

        # 输出投影行并行线性层, shape [7168, 16384], Float8_e4m3fnTensor
        self.wo = RowParallelLinear_rescale_int(layer_id, self.n_heads * self.v_head_dim, self.dim, 1, 1, 1, torch.int32)
        # softmax缩放系数, qk_head_dim = 192
        # self.softmax_scale = self.qk_head_dim ** -0.5
        # # max_seq_len = 4096 * 4, original_seq_len = 4096
        # if args.max_seq_len > args.original_seq_len:
        #     # mscale = 1.0, rope_factor = 40, math.log = ln 自然对数
        #     mscale = 0.1 * args.mscale * math.log(args.rope_factor) + 1.0
        #     self.softmax_scale = self.softmax_scale * mscale * mscale
        self.softmax_scale1 = 94
        self.softmax_scale2 = 695

        if attn_impl == "naive":
            self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.qk_head_dim), persistent=False)
            self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_heads, self.v_head_dim), persistent=False)
        else:
            # 缓存key和value向下投影表示
            # self.register_buffer("kv_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.kv_lora_rank), persistent=False)
            # self.register_buffer("kv_cache", torch.zeros(1, args.max_seq_len, self.kv_lora_rank), persistent=False)
            self.register_buffer("kv_cache", torch.zeros(1, args.max_seq_len, self.kv_lora_rank, dtype=torch.int64), persistent=False)
            # 缓存key执行rope操作后的表示
            # self.register_buffer("pe_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.qk_rope_head_dim), persistent=False)
            # self.register_buffer("pe_cache", torch.zeros(1, args.max_seq_len, self.qk_rope_head_dim), persistent=False)
            self.register_buffer("pe_cache", torch.zeros(1, args.max_seq_len, self.qk_rope_head_dim, dtype=torch.int64), persistent=False)

    # x shape [1, seqLen, 7168], x 的resacle 为 2^21
    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]):
        """
        Forward pass for the Multi-Headed Attention Layer (MLA).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, dim).
            start_pos (int): Starting position in the sequence for caching.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor with the same shape as the input.
        """

        # 从输入获取batch size和序列长度seqlen，并根据输入序列的起始位置计算输入序列的结束位置end_pos=start_pos+seqlen；
        bsz, seqlen, _ = x.size()
        end_pos = start_pos + seqlen

        # 获取query的投影表示：如果对query投影矩阵进行压缩(即q_lora_rank不为0)，则将输入乘以query的向下投影矩阵wq_a，然后经过归一化层q_norm，
        # 再乘以向上投影矩阵wq_b，否则直接乘以原始投影矩阵wq；将其维度调整为[batchsize, n_local_threads, qk_head_dim]；
        if self.q_lora_rank == 0:
            q = self.wq(x)
        else:
            # query向下投影矩阵, shape [7168, 1536], Float8_e4m3fnTensor
            # x(也就是 attn_normed) 的 scale 为 2^21, wq_a weight 的scale 为 2^30, q_down 的 scale 为 2^21
            q_down, q_down_rem = self.wq_a(x)
            # q_down = self.wq_a(x)

            if snark:
                dirStr = f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}'
                os.makedirs(dirStr, exist_ok=True)
                saveTensor(f'{dirStr}/wq_a_x.bin', x.cpu())
                saveTensor(f'{dirStr}/wq_a_w.bin', self.wq_a.weight.view(torch.uint32).cpu())
                saveTensor(f'{dirStr}/wq_a_y.bin', q_down.cpu())
                saveTensor(f'{dirStr}/q_norm_r.bin', q_down_rem.cpu())
                # q_down = (q_down.detach().to(torch.float32) * (2 ** -23)).to(torch.bfloat16)

            # q_norm 的 rescale 为 2^19
            (q_normed, rms) = self.q_norm(q_down)

            if snark:
                dirStr = f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}'
                os.makedirs(dirStr, exist_ok=True)
                saveTensor(f'{dirStr}/q_norm_x.bin', q_down.cpu())
                saveTensor(f'{dirStr}/q_norm_weight.bin', self.q_norm.weight.view(torch.uint32).cpu())
                saveTensor(f'{dirStr}/q_norm_rms.bin', rms.cpu())
                saveTensor(f'{dirStr}/q_norm_y.bin', q_normed.cpu())

            # q 的 rescale 为 2^19
            # q = self.wq_b(q_normed)
            q_nope = self.wq_b1(q_normed)
            q_pe = self.wq_b2(q_normed)

        # 在pytorch中view函数的作用为重构张量的维度
        # q = q.view(bsz, seqlen, self.n_local_heads, self.qk_head_dim)
        q_nope = q_nope.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim)
        q_pe = q_pe.view(bsz, seqlen, self.n_local_heads, self.qk_rope_head_dim)

        # 将query的投影表示按照最后一个维度拆分，前面qk_nope_head_dim维(128)作为query不包含位置信息的表示q_nope，后面qk_rope_head_dim维(64)添加rope位置信息
        # (调用apply_rotary_emb函数，参考秀才经商：DeepSeek源码解析之RoPE)作为query包含位置信息的表示q_pe(即公式39)；
        # q_nope 的维度[batch, seqLen, 128, 128], q_pe 的维度 [batch, seqLen, 128, 64]
        # q_nope, q_pe 的 rescale 为 2^19
        # q_nope, q_pe = torch.split(q, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        # freqs_cis  的 rescale 为 2^42, 计算之后 q_pe 的 rescale 为 2^19

        if snark:
            saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/q_pe_x.bin', q_pe.cpu())
            saveTensor(f'{zkDataDir}/freqs_cis.bin', freqs_cis.cpu())

        q_pe = apply_rotary_emb(q_pe, freqs_cis)

        if snark:
            saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/q_pe_y.bin', self.q_norm.weight.view(torch.uint32).cpu())

        # 获取key和value的联合表示kv(即公式41中的)和包含位置信息的key表示k_pe(即公式43中的)：输入乘以向下投影矩阵wkv_a后，按照最后一个维度拆分，
        # 前面kv_lora_rank维作为key和value的联合表示，后面qk_rope_head_dim维添加rope位置信息(调用apply_rotary_emb)后得到包含rope位置信息的key表示；

        # x 的resacle 为 2^21, kv shape [batch, seqLen, 512], kv 的resacle 为 2^21
        kv, kv_rem = self.wkv_a1(x)

        if snark:
            dirStr = f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}'
            os.makedirs(dirStr, exist_ok=True)
            saveTensor(f'{dirStr}/wkv_a1_x.bin', x.cpu())
            saveTensor(f'{dirStr}/wkv_a1_w.bin', self.wkv_a1.weight.view(torch.uint32).cpu())
            saveTensor(f'{dirStr}/wkv_a1_y.bin', kv.cpu())
            saveTensor(f'{dirStr}/wkv_a1_r.bin', kv_rem.cpu())

        k_pe, _ = self.wkv_a2(x)

        # print(f'k_pe 1 shape: {k_pe.shape}', flush=True)
        # unsqueeze()用于增加一个维度, k_pe.unsqueeze(2) 把 k_pe reshape 成 [batch, seqLen, 1, dim]
        # # kv, k_pe 的resacle 为 2^21
        k_pe = apply_rotary_emb(k_pe.unsqueeze(2), freqs_cis)
        # print(f'k_pe 2 shape: {k_pe.shape}', flush=True)

        if attn_impl == "naive":
            q = torch.cat([q_nope, q_pe], dim=-1)
            kv = self.wkv_b(self.kv_norm(kv))
            kv = kv.view(bsz, seqlen, self.n_local_heads, self.qk_nope_head_dim + self.v_head_dim)
            k_nope, v = torch.split(kv, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
            k = torch.cat([k_nope, k_pe.expand(-1, -1, self.n_local_heads, -1)], dim=-1)
            self.k_cache[:bsz, start_pos:end_pos] = k
            self.v_cache[:bsz, start_pos:end_pos] = v
            scores = torch.einsum("bshd,bthd->bsht", q, self.k_cache[:bsz, :end_pos]) * self.softmax_scale
        else:
            # 计算query和key的注意力：
            # query中不包含位置信息的q_nope(乘以了key的向上投影矩阵后)与缓存kv_cache中的key表示求内积；
            # query中包含位置信息的q_pe与缓存pe_cache中的key表示求内积；
            # 两者相加后乘以softmax缩放系数softmax_scale

            # q_nope 的维度[batch, seqLen, 128, 128], wkv_b_1 shape: [128, 128, 512]
            # q_nope rescale 2^19, wkv_b_1 rescale 2 ** 32
            # q_nope = torch.einsum("bshd,hdc->bshc", q_nope, wkv_b_1)
            # 调用 einsum_bshd_hdc_bshc 之后, q_nope维度 [batch, seqLen, 128, 512]
            wkv_b_1 = self.wkv_b_1.weight.view(self.n_local_heads, -1, self.kv_lora_rank)
            q_nope = einsum_bshd_hdc_bshc(q_nope.contiguous(), wkv_b_1.contiguous(), self.wkv_b_1.scale.item())
            # print('q_nope type: ' + str(q_nope.type()))
            # print('q_nope shape: ' + str(q_nope.shape))

            # kv_normed 的 rescale 为 2^23
            (kv_normed, rms) = self.kv_norm(kv)

            # kv_cache 的 rescale 为 2^23, shape [batch, seqLen, 512],
            self.kv_cache[:bsz, start_pos:end_pos] = kv_normed
            # self.kv_cache[:bsz, start_pos:end_pos] = kv2

            # kv = (kv.detach().to(torch.float32) * (2 ** -23)).to(torch.bfloat16)
            # pe_cache 的 rescale 为 2^21
            self.pe_cache[:bsz, start_pos:end_pos] = k_pe.squeeze(2)

            # q_nope rescale: 2^19, kv_cache rescale: 2^23
            # q_nope 的维度 [batch, seqLen, 128, 512], kv_cache 维度 (batch, args.max_seq_len, 512)
            # score1 = torch.einsum("bshc,btc->bsht", q_nope, self.kv_cache[:bsz, :end_pos])
            kv_cache1 = self.kv_cache[:bsz, :end_pos]
            # score1 = einsum_bshc_btc_bsht(q_nope.contiguous(), kv_cache1.contiguous(), 25)
            # score1 的 rescale 为 2^19
            score1 = einsum_bshc_btc_bsht(q_nope.contiguous(), kv_cache1.contiguous(), 23)
            # print(f'kv_cache1 type: {kv_cache1.type()}, shape: {kv_cache1.shape}', flush=True)
            # score1 = (score1.detach().to(torch.float32) * (2 ** -21)).to(torch.bfloat16)

            # score2 = torch.einsum("bshr,btr->bsht", q_pe, self.pe_cache[:bsz, :end_pos])
            pe_cache1 = self.pe_cache[:bsz, :end_pos]
            # score2 = einsum_bshc_btc_bsht(q_pe.contiguous(), pe_cache1.contiguous(), 23)
            # q_pe 的 rescale 为 2^19, scores2 的rescale 为  2^19
            score2 = einsum_bshc_btc_bsht(q_pe.contiguous(), pe_cache1.contiguous(), 21)
            # score2 = (score2.detach().to(torch.float32) * (2 ** -21)).to(torch.bfloat16)

            # scores = (score1 + score2) * self.softmax_scale
            # scores  的 rescale 为 2 ** 19
            scores = (score1 + score2) * self.softmax_scale1 // self.softmax_scale2
            # scores = torch.round(((score1 + score2) * self.softmax_scale1).to(torch.float32) / self.softmax_scale2).to(torch.int64)


        # mask 在 unsqueeze(1) 之后的 shape 为 [seqLen, 1, senLen], scores 的shape 为 [batch, seqLen, heads , t]
        if mask is not None:
            # print('mask type: ' + str(mask.type()))
            # print('mask shape: ' + str(mask.shape))
            scores += mask.unsqueeze(1)
        # query和key的内积按照最后一个维度计算softmax值；
        # scores = scores.softmax(dim=-1, dtype=torch.float32).type_as(x)
        scores_new = torch.empty_like(scores, dtype=torch.int64, device='cuda')
        # scores 和  scores_new 的 rescale 为 2 ** 19, shape: [bsz, seqLen, headCount, seqLen]

        # # softmax_q19 会破坏 scores 的原始数据，先拷贝一份数据
        if snark:
            saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/scores_softmax_x.bin', scores.contiguous().cpu())

        softmax_q19(scores.contiguous(), scores_new)

        if snark:
            saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/scores_softmax_y.bin', scores_new.cpu())

        if attn_impl == "naive":
            x = torch.einsum("bsht,bthd->bshd", scores, self.v_cache[:bsz, :end_pos])
        else:

            kv_cache2 = self.kv_cache[:bsz, :end_pos]
            # kv_cache2 = (kv_cache2.detach().to(torch.float32) * (2 ** -25)).to(torch.bfloat16)

            # x = (x.detach().to(torch.float32) * (2 ** -23)).to(torch.bfloat16)

            # 计算最终输出：
            # 注意力分数乘以kv缓存后，再乘以value的向上投影矩阵wkv_b(实现公式45和46)；
            # 乘以输出投影矩阵wo(公式47)；
            # x = torch.einsum("bsht,btc->bshc", scores_new, kv_cache2)
            # scores_new 的 rescale 为 2^19, kv_cache2 的 rescale 为 2^23, bshc 的 rescale 为 2^19
            # scores_new shape: [1, 8, 128, 8], bshc shape: [1, 8, 128, 512]
            # bshc = einsum_bsht_btc_bshc(scores_new.contiguous(), kv_cache2.contiguous(), 25)
            bshc = einsum_bsht_btc_bshc(scores_new.contiguous(), kv_cache2.contiguous(), 23)

            # # v_head_dim = 128, kv_lora_rank = 512, n_local_heads = 128
            # wkv_b_2 = wkv_b[:, -self.v_head_dim:]
            # # print('wkv_b 2 type: ' + str(wkv_b_2.type()))
            # # print('wkv_b 2 shape: ' + str(wkv_b_2.shape))
            wkv_b_2 = self.wkv_b_2.weight
            wkv_b_2 = wkv_b_2.view(self.n_local_heads, -1, self.kv_lora_rank)

            # wkv_b_2 = (wkv_b_2.detach().to(torch.float32) * (2 ** -self.wkv_b_2.scale.item())).to(torch.bfloat16)

            # x = torch.einsum("bshc,hdc->bshd", x, wkv_b_2)
            # bshc 的 rescale 为 2^19, wkv_b_2 的 rescale 为  self.wkv_b_2.scale
            # x 的 rescale 为 2 ** 19
            # bshc shape: [1, seqLen, 128, 512], wkv_b_2 shape: [128, 128, 512]
            x = einsum_bshc_hdc_bshd(bshc.contiguous(), wkv_b_2.contiguous(), self.wkv_b_2.scale.item())
            # x = (x.detach().to(torch.float32) * (2 ** -21)).to(torch.bfloat16)

        # x 返回的的 shape [1, seqLen, 7168]
        x = self.wo(x.flatten(2))

        return x

class MLP(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, layer_id, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.w1 = ColumnParallelLinear(layer_id, dim, inter_dim)
        self.w2 = RowParallelLinear(layer_id, inter_dim, dim)
        self.w3 = ColumnParallelLinear(layer_id, dim, inter_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class MLP_int(nn.Module):
    """
    Multi-Layer Perceptron (MLP) used as a feed-forward layer.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, layer_id, dim: int, inter_dim: int):
        """
        Initializes the MLP layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        self.layer_id = layer_id
        self.w1 = ColumnParallelLinear_rescale_int(layer_id, dim, inter_dim, 1, 1, torch.int32)
        self.w2 = RowParallelLinear_rescale_int(layer_id, inter_dim, dim, 1, 1, 1, torch.int32)
        self.w3 = ColumnParallelLinear_rescale_int(layer_id, dim, inter_dim, 1, 1, torch.int32)

    # 输入的 x 的rescale 为 2^23, [bsz, seqLen, 7168]
    def forward(self, start_pos: int, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MLP layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after MLP computation.
        """
        # r1 shape: [bsz, seqLen, inter_dim], r1 rescale: 2^23
        r1 = self.w1(x)

        # s1 = F.silu(r1)
        # s1 shape: [bsz, seqLen, inter_dim], s1 rescale: 2^23
        s1 = torch.empty_like(r1, dtype=torch.int64, device='cuda')
        # silu_q25(r1, s1)

        if snark:
            saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/mlp_silu_x.bin', r1.contiguous().cpu())

        silu_q23(r1, s1)

        if snark:
            saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/mlp_silu_y.bin', s1.cpu())

        # r2 rescale: 2^23, shape: [1, seqLen, inter_dim]
        r2 = self.w3(x)

        # 返回的 shape [bsz, seqLen, dim]
        q = self.w2(s1 * r2 // (1 << 23))
        return q


class Gate(nn.Module):
    """
    Gating mechanism for routing inputs in a mixture-of-experts (MoE) model.

    Attributes:
        dim (int): Dimensionality of input features.
        topk (int): Number of top experts activated for each input.
        n_groups (int): Number of groups for routing.
        topk_groups (int): Number of groups to route inputs to.
        score_func (str): Scoring function ('softmax' or 'sigmoid').
        route_scale (float): Scaling factor for routing weights.
        weight (torch.nn.Parameter): Learnable weights for the gate.
        bias (Optional[torch.nn.Parameter]): Optional bias term for the gate.
    """
    def __init__(self, layer_id: int, args: ModelArgs):
        """
        Initializes the Gate module.

        Args:
            args (ModelArgs): Model arguments containing gating parameters.
        """
        super().__init__()

        self.layer_id = layer_id

        self.dim = args.dim
        # n_activated_experts = 8
        self.topk = args.n_activated_experts
        # n_expert_groups = 8
        self.n_groups = args.n_expert_groups
        # n_limited_groups = 4
        self.topk_groups = args.n_limited_groups
        # score_func = 'sigmoid'
        self.score_func = args.score_func
        # route_scale = 2.5
        self.route_scale = args.route_scale
        # n_routed_experts = 256
        # self.weight = nn.Parameter(torch.empty(args.n_routed_experts, args.dim))
        self.register_buffer("weight", torch.empty(args.n_routed_experts, args.dim, dtype=torch.int32))
        self.register_buffer("scale", torch.tensor(0, dtype=torch.int32))
        # self.bias = nn.Parameter(torch.empty(args.n_routed_experts, dtype=torch.int32)) if self.dim == 7168 else None
        if self.dim == 7168:
            self.register_buffer("bias", torch.empty(args.n_routed_experts, dtype=torch.int32))
        else:
            self.bias = None

    # x 的 rescale 为 2^23
    def forward(self, start_pos: int, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for the gating mechanism.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Routing weights and selected expert indices.
        """

        x = x.view(1, -1, self.dim)

        # scores = linear(x, self.weight)
        # self.weight shape: [256, 7168]
        # 当前 scores shape: [1, seqLen, 256]
        # rescale = 2 ** self.scale.item()
        rescale = self.scale.item()

        # scores 的 rescale 为 2^23
        scores, scores_rem = linear_int(x, self.weight, 1, 1, rescale)
        # scores = int64_bmm_with_bias(x, self.weight, bias, 1, 1, self.scale)

        # x shape: [seqLen, 7168]
        x = x.view(-1, self.dim)

        if self.score_func == "softmax":
            scores = scores.softmax(dim=-1, dtype=torch.float32)
        else:
            # scores = scores.sigmoid()
            C = torch.empty_like(scores, dtype=torch.int64, device='cuda')

            if snark:
                saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/sigmoid_gate_x.bin', scores.cpu())
                saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/sigmoid_gate_r.bin', scores_rem.cpu())

            sigmoid_q23(scores, C)

            if snark:
                saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/sigmoid_gate_y.bin', C.cpu())

            # 当前 scores shape: [seqLen, 256]
            scores = C.squeeze(0)

        # bias的rescale为2^23
        original_scores = scores
        if self.bias is not None:
            # scores = scores + self.bias
            # 当前 scores shape: [seqLen, 256]
            scores = scores + self.bias

        if snark:
            saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/gate_original_scores.bin', original_scores.contiguous().cpu())
            saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/gate_bias.bin', self.bias.view(torch.uint32).cpu())

        # n_groups = 8
        if self.n_groups > 1:
            # x.size(0) = 8，当前 scores shape: [seqLen, 8, 32]
            scores = scores.view(x.size(0), self.n_groups, -1)
            # print(f'scores shape 111: {scores.shape}', flush=True)
            if self.bias is None:
                group_scores = scores.amax(dim=-1)
            else:
                # topk 返回 -1维度上 最大的 前 2 个值，同时返回值和索引，[0] 表示 取值，sum(-1) 再把最大的两个值相加.
                # 256维，分成8个组，每个组挑最大的两个数相加，得到 [seqLen, 8] 的结果，代表 8 个组的 最大两个值的和。
                # group_scores 的 shape: [8, 8]
                group_scores = scores.topk(2, dim=-1)[0].sum(dim=-1)
                # print(group_scores[0], flush=True)
                # print(f'group_scores shape: {group_scores.shape}')

            # topk_groups = 4, 从 8 个group中选择最大的 4个，返回其索引，比如返回 [[0, 2, 4, 6], ...]
            # indices shape: [seqLen, 4]
            indices = group_scores.topk(self.topk_groups, dim=-1)[1]
            # print(indices[0], flush=True)

            # mask shape: [seqLen, 8]
            # scatter_: 按照给定索引，把某个源张量的值写入到目标张量对应位置。 Tensor.scatter_(dim, index, src, reduce=None)
            # 比如 mask 为[[False, True, False, True, False, True, False, True], ...]
            # mask: 每一行最大的4个值相对应的 mask 为 False
            mask = scores.new_ones(x.size(0), self.n_groups, dtype=bool).scatter_(1, indices, False)
            # print(mask[0], flush=True)
            # 把满足布尔 mask 的位置替换成 "-inf", mask.unsqueeze(-1) shape: [8, 8, 1]
            # 把 scores 中 淘汰掉的4个group中的每一个值设置为 "-inf",总共设置 128个 "-inf"，占每一行中的一半
            # scores shape: [seqLen, 256]
            # scores = scores.masked_fill_(mask.unsqueeze(-1), float("-inf")).flatten(1)
            scores = scores.masked_fill_(mask.unsqueeze(-1), -(1 << 42)).flatten(1)

        # 没有淘汰掉的group中的 128个值中，选择最大的8个值，返回其下标
        # self.topk = 8, indices shape: [8, 8]
        indices = torch.topk(scores, self.topk, dim=-1)[1]
        # print(indices[0], flush=True)

        # gather 用来按照索引从一个张量中取值，按照8个最大值的下标，获取其值
        # weights shape: [8, 8]
        weights = original_scores.gather(1, indices)

        if snark:
            saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/gate_indices.bin', indices.contiguous().cpu())
            saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/gate_weights.bin', weights.contiguous().cpu())

        # print(f'weights shape: {weights.shape}')
        if self.score_func == "sigmoid":
            sum1 = weights.sum(dim=-1, keepdim=True)
            # weights = (weights * (2 ** 25) + sum1 // 2) // sum1
            weights = (weights * (2 ** 23)) // sum1
            # weights /= weights.sum(dim=-1, keepdim=True)

        #self.route_scale = 2.5
        # weights *= self.route_scale
        weights = weights * 5 // 2

        # weights = (weights.to(torch.float32) * (2 ** -23)).to(torch.bfloat16)
        # return weights.type_as(x), indices
        return weights, indices


class Expert_int(nn.Module):
    """
    Expert layer for Mixture-of-Experts (MoE) models.

    Attributes:
        w1 (nn.Module): Linear layer for input-to-hidden transformation.
        w2 (nn.Module): Linear layer for hidden-to-output transformation.
        w3 (nn.Module): Additional linear layer for feature transformation.
    """
    def __init__(self, layer_id, idx, dim: int, inter_dim: int):
        """
        Initializes the Expert layer.

        Args:
            dim (int): Input and output dimensionality.
            inter_dim (int): Hidden layer dimensionality.
        """
        super().__init__()
        # # w1 shape: [2048, 7168]
        # self.w1 = Linear(layer_id, dim, inter_dim)
        # # w2 shape: [7168, 2048]
        # self.w2 = Linear(layer_id, inter_dim, dim)
        # # w3 shape: [2048, 7168]
        # self.w3 = Linear(layer_id, dim, inter_dim)

        self.layer_id = layer_id
        self.idx = idx

        self.w1 = Linear_rescale_int(layer_id, dim, inter_dim, 1, 1, torch.int32)
        self.w2 = Linear_rescale_int(layer_id, inter_dim, dim, 1, 1, torch.int32)
        self.w3 = Linear_rescale_int(layer_id, dim, inter_dim, 1, 1, torch.int32)

    def forward(self, start_pos: int, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the Expert layer.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert computation.
        """

        # 返回的 shape [bsz, seqLen, 7168]
        # return self.w2(F.silu(self.w1(x)) * self.w3(x))
        # r1 shape: [bsz, seqLen, 18432], r1 rescale: 2^23
        r1 = self.w1(x)

        # s1 = F.silu(r1)
        # s1 shape: [bsz, seqLen, 18432], s1 rescale: 2^23
        s1 = torch.empty_like(r1, dtype=torch.int64, device='cuda')
        # silu_q25(r1, s1)

        if snark:
            saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/expert_{self.idx}_silu_x.bin', r1.contiguous().cpu())

        silu_q23(r1, s1)

        if snark:
            saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/expert_{self.idx}_silu_y.bin', s1.cpu())

        # r2 rescale: 2^23
        r2 = self.w3(x)

        # 返回的 shape [bsz, seqLen, 7168]
        q = self.w2((s1 * r2) >> 23)
        return q


class MoE(nn.Module):
    """
    Mixture-of-Experts (MoE) module.

    Attributes:
        dim (int): Dimensionality of input features.
        n_routed_experts (int): Total number of experts in the model.
        n_local_experts (int): Number of experts handled locally in distributed systems.
        n_activated_experts (int): Number of experts activated for each input.
        gate (nn.Module): Gating mechanism to route inputs to experts.
        experts (nn.ModuleList): List of expert modules.
        shared_experts (nn.Module): Shared experts applied to all inputs.
    """
    def __init__(self, layer_id, args: ModelArgs, ckpt_path):
        """
        Initializes the MoE module.

        Args:
            args (ModelArgs): Model arguments containing MoE parameters.
        """
        super().__init__()
        self.layer_id = layer_id
        self.ckpt_path = ckpt_path
        self.dim = args.dim
        self.moe_inter_dim = args.moe_inter_dim
        assert args.n_routed_experts % world_size == 0, f"Number of experts must be divisible by world size (world_size={world_size})"
        self.n_routed_experts = args.n_routed_experts
        self.n_local_experts = args.n_routed_experts // world_size
        self.n_activated_experts = args.n_activated_experts
        self.experts_start_idx = rank * self.n_local_experts
        self.experts_end_idx = self.experts_start_idx + self.n_local_experts
        self.gate = Gate(layer_id, args)
        # moe_inter_dim = 2048
        # self.experts = nn.ModuleList([Expert(layer_id, args.dim, args.moe_inter_dim) if self.experts_start_idx <= i < self.experts_end_idx else None
        #                               for i in range(self.n_routed_experts)])
        # self.experts = torch.nn.ModuleList()

        # dim = 7168, n_shared_experts = 1, moe_inter_dim = 2048
        self.shared_experts = MLP_int(layer_id, args.dim, args.n_shared_experts * args.moe_inter_dim)

    # x 的 rescale 为 2^23, shape: [1, seqLen, 7168]
    def forward(self, start_pos: int, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the MoE module.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor after expert routing and computation.
        """
        # ffn_normed 的 rescale 为 2^23
        # x = (x.to(torch.float32) * (2 ** -23)).to(torch.bfloat16)

        # z rescale: 2^23, z 的 shape [seqLen, 7168]
        z = self.shared_experts(start_pos, x)

        # x shape 之前为: [bsz, seqLen, 7168], 之后为 [8, 7168]
        shape = x.size()
        x = x.view(-1, self.dim)

        # weights shape: [seqLen, 8], indices shape: [seqLen, 8]
        # weights 的 rescale 为 2^23
        weights, indices = self.gate(start_pos, x)

        # y shape: [seqLen, 7168]
        y = torch.zeros_like(x)
        # torch.bincount 用来统计非负整数张量中各个数值出现的次数，类似于直方图计数
        # torch.bincount(input, weights=None, minlength=0) -> Tensor, weights: 可选的一维浮点张量，和 input 形状一致。若提供，就不是“次数统计”，而是“权重和”
        # 统计 256 个 专家 出现的次数
        counts = torch.bincount(indices.flatten(), minlength=self.n_routed_experts).tolist()
        for i in range(self.experts_start_idx, self.experts_end_idx):
            if counts[i] == 0:
                continue
            # expert = self.experts[i]
            with torch.device("cuda"):
                expert = Expert_int(self.layer_id, i, self.dim, self.moe_inter_dim)
            # load_model(expert, f'/data3/DeepSeek-V3-Demo1/experts-{self.layer_id}/{i}.safetensors')
            expertModelPath = os.path.join(self.ckpt_path, f"experts-{self.layer_id}/{i}.safetensors")
            load_model(expert, expertModelPath)

            # 第 idx 个 token, 专家 i 出现的编号是 top
            # 比如
            # [0, 1, 3, 2, 5, 4, 6, 9]
            # [7, 8, 3, 12, 5, 11, 6, 1]
            # [16, 10, 3, 2, 15, 4, 6, 9]
            # [10, 21, 3, 2, 5, 4, 1, 9]
            # torch.where(indices == 1) 返回的结果是 ([0, 1, 3], [1, 7, 6])
            idx, top = torch.where(indices == i)
            # expert(x[idx]) 返回的 shape [seqLen, 2048], weights[idx, top, None] 的 shape 为 [seqLen, 1], 包含一个 weight 值
            # y[idx] += expert(x[idx]) * weights[idx, top, None]
            x2 = x[idx].unsqueeze(0)
            y2 = expert(start_pos, x2)
            y2 = y2.view(-1, self.dim)
            # y[idx] += y2 * weights[idx, top, None] // (1 << 25)
            y[idx] += y2 * weights[idx, top, None] // (1 << 23)
        # z = self.shared_experts(x)
        if world_size > 1:
            dist.all_reduce(y)
        return (y + z).view(shape)

def getBF8PrintStr(ele):
    v = int(ele.cpu().view(torch.uint8).item())
    ex = v >> 3 & 0xF
    r = v & 0x7

    if ex == 15 and r == 7:
        print(f'BF8 Nan: {ex} {r} !!!', flush=True)
    elif ex == 0:
        print(f'BF8 subnormal: {ex} {r} !!!', flush=True)

    if v & 0x80:
        vstr = f'-{ex} {r}'
    else:
        vstr =  f'{ex} {r}'
    return vstr

class Block(nn.Module):
    """
    Transformer block combining attention and feed-forward layers.

    Attributes:
        attn (nn.Module): Attention layer (MLA).
        ffn (nn.Module): Feed-forward network (MLP or MoE).
        attn_norm (nn.Module): Layer normalization for attention.
        ffn_norm (nn.Module): Layer normalization for feed-forward network.
    """
    def __init__(self, layer_id: int, args: ModelArgs, ckpt_path):
        """
        Initializes the Transformer block.

        Args:
            layer_id (int): Layer index in the transformer.
            args (ModelArgs): Model arguments containing block parameters.
        """
        super().__init__()
        self.layer_id = layer_id
        self.ckpt_path = ckpt_path
        self.attn = MLA(layer_id, args)
        self.ffn = MLP_int(layer_id, args.dim, args.inter_dim) if layer_id < args.n_dense_layers else MoE(layer_id, args, ckpt_path)
        # print('args.dim: ' + str(args.dim))
        # args.dim = 7168
        self.attn_norm = RMSNorm_int(args.dim, torch.int32)
        self.ffn_norm = RMSNorm_int(args.dim, torch.int32)
        # self.ffn_norm = RMSNorm(args.dim)

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
        """
        Forward pass for the Transformer block.

        Args:
            x (torch.Tensor): Input tensor.
            start_pos (int): Starting position in the sequence.
            freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary embeddings.
            mask (Optional[torch.Tensor]): Mask tensor to exclude certain positions from attention.

        Returns:
            torch.Tensor: Output tensor after block computation.
        """

        x_abs = x.abs()
        x_abs_min = x_abs.min().item()
        x_abs_max = x_abs.max().item()
        print(f'x abs min: {x_abs_min}, max: {x_abs_max}', flush=True)

        # self.attn_norm(x): 在进行attention之前，先将7168维的embeding 进行 归一化
        # attn_norm 的 scale 为 2^21, x 的 scale 为 2^31
        (atten_normed, rms) = self.attn_norm(x)

        if snark:
            os.makedirs(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}', exist_ok=True)
            saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/attn_norm_x.bin', x.cpu())
            saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/attn_norm_weight.bin', self.attn_norm.weight.view(torch.uint32).cpu())
            saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/attn_norm_y.bin', atten_normed.cpu())
            saveTensor(f'{zkDataDir}/pos_{start_pos}/layer_{self.layer_id}/attn_norm_rms.bin', rms.cpu())

        # attned 的 rescale 是 2^19, shape: [1, seqLen, 7168]
        attned = self.attn(atten_normed, start_pos, freqs_cis, mask)

        # 调整 rescale，因为 x 的 rescale 是 2^31, attned 的 rescale 是 2^19，因此要乘以 2^12
        # x = x + attned * (2 ** 10)
        x = x + attned * (2 ** 12)

        # ffn_normed 的 rescale 为 2^23
        (ffn_normed, rms) = self.ffn_norm(x)

        ffned = self.ffn(start_pos, ffn_normed)
        # x = x + ffned * (2 ** 6)
        x = x + ffned * (2 ** 8)

        # 返回的 x 的rescale 为 2^31
        return x

# Transformer 类在初始化中就已经明确好了自己的进程（rank），并且可以发现它是由比较经典的transformer组件构成的：
# embedding层（self.embed）、堆叠的decoding block（self.layers），标准的RMSnorm层（self.norm）与最后将隐藏状态投射到词表分布的output层（self.head）
# 根据前面提及的初始化的参数来看，词表大小为129280，模型的hidden dim为7168，堆叠的decode block一共有61个。维度变换会在下面举例说明。
# Transformer 由61个Block组成，每个Block有 attn 和 ffd
# Transformer类在初始化中就已经明确好了自己的进程（rank），并且可以发现它是由比较经典的transformer组件构成的
# embedding层（self.embed）、堆叠的decoding block（self.layers），标准的RMSnorm层（self.norm）与最后将隐藏状态投射到词表分布的output层（self.head）。
class Transformer(nn.Module):
    """
    Transformer model with positional embeddings, multiple layers, and output projection.

    Attributes:
        max_seq_len (int): Maximum sequence length for the transformer.
        embed (nn.Module): Embedding layer for input tokens.
        layers (torch.nn.ModuleList): List of transformer blocks.
        norm (nn.Module): Layer normalization applied after all blocks.
        head (nn.Module): Output projection layer mapping to vocabulary size.
        freqs_cis (torch.Tensor): Precomputed complex exponential values for rotary(旋转的) embeddings.
    """
    def __init__(self, args: ModelArgs):
        """
        Initializes the Transformer model.

        Args:
            args (ModelArgs): Model arguments containing transformer parameters.
        """
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        Linear.dtype = torch.float8_e4m3fn if args.dtype == "fp8" else torch.bfloat16
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.embed = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = torch.nn.ModuleList()
        for layer_id in range(args.n_layers):
        #     self.layers.append(Block(layer_id, args))
            self.layers.append(nn.Module())

        self.norm = RMSNorm_int(args.dim, torch.int64)
        # self.head = ColumnParallelLinear(-1, args.dim, args.vocab_size, dtype=torch.get_default_dtype())
        # 模型中的 head 的 rescale 为 2^43, 使用的过程中的rescale为 2^35, head 输入的 rescale为 2^15, 输出的 rescale为 2^21
        # self.head = ColumnParallelLinear_int(-1, args.dim, args.vocab_size, 1, (1 << 8), (1 << 29), torch.int64)
        self.head = ColumnParallelLinear_int(-1, args.dim, args.vocab_size, 1, (1 << 8), 29, torch.int64)
        # self.head = ColumnParallelLinear_int(-1, args.dim, args.vocab_size, 1, (1 << 8), (1 << 31), torch.int64)
        # self.head = ColumnParallelLinear_int(-1, args.dim, args.vocab_size, (1 << 5), (1 << 11), (1 << 21), torch.int64)
        # register_buffer()注册了名为 "freqs_cis" 的缓冲区，缓冲区的值由 precompute_freqs_cis(args) 提供，并且由于设置了 persistent=False，
        # 该缓冲区不会被保存到模型的状态字典中。缓冲区注册的张量是该Transformer类的位置编码。
        # register_buffer 用于注册一个非参数张量（tensor），这个张量虽然不是模型的可学习参数，但仍然是模型状态的一部分。
        # 与参数不同，缓冲区不会在反向传播中计算梯度，也不会被优化器更新，但它会随模型一起移动到相应的设备（如 GPU）上。
        # persistent=False表示这个参数表示该缓冲区不属于持久状态（persistent state）。也就是说，当你调用 model.state_dict() 保存模型时，
        # 这个缓冲区不会被包含进去。位置编码可以在模型加载后重新计算，不需要存储。
        self.register_buffer("freqs_cis", precompute_freqs_cis(args), persistent=False)

    @torch.inference_mode()
    def prep_inference(self, tokens: torch.Tensor, start_pos: int = 0):
        # softmax_init()
        softmax_init_q19()
        softmax_init_q21()
        silu_init_q23()

        seqlen = tokens.size(1)

        # h 是经过embed之后的结果，embed将文本表达转化为词嵌入，h的形状为 (batch_size, seq_len, 7168)
        h = self.embed(tokens)
        # h = h.to(torch.bfloat16) * (1.0 / (1 << 44))

        return (h, start_pos, seqlen)

    @torch.inference_mode()
    def layer_inference(self, layer_id, h, start_pos, seqlen):
        freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
        mask = None

        # triu = triangle up
        # 返回上三角矩阵
        # 参数 k=0 代表主对角线，k 为正数则从主对角线开始向上数第 k 条，k 为负数则从主对角线开始向下数第 k 条
        if seqlen > 1:
            # mask = torch.full((seqlen, seqlen), float("-inf"), device="cuda").triu_(1)
            mask = torch.full((seqlen, seqlen), -(64 << 36), dtype=torch.int64, device="cuda").triu_(1)

        h = self.layers[layer_id](h, start_pos, freqs_cis, mask)

        h_abs = (h.to(torch.float32) * (2 ** -31)).to(torch.bfloat16).abs()
        h_abs_max = h_abs.max()
        h_abs[h_abs < (2 ** -125)] = h_abs_max
        h_abs_min = h_abs.min()
        h_abs_min_str = getBF16PrintStr(h_abs_min)
        h_abs_max_str = getBF16PrintStr(h_abs_max)
        print(f'h_abs min: {h_abs_min_str}, max: {h_abs_max_str}')

        # 返回的 h 的rescale 为 2^31
        return h

    @torch.inference_mode()
    def finish_inference(self, h):
        # norm的结果的scale = 2^15, h 的 scale = 2^15
        h = self.norm(h)[0][:, -1]

        # logits 的rescale 为 2^21
        logits = self.head(h[None, :])
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)

        # logits 的 scale = 2^21
        return logits

    # # 这里开始推理了，torch.inference_mode 这句话 关闭梯度计算 并 禁止 autograd 构建计算图，同时比 torch.no_grad() 还高效，专门为推理场景优化
    # @torch.inference_mode()
    # def forward(self, tokens: torch.Tensor, start_pos: int = 0):
    #     """
    #     Forward pass for the Transformer model.

    #     Args:
    #         tokens (torch.Tensor): Input tensor of token IDs with shape (batch_size, seq_len).
    #         start_pos (int, optional): Starting position in the sequence for rotary(旋转的) embeddings. Defaults to 0.

    #     Returns:
    #         torch.Tensor: Logits tensor of shape (batch_size, vocab_size).
    #     """
    #     seqlen = tokens.size(1)
    #     # h 是经过embed之后的结果，embed将文本表达转化为词嵌入，h的形状为 (batch_size, seq_len, 7168)
    #     h = self.embed(tokens)
    #     freqs_cis = self.freqs_cis[start_pos:start_pos+seqlen]
    #     print('freqs_cis: ' + str(freqs_cis.tolist()))

    #     mask = None

    #     # triu = triangle up
    #     # 返回上三角矩阵
    #     # 参数 k=0 代表主对角线，k 为正数则从主对角线开始向上数第 k 条，k 为负数则从主对角线开始向下数第 k 条
    #     if seqlen > 1:
    #         mask = torch.full((seqlen, seqlen), float("-inf"), device=tokens.device).triu_(1)

    #     for layer in self.layers:
    #         h = layer(h, start_pos, freqs_cis, mask)

    #     # 只取最后一个 token
    #     h = self.norm(h)[:, -1]
    #     logits = self.head(h)
    #     if world_size > 1:
    #         all_logits = [torch.empty_like(logits) for _ in range(world_size)]
    #         dist.all_gather(all_logits, logits)
    #         logits = torch.cat(all_logits, dim=-1)
    #     return logits


if __name__ == "__main__":
    torch.set_default_dtype(torch.bfloat16)
    torch.set_default_device("cuda")
    torch.manual_seed(0)
    args = ModelArgs()
    x = torch.randint(0, args.vocab_size, (2, 128))
    model = Transformer(0, args)
    print(model(x).size())
