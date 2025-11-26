// int64_gemm.cu
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

extern "C" __global__ void int64_32_bmm_broadcast_kernel(
    const int64_t* __restrict__ A,  // (B, M, K)
    const int32_t* __restrict__ B,  // (N, K)
    int64_t* __restrict__ C,        // (B, M, N)
    int64_t* __restrict__ R,        // remainer (B, M, N)
    const int64_t a_rescale,
    const int64_t b_rescale,
    const int64_t c_rescale,
    int Bdim, int M, int K, int N)
{
    int b = blockIdx.z;                              // batch
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N

    if (row < M && col < N) {
        __int128_t sum = 0;
        __int128_t rescale = (1 << c_rescale) - 1;
        for (int k = 0; k < K; ++k) {
            int64_t a_val = A[b * M * K + row * K + k];   // A[b, row, k]
            int32_t b_val = B[col * K + k];               // B[col, k]
            sum += __int128_t(a_val / a_rescale) * __int128_t(b_val / b_rescale);
        }
        int ind = b * M * N + row * N + col;
        // C[ind] = sum / c_rescale;  // C[b, row, col]
        // R[ind] = sum % c_rescale;  // R[b, row, col]
        C[ind] = int64_t(sum >> c_rescale);  // C[b, row, col]
        R[ind] = int64_t(sum & rescale);  // R[b, row, col]
    }
}

extern "C" void int64_32_bmm_broadcast_launcher(
    const int64_t* A, const int32_t* B, int64_t* C, int64_t* R,
    const int64_t a_rescale, const int64_t b_rescale, const int64_t c_rescale,
    int Bdim, int M, int K, int N)
{
    dim3 threads(32, 32);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y,
                Bdim);

    int64_32_bmm_broadcast_kernel<<<blocks, threads>>>(A, B, C, R, a_rescale, b_rescale, c_rescale, Bdim, M, K, N);
}

extern "C" __global__ void int64_64_bmm_broadcast_kernel(
    const int64_t* __restrict__ A,  // (B, M, K)
    const int64_t* __restrict__ B,  // (N, K)
    int64_t* __restrict__ C,        // (B, M, N)
    int64_t* __restrict__ R,        // remainer (B, M, N)
    const int64_t a_rescale,
    const int64_t b_rescale,
    const int64_t c_rescale,
    int Bdim, int M, int K, int N)
{
    int b = blockIdx.z;                              // batch
    int row = blockIdx.y * blockDim.y + threadIdx.y; // M
    int col = blockIdx.x * blockDim.x + threadIdx.x; // N

    if (row < M && col < N) {
        __int128_t sum = 0;
        __int128_t rescale = (1 << c_rescale) - 1;
        for (int k = 0; k < K; ++k) {
            int64_t a_val = A[b * M * K + row * K + k];   // A[b, row, k]
            int64_t b_val = B[col * K + k];               // B[col, k]
            sum += __int128_t(a_val / a_rescale) * __int128_t(b_val / b_rescale);
        }
        int ind = b * M * N + row * N + col;
        // C[ind] = sum / c_rescale;  // C[b, row, col]
        // R[ind] = sum % c_rescale;  // R[b, row, col]
        C[ind] = int64_t(sum >> c_rescale);  // C[b, row, col]
        R[ind] = int64_t(sum & rescale);  // R[b, row, col]
    }
}

extern "C" void int64_64_bmm_broadcast_launcher(
    const int64_t* A, const int64_t* B, int64_t* C, int64_t* R,
    const int64_t a_rescale, const int64_t b_rescale, const int64_t c_rescale,
    int Bdim, int M, int K, int N)
{
    dim3 threads(32, 32);
    dim3 blocks((N + threads.x - 1) / threads.x,
                (M + threads.y - 1) / threads.y,
                Bdim);

    int64_64_bmm_broadcast_kernel<<<blocks, threads>>>(A, B, C, R, a_rescale, b_rescale, c_rescale, Bdim, M, K, N);
}

extern "C" __global__ void bf16_to_int32_2d_kernel(const uint16_t* input, int32_t* output, int rows, int cols, int rescale)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;

        int v0 = input[idx];
        int ex0 = ((v0 >> 7) & 0xFF) - 127;
        int r0 = v0 & 0x7F;

        if (ex0 == -127 && r0 == 0) {
            output[idx] = 0;
            return;
        }

        int ex2 = ex0 + rescale;
        int r2 = r0 + 128;
        uint32_t v = 0;
        if(ex2 >= 0) {
            v = r2 * (1 << ex2);
        } else {
            v = r2 / (1 << -ex2);
        }

        if (v0 & 0x8000) {
            v = -v;
        }

        output[idx] = v;
    }
}

extern "C" void bf16_to_int32_2d(const uint16_t* input, int32_t* output, int rows, int cols, int rescale) {
    dim3 threads(32, 32);
    dim3 blocks((cols + threads.x - 1) / threads.x,
                (rows + threads.y - 1) / threads.y);

    bf16_to_int32_2d_kernel<<<blocks, threads>>>(input, output, rows, cols, rescale);
}

extern "C" __global__ void wkv_b_bf16_to_int32_kernel(const uint16_t* input, int32_t* output, int rows, int cols)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;

        int v0 = input[idx];
        int ex0 = ((v0 >> 7) & 0xFF) - 127;
        int r0 = v0 & 0x7F;

        if (ex0 == -127 && r0 == 0) {
            output[idx] = 0;
            return;
        }

        if (ex0 >= -1) {
            output[idx] = 0x7FFFFFFF;
            return;
        }

        int ex2 = ex0 + 25;
        int r2 = r0 + 128;
        uint32_t v = 0;
        if(ex2 >= 0) {
            v = r2 * (1 << ex2);
        } else {
            v = r2 / (1 << -ex2);
        }

        if (v0 & 0x8000) {
            v = -v;
        }

        output[idx] = v;
    }
}

extern "C" void wkv_b_bf16_to_int32(const uint16_t* input, int32_t* output, int rows, int cols) {
    dim3 threads(32, 32);
    dim3 blocks((cols + threads.x - 1) / threads.x,
                (rows + threads.y - 1) / threads.y);

    wkv_b_bf16_to_int32_kernel<<<blocks, threads>>>(input, output, rows, cols);
}

extern "C" __global__ void float32_to_int64_2d_kernel(const uint32_t* input, int64_t* output, int rows, int cols, int rescale)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < rows && col < cols) {
        int idx = row * cols + col;

        uint32_t v0 = input[idx];
        int ex0 = ((v0 >> 23) & 0xFF) - 127;
        int r0 = v0 & 0x7FFFFF;

        if (ex0 == -127 && r0 == 0) {
            output[idx] = 0;
            return;
        }

        int ex2 = ex0 + rescale;
        int64_t r2 = r0 + 8388608;
        int64_t v = 0;
        if(ex2 >= 0) {
            v = r2 * (1 << ex2);
        } else {
            v = r2 / (1 << -ex2);
        }

        if (v0 & 0x80000000) {
            v = -v;
        }

        output[idx] = v;
    }
}

extern "C" void float32_to_int64_2d(const uint32_t* input, int64_t* output, int rows, int cols, int rescale) {
    dim3 threads(32, 32);
    dim3 blocks((cols + threads.x - 1) / threads.x,
                (rows + threads.y - 1) / threads.y);

    float32_to_int64_2d_kernel<<<blocks, threads>>>(input, output, rows, cols, rescale);
}

extern "C" __global__ void complex_int64_mul_kernel(
    const int64_t* __restrict__ A,
    const int64_t* __restrict__ B,
    int64_t* __restrict__ C,
    // int64_t high_rescale, int64_t row_rescale,
    int batchSize, int seqLen, int headCount, int headDim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batchSize * seqLen * headCount * headDim;
    if (idx >= total) return;

    // 计算 A 的索引
    int i = idx;
    int dimId = i % headDim; i /= headDim;
    int headId = i % headCount; i /= headCount;
    int seqId = i % seqLen; i /= seqLen;
    int batchId = i;

    // A 索引
    int a_idx = ((batchId * seqLen + seqId) * headCount + headId) * headDim + dimId;

    // B 索引 (广播)
    int b_idx = ((0 * seqLen + seqId) * 1 + 0) * headDim + dimId;

    int64_t a0 = A[2 * a_idx];
    int64_t a1 = A[2 * a_idx + 1];
    int64_t b0 = B[2 * b_idx];
    int64_t b1 = B[2 * b_idx + 1];

    // C[2 * a_idx] = (a0 * b0 - a1 * b1) / c_resacle;
    // C[2 * a_idx + 1] = (a0 * b1 + a1 * b0) / c_resacle;

    // C[2 * a_idx] = __mul64hi(a0, b0) * high_rescale + a0 * b0 / row_rescale) - (__mul64hi(a1, b1) * high_rescale + a1 * b1 / row_rescale);
    // C[2 * a_idx + 1] = (__mul64hi(a0, b1) * high_rescale + a0 * b1 / row_rescale) + (__mul64hi(a1, b0) * high_rescale + a1 * b0 / row_rescale);
    int64_t a0b0 = ((__mul64hi(a0, b0) & 0x3FFFFFFFFFF) << 22) | (((a0 * b0) >> 42) & 0x3FFFFF);
    int64_t a1b1 = ((__mul64hi(a1, b1) & 0x3FFFFFFFFFF) << 22) | (((a1 * b1) >> 42) & 0x3FFFFF);
    int64_t a0b1 = ((__mul64hi(a0, b1) & 0x3FFFFFFFFFF) << 22) | (((a0 * b1) >> 42) & 0x3FFFFF);
    int64_t a1b0 = ((__mul64hi(a1, b0) & 0x3FFFFFFFFFF) << 22) | (((a1 * b0) >> 42) & 0x3FFFFF);

    C[2 * a_idx] = a0b0 - a1b1;
    C[2 * a_idx + 1] = a0b1 + a1b0;

    // if(idx == 32) {
    //     printf("%d %d %d, %d %d %d %d (%d %d %d %d): (%ld, %ld i) * (%ld, %ld i) = (%ld, %ld i)\n",
    //     idx, a_idx, b_idx,
    //     batchSize, seqLen, headCount, headDim,
    //     batchId, seqId, headId, dimId,
    //     a0, a1, b0, b1, C[2 * a_idx], C[2 * a_idx + 1]);
    // }
}

extern "C" void complex_int64_mul(
    const int64_t* A, const int64_t* B, int64_t* C,
    // const int64_t high_rescale, const int64_t row_rescale,
    int batchSize, int seqLen, int headCount, int headDim)
{
    int total = batchSize * seqLen * headCount * headDim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    complex_int64_mul_kernel<<<blocks, threads>>>(A, B, C,
        // high_rescale, row_rescale,
        batchSize, seqLen, headCount, headDim);
}


extern "C" __global__ void rms_norm_kernel_32(
    const int64_t* __restrict__ A,
    const int32_t* __restrict__ W,
    const int64_t* __restrict__ rms,
    int64_t* __restrict__ C,
    int seqLen, int Dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seqLen * Dim;
    if (idx >= total) return;

    // 计算 A 的索引
    int dimId = idx % Dim;
    int seqId = idx / Dim;

    // A 索引
    int a_idx = seqId * Dim + dimId;

    // W 索引 (广播)
    int w_idx = dimId;

    int64_t a = A[a_idx];
    int32_t w = W[w_idx];
    int64_t r = rms[seqId];

    __int128 prod = ( __int128)a * ( __int128)w;  // 在 128 位里计算乘积，不溢出
    __int128 qq = prod / (__int128)r;            // 整数除法
    __int128 rr = prod % (__int128)r;            // 整数取模
    if(rr < 0) {
        qq = qq - 1;
        rr = rr + r;
    }

    int64_t res = (int64_t)qq;

    C[a_idx] = res;
}

extern "C" __global__ void rms_norm_kernel_64(
    const int64_t* __restrict__ A,
    const int64_t* __restrict__ W,
    const int64_t* __restrict__ rms,
    int64_t* __restrict__ C,
    int seqLen, int Dim)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seqLen * Dim;
    if (idx >= total) return;

    // 计算 A 的索引
    int dimId = idx % Dim;
    int seqId = idx / Dim;

    // A 索引
    int a_idx = seqId * Dim + dimId;

    // W 索引 (广播)
    int w_idx = dimId;

    int64_t a = A[a_idx];
    int64_t w = W[w_idx];
    int64_t r = rms[seqId];

    __int128 prod = ( __int128)a * ( __int128)w;  // 在 128 位里计算乘积，不溢出
    __int128 qq = prod / (__int128)r;            // 整数除法
    __int128 rr = prod % (__int128)r;            // 整数取模
    if(rr < 0) {
        qq = qq - 1;
        rr = rr + r;
    }

    int64_t res = (int64_t)qq;

    C[a_idx] = res;
}

extern "C" void rms_norm_32(
    const int64_t* A, const int32_t* W, const int64_t* rms, int64_t* C,
    int seqLen, int Dim)
{
    int total = seqLen * Dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    rms_norm_kernel_32<<<blocks, threads>>>(A, W, rms, C, seqLen, Dim);
}

extern "C" void rms_norm_64(
    const int64_t* A, const int64_t* W, const int64_t* rms, int64_t* C,
    int seqLen, int Dim)
{
    int total = seqLen * Dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    rms_norm_kernel_64<<<blocks, threads>>>(A, W, rms, C, seqLen, Dim);
}

extern "C" __global__ void einsum_bshd_hdc_bshc_kernel(
    const int64_t* q_nope,  // [B, S, H, D]
    const int32_t* wkv_b_1, // [H, D, C]
    int64_t* out,           // [B, S, H, C]
    int64_t rescale,
    int B, int S, int H, int D, int C)
{
    int b = blockIdx.x;   // batch
    int s = blockIdx.y;   // sequence
    int h = blockIdx.z;   // head
    int c = threadIdx.x;  // output channel

    if (c >= C) return;

    __int128_t sum = rescale / 2;
    int q_base = ((b * S + s) * H + h) * D;
    int w_base = h * D * C + c;
    for (int d = 0; d < D; d++) {
        // int w_idx = (h * D + d) * C + c;
        sum += __int128_t(q_nope[q_base + d]) * __int128_t(wkv_b_1[w_base +  d * C]);
    }

    // sum /= rescale;
    int64_t sum2 = int64_t(sum >> rescale);

    int out_idx = ((b * S + s) * H + h) * C + c;
    out[out_idx] = sum2;
}

extern "C" void einsum_bshd_hdc_bshc(const int64_t* q_nope, const int32_t* wkv_b_1, int64_t* out,
    int64_t rescale, int B, int S, int H, int D, int C) {

    dim3 grid(B, S, H);
    dim3 block(C);

    einsum_bshd_hdc_bshc_kernel<<<grid, block>>>(
        q_nope, wkv_b_1, out, rescale,
        B, S, H, D, C);
}

extern "C" __global__ void einsum_bshc_btc_bsht_kernel(
    const int64_t* __restrict__ A,  // [B, S, H, C]
    const int64_t* __restrict__ B,  // [B, T, C]
    int64_t* __restrict__ C,        // [B, S, H, T]
    int64_t rescale,
    int Bsz, int S, int H, int T, int Cdim)
{
    int b = blockIdx.x;
    int s = blockIdx.y;
    int h = blockIdx.z;
    int t = threadIdx.x;

    if (t >= T) return;

    // 计算 A[b,s,h,:] 和 B[b,t,:] 的内积
    __int128_t sum = rescale / 2;

    int A_base = ((b * S + s) * H + h) * Cdim;
    int B_base = (b * T + t) * Cdim;
    for (int c = 0; c < Cdim; c++) {
        // int idxB = (b * T + t) * Cdim + c;
        sum += __int128_t(A[A_base + c]) * __int128_t(B[B_base + c]);
    }

    // sum /= rescale;
    int64_t sum2 =  int64_t(sum >> rescale);

    int idxC = ((b * S + s) * H + h) * T + t;
    C[idxC] = sum2;
}

extern "C" void einsum_bshc_btc_bsht(const int64_t* A, const int64_t* B, int64_t* C,
    int64_t rescale, int Bsz, int S, int H, int T, int Cdim)
{
    dim3 grid(Bsz, S, H);
    dim3 block(T);

    einsum_bshc_btc_bsht_kernel<<<grid, block>>>(
        A, B, C, rescale,
        Bsz, S, H, T, Cdim);
}

extern "C" __global__ void einsum_bsht_btc_bshc_kernel(
    const int64_t* __restrict__ A,
    const int64_t* __restrict__ B,
    int64_t* __restrict__ C,
    int64_t rescale,
    int Bsz, int S, int H, int T, int Cdim)
{
    int b = blockIdx.x;
    int s = blockIdx.y;
    int h = blockIdx.z;
    int c = threadIdx.x;

    if (c >= Cdim) return;

    __int128_t sum = rescale / 2;

    int A_base = ((b * S + s) * H + h) * T;
    int B_base = b * T * Cdim + c;
    for (int t = 0; t < T; ++t) {
        // int idxB = (b * T + t) * Cdim + c;
        sum += __int128_t(A[A_base + t]) * __int128_t(B[B_base +  t * Cdim]);
    }

    // sum /= rescale;
    int64_t sum2 = int64_t(sum >> rescale);

    const int idxC = ((b * S + s) * H + h) * Cdim + c;
    C[idxC] = sum2;
}

extern "C" void einsum_bsht_btc_bshc(
    const int64_t* A, const int64_t* B, int64_t* C,
    int64_t rescale, int Bsz, int S, int H, int T, int Cdim)
{
    dim3 grid(Bsz, S, H);
    dim3 block(Cdim);

    einsum_bsht_btc_bshc_kernel<<<grid, block>>>(
        A, B, C, rescale,
        Bsz, S, H, T, Cdim);
}

extern "C" __global__ void einsum_bshc_hdc_bshd_kernel(
    const int64_t* __restrict__ A,
    const int32_t* __restrict__ B,
    int64_t* __restrict__ C,
    int64_t rescale,
    int Bsz, int S, int H, int D, int Cdim)
{
    int b = blockIdx.x;
    int s = blockIdx.y;
    int h = blockIdx.z;
    int d = threadIdx.x;

    if (d >= D) return;

    __int128_t sum = 0;
    int A_base = ((b * S + s) * H + h) * Cdim;
    int B_base = (h * D + d) * Cdim;
    for (int c = 0; c < Cdim; ++c) {
        sum += __int128_t(A[A_base + c]) * __int128_t(B[B_base + c]);
    }

    // sum = (sum + rescale / 2) / rescale;
    int64_t sum2 = int64_t(sum >> rescale);

    const int idxC = ((b * S + s) * H + h) * D + d;
    C[idxC] = sum2;
}

extern "C" void  einsum_bshc_hdc_bshd(const int64_t* A, const int32_t* B, int64_t* C,
    int64_t rescale, int Bsz, int S, int H, int D, int Cdim)
{
    dim3 grid(Bsz, S, H);
    dim3 block(D);

    einsum_bshc_hdc_bshd_kernel<<<grid, block>>>(
        A, B, C, rescale,
        Bsz, S, H, D, Cdim
    );
}

// static const int64_t LOG2E_Q32 = 6196328019ULL; // log2(e)*2^32
static const int64_t LOG2E_Q21 = 3025551; // log2(e)*2^21
static const int64_t LOG2E_Q19 = 756388; // log2(e)*2^19
// static const int LOG_TABLE_SIZE = 10;
static const int LOG_TABLE_SIZE = 8;
// static uint64_t EXP2_FRAC_LUT[256] = { /* 预生成：round(2^(i/256)*2^32) */ };
// static int64_t EXP2_FRAC_LUT[256] = { /* 预生成：round(2^(i/256)*2^32) */ };
// EXP2_FRAC_LUT = torch.zeros([256, ], dtype=torch.int64, device="cuda")

// extern "C" void softmax_q21_to_probs(const int64_t* R, int n, int64_t* P_q21) {
//     int32_t Rmax = R[0];
//     for (int i = 1; i < n; ++i) if (R[i] > Rmax) Rmax = R[i];

//     // printf("Rmax: %d\n", Rmax);

//     int64_t sumW = 0;
//     static thread_local int64_t Wbuf[4096]; // 或动态分配
//     for (int i = 0; i < n; ++i) {
//         int64_t d = R[i] - Rmax; // Δ_i (<=0)

//         // 剪裁：小于 -16 的差值近似为 0
//         if (d < -(16 << 21))
//         {
//             Wbuf[i] = 0;
//             continue;
//         }

//         // y = d * log2(e) / 2^21  (Q32)
//         int64_t y = (d * LOG2E_Q21) >> 21;

//         // printf("y: %ld\n", y);

//         int64_t k = (-y) >> 21;        // 整数部分 取正数（k > 0）
//         int64_t f = (-y) & 0x1FFFFF; // 小数部分 取正数（Q21, f > 0）

//         // printf("k: %ld, f: %ld\n", k, f);

//         int64_t t = EXP2_FRAC_LUT[ f >> 13 ]; // 2^(frac(y)) in Q21, 13 = 21 - 8, 取小数部分 转换成整数之后的 高8位
//         // int64_t t = 0;

//         int64_t wi = (k >= 32) ? 0u : (t >> k); // 2^(-k) * t, 右移
//         Wbuf[i] = wi;
//         sumW += wi;
//     }

//     // 归一化到 Q21 概率
//     for (int i = 0; i < n; ++i) {
//         int64_t num = Wbuf[i] << 21; // 提升精度
//         P_q21[i] = sumW ? (num / sumW) : 0;
//     }
// }

// start of softmax_q21 ------------------------

extern "C" __global__ void softmax_kernel_q21(
    int64_t* R,  // [B, S, H, T]
    int64_t* C,  // [B, S, H, T]
    int64_t LOG2E_Q21, int64_t* EXP2_FRAC_LUT,
    int Bsz, int S, int H, int T)
{
    int b = blockIdx.x;
    int s = blockIdx.y;
    int h = threadIdx.x;

    if (h >= H) return;

    int idxbase = ((b * S + s) * H + h) * T;

    int64_t Rmax = R[idxbase];
    for (int i = 1; i < T; ++i) if (R[idxbase + i] > Rmax) Rmax = R[idxbase + i];

    int64_t sumW = 0;
    for (int i = 0; i < T; i++) {
        int64_t d = R[idxbase + i] - Rmax; // Δ_i (d <= 0)

        // 剪裁：小于 -64 的差值近似为 0
        if (d < -(64 << 21))
        {
            R[idxbase + i] = 0;
            continue;
        }

        // y = d * log2(e) / 2^21  (Q21, y <= 0)
        int64_t y = (d * LOG2E_Q21 + (1 << 20)) >> 21;

        int64_t k = (-y) >> 21;        // 整数部分 取正数（k > 0）
        int64_t f = (-y) & 0x1FFFFF; // 小数部分 取正数（Q21）

        // printf("k: %ld, f: %ld\n", k, f);

        int64_t t = EXP2_FRAC_LUT[ f >> (21 - LOG_TABLE_SIZE) ]; // 2^(frac(y)) in Q21, 取小数部分 转换成整数之后的高 LOG_TABLE_SIZE 位
        // int64_t t = 0;

        int64_t wi = (k >= 64) ? 0 : (t >> k); // 2^(-k) * t, 右移

        // if(b == 0 && s == 3 && h == 127) {
        //     printf("i: %d, d: %ld, y: %ld, k: %ld, f: %ld, t: %ld, wi: %ld\n", i, d, y, k, f, t, wi);
        // }

        R[idxbase + i] = wi;
        sumW += wi;
    }

    // 归一化到 Q21 概率
    for (int i = 0; i < T; ++i) {
        int64_t num = R[idxbase + i] << 21; // 提升精度
        C[idxbase + i] = sumW ? ((num + sumW / 2) / sumW) : 0;
    }
}

extern "C" void softmax_q21(int64_t* R, int64_t* C, int64_t* EXP2_FRAC_LUT,
    int Bsz, int S, int H, int T)
{
    dim3 grid(Bsz, S);
    dim3 block(H);

    softmax_kernel_q21<<<grid, block>>>(
        R, C, LOG2E_Q21, EXP2_FRAC_LUT,
        Bsz, S, H, T);
}

extern "C" void softmax_init_q21(int64_t* EXP2_FRAC_LUT)
{
    // printf("inited!\n");
    int x2_21 = 1 << 21;
    for(int i = 0; i < (1 << LOG_TABLE_SIZE); i++) {
        // EXP2_FRAC_LUT[i] = uint64_t(std::pow(2, i / 256.0) * 4294967296);
        EXP2_FRAC_LUT[i] = int64_t(std::pow(2, i * (-1.0f) / (1 << LOG_TABLE_SIZE)) * x2_21);
    }
}
// -- end of softmax_q21 -----------------------------

// start of softmax_q19 ------------------------
extern "C" __global__ void softmax_kernel_q19(
    int64_t* R,  // [B, S, H, T]
    int64_t* C,  // [B, S, H, T]
    int64_t LOG2E_Q19, int64_t* EXP2_FRAC_LUT,
    int Bsz, int S, int H, int T)
{
    int b = blockIdx.x;
    int s = blockIdx.y;
    int h = threadIdx.x;

    if (h >= H) return;

    int idxbase = ((b * S + s) * H + h) * T;

    int64_t Rmax = R[idxbase];
    for (int i = 1; i < T; ++i) if (R[idxbase + i] > Rmax) Rmax = R[idxbase + i];

    int64_t sumW = 0;
    for (int i = 0; i < T; i++) {
        u_int64_t d = Rmax - R[idxbase + i]; // Δ_i (d >= 0)

        // 剪裁：小于 -64 的差值近似为 0, d > (64 << 19)的条件 比 (k >= 64) 要宽松
        if (d > (64 << 19))
        {
            R[idxbase + i] = 0;
            continue;
        }

        // y = d * log2(e) / 2^19  (Q19, y >= 0)
        int64_t y = int64_t((__int128_t(d) * __int128_t(LOG2E_Q19)) >> 19);

        int64_t k = y >> 19;        // 整数部分 取正数（k > 0）
        int64_t f = y & 0x7FFFF; // 小数部分 取正数（Q19）

        // printf("k: %ld, f: %ld\n", k, f);

        int64_t t = EXP2_FRAC_LUT[ f >> (19 - LOG_TABLE_SIZE) ]; // 2^(frac(y)) in Q19, 取小数部分 转换成整数之后的高 LOG_TABLE_SIZE 位
        // int64_t t = 0;

        int64_t wi = (k >= 64) ? 0 : (t >> k); // 2^(-k) * t, 右移

        // if(b == 0 && s == 2 && h == 2) {
        //     printf("i: %d, d: %ld, y: %ld, k: %ld, f: %ld, t: %ld, wi: %ld\n", i, d, y, k, f, t, wi);
        // }

        R[idxbase + i] = wi;
        sumW += wi;
    }

    // if(b == 0 && s == 9 && h == 7) {
    //     printf("sumW: %ld\n", sumW);
    // }

    // 归一化到 Q19 概率
    for (int i = 0; i < T; ++i) {
        int64_t num = R[idxbase + i] << 19; // 提升精度
        C[idxbase + i] = sumW ? (num / sumW) : 0;

        // if(b == 0 && s == 9 && h == 7) {
        //     printf("i: %d, r: %ld, num: %ld, c: %ld\n", i, R[idxbase + i], num, C[idxbase + i]);
        // }
    }
}

extern "C" void softmax_q19(int64_t* R, int64_t* C, int64_t* EXP2_FRAC_LUT,
    int Bsz, int S, int H, int T)
{
    dim3 grid(Bsz, S);
    dim3 block(H);

    softmax_kernel_q19<<<grid, block>>>(
        R, C, LOG2E_Q19, EXP2_FRAC_LUT,
        Bsz, S, H, T);
}

extern "C" void softmax_init_q19(int64_t* EXP2_FRAC_LUT)
{
    // printf("inited!\n");
    int x2_19 = 1 << 19;
    for(int i = 0; i < (1 << LOG_TABLE_SIZE); i++) {
        // EXP2_FRAC_LUT[i] = uint64_t(std::pow(2, i / 256.0) * 4294967296);
        EXP2_FRAC_LUT[i] = int64_t(std::pow(2, i * (-1.0f) / (1 << LOG_TABLE_SIZE)) * x2_19);
    }
}
// -- end of softmax_q19 -----------------------------

// -- start of silu_q25 -----------------------------
static const int64_t LOG2E_Q25 = 48408813; // round(log2(e)*2^25)
static const int64_t exp2_25 = 33554432; // 1 << 25;
static const int64_t exp2_50 = 1125899906842624; // 1 << 50;

extern "C" __global__ void silu_kernel_q25(
    int64_t* R,  // [B, S, Dim]
    int64_t* C,  // [B, S, Dim]
    int64_t LOG2E_Q25, int64_t* EXP2_FRAC_LUT_Q25,
    int Bsz, int S, int Dim)
{
    int b = blockIdx.x;
    int s = blockIdx.y;
    int d = blockIdx.z * blockDim.x + threadIdx.x;

    if (d >= Dim) return;

    int idx = (b * S + s) * Dim + d;
    int64_t r = R[idx];

    // 饱和区裁剪（可调阈值 64）
    const int64_t LIM = (int64_t)64 << 25;
    if (r >= LIM) // σ≈1 -> SiLU ~= x
    {
        C[idx] = r;
        return;
    }
    if (r <= -LIM) // σ≈0 -> SiLU ~= 0
    {
        C[idx] = 0;
        return;
    }

    // y = - x * log2(e) / 2^25   (Q25)
    int64_t y = -int64_t((__int128_t(r) * __int128_t(LOG2E_Q25)) >> 25);

    // u ≈ 2^y = e^{-x} = 2^k * 2^f  (Q25)
    int64_t u = 0;
    int64_t k = y >> 25;                  // 整数部分
    int64_t f = y & 0x1FFFFFF;         // 小数部分 (Q25)
    int64_t t = EXP2_FRAC_LUT_Q25[f >> (25 - LOG_TABLE_SIZE)];             // 2^(frac) in Q25
    if (k > -63)
        u = (k < 0) ? (t >> (-k)) : (t << k);             // 一般 k<=0

    // σ = 1 / (1 + u)   (Q25)
    int64_t q = exp2_50 / (exp2_25 + u);     // Q25

    // SiLU = x * σ     : (Q25 * Q25) >> 25 → Q25
    C[idx] = ((r * q) >> 25);
}

extern "C" void silu_q25(int64_t* R, int64_t* C, int64_t* EXP2_FRAC_LUT_25,
    int Bsz, int S, int Dim)
{
    dim3 grid(Bsz, S, (Dim + 255) / 256);
    dim3 block(256, 1, 1);

    silu_kernel_q25<<<grid, block>>>(
        R, C, LOG2E_Q25, EXP2_FRAC_LUT_25,
        Bsz, S, Dim);
}

extern "C" void silu_init_q25(int64_t* EXP2_FRAC_LUT)
{
    int tableSize = 1 << LOG_TABLE_SIZE;
    for(int i = 0; i < tableSize; i++) {
        // EXP2_FRAC_LUT[i] = uint64_t(std::pow(2, i / 1024.0) * 2^25);
        EXP2_FRAC_LUT[i] = int64_t(std::pow(2, i * 1.0f / tableSize) * exp2_25);
    }
}

extern "C" __global__ void sigmoid_kernel_q25(
    int64_t* R,  // [B, S, Dim]
    int64_t* C,  // [B, S, Dim]
    int64_t LOG2E_Q25, int64_t* EXP2_FRAC_LUT_Q25,
    int Bsz, int S, int Dim)
{
    int b = blockIdx.x;
    int s = blockIdx.y;
    int d = blockIdx.z * blockDim.x + threadIdx.x;

    if (d >= Dim) return;

    int idx = (b * S + s) * Dim + d;
    int64_t r = R[idx];

    // 饱和区裁剪（可调阈值 64）
    const int64_t LIM = (int64_t)64 << 25;
    if (r >= LIM) // σ≈1 -> SiLU ~= x
    {
        C[idx] = r;
        return;
    }
    if (r <= -LIM) // σ≈0 -> SiLU ~= 0
    {
        C[idx] = 0;
        return;
    }

    // y = - x * log2(e) / 2^25   (Q25)
    int64_t y = -int64_t((__int128_t(r) * __int128_t(LOG2E_Q25)) >> 25);

    // u ≈ 2^y = e^{-x} = 2^k * 2^f  (Q25)
    int64_t u = 0;
    int64_t k = y >> 25;                  // 整数部分
    int64_t f = y & 0x1FFFFFF;         // 小数部分 (Q25)
    int64_t t = EXP2_FRAC_LUT_Q25[f >> (25 - LOG_TABLE_SIZE)];             // 2^(frac) in Q25
    if (k > -63)
        u = (k < 0) ? (t >> (-k)) : (t << k);             // 一般 k<=0

    // σ = 1 / (1 + u)   (Q25)
    C[idx] = exp2_50 / (exp2_25 + u);
}

extern "C" void sigmoid_q25(int64_t* R, int64_t* C, int64_t* EXP2_FRAC_LUT_25,
    int Bsz, int S, int Dim)
{
    dim3 grid(Bsz, S, (Dim + 255) / 256);
    dim3 block(256, 1, 1);

    sigmoid_kernel_q25<<<grid, block>>>(
        R, C, LOG2E_Q25, EXP2_FRAC_LUT_25,
        Bsz, S, Dim);
}

// -- end of silu_q25 -----------------------------


// -- start of silu_q23 -----------------------------
static const int64_t LOG2E_Q23 = 12102203; // round(log2(e)*2^23)
static const int64_t exp2_23 = 8388608; // 1 << 23;
static const int64_t exp2_46 = 70368744177664; // 1 << 46;

extern "C" __global__ void silu_kernel_q23(
    int64_t* R,  // [B, S, Dim]
    int64_t* C,  // [B, S, Dim]
    int64_t LOG2E_Q23, int64_t* EXP2_FRAC_LUT_Q23,
    int Bsz, int S, int Dim)
{
    int b = blockIdx.x;
    int s = blockIdx.y;
    int d = blockIdx.z * blockDim.x + threadIdx.x;

    if (d >= Dim) return;

    int idx = (b * S + s) * Dim + d;
    int64_t r = R[idx];

    // 饱和区裁剪（可调阈值 64）
    const int64_t LIM = (int64_t)64 << 23;
    if (r >= LIM) // σ≈1 -> SiLU ~= x
    {
        C[idx] = r;
        return;
    }
    if (r <= -LIM) // σ≈0 -> SiLU ~= 0
    {
        C[idx] = 0;
        return;
    }

    // y = - x * log2(e) / 2^23   (Q23)
    int64_t y = int64_t((__int128_t(-r) * __int128_t(LOG2E_Q23)) >> 23);

    // u ≈ 2^y = e^{-x} = 2^k * 2^f  (Q23)
    int64_t u = 0;
    int64_t k = y >> 23;                  // 整数部分
    int64_t f = y & 0x7FFFFF;         // 小数部分 (Q23)
    int64_t t = EXP2_FRAC_LUT_Q23[f >> (23 - LOG_TABLE_SIZE)];             // 2^(frac) in Q23
    if (k > -63)
        u = (k < 0) ? (t >> (-k)) : (t << k);             // 一般 k<=0

    // σ = 1 / (1 + u)   (Q23)
    int64_t q = exp2_46 / (exp2_23 + u);     // Q23

    // SiLU = x * σ     : (Q23 * Q23) >> 23 → Q23
    C[idx] = ((r * q) >> 23);
}

extern "C" void silu_q23(int64_t* R, int64_t* C, int64_t* EXP2_FRAC_LUT_23,
    int Bsz, int S, int Dim)
{
    dim3 grid(Bsz, S, (Dim + 255) / 256);
    dim3 block(256, 1, 1);

    silu_kernel_q23<<<grid, block>>>(
        R, C, LOG2E_Q23, EXP2_FRAC_LUT_23,
        Bsz, S, Dim);
}

extern "C" void silu_init_q23(int64_t* EXP2_FRAC_LUT)
{
    int tableSize = 1 << LOG_TABLE_SIZE;
    for(int i = 0; i < tableSize; i++) {
        // EXP2_FRAC_LUT[i] = uint64_t(std::pow(2, i / 1024.0) * 2^23);
        EXP2_FRAC_LUT[i] = int64_t(std::pow(2, i * 1.0f / tableSize) * exp2_23);
    }
}

extern "C" __global__ void sigmoid_kernel_q23(
    int64_t* R,  // [B, S, Dim]
    int64_t* C,  // [B, S, Dim]
    int64_t LOG2E_Q23, int64_t* EXP2_FRAC_LUT_Q23,
    int Bsz, int S, int Dim)
{
    int b = blockIdx.x;
    int s = blockIdx.y;
    int d = blockIdx.z * blockDim.x + threadIdx.x;

    if (d >= Dim) return;

    int idx = (b * S + s) * Dim + d;
    int64_t r = R[idx];

    // 饱和区裁剪（可调阈值 64）
    const int64_t LIM = (int64_t)64 << 23;
    if (r >= LIM) // σ≈1 -> SiLU ~= x
    {
        printf("r: %ld >= LIM", r);
        C[idx] = r;
        return;
    }
    if (r <= -LIM) // σ≈0 -> SiLU ~= 0
    {
        printf("r: %ld <= -LIM", r);
        C[idx] = 0;
        return;
    }

    // y = - x * log2(e) / 2^23   (Q23)
    int64_t y = int64_t((__int128_t(-r) * __int128_t(LOG2E_Q23)) >> 23);

    // u ≈ 2^y = e^{-x} = 2^k * 2^f  (Q23)
    int64_t u = 0;
    int64_t k = y >> 23;                  // 整数部分
    int64_t f = y & 0x7FFFFF;         // 小数部分 (Q23)
    int64_t t = EXP2_FRAC_LUT_Q23[f >> (23 - LOG_TABLE_SIZE)];             // 2^(frac) in Q23
    if (k > -63)
        u = (k < 0) ? (t >> (-k)) : (t << k);             // 一般 k<=0

    // if(s == 0 && d == 4)
    // {
    //     printf("s: %d, d: %d, x: %ld, y: %ld, k: %ld, f: %ld, t: %ld, u: %ld\n", s, d, r, y, k, f, t, u);
    // }

    // σ = 1 / (1 + u)   (Q23)
    C[idx] = exp2_46 / (exp2_23 + u);
}

extern "C" void sigmoid_q23(int64_t* R, int64_t* C, int64_t* EXP2_FRAC_LUT_23,
    int Bsz, int S, int Dim)
{
    dim3 grid(Bsz, S, (Dim + 255) / 256);
    dim3 block(256, 1, 1);

    sigmoid_kernel_q23<<<grid, block>>>(
        R, C, LOG2E_Q23, EXP2_FRAC_LUT_23,
        Bsz, S, Dim);
}

// -- end of silu_q23 -----------------------------