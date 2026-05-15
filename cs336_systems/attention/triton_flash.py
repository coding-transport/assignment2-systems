# _*_ coding : UTF-8 _*_
# @Time : 2026/5/12 17:05
# @Author : Yif Wang
# @file : triton_flash

import torch
import triton
import triton.language as tl


@triton.jit()
def _flash_attn_fwd_kernel(
    Q, K, V, Out, L,
    stride_qb, stride_qn, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_on, stride_od,
    B, N,
    D: tl.constexpr,          # 修改 1: 声明为 constexpr
    IS_CAUSAL: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    batch_id = tl.program_id(0)

    # 定位基础指针
    q_ptr = Q + batch_id * stride_qb
    k_ptr = K + batch_id * stride_kb
    v_ptr = V + batch_id * stride_vb
    o_ptr = Out + batch_id * stride_ob

    cols_d = tl.arange(0, D)
    # 缩放因子
    qk_scale = 1.0 / (D ** 0.5)

    # 外层循环遍历 Q 块
    for start_q in range(0, N, BLOCK_N):
        rm = start_q + tl.arange(0, BLOCK_N)

        # 初始化当前 Q 块的统计量
        m_i = tl.zeros([BLOCK_N], dtype=tl.float32) - float('inf')
        l_i = tl.zeros([BLOCK_N], dtype=tl.float32)
        acc = tl.zeros([BLOCK_N, D], dtype=tl.float32)

        # 加载 Qi 块并预缩放
        qi = tl.load(q_ptr + rm[:, None] * stride_qn + cols_d[None, :] * stride_qd, mask=rm[:, None] < N)
        qi = (qi * qk_scale).to(tl.float16)

        # 内层循环遍历 KV 块
        for start_k in range(0, N, BLOCK_N):
            rn = start_k + tl.arange(0, BLOCK_N)

            # --- 掩码机制优化 ---
            # 如果是因果模式且当前 K 块完全在 Q 块之后，直接跳过整个 K 块
            if IS_CAUSAL and start_k > start_q + BLOCK_N - 1:
                break

            ki = tl.load(k_ptr + rn[None, :] * stride_kn + cols_d[:, None] * stride_kd, mask=rn[None, :] < N)
            vi = tl.load(v_ptr + rn[:, None] * stride_vn + cols_d[None, :] * stride_vd, mask=rn[:, None] < N)

            # 计算 S = QK^T
            sij = tl.dot(qi, ki.to(tl.float16))

            # --- 应用因果掩码 ---
            if IS_CAUSAL:
                # 比较全局索引 rm (Q) 和 rn (K)
                mask = rm[:, None] >= rn[None, :]
                sij = tl.where(mask, sij, float("-inf"))

            # Online Softmax 逻辑
            m_ij = tl.max(sij, 1)
            p = tl.exp(sij - m_ij[:, None])
            l_ij = tl.sum(p, 1)

            m_next = tl.maximum(m_i, m_ij)
            alpha = tl.exp(m_i - m_next)
            beta = tl.exp(m_ij - m_next)

            l_i = l_i * alpha + l_ij * beta
            acc = acc * alpha[:, None] + tl.dot(p.to(tl.float16), vi.to(tl.float16)) * beta[:, None]
            m_i = m_next

        # 写回结果
        tl.store(o_ptr + rm[:, None] * stride_on + cols_d[None, :] * stride_od,
                 (acc / l_i[:, None]).to(tl.float16), mask=rm[:, None] < N)

        # 写回 LSE
        tl.store(L + batch_id * N + rm, m_i + tl.log(l_i), mask=rm < N)


class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        # 确保输入是连续的或者符合 stride 预期
        B, N, D = q.shape
        BLOCK_N = 128

        out = torch.empty_like(q)
        lse = torch.empty((B, N), device=q.device, dtype=torch.float32)

        grid = (B,)

        _flash_attn_fwd_kernel[grid](
            q, k, v, out, lse,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            B, N, D,
            IS_CAUSAL=is_causal,
            BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = is_causal
        return out