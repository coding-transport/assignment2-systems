# _*_ coding : UTF-8 _*_
# @Time : 2026/5/12 17:05
# @Author : Yif Wang
# @file : triton_flash

import torch
import triton
import triton.language as tl


@triton.jit
def _flash_attn_fwd_kernel(
        Q, K, V, Out, L,
        stride_qb, stride_qn, stride_qd,
        stride_kb, stride_kn, stride_kd,
        stride_vb, stride_vn, stride_vd,
        stride_ob, stride_on, stride_od,
        B, N,
        D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    batch_id = tl.program_id(0)

    # 定位基础指针
    q_ptr = Q + batch_id * stride_qb
    k_ptr = K + batch_id * stride_kb
    v_ptr = V + batch_id * stride_vb
    o_ptr = Out + batch_id * stride_ob

    cols_d = tl.arange(0, D)
    qk_scale = 1.0 / (D ** 0.5)

    # 外层循环遍历 Q 块 (每次前进 BLOCK_M)
    for start_q in range(0, N, BLOCK_M):
        rm = start_q + tl.arange(0, BLOCK_M)

        # 初始化当前 Q 块的统计量
        m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float('inf')
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        acc = tl.zeros([BLOCK_M, D], dtype=tl.float32)

        # 加载 Qi 块并预缩放
        qi = tl.load(q_ptr + rm[:, None] * stride_qn + cols_d[None, :] * stride_qd, mask=rm[:, None] < N)
        qi = (qi * qk_scale).to(tl.float16)

        # 内层循环遍历 KV 块 (每次前进 BLOCK_N)
        for start_k in range(0, N, BLOCK_N):
            rn = start_k + tl.arange(0, BLOCK_N)

            # --- 安全掩码优化（完美避开 break 硬件不支持问题） ---
            if (not IS_CAUSAL) or (start_k <= start_q + BLOCK_M - 1):
                ki = tl.load(k_ptr + rn[None, :] * stride_kn + cols_d[:, None] * stride_kd, mask=rn[None, :] < N)
                vi = tl.load(v_ptr + rn[:, None] * stride_vn + cols_d[None, :] * stride_vd, mask=rn[:, None] < N)

                # 计算 S = QK^T
                sij = tl.dot(qi, ki.to(tl.float16))

                # --- 应用因果掩码 ---
                if IS_CAUSAL:
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


@triton.jit
def _flash_attn_bwd_kernel(
        Q, K, V, Out, LSE,
        DO, DQ, DK, DV,
        stride_qb, stride_qn, stride_qd,
        stride_kb, stride_kn, stride_kd,
        stride_vb, stride_vn, stride_vd,
        stride_ob, stride_on, stride_od,  # 修复：引入正规的 Out 步长
        stride_dob, stride_don, stride_dod,
        stride_dqb, stride_dqn, stride_dqd,
        stride_dkb, stride_dkn, stride_dkd,
        stride_dvb, stride_dvn, stride_dvd,
        B, N,
        D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
):
    batch_id = tl.program_id(0)

    # 计算当前 Batch 的准确指针偏移（基于传入的真实 stride）
    q_ptr = Q + batch_id * stride_qb
    k_ptr = K + batch_id * stride_kb
    v_ptr = V + batch_id * stride_vb
    o_ptr = Out + batch_id * stride_ob
    do_ptr = DO + batch_id * stride_dob

    dq_ptr = DQ + batch_id * stride_dqb
    dk_ptr = DK + batch_id * stride_dkb
    dv_ptr = DV + batch_id * stride_dvb

    lse_ptr = LSE + batch_id * N

    cols_d = tl.arange(0, D)
    qk_scale = 1.0 / (D ** 0.5)

    # 外层循环：遍历 KV 块 (计算 dK 和 dV)
    for start_n in range(0, N, BLOCK_N):
        rn = start_n + tl.arange(0, BLOCK_N)

        ki = tl.load(k_ptr + rn[:, None] * stride_kn + cols_d[None, :] * stride_kd, mask=rn[:, None] < N)
        vi = tl.load(v_ptr + rn[:, None] * stride_vn + cols_d[None, :] * stride_vd, mask=rn[:, None] < N)

        dki = tl.zeros([BLOCK_N, D], dtype=tl.float32)
        dvi = tl.zeros([BLOCK_N, D], dtype=tl.float32)

        # 内层循环：遍历 Q 块
        # 内层循环：遍历 Q 块
        for start_m in range(0, N, BLOCK_M):
            rm = start_m + tl.arange(0, BLOCK_M)

            # --- 掩码机制优化（安全修复版：用 if 包裹，绝不用 continue） ---
            if (not IS_CAUSAL) or (start_n <= start_m + BLOCK_M - 1):

                # ─── 把原本内层循环里所有的计算逻辑移到这里面 ───
                qi = tl.load(q_ptr + rm[:, None] * stride_qn + cols_d[None, :] * stride_qd, mask=rm[:, None] < N)
                doi = tl.load(do_ptr + rm[:, None] * stride_don + cols_d[None, :] * stride_dod, mask=rm[:, None] < N)
                oi = tl.load(o_ptr + rm[:, None] * stride_on + cols_d[None, :] * stride_od, mask=rm[:, None] < N)
                lse = tl.load(lse_ptr + rm, mask=rm < N)

                di = tl.sum(doi * oi, 1)

                sij = tl.dot(qi, tl.trans(ki)) * qk_scale

                if IS_CAUSAL:
                    mask = rm[:, None] >= rn[None, :]
                    sij = tl.where(mask, sij, float("-inf"))

                p = tl.exp(sij - lse[:, None])

                dvi += tl.dot(tl.trans(p.to(tl.float16)), doi.to(tl.float16))

                dp = tl.dot(doi.to(tl.float16), tl.trans(vi.to(tl.float16)))

                ds = p * (dp - di[:, None]) * qk_scale

                dki += tl.dot(tl.trans(ds.to(tl.float16)), qi.to(tl.float16))

                dqi = tl.dot(ds.to(tl.float16), ki.to(tl.float16))
                tl.atomic_add(dq_ptr + rm[:, None] * stride_dqn + cols_d[None, :] * stride_dqd,
                              dqi.to(tl.float16), mask=rm[:, None] < N)
        # 写回当前 KV 块对应的 dK 和 dV
        tl.store(dk_ptr + rn[:, None] * stride_dkn + cols_d[None, :] * stride_dkd, dki.to(tl.float16),
                 mask=rn[:, None] < N)
        tl.store(dv_ptr + rn[:, None] * stride_dvn + cols_d[None, :] * stride_dvd, dvi.to(tl.float16),
                 mask=rn[:, None] < N)


class TritonFlashAttention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        B, N, D = q.shape

        # 🎯 强行锁死 64 分块，完美卡进你的 99KB 硬件 Shared Memory
        BLOCK_M = 64
        BLOCK_N = 64

        out = torch.empty_like(q)
        lse = torch.empty((B, N), device=q.device, dtype=torch.float32)

        grid = (B, 1, 1)

        # 🚀 使用纯位置传参，显式控制 num_stages=2 避免编译器过度追求吞吐而导致显存溢出
        _flash_attn_fwd_kernel[grid](
            q, k, v, out, lse,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),
            B, N,
            D,
            is_causal,
            BLOCK_M,
            BLOCK_N,
            num_stages=2,
            num_warps=4,
        )

        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = is_causal
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse = ctx.saved_tensors
        is_causal = ctx.is_causal

        B, N, D = q.shape

        dq = torch.zeros_like(q, dtype=torch.float32)
        dk = torch.empty_like(k)
        dv = torch.empty_like(v)

        BLOCK_M = 64
        BLOCK_N = 64

        grid = (B, 1, 1)

        # 🚀 反向同样锁死位置参数，并设定安全流水线深度 num_stages=2
        _flash_attn_bwd_kernel[grid](
            q, k, v, out, lse,
            grad_out, dq, dk, dv,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            out.stride(0), out.stride(1), out.stride(2),  # 正确传入 Out 的 stride
            grad_out.stride(0), grad_out.stride(1), grad_out.stride(2),
            dq.stride(0), dq.stride(1), dq.stride(2),
            dk.stride(0), dk.stride(1), dk.stride(2),
            dv.stride(0), dv.stride(1), dv.stride(2),
            B, N,
            D,
            is_causal,
            BLOCK_M,
            BLOCK_N,
            num_stages=2,
            num_warps=4,
        )

        return dq.to(q.dtype), dk, dv, None