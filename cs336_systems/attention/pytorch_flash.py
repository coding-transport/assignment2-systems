# _*_ coding : UTF-8 _*_
# @Time : 2026/5/12 17:05
# @Author : Yif Wang
# @file : pytorch_flash
import torch

class MyFlashAttnAutogradFunctionClass(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v):
        B, H, N, D = q.shape
        O = torch.zeros_like(q)
        m = torch.full((B, H, N, 1), float('-inf'), device=q.device)
        L = torch.zeros((B, H, N, 1), device=q.device)
        Br = 64
        Bc = 64
        inv = torch.sqrt(D)
        for j in range(0, N, Bc):
            K_j = k[:, :, j:j + Bc, :]  # 切片获取 K 块
            V_j = v[:, :, j:j + Bc, :]  # 切片获取 V 块

            # 内层循环遍历 Q, O (行块)
            for i in range(0, N, Br):
                # 获取当前行块对应的 Q, O 以及统计量
                Q_i = q[:, :, i:i + Br, :]
                m_old = m[:, :, i:i + Br, :]  # (B, H, Br, 1)
                L_old = L[:, :, i:i + Br, :]  # (B, H, Br, 1)
                O_old = O[:, :, i:i + Br, :]
                # --- 执行计算逻辑 ---
                # 算出新的 O_i, m_i, L_i
                score = Q_i @ K_j.transpose(-2, -1) * inv
                m_block = torch.max(score, dim=-1, keepdim=True).values  # (B, H, Br, 1)
                # 计算当前块的指数和
                p_ij = torch.exp(score - m_block)  # (B, H, Br, Bc)
                L_block = torch.sum(p_ij, dim=-1, keepdim=True)  # (B, H, Br, 1)
                max_b = torch.maximum(m_block, m_old)
                alpha = torch.exp(m_old - max_b)
                beta = torch.exp(m_block - max_b)
                L[:, :, i:i + Br, :] = L_old*alpha + L_block*beta
                O_new = (O_old*alpha*L_old + (p_ij @ V_j)*beta)/L[:, :, i:i + Br, :]
                O[:, :, i:i + Br, :] = O_new
                m[:, :, i:i + Br, :] = max_b

            # 将统计量保存到 ctx 供 backward 使用
        ctx.save_for_backward(q, k, v, O, m, L)
        return O