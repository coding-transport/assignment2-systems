import torch


class MyFlashAttnAutogradFunctionClass(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, sm_scale=None, is_causal=False):
        # 1. 基础维度获取
        B, N, D = q.shape
        if sm_scale is None:
            sm_scale = 1.0 / (D ** 0.5)

        # 初始化输出和辅助变量
        O = torch.zeros_like(q)
        m = torch.full((B, N), float('-inf'), device=q.device)  # 记录当前的 max
        L = torch.zeros((B, N), device=q.device)  # 记录当前的 exp 累加和

        # 分块大小（根据 GPU 显存建议设置，64 是常见选择）
        Br, Bc = 64, 64
        Tr, Tc = (N + Br - 1) // Br, (N + Bc - 1) // Bc

        # 2. 外层循环处理 Key 和 Value 的列分块
        for j in range(Tc):
            start_j, end_j = j * Bc, min((j + 1) * Bc, N)
            kj = k[:, start_j:end_j, :]
            vj = v[:, start_j:end_j, :]

            # 3. 内层循环处理 Query 的行分块
            for i in range(Tr):
                # 如果是因果掩码，且当前列分块的起点已经超过了行分块的终点，直接跳过
                if is_causal and (j * Bc > (i + 1) * Br - 1):
                    continue

                start_i, end_i = i * Br, min((i + 1) * Br, N)
                qi = q[:, start_i:end_i, :]
                oi = O[:, start_i:end_i, :]
                mi = m[:, start_i:end_i, :]
                li = L[:, start_i:end_i, :]

                # 计算注意力分数 S = Q * K.T * scale
                # (B, H, Br, D) @ (B, H, D, Bc) -> (B, H, Br, Bc)
                S_ij = torch.matmul(qi, kj.transpose(-2, -1)) * sm_scale

                # --- 4. 处理因官掩码 (Causal Masking) ---
                if is_causal:
                    # 创建局部掩码：行索引必须 >= 列索引
                    # 需要考虑全局索引：行 idx_i = start_i + row, 列 idx_j = start_j + col
                    rows = torch.arange(start_i, end_i, device=q.device).view(-1, 1)
                    cols = torch.arange(start_j, end_j, device=q.device).view(1, -1)
                    mask = rows >= cols
                    S_ij = S_ij.masked_fill(~mask, float('-inf'))

                # --- 5. Flash Attention 在线 Softmax 更新逻辑 ---
                m_ij, _ = torch.max(S_ij, dim=-1, keepdim=True)
                P_ij = torch.exp(S_ij - m_ij)
                l_ij = torch.sum(P_ij, dim=-1, keepdim=True)

                # 更新统计量
                m_new = torch.max(mi, m_ij)
                # 利用公式进行重标定更新
                li_new = torch.exp(mi - m_new) * li + torch.exp(m_ij - m_new) * l_ij

                # 更新输出 Oi
                # Oi = (Oi * exp(mi - m_new) * li + exp(m_ij - m_new) * P_ij * Vj) / li_new
                term1 = (torch.exp(mi - m_new) * li) * oi
                term2 = torch.exp(m_ij - m_new) * torch.matmul(P_ij, vj)
                oi = (term1 + term2) / li_new

                # 写回
                O[:, start_i:end_i, :] = oi
                m[:, start_i:end_i, :] = m_new
                L[:, start_i:end_i, :] = li_new

        # 保存上下文供反向传播
        ctx.save_for_backward(q, k, v, O, m, L)
        ctx.is_causal = is_causal

        return O

    @staticmethod
    def backward(ctx, grad_output):

        q, k, v, O, m, L = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        B, H, N, D = q.shape

        # 2. 预计算辅助梯度项 D = rowsum(grad_output * O)
        Di = torch.sum(grad_output * O, dim=-1, keepdim=True)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        Br, Bc = 64, 64

        # 3. 反向传播分块循环 (O(N) 逻辑)
        for i in range(0, N, Br):
            qi = q[:, :, i:i + Br, :]
            dOi = grad_output[:, :, i:i + Br, :]
            mi = m[:, :, i:i + Br, :]
            Li = L[:, :, i:i + Br, :]
            Di_row = Di[:, :, i:i + Br, :]

            dqi = torch.zeros_like(qi)

            for j in range(0, N, Bc):
                kj = k[:, :, j:j + Bc, :]
                vj = v[:, :, j:j + Bc, :]

                # 重计算局部 Softmax 概率
                score = (qi @ kj.transpose(-2, -1)) * sm_scale
                p_ij = torch.exp(score - mi) / Li

                # 计算梯度累加
                dv[:, :, j:j + Bc, :] += p_ij.transpose(-2, -1) @ dOi
                dp_ij = dOi @ vj.transpose(-2, -1)
                ds_ij = p_ij * (dp_ij - Di_row) * sm_scale

                dqi += ds_ij @ kj
                dk[:, :, j:j + Bc, :] += ds_ij.transpose(-2, -1) @ qi

            dq[:, :, i:i + Br, :] = dqi

        # 返回值必须对应 forward 的输入 q, k, v
        return dq, dk, dv