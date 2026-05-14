import torch


class MyFlashAttnAutogradFunctionClass(torch.autograd.Function):

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False, sm_scale=None):
        B, N, D = q.shape
        if sm_scale is None:
            sm_scale = 1.0 / (D ** 0.5)

        # 1. 核心修正：初始化 m 和 L 为 (B, N, 1)
        # 这样在计算 (B, Br, D) / (B, Br, 1) 时，维度能完美自动对齐，不会出现广播歧义
        O = torch.zeros_like(q)
        m = torch.full((B, N, 1), float('-inf'), device=q.device)
        L = torch.zeros((B, N, 1), device=q.device)

        Br, Bc = 64, 64
        Tr, Tc = (N + Br - 1) // Br, (N + Bc - 1) // Bc

        for j in range(Tc):
            start_j, end_j = j * Bc, min((j + 1) * Bc, N)
            kj = k[:, start_j:end_j, :]
            vj = v[:, start_j:end_j, :]

            for i in range(Tr):
                start_i, end_i = i * Br, min((i + 1) * Br, N)

                # 因果掩码剪枝
                if is_causal and (start_j >= end_i):
                    continue

                qi = q[:, start_i:end_i, :]
                oi = O[:, start_i:end_i, :]
                mi = m[:, start_i:end_i, :]
                li = L[:, start_i:end_i, :]

                # 计算注意力分数
                S_ij = torch.matmul(qi, kj.transpose(-2, -1)) * sm_scale
                print(S_ij, sm_scale)
                # 2. 修正：因果掩码必须使用全局索引
                if is_causal:
                    rows = torch.arange(start_i, end_i, device=q.device).view(-1, 1)
                    cols = torch.arange(start_j, end_j, device=q.device).view(1, -1)
                    S_ij = S_ij.masked_fill(rows < cols, -1e9)

                # 计算当前块统计量 (B, Br, 1)
                m_ij, _ = torch.max(S_ij, dim=-1, keepdim=True)
                P_ij = torch.exp(S_ij - m_ij)
                l_ij = torch.sum(P_ij, dim=-1, keepdim=True)

                # --- 3. 核心修正：在线 Softmax 更新公式 ---
                m_new = torch.maximum(mi, m_ij)

                # 计算重标定因子
                alpha = torch.exp(mi - m_new)
                beta = torch.exp(m_ij - m_new)

                li_new = alpha * li + beta * l_ij

                term1 = (alpha * li) * oi
                term2 = beta * torch.matmul(P_ij, vj)
                oi_new = (term1 + term2) / li_new

                # 写回全局
                O[:, start_i:end_i, :] = oi_new
                m[:, start_i:end_i, :] = m_new
                L[:, start_i:end_i, :] = li_new

        # --- 4. 关键修正：满足测试脚本对 saved_tensors 的断言 ---
        # 测试脚本要求找到且仅找到一个形状为 (B, N) 的 tensor
        lse = m + torch.log(L)

        # 必须执行 .squeeze(-1) 否则形状是 (B, N, 1)，测试会报错 found 0 或者形状不匹配
        ctx.save_for_backward(q, k, v, O, lse.squeeze(-1))

        ctx.is_causal = is_causal
        ctx.sm_scale = sm_scale

        return O

    @staticmethod
    def backward(ctx, grad_output):
        # 1. 获取正向传播保存的张量
        # 注意：根据之前的修正，这里存的是 q, k, v, O, lse (B, N)
        q, k, v, O, lse = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        is_causal = ctx.is_causal
        B, N, D = q.shape

        # 将 lse 展回 (B, N, 1) 方便广播计算
        lse = lse.unsqueeze(-1)

        # 2. 预计算辅助梯度项 Di = rowsum(grad_output * O)
        Di = torch.sum(grad_output * O, dim=-1, keepdim=True)  # (B, N, 1)

        dq = torch.zeros_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        # 分块大小（通常反向传播也会采用分块以节省显存）
        Br, Bc = 64, 64
        Tr, Tc = (N + Br - 1) // Br, (N + Bc - 1) // Bc

        # 外层循环遍历 Key/Value 块 (j)
        for j in range(Tc):
            start_j, end_j = j * Bc, min((j + 1) * Bc, N)
            kj = k[:, start_j:end_j, :]
            vj = v[:, start_j:end_j, :]
            dkj = torch.zeros_like(kj)
            dvj = torch.zeros_like(vj)

            # 内层循环遍历 Query 块 (i)
            for i in range(Tr):
                start_i, end_i = i * Br, min((i + 1) * Br, N)

                # 因果掩码剪枝
                if is_causal and (start_j > end_i - 1):
                    continue

                qi = q[:, start_i:end_i, :]
                dO_i = grad_output[:, start_i:end_i, :]
                LSE_i = lse[:, start_i:end_i, :]
                Di_i = Di[:, start_i:end_i, :]

                # --- 重新计算 S_ij 并应用掩码 ---
                S_ij = torch.matmul(qi, kj.transpose(-2, -1)) * sm_scale
                if is_causal:
                    rows = torch.arange(start_i, end_i, device=q.device).view(-1, 1)
                    cols = torch.arange(start_j, end_j, device=q.device).view(1, -1)
                    S_ij = S_ij.masked_fill(rows < cols, float('-inf'))

                # --- 计算 P_ij (利用正向传播的 LSE 保证数值稳定) ---
                P_ij = torch.exp(S_ij - LSE_i)  # (B, Br, Bc)

                # --- 计算各部分梯度 ---
                # 1. dvj
                dvj += torch.matmul(P_ij.transpose(-2, -1), dO_i)

                # 2. dP_ij 和 dS_ij
                dP_ij = torch.matmul(dO_i, vj.transpose(-2, -1))
                dS_ij = P_ij * (dP_ij - Di_i)  # (B, Br, Bc)

                # 3. dqi 和 dkj
                # 注意需要乘上 sm_scale
                dq[:, start_i:end_i, :] += torch.matmul(dS_ij, kj) * sm_scale
                dkj += torch.matmul(dS_ij.transpose(-2, -1), qi) * sm_scale

            # 写回全局梯度
            dk[:, start_j:end_j, :] = dkj
            dv[:, start_j:end_j, :] = dvj

        # 返回值必须匹配 forward 的参数个数: q, k, v, sm_scale, is_causal
        # 因为 sm_scale 和 is_causal 不需要梯度，返回 None
        return dq, dk, dv, None, None