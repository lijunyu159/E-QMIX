# -*-coding:utf-8-*-
import torch
import numpy as np
import torch.nn.functional as F

'''
Loss function of VAE with options to schedule the beta coefficient
'''


class CVAE_loss(object):
    def __init__(self, config, schedule_kl_method=None):
        assert schedule_kl_method in ['constant', 'linear', 'logistic', 'cyclical']
        # self.batch_size = config.pretext['batch_size']
        self.max_seq_len = config.pretext['num_steps']

        # whether to schedule to weight of kl loss to prevent KL vanishing problem
        self.schedule_kl_method = schedule_kl_method  # linear, logistic, or None

        if self.schedule_kl_method == 'linear':
            self.x0 = 500
        elif self.schedule_kl_method == 'logistic':
            self.x0 = 300
        elif self.schedule_kl_method == 'cyclical':
            self.x0 = 250
            self.cur_period_num = 0
            self.period_len = config.pretext['epoch_num'] // 4
            self.total_period_num = config.pretext['epoch_num'] // self.period_len + 1
            self.step_in_period = 0.
            # initialize all betas within a period
            self.betas = np.zeros(self.period_len)
            ratio = 0.8
            # first half
            self.betas[:int(self.period_len * ratio)] = 0.02 / (
                        1 + np.exp(-0.05 * (np.arange(int(self.period_len * ratio)) - self.x0)))
            # second half
            self.betas[int(self.period_len * ratio):] = self.betas[int(self.period_len * ratio - 1)]

        # when schedule_kl_weight() is called for the first time, step_counter will be 0
        self.step_counter = -1.

    '''
    Given a list of sequence lengths, create a mask to indicate which indices are padded
    e.x. Input: [3, 1, 4], max_human_num = 5
    Output: [[1, 1, 1, 0, 0], [1, 0, 0, 0, 0], [1, 1, 1, 1, 0]]
    '''

    def create_bce_mask(self, each_seq_len, N):
        B = len(each_seq_len)
        # why +1: use a sentinel in the end to handle the case when each_seq_len = 20
        mask = torch.zeros(B, max(each_seq_len) + 1).cuda()  # [1024, 21]
        mask[torch.arange(B), each_seq_len] = 1.
        mask = torch.logical_not(mask.cumsum(dim=1))
        # remove the sentinel
        mask = mask[:, :-1]  # [1024, 20]
        mask = mask.unsqueeze(-1).expand(B, int(max(each_seq_len)), N)
        return mask

    '''
    return the current value of beta (weight of the KL loss)
    '''

    def get_kl_weight(self):
        if self.schedule_kl_method == 'linear':
            return min(1., self.step_counter / self.x0)
        elif self.schedule_kl_method == 'logistic':
            # return 0.00001 / (1 + np.exp(-0.05 * (self.step_counter - self.x0)))
            return 1.0 / (1 + np.exp(-0.05 * (self.step_counter - self.x0)))
        elif self.schedule_kl_method == 'cyclical':
            self.step_in_period = self.step_counter % self.period_len
            # print('step counter', self.step_counter, 'step_in_period', self.step_in_period, 'cure period num', self.cur_period_num)
            return self.betas[int(self.step_in_period)]

        else:
            return 5e-7  # beta <1 更重视重构，z 更愿意携带细节; beta > 1,更强约束，z 更“省信息”，更偏向抓大结构/关键因素，细节会被牺牲

    '''
    update the step_counter for scheduling
    '''

    def schedule_kl_weight(self):
        if self.schedule_kl_method == 'cyclical':
            if self.step_counter % self.period_len == 0 and self.step_counter > 0:
                self.cur_period_num = self.cur_period_num + 1
        self.step_counter = self.step_counter + 1

    '''
    calculate the VAE loss = beta * KL loss + reconstruction loss (MSE loss in our case)

    If seq_len is not None, create a mask to mask out padded sequences from contributing to the loss
    If train = True, schedule KL loss weight; else the weight is 1
    '''

    def forward(self, true_act, pred_act, z_mean, z_log_var,
                each_seq_len=None, veh_mask=None, train=True):

        B = true_act.size(0)
        T = pred_act.shape[1]
        N = pred_act.shape[2]
        pred_act = pred_act.reshape(B, pred_act.shape[1], -1, pred_act.shape[3])  # -> [B,T,N,F]

        # -------------------------
        # 1) 逐项 MSE（不做 reduction）
        # -------------------------
        # mse = F.mse_loss(pred_act, true_act, reduction="none")  # [B,T,N,F]
        ce = F.cross_entropy(pred_act.reshape(-1, pred_act.shape[-1]),
                             true_act.reshape(-1),
                             reduction='none').reshape(B, T, N)
        # per_btn = mse.mean(dim=-1)  # [B,T,N]（对 F 取均值）

        # -------------------------
        # 2) 时间掩码：from each_seq_len -> [B,T] 或 [B,T,N]
        # -------------------------
        if each_seq_len is not None:
            time_mask = self.create_bce_mask(each_seq_len, N)  # 你这里建议传 T（时间长度）
            time_mask = time_mask.to(ce.device).bool()

            # 兼容两种常见输出：
            # (a) [B, T]
            if time_mask.dim() == 2 and time_mask.shape == (B, T):
                time_mask = time_mask[:, :, None].expand(B, T, N)  # -> [B,T,N]
            # (b) [B, T, N]
            elif time_mask.dim() == 3 and time_mask.shape == (B, T, N):
                pass
            else:
                raise RuntimeError(f"Unexpected time_mask shape: {time_mask.shape}, expect (B,T) or (B,T,N)")
        else:
            time_mask = torch.ones((B, T, N), device=ce.device, dtype=torch.bool)

        # -------------------------
        # 3) 车辆掩码：veh_mask -> [B,T,N]
        # -------------------------
        if veh_mask is not None:
            veh_mask = veh_mask.to(ce.device).bool()
            if veh_mask.dim() == 4 and veh_mask.size(-1) == 1:
                veh_mask = veh_mask.squeeze(-1)  # [B,T,N,1] -> [B,T,N]
            if veh_mask.shape != (B, T, N):
                raise RuntimeError(f"veh_mask shape {veh_mask.shape} != (B,T,N)=({B},{T},{N})")
        else:
            veh_mask = torch.ones((B, T, N), device=ce.device, dtype=torch.bool)

        # -------------------------
        # 4) 合并 mask，并做 masked mean
        # -------------------------
        mask = time_mask & veh_mask  # [B,T,N]
        mask_f = mask.float()

        # BCE_1 = (per_btn * time_mask).sum() / time_mask.sum().clamp_min(1.0)
        BCE = (ce * mask_f).sum() / mask_f.sum().clamp_min(1.0)
        BCE = BCE * 10

        # -------------------------
        # 5) KL 不变
        # -------------------------
        kld_per_dim = -0.5 * (1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        # kld_per_sample = kld_per_dim.sum(dim=1)  # [B]
        # KLD = kld_per_sample.mean()
        # kld_bn = kld_per_dim.sum(dim=-1)  # [B,N]

        # m = veh_mask.float()  # [B,N]
        # KLD = (kld_bn * m).sum() / m.sum().clamp_min(1.0)
        # KLD = kld_bn.mean()
        if kld_per_dim.dim() == 4:
            # [B,T,N,Z] -> [B,T,N]
            kld_btn = kld_per_dim.sum(dim=-1)
            KLD = (kld_btn * mask_f).sum() / mask_f.sum().clamp_min(1.0)
        else:
            # [B,N,Z] -> [B,N]；按“车是否在该轨迹出现过”做 mask
            kld_bn = kld_per_dim.sum(dim=-1)
            veh_valid = mask.any(dim=1).float()  # [B,N]
            KLD = (kld_bn * veh_valid).sum() / veh_valid.sum().clamp_min(1.0)
        self.beta = 5e-7
        return BCE + KLD * self.beta, BCE, KLD * self.beta, KLD

    # def forward(self, true_act, pred_act, z_mean, z_log_var, each_seq_len=None, veh_mask=None, train=True):
    #     act_idx = true_act.long().squeeze(-1)
    #     B = act_idx.shape[0]
    #     pred_act = pred_act.reshape(B, pred_act.shape[1], -1, pred_act.shape[3])
    #     # logits = pred_act
    #     if each_seq_len:  # mask out the padded sequences from loss calculation
    #         mask = self.create_bce_mask(each_seq_len, act_idx.shape[2])  # (1024, T)
    #         # BCE = F.gaussian_nll_loss(act_mean, true_act, act_var)
    #         BCE = F.mse_loss(pred_act[mask], true_act[mask]) * 10
    #     else:
    #         BCE = F.mse_loss(pred_act, true_act)
    #     # KLD = -torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    #     kld_per_dim = -0.5 * (1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    #     kld_per_sample = kld_per_dim.sum(dim=1)  # [B]
    #     KLD = kld_per_sample.mean()
    #     # if train:
    #     #     self.beta = self.get_kl_weight()
    #     # else:
    #     self.beta = 1.
    #     return BCE + KLD * self.beta, BCE, KLD * self.beta, KLD
    # def forward(self, true_act, pred_act, z_mean, z_log_var, each_seq_len=None, train=True):
    #     # true_act: (B, T, N, 1)  ->  (B, T, N)
    #     act_idx = true_act.long().squeeze(-1)  # [B, T, N]
    #     B = act_idx.shape[0]
    #     pred_act = pred_act.reshape(B,  pred_act.shape[1],-1, pred_act.shape[2])
    #     # pred_act: (B, T, N, n_actions)
    #     logits = pred_act
    #
    #     if each_seq_len is not None:
    #         # 根据每个序列长度做 mask，避免 padding 位置参与 loss
    #         mask = self.create_bce_mask(each_seq_len, act_idx.shape[2])  # 形状需要你自己保证对齐 [B, T, N] 或 [B, T]
    #         # 展开维度
    #         mask = mask.bool().reshape(-1)  # [B*T*N]
    #         logits_flat = logits.reshape(-1, logits.size(-1))[mask]  # [有效step数, n_actions]
    #         target_flat = act_idx.reshape(-1)[mask]  # [有效step数]
    #     else:
    #         logits_flat = logits.reshape(-1, logits.size(-1))  # [B*T*N, n_actions]
    #         target_flat = act_idx.reshape(-1)  # [B*T*N]
    #
    #     CE = F.cross_entropy(logits_flat, target_flat, reduction='mean')
    #
    #     # KLD 不变
    #     KLD = -torch.mean(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
    #
    #     # if train:
    #     #     self.beta = self.get_kl_weight()
    #     # else:
    #     self.beta = 1.0
    #
    #     loss = CE + self.beta * KLD
    #     return loss, CE, self.beta * KLD, KLD

    # for debugging only
    # def forward(self):
    #     return self.get_kl_weight()

