import copy

import torch
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
# from modules.mixers.vdn import VDNMixer
# from modules.mixers.qatten import QattenMixer
# from modules.mixers.conv_mix import ConvMixer
# from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num
import torch.optim.lr_scheduler as lr_scheduler

# ====== 引入VAE相关代码 ====== #
from src.pretext.pretext_models.cvae_model import *
from src.pretext.loss import *
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import torch.optim as optim
import os
# =========================== #

# ====== 引入RND相关代码 ====== #
from src.modules.layer.rnd import RND, VAE_RND
# =========================== #

# ====== running-std标准化 ====== #
class RunningMeanStd:
    def __init__(self, eps=1e-4, device="cuda:0"):
        self.mean = torch.zeros((), device=device)
        self.var = torch.ones((), device=device)
        self.count = torch.tensor(eps, device=device)

    @torch.no_grad()
    def update(self, x: torch.Tensor):
        x = x.detach().float().reshape(-1)
        x = x[torch.isfinite(x)]
        if x.numel() == 0:
            return

        batch_mean = x.mean()
        batch_var = x.var(unbiased=False)
        batch_count = torch.tensor(float(x.numel()), device="cuda:0")

        delta = batch_mean - self.mean
        tot_count = self.count + batch_count

        new_mean = self.mean + delta * batch_count / tot_count

        m_a = self.var * self.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + delta * delta * self.count * batch_count / tot_count
        new_var = M2 / tot_count

        self.mean = new_mean
        self.var = new_var
        self.count = tot_count

    @property
    def std(self):
        return torch.sqrt(self.var + 1e-8)

@torch.no_grad()
def normalize_rnd(raw: torch.Tensor, rms: RunningMeanStd, mask,
                  clip_after_norm: float):
    """
    raw: rnd error tensor
    mask: valid mask (same shape or broadcastable)
    returns normalized tensor with same shape as raw
    """
    x = raw.detach().float()

    # 更新 running stats：只用有效样本
    upd = x
    if mask is not None:
        upd = upd[mask.bool()]
    # 可选：防极端尖峰破坏 running std（建议保守一点）
    upd = upd.clamp(min=0.0, max=1e6)
    rms.update(upd)  # CPU 统计更省显存；若你想全 GPU 就去掉 .cpu()

    # 标准化（只除以 std；如需减均值可改成 (x - mean)/std）
    std = rms.std.to("cuda:0")
    x_norm = x / (std + 1e-8)

    # 标准化后 clip（更常用）
    if clip_after_norm is not None:
        x_norm = x_norm.clamp(max=clip_after_norm)

    return x_norm

# ====== 用分位数记录奖励 ====== #
def summarize_tensor(x, mask):
    # x: 任意 shape 的 tensor，已经是 loss/reward 的逐样本值
    x = x.detach().float()
    if mask is not None:
        mask = mask.detach().bool()
        x = x[mask]
    x = x.reshape(-1)
    # 防止空 tensor 或 NaN 影响
    if x.numel() == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"),
                "p99": float("nan"), "max": float("nan")}
    x = x[torch.isfinite(x)]
    if x.numel() == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"),
                "p99": float("nan"), "max": float("nan")}

    q = torch.quantile(x, torch.tensor([0.5, 0.9, 0.99], device=x.device))
    return {
        "mean": x.mean().item(),
        "p50": q[0].item(),
        "p90": q[1].item(),
        "p99": q[2].item(),
        "max": x.max().item(),
    }
# =========================== #

class NQLearnerLRDecay:
    def __init__(self, mac, scheme, logger, args):
        self.args = args
        self.mac = mac
        self.logger = logger
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')
        self.params = list(mac.parameters())

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        elif args.mixer == "conv":
            self.mixer = ConvMixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params,  lr=args.lr)
            self.scheduler_lr = lr_scheduler.ExponentialLR(self.optimiser, gamma=self.args.gamma_lr)
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_mac = copy.deepcopy(mac)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')

        # ====== 静态RND ====== #
        self.static_rnd = RND(args, scheme)
        self.static_rnd_optim = Adam(params=self.static_rnd.parameters(),
                                  lr=args.rnd_lr,
                                  weight_decay=getattr(args, "weight_decay", 0))
        # ====== 动态RND ====== #
        self.dynamic_rnd = VAE_RND(args, scheme)
        self.dynamic_rnd_optim = Adam(params=self.dynamic_rnd.parameters(),
                                      lr=args.rnd_lr,
                                      weight_decay=getattr(args, "weight_decay", 0))

        # ====== 训练VAE用到的参数 ====== #
        self.vae = CVAECoordinationPatterns({}, task='pretext_predict',
                                            decoder_base=self.args.pretext['cvae_decoder'],
                                            config=self.args)
        self.loss_func = CVAE_loss(config=self.args, schedule_kl_method='constant')
        self.vae_optim = optim.Adam(self.vae.parameters(),
                                   lr=float(self.args.pretext['lr']),
                                   weight_decay=float(1e-6))
        # self.vae.load_state_dict(torch.load("/home/heuristic_based_qmix_republic_safe_module_VAE"))
        # ============================= #

        self.rms_stat = RunningMeanStd(device="cuda:0")  # 或与你训练同 device
        self.rms_dyna = RunningMeanStd(device="cuda:0")

        # ====== 记录奖励的不同成分，目的是确定奖励前的权重 ====== #
        self.reward_log_path = './reward_terms.csv'
        self.reward_log_every = 10
        if not os.path.exists(self.reward_log_path):
            with open(self.reward_log_path, 'w', encoding='utf-8') as f:
                f.write("episode_num, reward_mean, "
                        "rnd_mean,"
                        "vae_mean\n")
        # ====== 初始化一个损失收集文件 ====== #
        save_dir = './'
        os.makedirs(save_dir, exist_ok=True)
        self.save_path_loss = os.path.join(save_dir, "vae_loss_stream.txt")
        # 先写表头
        with open(self.save_path_loss, "w", encoding="utf-8") as f:
            f.write("vae_loss\tvae_act_loss\tvae_kl_loss\tr_imm_loss\n")

    # ====== 导出函数 ====== #
    def get_static_rnd_state_cpu(self):
        # 注意：用 cpu() 方便 runner load（runner 可能在 cpu 或 gpu）
        return {k: v.detach().cpu() for k, v in self.static_rnd.state_dict().items()}

    def get_dynamic_rnd_state_cpu(self):
        return {k: v.detach().cpu() for k, v in self.dynamic_rnd.state_dict().items()}

    def get_vae_state_cpu(self):
        return {k: v.detach().cpu() for k, v in self.vae.state_dict().items()}

    # ====== 训练VAE ====== #
    def train_vae(self, batch):

        self.vae.cuda().train()
        cudnn.benchmark = True

        seq = batch["filled"][:,:-1].squeeze(-1)
        ep_lens = seq.sum(dim=1).long().tolist()

        rnn_hxs_encoder = {}
        rnn_hxs_encoder['rnn'] = torch.zeros(
            self.args.batch_size * self.args.n_agents,
            self.args.network['rnn_hidden_size'],
            device='cuda:0'
        )
        rnn_hxs_decoder = {}
        rnn_hxs_decoder['rnn'] = torch.zeros(
            self.args.batch_size * self.args.n_agents,
            self.args.network['rnn_hidden_size'],
            device='cuda:0'
        )

        # self.vae_optim.zero_grad()

        # state_dict = batch["obs"][:, :-1]
        actions = batch["actions"][:, :-1].squeeze(-1).long()
        # a_one_hot = F.one_hot(actions, num_classes=7).float().squeeze(-2)
        B = actions.shape[0]

        alive_veh = 1. * (th.sum(batch["obs"][:, :-1], dim=3) > 0).view(B, -1, self.args.n_agents)
        # alive_veh_mask = th.bmm(alive_veh.unsqueeze(2), alive_veh.unsqueeze(1))

        pred_a, z_mean, z_log_var, rnn_hxs_encoder, rnn_hxs_decoder, z = self.vae(
            batch["obs"][:,:-1],
            rnn_hxs_encoder,
            rnn_hxs_decoder,
            seq_len=ep_lens,
            veh_mask=alive_veh
        )

        vae_loss, act_loss, kl_loss, kl_loss_nonweight = self.loss_func.forward(
            actions, pred_a, z_mean, z_log_var, ep_lens, alive_veh
        )


        # vae_loss.backward()
        # self.vae_optim.step()

        # ====== 记录损失 ====== #
        self.all_loss = []
        self.all_act_loss = []
        self.all_kl_loss = []
        self.all_r_imm = []

        self.all_loss.append(vae_loss.item())
        self.all_act_loss.append(act_loss.item())
        self.all_kl_loss.append(kl_loss.item())
        self.all_r_imm.append(kl_loss_nonweight.item())

        with open(self.save_path_loss, 'a', encoding='utf-8') as f:
            f.write(
                f"{sum(self.all_loss) / len(self.all_loss)}\t"
                f"{sum(self.all_act_loss) / len(self.all_act_loss)}\t"
                f"{sum(self.all_kl_loss) / len(self.all_kl_loss)}\t"
                f"{sum(self.all_r_imm) / len(self.all_r_imm)}\n"
            )
        # ===================== #
        # ====== 将KL损失作为损失约束的一部分 ======
        self.kl_loss = kl_loss_nonweight
        self.act_loss = act_loss
        self.vae_loss = vae_loss

        # ====== 保存VAE权重 ====== #
        torch.save({
            "model_state_dict": self.vae.state_dict(),
        }, "vae_checkpoint.pt")  # 隔一段记录一次，总记录拖慢训练进程

    # # ====== 反事实计算贡献，避免内在奖励将不合理的动作奖励值抬高 ====== #
    # @th.no_grad()
    # def _cf_weighted_team_bonus_from_dyna_BN(
    #         self,
    #         q_all,  # (B,T,N,A) = mac_out[:, :-1]
    #         chosen_q,  # (B,T,N)   = gather(mac_out, actions).squeeze
    #         states,  # (B,T,S)
    #         avail,  # (B,T,N,A)
    #         alive_veh,  # (B,T,N)   1/0
    #         filled_mask,  # (B,T,1)   1/0  (= mask before expand_as td_error)
    #         dyna_rnd_loss_agent,  # (B,N)     动态内在奖励（每条轨迹每个agent一个）
    # ):
    #     """
    #     返回 team bonus: (B,T,1)
    #     """
    #     B, T, N, A = q_all.shape
    #     eps = 1e-6
    #
    #     # ---------------- 1) 反事实优势 A_i^{cf}(t) ----------------
    #     q_masked = q_all.clone()
    #     q_masked[avail == 0] = -1e9
    #
    #     tau = getattr(self.args, "cf_tau", 1.0)
    #     pi = th.softmax(q_masked / max(tau, 1e-6), dim=-1)  # (B,T,N,A)
    #
    #     q_tot_exec = self.mixer(chosen_q, states)  # (B,T,1)
    #
    #     # 展开成 (B, T*A, ·) 以减少 mixer 调用次数（每个 agent 一次）
    #     state_rep = states.unsqueeze(2).expand(B, T, A, states.shape[-1]).reshape(B, T * A, -1)  # (B,T*A,S)
    #     chosen_rep = chosen_q.unsqueeze(2).expand(B, T, A, N)  # (B,T,A,N)
    #
    #     adv_list = []
    #     for i in range(N):
    #         q_cf = chosen_rep.clone()  # (B,T,A,N)
    #         q_cf[:, :, :, i] = q_masked[:, :, i, :]  # 替换第 i 个 agent 的动作维 (B,T,A)
    #         q_cf = q_cf.reshape(B, T * A, N)  # (B,T*A,N)
    #
    #         q_tot_cf = self.mixer(q_cf, state_rep)  # (B,T*A,1)
    #         q_tot_cf = q_tot_cf.reshape(B, T, A, 1)  # (B,T,A,1)
    #
    #         baseline_i = (pi[:, :, i, :].unsqueeze(-1) * q_tot_cf).sum(dim=2)  # (B,T,1)
    #         adv_i = q_tot_exec - baseline_i  # (B,T,1)
    #         adv_list.append(adv_i)
    #
    #     adv = th.stack(adv_list, dim=2).squeeze(-1)  # (B,T,N)
    #
    #     # ---------------- 2) 计算权重 w_{t,i} 并按“每个agent沿时间”归一化 ----------------
    #     good = alive_veh * filled_mask.squeeze(-1).unsqueeze(-1)  # (B,T,N)
    #
    #     if getattr(self.args, "cf_hard_gate", False):
    #         w = (adv > 0).float()
    #     else:
    #         temp = getattr(self.args, "cf_adv_temp", 1.0)
    #         w = th.sigmoid(adv / max(temp, 1e-6))  # (B,T,N) in (0,1)
    #
    #     w = w * good
    #     k = getattr(self.args, "cf_topk", 5)  # 例如 5 或 10
    #     # w: (B,T,N)
    #     w2 = w.clone()
    #     # 对每个 (B,N) 在 T 维取 topk
    #     topv, topi = th.topk(w2, k=min(k, T), dim=1)
    #     mask_topk = th.zeros_like(w2).scatter_(1, topi, 1.0)
    #     w2 = w2 * mask_topk
    #
    #     w_sum = w2.sum(dim=1, keepdim=True).clamp(min=1e-6)
    #     w_norm = w2 / w_sum
    #
    #     # # 对每个 agent：sum_t w = 1（避免“拖久了拿更多内在奖励”）
    #     # w_sum = w.sum(dim=1, keepdim=True).clamp(min=1.0)  # (B,1,N)
    #     # w_norm = w / w_sum  # (B,T,N)
    #
    #     # ---------------- 3) 处理 dyna_rnd_loss_agent (B,N)：标准化 + 轨迹内零均值化 ----------------
    #     # 每个 agent 是否在该轨迹里出现过（至少一个时间步 alive）
    #     agent_present = alive_veh.any(dim=1)  # (B,N) bool
    #
    #     clip_after = getattr(self.args, "rnd_clip_after_norm", 5.0)
    #     dyna_norm = normalize_rnd(dyna_rnd_loss_agent, self.rms_dyna, agent_present,
    #                               clip_after_norm=clip_after)  # (B,N)
    #
    #     # 轨迹内对 agent 维度减均值（只在 present 的 agent 上）
    #     denom_agent = agent_present.float().sum(dim=1, keepdim=True).clamp(min=1.0)  # (B,1)
    #     mu_agent = (dyna_norm * agent_present.float()).sum(dim=1, keepdim=True) / denom_agent
    #     dyna_center = (dyna_norm - mu_agent) * agent_present.float()  # (B,N)
    #
    #     # ---------------- 4) 按权重分摊到时间步，并聚合成 team (B,T,1) ----------------
    #     # 每步每agent的 bonus
    #     beta = getattr(self.args, "cf_rnd_beta", 0.0005)
    #     cap = getattr(self.args, "cf_rnd_cap", 0.002)
    #
    #     bonus_agent_t = beta * w_norm * dyna_center.unsqueeze(1)  # (B,T,N)
    #
    #     # 聚合成 team：按本步 active agent 数平均（避免 N 变化导致尺度漂移）
    #     active = good.sum(dim=2, keepdim=True).clamp(min=1.0)  # (B,T,1)
    #     bonus_team = bonus_agent_t.sum(dim=2, keepdim=True) / active  # (B,T,1)
    #
    #     bonus_team = bonus_team.clamp(min=-cap, max=cap)
    #     return bonus_team

    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        obs = batch["obs"][:, :-1]

        # ====== 一、计算动态RND ====== #
        # ====== 1、利用已有的VAE计算z ====== #
        seq = batch["filled"][:, :-1].squeeze(-1)
        ep_lens = seq.sum(dim=1).long().tolist()

        alive_veh = 1. * (th.sum(batch["obs"][:, :-1], dim=3) > 0).view(obs.shape[0], -1, self.args.n_agents)
        # alive_veh_mask = th.bmm(alive_veh.unsqueeze(2), alive_veh.unsqueeze(1))

        rnn_hxs_encoder = {}
        rnn_hxs_encoder['rnn'] = torch.zeros(
            self.args.batch_size * self.args.n_agents,
            self.args.network['rnn_hidden_size'],
            device='cuda:0'
        )
        rnn_hxs_decoder = {}
        rnn_hxs_decoder['rnn'] = torch.zeros(
            self.args.batch_size * self.args.n_agents,
            self.args.network['rnn_hidden_size'],
            device='cuda:0'
        )
        pred_s, z_mean, z_log_var, rnn_hxs_encoder, rnn_hxs_decoder, z = self.vae(
            obs,
            rnn_hxs_encoder,
            rnn_hxs_decoder,
            seq_len=ep_lens,
            veh_mask=alive_veh
        )
        # ====== 2、利用z得到动态RND ====== #
        self.dynamic_rnd_optim.zero_grad()
        z_det = z.detach()
        dyna_rnd_loss = self.dynamic_rnd(z_det)
        # if self.dynamic_rnd.update_freq >= 2: # 这里缓慢更新，感觉可以避免非平稳
        dyna_rnd_loss.mean().backward()
            # self.dynamic_rnd.update_freq = 0
        # else:
        #     self.dynamic_rnd.update_freq += 1
        self.dynamic_rnd_optim.step()
        # dyna_rnd_loss = dyna_rnd_loss.detach()
        # ======================== #

        # ====== 二、计算静态RND ====== #
        self.static_rnd_optim.zero_grad()
        obs_rnd = obs.clone()
        # obs_rnd[..., 3:] = 0  # 这里可能不仅是单纯的观测，可能还需要加上ID和动作信息
        rnd_input = []
        rnd_mask = []
        for rnd_t in range(obs.shape[1]):
            rnd_input.append(obs_rnd[:, rnd_t,:,:3])
            rnd_input.append(actions[:, rnd_t,:,:])
            rnd_input.append(th.eye(self.args.n_agents, device='cuda:0').unsqueeze(0).expand(obs_rnd.shape[0], -1, -1))
            rnd_mask.append(1. * (th.sum(obs_rnd[:, rnd_t, :, :3], dim=2) == 0).view(obs.shape[0], -1, self.args.n_agents))
        rnd_input = th.cat([x.reshape(obs_rnd.shape[0], self.args.n_agents, -1) for x in rnd_input], dim=-1)
        rnd_mask = th.cat([x.reshape(obs_rnd.shape[0], self.args.n_agents, -1) for x in rnd_mask], dim=-1)
        rnd_input = rnd_input.reshape(obs_rnd.shape[0], obs_rnd.shape[1], obs_rnd.shape[2], -1)
        rnd_mask = rnd_mask.reshape(obs_rnd.shape[0], obs_rnd.shape[1], obs_rnd.shape[2], -1)
        stat_rnd_loss = self.static_rnd(rnd_input * (1-rnd_mask))
        # if self.static_rnd.update_freq >= 2:  # 因为仅在数据收集端使用了RND，所以应该不用避免更新非平稳
        stat_rnd_loss.mean().backward()
            # self.static_rnd.update_freq = 0
        # else:
        #     self.static_rnd.update_freq += 1
        self.static_rnd_optim.step()
        # stat_rnd_loss = stat_rnd_loss.detach()
        # ============================ #

        # # ====== 构建新的rewards(这里实际上没有起作用，因为runner端已经计算了rewards with intrinsic/仅在动作收集时使用RND) ====== #
        # # if not getattr(self.args, "intrinsic_in_runner", True):
        # if t_env < 1500000:
        #     # stat_step = stat_rnd_loss.detach().float().mean(dim=-1)  # (B,T)
        #     dyna_per_sample = dyna_rnd_loss.mean(dim=-1) # (B,)
        #
        #     # stat_mask = mask.detach().bool().squeeze(-1) # (B,T)
        #     dyna_mask = mask.detach().bool().squeeze(-1).any(dim=1)
        #
        #     # stat_step_norm = normalize_rnd(stat_step, self.rms_stat, stat_mask, clip_after_norm=5.0)
        #     # dyna_norm = normalize_rnd(dyna_per_sample, self.rms_dyna, dyna_mask, clip_after_norm=5.0)
        #     # stat_step *= stat_mask
        #     dyna_per_sample *= dyna_mask
        #
        #     reward_mask = (rewards != -0.005)
        #     # rewards = torch.where(reward_mask, rewards + \
        #     #           torch.clamp(self.args.stat_coef * stat_step.unsqueeze(-1), max=0.001) + \
        #     #           torch.clamp(self.args.dyna_coef * dyna_per_sample.unsqueeze(-1).unsqueeze(-1),
        #     #                       max=0.001), rewards)  # 不让内在奖励太大
        #     rewards = torch.where(reward_mask, rewards+torch.clamp(self.args.dyna_coef * dyna_per_sample.unsqueeze(-1).unsqueeze(-1), max=0.001),rewards)
        #     # ====== 记录奖励 ====== #
        #     # stat_s = summarize_tensor(stat_step_norm, stat_mask)
        #     # dyna_s = summarize_tensor(dyna_norm, dyna_mask)
        #
        #     # rewards_mean = batch["reward"][:, :-1].detach().float().mean().item()
        #     # stat_rnd = stat_rnd_loss.mean(dim=-1, keepdim=True).detach().float().mean().item()
        #     # dyna_rnd = dyna_rnd_loss.mean(dim=-1, keepdim=True).detach().float().mean().item()
        #
        #     # with open(self.reward_log_path, "a", encoding='utf-8') as f:
        #     #     f.write(f"{episode_num},{rewards_mean:.6g},{stat_rnd:.6g},{dyna_rnd:.6g}\n")
        #         # f.write(
        #         #     f"{episode_num},"
        #         #     f"{rewards_mean:.6g},"
        #         #     f"{stat_s['mean']:.6g},{stat_s['p50']:.6g},{stat_s['p90']:.6g},{stat_s['p99']:.6g},{stat_s['max']:.6g},"
        #         #     f"{dyna_s['mean']:.6g},{dyna_s['p50']:.6g},{dyna_s['p90']:.6g},{dyna_s['p99']:.6g},{dyna_s['max']:.6g}\n"
        #         # )
        # # else:
        # #     rewards = rewards + \
        # #               self.args.stat_coef * stat_rnd_loss.mean(dim=-1, keepdim=True)
        # # ============================ #

        # ====== DRL模型训练 ====== #
        # Calculate estimated Q-Values
        self.mac.agent.train()
        mac_out = []
        self.mac.init_hidden(batch.batch_size)
        for t in range(batch.max_seq_length):
            agent_outs = self.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
        mac_out = th.stack(mac_out, dim=1)  # Concat over time

        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)  # Remove the last dim
        chosen_action_qvals_ = chosen_action_qvals

        # # ======
        # rewards_used = rewards
        # if (t_env < 1500000) and getattr(self.args, "use_cf_rnd", True):
        #     q_all = mac_out[:, :-1].detach()  # (B,T,N,A)
        #     states_t = batch["state"][:, :-1].detach()  # (B,T,S)
        #     avail_t = avail_actions[:, :-1].detach()  # (B,T,N,A)
        #     chosen_q = chosen_action_qvals.detach()  # (B,T,N)
        #
        #     # alive_veh 你前面已经算过：(B,T,N) :contentReference[oaicite:7]{index=7}
        #     bonus_team = self._cf_weighted_team_bonus_from_dyna_BN(
        #         q_all=q_all,
        #         chosen_q=chosen_q,
        #         states=states_t,
        #         avail=avail_t,
        #         alive_veh=alive_veh.detach(),
        #         filled_mask=mask.detach(),  # 注意：这里 mask 还没 expand_as(td_error2) 之前用
        #         dyna_rnd_loss_agent=dyna_rnd_loss.detach(),  # (B,T,N)
        #     )
        #
        #     rewards = rewards_used + bonus_team
        # # ===============

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_mac.agent.train()
            target_mac_out = []
            self.target_mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_mac.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            # We don't need the first timesteps Q-Value estimate for calculating targets
            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values/ Double q learning
            mac_out_detach = mac_out.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                    self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        loss = L_td = masked_td_error.sum() / mask.sum()
        loss = loss + self.kl_loss + self.act_loss  # 希望约束KL损失

        # Optimise
        self.vae_optim.zero_grad()
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()
        self.vae_optim.step()
        self.scheduler_lr.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.log_stats_t = t_env
            
            # print estimated matrix
            # if self.args.env == "one_step_matrix_game":
            #     print_matrix_status(batch, self.mixer, mac_out)

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                / (self.priority_max - self.priority_min + 1e-5)
            else:
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                / th.sqrt(mask.sum(1))).detach().to('cpu')
        return info

    def _update_targets(self):
        self.target_mac.load_state(self.mac)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.mac.cuda()
        self.target_mac.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            
    def save_models(self, path):
        self.mac.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.mac.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_mac.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))
