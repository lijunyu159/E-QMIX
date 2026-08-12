import torch
from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
from src.modules.layer.rnd import RND
from src.modules.layer.rnd import VAE_RND
from src.pretext.pretext_models.cvae_model import *

from collections import Counter
import os

# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = []
        for i, worker_conn in enumerate(self.worker_conns):
            ps = Process(target=env_worker, 
                    args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
            self.ps.append(ps)

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0 if not args.use_curricula else args.t_env

        self.train_returns = []
        self.test_returns = []
        self.train_stats = {}
        self.test_stats = {}
        # self.max_return = -1000
        self.return_mean = None
        self.log_train_stats_t = -100000
        # ====== 记录可用动作 ====== #
        default_log_path = "./avail_action_dist.txt"
        self.avail_dist_log_path = getattr(self.args, "avail_action_log_path", default_log_path)

        # ========================= #
        self.stat_rnd = None
        self.dyna_rnd = None
        self._last_sent_actions = {}
        H = self.args.network["rnn_hidden_size"]
        self.vae_enc_hxs = [
            {"rnn": th.zeros(1, 8, H, device=self.args.device)}
            for _ in range(self.batch_size)
        ]
        # ========================= #

    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

        # ====== 实例化RND，并置为eval ====== #
        self.stat_rnd = RND(self.args, scheme).to("cuda:0")
        self.stat_rnd.eval()
        self.vae = CVAECoordinationPatterns({}, task='pretext_predict',
                                            decoder_base=self.args.pretext['cvae_decoder'],
                                            config=self.args)
        self.vae.eval()
        self.dyna_rnd = VAE_RND(self.args, scheme).to("cuda:0")
        self.dyna_rnd.eval()

    # ====== Runner <- Learner接口 ====== #
    def set_vae_state(self, state_cpu):
        if getattr(self, "vae", None) is None:
            return
        self.vae.load_state_dict(state_cpu, strict=True)
        self.vae.to(self.args.device)
        self.vae.eval()

    def set_static_rnd_state(self, state_cpu):
        if self.stat_rnd is None:
            return
        # state_cpu 是 cpu tensor；load 后再搬到 args.device
        self.stat_rnd.load_state_dict(state_cpu, strict=True)
        self.stat_rnd.to(self.args.device)
        self.stat_rnd.eval()

    def set_dynamic_rnd_state(self, state_cpu):
        if self.dyna_rnd is None:
            return
        self.dyna_rnd.load_state_dict(state_cpu, strict=True)
        self.dyna_rnd.to(self.args.device)
        self.dyna_rnd.eval()
    # ================================== #

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"])
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
        # print(pre_transition_data["avail_actions"])
        # ====== 记录t_ep = 0 的分布 ====== #
        self._log_avail_action_dist(pre_transition_data["avail_actions"], t_ep=0, test_mode=False)
        # =============================== #
        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0
        for i in range(self.batch_size):
            self.vae_enc_hxs[i]["rnn"].zero_()

    # ====== 统计可用动作分布 ====== #
    def _log_avail_action_dist(self, avail_actions_list, t_ep, test_mode):
        """
        avail_actions_list: List[env] -> avail_actions (List[n_agents] of 0/1 masks)
        """
        if not avail_actions_list:
            return

        all_counts = []
        masked_counts = []

        for env_avail in avail_actions_list:
            if env_avail is None:
                continue
            if len(env_avail) == 0:
                continue

            action_dim = len(env_avail[0])
            counts = [int(np.sum(a)) for a in env_avail]  # 每车可用动作数
            all_counts.extend(counts)

            # 更关注“真的被 mask 影响”的车辆（排除永远全 1 的那些）
            masked_counts.extend([c for c in counts if c < action_dim])

        if len(all_counts) == 0:
            return


        # 分布（Counter）
        hist_all = Counter(all_counts)
        hist_masked = Counter(masked_counts) if len(masked_counts) > 0 else Counter()
        # 前缀字符串（终端/文件用）
        phase_str = "TEST" if test_mode else "TRAIN"
        # 组成一行可读的日志
        line = (
            f"[{phase_str}] t_ep={t_ep}, t_env={self.t_env} | "
            f"avail# dist(all)={dict(sorted(hist_all.items()))} | "
            f"dist(masked_only)={dict(sorted(hist_masked.items()))}\n"
        )
        # 1）按需在终端打印
        if getattr(self.args, "print_avail_dist", False):
            # 不想在终端看到的话，在配置里把 print_avail_dist 设为 False 即可
            print(line, end="")
        # 2）追加写入文本文件
        try:
            with open(self.avail_dist_log_path, "a", encoding="utf-8") as f:
                f.write(line)
        except Exception as e:
            # 避免因为写文件失败导致训练崩掉，只打印一个提示
            print(f"[WARN] Failed to write avail_action_dist log: {e}")
        # 记录成标量（便于 tensorboard）
        prefix = "test_" if test_mode else ""
        mean_all = float(np.mean(all_counts))
        le2_all = float(np.mean(np.array(all_counts) <= 2))
        self.logger.log_stat(prefix + "avail_actions_mean", mean_all, self.t_env)
        self.logger.log_stat(prefix + "avail_actions_le2_frac", le2_all, self.t_env)
        if len(masked_counts) > 0:
            mean_masked = float(np.mean(masked_counts))
            self.logger.log_stat(prefix + "avail_actions_masked_mean", mean_masked, self.t_env)
            self.logger.log_stat(prefix + "avail_actions_masked_frac",
                                 float(len(masked_counts)) / float(len(all_counts)), self.t_env)


    # ====================================== #
    # # ====== 计算加入Q值的RND，目的是只修改动作收集时的RND ====== #
    # def _compute_rnd_action_bonus(self, obs_t, avail_actions):
    #     """
    #         obs_t: (bs, n_agents, obs_dim)
    #         avail_actions: (bs, n_agents, n_actions)
    #         return: (bs, n_agents, n_actions)  # 将加到 Q 上
    #         """
    #     device = next(self.stat_rnd.parameters()).device
    #     obs_t = obs_t.to(device).float()
    #     avail_actions = avail_actions.to(device)
    #
    #     bs, n_agents, _ = obs_t.shape
    #     n_actions = avail_actions.shape[-1]
    #
    #     # 取与你 learner 一致的 obs[:3]
    #     obs_feat = obs_t[:, :, :3]  # (bs,n_agents,3)
    #
    #     # 候选动作索引（用“原始 action index”还是“归一化 action”，要与你 learner 一致）
    #     a = th.arange(n_actions, device=device).view(1, 1, n_actions, 1).float()
    #     a = a.expand(bs, n_agents, n_actions, 1)  # (bs,n_agents,n_actions,1)
    #     # agent id one-hot
    #     agent_id = th.eye(n_agents, device=device).view(1, n_agents, 1, n_agents)
    #     agent_id = agent_id.expand(bs, n_agents, n_actions, n_agents)
    #
    #     x = th.cat([
    #         obs_feat.unsqueeze(2).expand(bs, n_agents, n_actions, 3),
    #         a,
    #         agent_id
    #     ], dim=-1)  # (bs,n_agents,n_actions, 3+1+n_agents)
    #
    #     # 折叠成 (bs, 1, N, dim) 以复用你 static_rnd 的写法
    #     x = x.reshape(bs, 1, n_agents * n_actions, -1)
    #
    #     with th.no_grad():
    #         e = self.stat_rnd(x)  # 期望输出逐样本误差
    #         # 适配不同输出形状：常见为 (bs,1,N,1) 或 (bs,1,N)
    #     if e.dim() == 4 and e.size(-1) == 1:
    #         e = e.squeeze(-1)
    #     e = e.reshape(bs, n_agents, n_actions)
    #
    #     # ---- 关键：做“相对偏置”，避免整体抬高或整体压低 ----
    #     # 每个 agent 内，对 action 维度零均值 + 标准化
    #     e = e - e.mean(dim=-1, keepdim=True)
    #     e = e / (e.std(dim=-1, keepdim=True) + 1e-6)
    #
    #     clip = getattr(self.args, "rnd_action_clip", 2.0)
    #     beta = getattr(self.args, "rnd_action_beta", 0.05)  # 推荐从 0.01~0.1 试
    #
    #     bonus = (beta * e).clamp(-clip * beta, clip * beta)
    #
    #     # 不可用动作置为很小，避免 greedy 选到
    #     bonus = bonus.masked_fill(avail_actions == 0, -1e9)
    #     return bonus

    def run(self, test_mode=False, phase=0):
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)] # 这个batch_size是同时运行的进程 [0]
        # ====== 捕捉外在环境返回值，方便比较加入内在奖励函数后的return_mean ====== #
        episode_returns_ext = [0.0 for _ in range(self.batch_size)]
        # =================================================================== #
        episode_lengths = [0 for _ in range(self.batch_size)]
        complete_flags = [0.0 for _ in range(self.batch_size)]
        collisions = [0.0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION
        
        save_probs = getattr(self.args, "save_probs", False)
        while True:

            # Pass the entire batch of experiences up till now to the agents
            # Receive the actions for each agent at this timestep in a batch for each un-terminated env
            # # ====== 将RND仅加到动作收集阶段 ====== #
            # rnd_bonus = None
            # if (not test_mode) and getattr(self.args, "use_rnd_explore", True):
            #     obs_t = self.batch["obs"][envs_not_terminated, self.t]  # (bs,n_agents,obs_dim)
            #     avail = self.batch["avail_actions"][envs_not_terminated, self.t]  # (bs,n_agents,n_actions)
            #     rnd_bonus = self._compute_rnd_action_bonus(obs_t, avail)
            #
            # actions = self.mac.select_actions(
            #     self.batch, t_ep=self.t, t_env=self.t_env,
            #     bs=envs_not_terminated, test_mode=test_mode,
            #     rnd_bonus=rnd_bonus
            # )
            # # ================================== #
            if save_probs:
                actions, probs = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            else:
                actions = self.mac.select_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
                # actions = self.actions_selection_curricula(test_mode=test_mode, stage=phase)
            cpu_actions = actions.to("cpu").numpy()

            # Update the actions taken
            actions_chosen = {
                "actions": actions.unsqueeze(1).to("cpu"),
            }
            if save_probs:
                actions_chosen["probs"] = probs.unsqueeze(1).to("cpu")
            
            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", cpu_actions[action_idx]))
                        self._last_sent_actions[idx] = cpu_actions[action_idx]
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # ====== 收集消极动作车辆 ======= #
                    info = data["info"]
                    slow = info["slow_mask"]  # (N,1) np.float32
                    danger = info["danger_mask"]  # (N,1)
                    alive = info["alive_mask"]  # (N,1)
                    # 这块不删的话后面记录会报错
                    for key in ["slow_mask", "danger_mask", "alive_mask"]:
                        if key in info:
                            del info[key]
                    # ============================= #
                    # ====== 在动作收集端计算RND和VAE_rnd ====== #
                    if (self.stat_rnd is not None) and (not test_mode):
                        base_r = float(data["reward"])
                        if base_r != -0.005: # 只有在当前步奖励不等于-0.005时才鼓励
                            # ====== 收集静态RND ====== #
                            a_np = self._last_sent_actions.get(idx, None)
                            # if a_np is not None:
                                # data["obs"] 是 next_obs（env.step 后返回的 obs）
                            obs_np = np.asarray(data["obs"], dtype=np.float32)  # [n_agents, obs_dim]
                            obs_t = th.from_numpy(obs_np).to(self.args.device)  # [n_agents, obs_dim]

                            # alive: (N,1) -> [1,1,N] bool mask
                            alive_btn = th.from_numpy(np.asarray(alive, dtype=np.float32)).to(self.args.device)
                            alive_btn = (alive_btn.view(1, 1, self.args.n_agents) > 0.5)

                            # === 每步更新 VAE encoder，并取当步 z_mean 作为 z_t ===
                            # 输入给 encoder: [B=1, T=1, N, F]
                            obs_step = obs_t.unsqueeze(0).unsqueeze(0)

                            # 按 learner 逻辑只取前三维 obs :contentReference[oaicite:10]{index=10}
                            obs_feat = obs_t[:, :3]  # [n_agents, 3]

                            act_t = th.from_numpy(np.asarray(a_np)).to(self.args.device).float()
                            act_t = act_t.view(self.args.n_agents, -1)  # [n_agents, act_dim(=1)]

                            agent_id = th.eye(self.args.n_agents, device=self.args.device)  # [n_agents, n_agents]

                            rnd_in = th.cat([obs_feat, act_t, agent_id], dim=-1)  # [n_agents, 3+1+n_agents]
                            # rnd_in = rnd_in.unsqueeze(0).unsqueeze(0)
                            with torch.no_grad():

                                z_mean_step, z_logvar_step, self.vae_enc_hxs[idx] = self.vae.encoder(
                                    obs_step,
                                    self.vae_enc_hxs[idx],
                                    seq_len=[1],
                                    veh_mask=alive_btn
                                )
                                # z-RND 的输入：只用 z_mean_step（不要用采样 z，避免噪声当新奇度）
                                z_rnd_in = z_mean_step.detach()  # [1,1,N,Z]

                                z_rnd_bonus = self.dyna_rnd(z_rnd_in)  # 期望输出 [1,1,1] 或 [1,1,N]

                                stat_rnd_bonus = self.stat_rnd(rnd_in)  # (B,T,N)

                            good = alive * (1-slow) * (1-danger)
                            stat_rnd_bonus = stat_rnd_bonus * torch.from_numpy(good).float().squeeze(-1).to(self.args.device)
                            z_rnd_bonus = z_rnd_bonus * th.from_numpy(good).float().squeeze(-1).to(self.args.device)
                            stat_rnd_bonus = (self.args.stat_coef * stat_rnd_bonus).clamp(max=0.001)
                            z_rnd_bonus = (self.args.dyna_coef * z_rnd_bonus).clamp(max=0.001)

                            # =========================== #
                            data["reward"] = base_r + float(stat_rnd_bonus.mean().item()) + float(z_rnd_bonus.mean().item())
                            # data["reward"] = base_r + float(stat_rnd_bonus.mean().item())
                            # ====== 记录不加入内在奖励时的奖励 ====== #
                            episode_returns_ext[idx] += base_r
                            # ===================================== #
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                        self.vae_enc_hxs[idx]["rnn"].zero_()
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])

            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1
            # ====== 这里加：每 N 步记录一次（避免日志太密）====== #
            avail_log_interval = getattr(self.args, "avail_log_interval", 1)
            if (self.t % avail_log_interval) == 0:
                self._log_avail_action_dist(pre_transition_data["avail_actions"], t_ep=self.t, test_mode=test_mode)
            # ============================================= #

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats", None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)
        # print(self.mac.action_selector.epsilon)
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        infos = [cur_stats] + final_env_infos
        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)
        # store the current mean return
        self.return_mean = np.mean(cur_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker(remote, env_fn):
    # Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset()
            # print(env.comm_lag)
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

