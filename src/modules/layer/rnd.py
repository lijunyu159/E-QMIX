# -*-coding:utf-8-*-
import torch
import torch.nn as nn

class RND(nn.Module):
    def __init__(self, args, scheme=None):
        super(RND, self).__init__()
        # input_size = scheme['obs']['vshape'] + args.env_config['ob_space']['latent_size']
        input_size = 12 # x,y,v + id(1) + action(8)
        output_size = args.mixing_embed_dim  # 32
        self.update_freq = 0
        self.target = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        ).to(args.device)
        self.predictor = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        ).to(args.device)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, requires_grad=True):
        target_output = self.target(x)
        predict_output = self.predictor(x)
        intrinsic_reward = ((target_output - predict_output) ** 2).sum(-1)
        if not requires_grad:
            return intrinsic_reward.detach()
        return intrinsic_reward

class VAE_RND(nn.Module):
    def __init__(self, args, scheme=None):
        super(VAE_RND, self).__init__()
        # input_size = scheme['obs']['vshape'] + args.env_config['ob_space']['latent_size']
        input_size = args.env_config["ob_space"]["latent_size"]
        output_size = args.mixing_embed_dim  # 32
        self.update_freq = 0
        self.target = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        ).to(args.device)
        self.predictor = nn.Sequential(
            nn.Linear(input_size, output_size),
            nn.ReLU(),
            nn.Linear(output_size, output_size)
        ).to(args.device)
        for p in self.target.parameters():
            p.requires_grad = False

    def forward(self, x, requires_grad=True):
        target_output = self.target(x)
        predict_output = self.predictor(x)
        intrinsic_reward = ((target_output - predict_output) ** 2).sum(-1)
        if not requires_grad:
            return intrinsic_reward.detach()
        return intrinsic_reward