# -*-coding:utf-8-*-
from src.pretext.pretext_models.pretext_rnn_base import LSTM_Pretext

from src.vae_utils import init
from .decoders import *

class CVAECoordinationPatterns(nn.Module):
    def __init__(self, obs_shape, task, decoder_base, config):
        super(CVAECoordinationPatterns, self).__init__()
        # pretext_predict: train/test the predictor with supervised learning
        # rl_predict: use the predictor to infer in rl
        assert task in ['pretext_predict', 'rl_predict']

        self.config = config
        self.update_freq = 0
        # initialize encoder
        self.encoder_base = LSTM_Pretext(obs_shape, config, task).to("cuda:0")

        # need to wrap the lstm or srnn encoder so that it can output mean and variance for vae
        # since we only have two classes of traits, latent size = 2 is more than enough
        # for latent_size, check driving_config.py
        # self.encoder = Encoder(config, self.encoder_base, latent_size=config.env_config['ob_space']['latent_size']).to("cuda:0")
        self.encoder = Encoder(
            config, self.encoder_base,
            latent_size=config.env_config['ob_space']['latent_size'],
            per_timestep_z=config.pretext.get('per_step_z', True)
        ).to("cuda:0")
        # initialize decoder
        if decoder_base == 'mlp':
            self.decoder = MLP(obs_shape, config, task, latent_size=config.env_config['ob_space']['latent_size'])
        elif decoder_base == 'lstm':
            self.decoder = LSTM_DECODER(obs_shape, config, task, latent_size=config.env_config['ob_space']['latent_size']).to("cuda:0")
        else:
            raise NotImplementedError


    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs_encoder, rnn_hxs_decoder, seq_len, veh_mask=None):
        # encoder
        z_mean, z_log_var, rnn_hxs_encoder = self.encoder(inputs, rnn_hxs_encoder, seq_len, veh_mask=veh_mask)
        # z.shape = B*N, N
        z = self.reparameterize(z_mean, z_log_var)
        # each timestep in a traj should use the same z
        # [batch_size, 2] -> [batch_size, num_steps, 2]
        max_seq_len = max(seq_len)
        # z_in = z.unsqueeze(1).repeat(1, max_seq_len, 1, 1)
        if z.dim() == 3:
            # 轨迹级 z: [B,N,Z] -> [B,T,N,Z]
            z_in = z.unsqueeze(1).repeat(1, max_seq_len, 1, 1)
        else:
            # 时间步级 z: [B,T,N,Z]
            z_in = z[:, :max_seq_len]
        # decoder  z为输入，a为输出
        if self.decoder.is_recurrent:
            reconstructed, rnn_hxs_decoder = self.decoder(z_in, rnn_hxs_decoder, max_seq_len)
        else:
            reconstructed = self.decoder(inputs, z_in)
        return reconstructed, z_mean, z_log_var, rnn_hxs_encoder, rnn_hxs_decoder, z

    # reparameterization trick
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

'''
vae encoder model
'''
class Encoder(nn.Module):
    # latent_size: the number of dimensions in latent state
    def __init__(self, config, base, latent_size, per_timestep_z=False):
        super(Encoder, self).__init__()
        self.base = base
        self.per_timestep_z = per_timestep_z
        # init the parameters
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        self.linear_means = init_(nn.Linear(config.network['rnn_hidden_size'], latent_size))
        self.linear_log_var = init_(nn.Linear(config.network['rnn_hidden_size'], latent_size))

    # forward function, used for pretext training and testing
    # inputs: state dict
    # x: output features from SRNN [nenv, seq_len, human_num]
    # all humans share the same weights for calculating mean and variance
    # means: [human_num, latent_size], log_vars: [human_num, latent_size]
    def forward(self, inputs, rnn_hxs, seq_len, veh_mask=None):
        # inputs: [pretext_nodes]: batch_size, 2
        # [pretext_spatial_edges]: batch_size, 2
        # [pretext_temporal_edges]: batch_size, 1
        # outputs: x: [batch_size, feature_size], rnn_hxs: [batch_size, feature_size]
        x, rnn_hxs = self.base.forward(inputs, rnn_hxs, seq_len, veh_mask=veh_mask, return_sequence=self.per_timestep_z)
        means = self.linear_means(x) # [nenv, human_num, 2]
        log_vars = self.linear_log_var(x) # [nenv, human_num, 2]
        if veh_mask is not None:
            # veh_mask: [B, T, N] (or [B, T, N, 1]); 1/True means in-scene/valid at that timestep
            veh_mask_bool = veh_mask
            if veh_mask_bool.dim() == 4 and veh_mask_bool.size(-1) == 1:
                veh_mask_bool = veh_mask_bool.squeeze(-1)
            veh_mask_bool = veh_mask_bool.to(device=means.device)
            if veh_mask_bool.dtype != torch.bool:
                veh_mask_bool = veh_mask_bool > 0.5
            Bm, Tm, Nm = veh_mask_bool.shape
            # time mask from seq_len (episode padding)
            if isinstance(seq_len, (list, tuple)):
                seq_len_t = torch.tensor(seq_len, device=means.device, dtype=torch.long)
            else:
                seq_len_t = seq_len.to(device=means.device, dtype=torch.long)
            time_mask = torch.arange(Tm, device=means.device).unsqueeze(0) < seq_len_t.unsqueeze(1)  # [B, T]
            valid_btn = veh_mask_bool & time_mask[:, :, None]  # [B, T, N]
            # veh_valid = valid_btn.any(dim=1).to(dtype=means.dtype)  # [B, N]
            # # zero-out invalid vehicles so they do not contribute to z / KL
            # means = means * veh_valid.unsqueeze(-1)
            # log_vars = log_vars * veh_valid.unsqueeze(-1)
            if means.dim() == 4:
                # [B,T,N,Z] 逐时间步 mask
                m = valid_btn.to(dtype=means.dtype).unsqueeze(-1)  # [B,T,N,1]
                means = means * m
                log_vars = log_vars * m
            else:
                # [B,N,Z] 轨迹级 mask（车是否出现过）
                veh_valid = valid_btn.any(dim=1).to(dtype=means.dtype)  # [B,N]
                means = means * veh_valid.unsqueeze(-1)
                log_vars = log_vars * veh_valid.unsqueeze(-1)
        return means, log_vars, rnn_hxs

    # predict function, used in rl training (see vec_pretext_normalize.py)
    def predict(self, inputs, rnn_hxs):
        means, log_vars, rnn_hxs = self.forward(inputs, rnn_hxs)
        z = self.reparameterize(means, log_vars)
        return z

    # reparameterization trick
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std