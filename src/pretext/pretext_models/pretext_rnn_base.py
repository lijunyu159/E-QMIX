# -*-coding:utf-8-*-
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from vae_utils import init

'''
GRU encoder network for trait prediction pretext task
can be used in VAE in our method and Morton et al
'''
class LSTM_Pretext(nn.Module):
    def __init__(self, obs_space_dict, config, task):
        super(LSTM_Pretext, self).__init__()
        # config settings
        self.config = config
        self.is_recurrent = True
        self.task = task
        # number of edges per agent = total number of spatial edges / total number of nodes
        # self.human_num = int(obs_space_dict['spatial_edges'].shape[0] // obs_space_dict['pretext_nodes'].shape[0])

        self.seq_length = config.pretext['num_steps']
        self.nenv = config.pretext['batch_size']
        self.nminibatch = 1
        self.output_size = config.network['rnn_hidden_size']

        # init the parameters
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), np.sqrt(2))

        # last layer of the encoder NN
        # for vae, we will add another linear layer in Encoder after output_linear
        self.output_linear = nn.Sequential(init_(nn.Linear(self.output_size, self.output_size)), nn.ReLU())

        # find the input size
        input_size = 10 # ego car's offset from start of traj, its offset from its front car
        # for Morton et al, the observation contains the actions of each car
        # if self.config.pretext['cvae_decoder'] == 'mlp':
        # # +1 since pretext_nodes also contains the agent's action_x in Morton et al
        #     input_size = input_size + 1

        # embedding layer for the inputs
        self.embedding = nn.Sequential(
            init_(nn.Linear(input_size, 32)), nn.ReLU(),
            init_(nn.Linear(32, config.network['embedding_size'])), nn.ReLU())

        # the RNN
        self.RNN = nn.GRU(config.network['embedding_size'], config.network['rnn_hidden_size'])
        # initialize rnn
        for name, param in self.RNN.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        self.train()

    '''
    forward util function for both rl and sl
    inputs: padded sequence, each_seq_len: list of original sequence lengths of inputs
    returns: output of rnn & new hidden state of rnn
    '''
    def _forward(self, inputs, rnn_hxs, seq_len, veh_mask=None, return_sequence=False):
        """
        inputs: [B, T, N, F] (BTNF)
        seq_len: list[int] or 1D tensor of length B (episode-level valid lengths, i.e. padding mask)
        veh_mask: optional [B, T, N] (or [B, T, N, 1]) mask where 1/True means the vehicle is valid (in-scene)
                  at that timestep. When provided, per-vehicle sequence lengths are computed from
                  (veh_mask & time_mask), so that vehicles that leave early are not encoded using padded zeros.
        returns:
            output: [B, N, H] encoded features for each vehicle
            h_new:  [B, N, H] new RNN hidden state for each vehicle
        """
        # Use delta-position as input feature (keep last 2 dims)
        # inputs = torch.diff(inputs[..., -2:], dim=1)
        # zero_pad = torch.zeros((inputs.shape[0], 1, inputs.shape[2], 2),
        #                        dtype=inputs.dtype, device=inputs.device)
        # inputs = torch.cat([zero_pad, inputs], dim=1)  # [B, T, N, 2]
        inputs = inputs
        B, T, N, F = inputs.shape

        # ---------
        # Build time mask from seq_len (episode padding)
        # ---------
        if isinstance(seq_len, (list, tuple)):
            seq_len_t = torch.tensor(seq_len, device=inputs.device, dtype=torch.long)
        else:
            seq_len_t = seq_len.to(device=inputs.device, dtype=torch.long)

        time_mask = torch.arange(T, device=inputs.device).unsqueeze(0) < seq_len_t.unsqueeze(1)  # [B, T]

        # ---------
        # Vehicle mask (in-scene) -> [B, T, N]
        # ---------
        if veh_mask is None:
            veh_mask_bool = torch.ones((B, T, N), device=inputs.device, dtype=torch.bool)
        else:
            veh_mask_bool = veh_mask
            if veh_mask_bool.dim() == 4 and veh_mask_bool.size(-1) == 1:
                veh_mask_bool = veh_mask_bool.squeeze(-1)
            veh_mask_bool = veh_mask_bool.to(device=inputs.device)
            if veh_mask_bool.dtype != torch.bool:
                veh_mask_bool = veh_mask_bool > 0.5
            if veh_mask_bool.size(1) != T:
                veh_mask_bool = veh_mask_bool[:, :T, :]
            if veh_mask_bool.shape != (B, T, N):
                raise RuntimeError(f"veh_mask shape {tuple(veh_mask_bool.shape)} != (B,T,N)=({B},{T},{N})")

        valid_btn = veh_mask_bool & time_mask[:, :, None]  # [B, T, N]

        # ---------
        # Initial hidden state: expect rnn_hxs['rnn'] as [B, N, H]
        # ---------
        if isinstance(rnn_hxs, dict):
            h0 = rnn_hxs.get('rnn', None)
        else:
            h0 = rnn_hxs

        if h0 is None:
            h0 = inputs.new_zeros(B, N, self.output_size)

        # normalize possible shapes
        if h0.dim() == 4 and h0.size(0) == 1:
            h0 = h0.squeeze(0)  # [B, N, H]
        if h0.dim() == 4 and h0.size(1) == 1:
            h0 = h0.squeeze(1)  # [B, N, H]
        if h0.dim() == 2:
            h0 = h0.reshape(B, N, -1)  # [B, N, H]
        if h0.dim() != 3 or h0.size(0) != B or h0.size(1) != N:
            raise RuntimeError(f"Expected rnn_hxs['rnn'] with shape [B,N,H]=({B},{N},H), got {tuple(h0.shape)}")

        h_new_list = []
        out_last_list = []
        out_seq_list = []

        for n in range(N):
            # [B, T, F] -> [T, B, F]
            state = inputs[:, :, n, :].permute(1, 0, 2)

            lengths_n = valid_btn[:, :, n].long().sum(dim=1)  # [B]
            has = lengths_n > 0

            h_full = inputs.new_zeros((B, self.output_size))
            out_last = inputs.new_zeros((B, self.output_size))
            out_seq = inputs.new_zeros((B, T, self.output_size))

            if has.any():
                state_valid = state[:, has, :]  # [T, M, F]
                emb_valid = self.embedding(state_valid)  # [T, M, E]
                packed = pack_padded_sequence(emb_valid, lengths_n[has].cpu(),
                                              enforce_sorted=False)

                h0_valid = h0[:, n, :][has].unsqueeze(0).contiguous()  # [1, M, H]
                # _, h_valid = self.RNN(packed, h0_valid)                # h_valid: [1, M, H]
                packed_out, h_valid = self.RNN(packed, h0_valid)
                h_valid = h_valid.squeeze(0)                           # [M, H]

                h_full[has] = h_valid
                # out_full = self.output_linear(h_full)
                # out_full[~has] = 0.0  # ensure invalid vehicles are exactly masked-out
                out_last = self.output_linear(h_full)
                out_last[~has] = 0.0
                # ---- per-timestep features ----
                out_unpacked, _ = pad_packed_sequence(packed_out, total_length=T)  # [T,M,H]
                out_unpacked = self.output_linear(out_unpacked)  # [T,M,H]
                out_seq[has] = out_unpacked.permute(1, 0, 2)  # [M,T,H] -> place into [B,T,H]

            h_new_list.append(h_full)
            out_last_list.append(out_last)
            out_seq_list.append(out_seq)

        # [B, N, H]
        # output = torch.stack(out_list, dim=1)
        if return_sequence:
            output = torch.stack(out_seq_list, dim=2)  # [B,T,N,H]
        else:
            output = torch.stack(out_last_list, dim=1)  # [B,N,H]
        h_new = torch.stack(h_new_list, dim=1)
        return output, h_new

    # forward function: returns new rnn hidden state
    # def forward(self, inputs, rnn_hxs, seq_len, veh_mask=None, infer=False):
    #     x, h_new = self._forward(inputs, rnn_hxs, seq_len, veh_mask=veh_mask)
    #     rnn_hxs['rnn'] = h_new  # [nenv, 1, 256]
    #
    #     # return x.squeeze(-1), rnn_hxs
    #     return x, rnn_hxs
    def forward(self, inputs, rnn_hxs, seq_len, veh_mask=None, infer=False, return_sequence=False):
        x, h_new = self._forward(inputs, rnn_hxs, seq_len, veh_mask=veh_mask, return_sequence=return_sequence)
        rnn_hxs['rnn'] = h_new
        return x, rnn_hxs