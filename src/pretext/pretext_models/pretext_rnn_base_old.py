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
        input_size = 2 # ego car's offset from start of traj, its offset from its front car
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
    def _forward(self, inputs, rnn_hxs, seq_len):
        # if self.task == 'rl_predict':
        #     robot_node = inputs['pretext_nodes']  # seq_length, nenv, 1
        #     humans_obs = inputs['pretext_spatial_edges'] # seq_length, nenv*human_num, 2
        # else:
        # robot_node = inputs['pretext_nodes'].permute(1, 0, 2)  # seq_length, nenv, 1
        # humans_obs = inputs['pretext_spatial_edges'].permute(1, 0, 2)  # seq_length, nenv*human_num, 2
        inputs = torch.diff(inputs[...,-2:], dim=1)
        zero_pad = torch.zeros((inputs.shape[0], 1, inputs.shape[2], 2), dtype=inputs.dtype, device=inputs.device)
        inputs = torch.cat([zero_pad,inputs], dim=1)  # (B, T, N, 2)
        B, T, N, F = inputs.shape
        h_new_mas = []
        output_mas = []
        for n in range(N):
            inputs_n = inputs[:,:,n]
            state = inputs_n.permute(1,0,2)
            # lengths = torch.full(
            #     (B,),
            #     T,
            #     dtype=torch.long,
            #     device='cpu')  # 如果B=1，那说明就一条轨迹，T = 96可以，但如果若干个批次呢？这里的T应该是这一批数据中的最大时间步
            hidden_states = rnn_hxs['rnn'].unsqueeze(0)

            # [seq_length, nenv, human_num, 1] + [seq_length, nenv, human_num, 1] = [seq_length, nenv, human_num, 2]
            # concat_state = torch.cat((robot_node, humans_obs[:, :, 0, None]), dim=-1)

            # [seq_length, nenv*human_num, 64]
            concat_state = self.embedding(state)
            concat_state = pack_padded_sequence(concat_state, seq_len, enforce_sorted=False)
            # h_new: [1, batch_size, 256]
            _, h_new = self.RNN(concat_state, hidden_states[:,n:n+1])
            h_new = h_new.squeeze(0)
            # [nenv*human_num, 1]
            output = self.output_linear(h_new)
            h_new_mas.append(h_new)
            output_mas.append(output)
        output = torch.stack(output_mas)
        h_new = torch.stack(h_new_mas)
        output = output.reshape(B, N, output.shape[2])
        h_new = h_new.reshape(B, N, h_new.shape[2])
        return output, h_new

    # forward function: returns new rnn hidden state
    def forward(self, inputs, rnn_hxs, seq_len, infer=False):
        x, h_new = self._forward(inputs, rnn_hxs, seq_len)
        rnn_hxs['rnn'] = h_new  # [nenv, 1, 256]

        # return x.squeeze(-1), rnn_hxs
        return x, rnn_hxs