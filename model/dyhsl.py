from model.dyhsl_model.model import FullModel
from torch import nn
import argparse
import torch


class DYHSL(nn.Module):
    def __init__(self, args):
        super(DYHSL, self).__init__()
        self.tod_idx=288
        self.dyhsl = FullModel(args.num_nodes, args.batch_size, args.predefined_adjs, args)

    def forward(self, x, y):  # B x T x N x D
        
        data = {'feat': x[:, :, :, :1], 'tod_idx': (x[:, :, 0, 1]*self.tod_idx).long(), 'dow_onehot': x[:, :, 0, 2:9], 'target': y}
        return self.dyhsl(data)


# def load_my_net(data_shape):
#     predefined_adjs = data_shape[1]
#     predefined_adjs = [torch.tensor(adj).to('cuda:0') for adj in predefined_adjs]
#     data_shape = data_shape[0]
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--window_sizes', default=[1, 2, 3, 4, 6, 12], help='J=6')
#     parser.add_argument('--num_hyper_edge', default=32, help='hyper edges')
#     parser.add_argument('--hidden_dim', default=64, help='the dimension for the hidden features')
#     parser.add_argument('--num_head_layers', default=2, help='the number of hidden layers')
#     args = parser.parse_args(args=list())
#     return DYHSL(data_shape[2], data_shape[0], predefined_adjs, args)


# def call_my_net(model_config, data_shape):  # data (B x T x N x D), adj
#     if model_config.type == "dyhsl":
#         return load_my_net(data_shape)

