from model.stssl_model import models
from torch import nn
import argparse
import torch
from model.stssl_model.lib.utils import dwa

class STSSL(nn.Module):
    def __init__(self, args):
        super(STSSL, self).__init__()
        self.stssl = models.STSSL(args.num_nodes, args.batch_size, args.output_size, args.pre_len, args)

    def forward(self, x, adj):  # x (B x T x N x D), adj
        return self.stssl(x[..., :1], adj)

    def loss(self, z1, z2, y_true, scaler, loss_weights):
        return self.stssl.loss(z1, z2, y_true, scaler, loss_weights)



