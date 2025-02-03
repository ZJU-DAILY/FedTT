from torch import nn
import torch

torch.manual_seed(42)


class Model(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
