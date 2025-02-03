from model.model import Model
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class GAT(Model):
    def __init__(self, args):
        super().__init__(args=args)
        self.convs = nn.ModuleList()
        self.convs.append(
            GATConv(self.args.input_dim, self.args.output_dim, heads=self.args.num_heads).to(self.args.device))
        for _ in range(self.args.num_layers - 2):
            self.convs.append(
                GATConv(self.args.output_dim * self.args.num_heads, self.args.output_dim, heads=self.args.num_heads).to(
                    self.args.device))
        self.convs.append(
            GATConv(self.args.output_dim * self.args.num_heads, self.args.output_dim, heads=self.args.num_heads).to(
                self.args.device))

    def forward(self, x, edge_index):
        for i in range(self.args.num_layers):
            x = self.convs[i](x, edge_index)
            if i != self.args.num_layers - 1:
                x = F.elu(x)
        return x
