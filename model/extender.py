from model.model import Model
from model.gat import GAT
from model.mlp import MLP
import torch
from torch_geometric.data import Data


class Extender(Model):
    def __init__(self, args):
        super().__init__(args=args)
        self.args.__setattr__('output_dim', 128)
        self.args.__setattr__('num_heads', 8)
        self.args.__setattr__('num_layers', 4)
        self.gat = GAT(args=self.args).to(self.args.device)
        self.args.__setattr__('input_size', self.args.output_dim * self.args.num_heads)
        self.args.__setattr__('hidden_size', [])
        self.e = MLP(args=self.args).to(self.args.device)

    def forward(self, x, shortest_paths):
        data = Data(x=shortest_paths, edge_index=shortest_paths.nonzero().t().contiguous()).to(self.args.device)
        feature = self.gat(data.x, data.edge_index)
        pred = torch.zeros(x.shape).to(self.args.device)
        for i in range(x.shape[-1]):
            if not torch.isnan(x[i]):
                y = x[i] * feature[i]
                for fc, acs in zip(self.e.fcs, self.e.acs):
                    y = acs(fc(y))
                pred += y
        return pred / torch.sum(torch.isnan(x).int())
