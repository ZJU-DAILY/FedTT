from model.mlp import MLP
from model.model import Model
import torch


class Discriminator(Model):
    def __init__(self, args):
        super().__init__(args=args)
        if self.args.target_dataset == 'pemsd4':
            self.args.__setattr__('input_size', 307)
        elif self.args.target_dataset == 'pemsd8':
            self.args.__setattr__('input_size', 170)
        elif self.args.target_dataset == 'ft_aed':
            self.args.__setattr__('input_size', 170)
        elif self.args.target_dataset == 'hk_traffic':
            self.args.__setattr__('input_size', 170)
        else:
            raise Exception('Unsupported target dataset.')
        self.args.__setattr__('output_size', 1)
        self.model = MLP(args=self.args).to(self.args.device)

    def classify(self, data, aggregated_data):
        all_data = torch.cat([data[0].view(-1, *data[0].shape[2:]), data[1].view(-1, *data[1].shape[2:]),
                              aggregated_data[0].view(-1, *aggregated_data[0].shape[2:]),
                              aggregated_data[1].view(-1, *aggregated_data[1].shape[2:])], dim=0)
        return self.forward(x=all_data)

    def forward(self, x):
        y = x
        for fc, acs in zip(self.model.fcs, self.model.acs):
            y = acs(fc(y))
        return y
