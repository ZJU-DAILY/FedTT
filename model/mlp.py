from model.model import Model
from torch import nn


class MLP(Model):
    def __init__(self, args):
        super().__init__(args=args)
        self.fcs = nn.ModuleList()
        self.acs = nn.ModuleList()
        prev_size = self.args.input_size
        for hidden_size in self.args.hidden_size:
            self.fcs.append(nn.Linear(prev_size, hidden_size).to(self.args.device))
            self.acs.append(nn.Sigmoid().to(self.args.device))
            prev_size = hidden_size
        self.fcs.append(nn.Linear(prev_size, self.args.output_size).to(self.args.device))
        self.acs.append(nn.Sigmoid().to(self.args.device))

    def forward(self, x):
        y = x
        for fc, acs in zip(self.fcs, self.acs):
            y = acs(fc(y))
        return y[:, :1, :]