from model.model import Model
from torch import nn


class GRU(Model):
    def __init__(self, args):
        super().__init__(args=args)
        self.gru = nn.GRU(input_size=self.args.input_size, num_layers=self.args.num_layers,
                          hidden_size=self.args.hidden_size, batch_first=True).to(self.args.device)
        self.linear = nn.Linear(self.args.hidden_size, self.args.output_size).to(self.args.device)

    def forward(self, x):  # B x T x N x D
        output, _ = self.gru(x[..., 0])
        return self.linear(output[:, -1:, :]).unsqueeze(3)
