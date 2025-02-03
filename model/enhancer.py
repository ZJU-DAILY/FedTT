from model.model import Model
from model.cnn import CNN
from model.dyhsl import DYHSL
from model.gru import GRU
from model.mlp import MLP
from model.pdformer import PDFormer
from model.stssl import STSSL


class Enhancer(Model):
    def __init__(self, args):
        super().__init__(args=args)
        if self.args.model == 'gru':
            self.args.__setattr__('num_layers', 2)
            self.args.__setattr__('hidden_size', 50)
            self.model = GRU(args=self.args).to(self.args.device)
        elif self.args.model == 'mlp':
            self.args.__setattr__('hidden_size', [50, 50])
            self.model = MLP(args=self.args).to(self.args.device)
        elif self.args.model == 'cnn':
            self.model = CNN(args=self.args).to(self.args.device)
        elif self.args.model == 'stssl':
            self.model = STSSL(args=self.args).to(self.args.device)
        elif self.args.model == 'dyhsl':
            self.model = DYHSL(args=self.args).to(self.args.device)
        elif self.args.model == 'pdformer':
            self.model = PDFormer(args=self.args).to(self.args.device)
        else:
            raise Exception('Unsupported model.')

    def forward(self, prev_data, late_data):
        return (self.model(prev_data) + self.model(late_data)) / 2
