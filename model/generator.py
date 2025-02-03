from model.mlp import MLP
from model.model import Model
import torch
from torch import nn
from torch import optim
from util import EarlyStopping


class Generator(Model):
    def __init__(self, args, road1, road2, prototype1, prototype2):
        super().__init__(args=args)
        self.model_x = MLP(args=self.args).to(self.args.device)
        if self.args.target_dataset == 'pemsd4':
            self.args.__setattr__('input_size', 307)
        elif self.args.target_dataset == 'pemsd8':
            self.args.__setattr__('input_size', 170)
        else:
            raise Exception('Unsupported target dataset.')
        self.model_g = MLP(args=self.args).to(self.args.device)
        self.model_p = MLP(args=self.args).to(self.args.device)
        self.road_conversion = self.road_gradient(road1=road1, road2=road2).to(self.args.device)
        self.prototype_conversion = self.prototype_gradient(prototype1=prototype1.to('cpu'),
                                                            prototype2=prototype2.to('cpu')).to(self.args.device)

    def road_gradient(self, road1, road2):
        early_stopping = EarlyStopping(args=self.args)
        A = torch.randn(road1.shape[0], road2.shape[0], requires_grad=True)
        optimizer = optim.Adam([A], lr=0.001)
        num_epochs = 10000
        r1 = torch.where(torch.isnan(road1), torch.zeros_like(road1), road1)
        r2 = torch.where(torch.isnan(road2), torch.zeros_like(road2), road2)

        def criterion(pred, target):
            mask = (target != 0).float()
            loss = nn.functional.mse_loss(pred * mask, target * mask, reduction='sum')
            num_valid_elements = torch.sum(mask)
            return loss / num_valid_elements if num_valid_elements > 0 else 0

        for epoch in range(num_epochs):
            optimizer.zero_grad()
            road = torch.matmul(torch.matmul(A.T, r1), A)
            loss = criterion(road, r2)
            loss.backward()
            optimizer.step()
            if early_stopping(loss=loss):
                print(f'Early stop as epoch {epoch}')
                break
        return A.detach()

    def prototype_gradient(self, prototype1, prototype2):
        early_stopping = EarlyStopping(args=self.args)
        A = torch.randn(prototype2.shape[0], prototype1.shape[0], requires_grad=True)
        optimizer = optim.Adam([A], lr=0.01)
        criterion = nn.MSELoss()
        num_epochs = 10000
        for epoch in range(num_epochs):
            optimizer.zero_grad()
            prototype = torch.matmul(A, prototype1)
            loss = criterion(prototype, prototype2)
            loss.backward()
            optimizer.step()
            if early_stopping(loss=loss):
                print(f'Early stop as epoch {epoch}')
                break
        return A.detach()

    def convert(self, data):
        return self.forward(x=data[0]), self.forward(x=data[1])

    def forward(self, x):
        y_x = x
        y_r = torch.matmul(x, self.road_conversion)
        y_p = torch.matmul(x, self.prototype_conversion.T)
        for fc_x, acs_x, fc_r, acs_r, fc_p, acs_p in zip(self.model_x.fcs, self.model_x.acs, self.model_g.fcs,
                                                         self.model_g.acs,
                                                         self.model_p.fcs,
                                                         self.model_p.acs):
            y_x = acs_x(fc_x(y_x))
            y_r = acs_r(fc_r(y_r))
            y_p = acs_p(fc_p(y_p))
        return y_x + y_r + y_p
