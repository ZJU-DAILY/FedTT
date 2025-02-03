from dataset.pemsd4 import PeMSD4
from dataset.pemsd8 import PeMSD8
from model.cnn import CNN
from model.discriminator import Discriminator
from model.dyhsl import DYHSL
from model.gru import GRU
from model.mlp import MLP
from model.pdformer import PDFormer
from model.stssl import STSSL
from party.party import Party
import secretflow as sf
import torch
from torch import nn
from torch import optim


class Server(Party):
    def __init__(self, args, party, clients):
        super().__init__(args=args, party=party)
        self.clients = clients
        self.dataset = None
        self.data_size = 0
        self.discriminator = None
        self.frozen_mask_data, self.frozen_loss = None, None
        self.model = None
        self.optimizer_t, self.optimizer_d = None, None

    def init(self):
        if self.args.target_dataset == 'pemsd4':
            self.dataset = PeMSD4(args=self.args)
            self.args.__setattr__('input_size', 307)
            self.args.__setattr__('output_size', 307)
        elif self.args.target_dataset == 'pemsd8':
            self.dataset = PeMSD8(args=self.args)
            self.args.__setattr__('input_size', 170)
            self.args.__setattr__('output_size', 170)
        elif self.args.target_dataset == 'ft_aed':
            self.dataset = PeMSD8(args=self.args)
            self.args.__setattr__('input_size', 196)
            self.args.__setattr__('output_size', 196)
        elif self.args.target_dataset == 'hk_traffic':
            self.dataset = PeMSD8(args=self.args)
            self.args.__setattr__('input_size', 411)
            self.args.__setattr__('output_size', 411)
        else:
            raise Exception('Unsupported dataset.')
        self.dataset.load_dataset()

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
        self.discriminator = Discriminator(args=self.args)
        self.optimizer_t = optim.Adam(params=self.model.parameters(), lr=self.args.learning_rate)
        self.optimizer_d = optim.Adam(params=self.discriminator.parameters(), lr=self.args.learning_rate)

    def train(self):
        prev_aggregated_data = None, None
        for epoch in range(self.args.epoch):
            clients_mask_data = list()
            loss = 0
            for client in self.clients:
                if epoch % self.args.frozen == 0:
                    mask_data, client_loss = client.party(client.train)(
                        epoch=epoch, prototype=self.dataset.prototype,
                        aggregated_data=sf.PYUObject(client.party, prev_aggregated_data))
                    self.frozen_mask_data, self.frozen_loss = mask_data.to(self.party), client_loss.to(self.party)
                else:
                    mask_data, client_loss = self.frozen_mask_data, self.frozen_loss
                clients_mask_data.append(mask_data)
                loss += client_loss
            mask_data = clients_mask_data[0]
            for i in range(1, len(self.clients)):
                mask_data = mask_data[0] + clients_mask_data[i][0], mask_data[1] + clients_mask_data[i][1]
            if epoch == 0:
                aggregated_data = mask_data
            else:
                aggregated_data = mask_data[0] - (len(self.clients) - 1) * prev_aggregated_data[0], \
                                  mask_data[1] - (len(self.clients) - 1) * prev_aggregated_data[1]
            aggregated_data = aggregated_data[0] / len(self.clients), aggregated_data[1] / len(self.clients)
            prev_aggregated_data = aggregated_data[0].detach(), aggregated_data[1].detach()

            if self.data_size >= len(self.dataset.train_loader):
                self.dataset.reset_train()
                self.data_size = 0
            batch_x, batch_y = next(self.dataset.train_iter)
            batch_x = batch_x.to(self.args.device)
            batch_y = batch_y.to(self.args.device)
            self.data_size += 1

            labels = torch.cat([torch.zeros(batch_x.numel() // batch_x.shape[-1], device=self.args.device),
                                torch.zeros(batch_y.numel() // batch_y.shape[-1], device=self.args.device),
                                torch.ones(aggregated_data[0].numel() // aggregated_data[0].shape[-1],
                                           device=self.args.device),
                                torch.ones(aggregated_data[1].numel() // aggregated_data[1].shape[-1],
                                           device=self.args.device)]).unsqueeze(1)
            criterion_d = nn.BCELoss()
            self.discriminator.train()
            self.optimizer_d.zero_grad()
            output = self.discriminator.classify(data=(batch_x, batch_y), aggregated_data=aggregated_data)
            loss = -0.7 * criterion_d(output, labels)

            criterion_t = nn.L1Loss()
            self.model.train()
            self.optimizer_t.zero_grad()
            output = self.model(aggregated_data[0])
            loss += criterion_t(
                output * (self.dataset.scalar['train']['max'] - self.dataset.scalar['train']['min']) +
                self.dataset.scalar['train']['min'],
                aggregated_data[1] * (self.dataset.scalar['train']['max'] - self.dataset.scalar['train']['min']) +
                self.dataset.scalar['train']['min'])

            output = self.model(batch_x)
            loss += criterion_t(
                output * (self.dataset.scalar['train']['max'] - self.dataset.scalar['train']['min']) +
                self.dataset.scalar['train']['min'],
                batch_y * (self.dataset.scalar['train']['max'] - self.dataset.scalar['train']['min']) +
                self.dataset.scalar['train']['min'])

            loss.backward()
            self.optimizer_t.step()
            self.optimizer_d.step()
            for client in self.clients:
                client.step()

            print(f'Epoch: {epoch}, loss: {loss.item()}')
