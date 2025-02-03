import argparse
from dataset.pemsd4 import PeMSD4
from dataset.pemsd8 import PeMSD8
import heapq
from model.discriminator import Discriminator
from model.extender import Extender
from model.generator import Generator
from model.enhancer import Enhancer
from party import Party
import torch
from torch import nn
from torch import optim


def dijkstra(graph, start):
    n = len(graph)
    distances = [float('inf')] * n
    distances[start] = 0
    heap = [(0, start)]
    while heap:
        current_dist, u = heapq.heappop(heap)
        if current_dist > distances[u]:
            continue
        for v, weight in enumerate(graph[u]):
            if weight > 0:
                distance = current_dist + weight
                if distance < distances[v]:
                    distances[v] = distance
                    heapq.heappush(heap, (distance, v))
    return distances


def shortest_distance_matrix(graph):
    n = len(graph)
    distance_matrix = [[0] * n for _ in range(n)]
    for i in range(n):
        distances = dijkstra(graph, i)
        distance_matrix[i] = distances
    return distance_matrix


class Client(Party):
    def __init__(self, args, index, party, server=None):
        super().__init__(args=args, party=party)
        self.dataset = None
        self.data_size = 0
        self.extender, self.enhancer = None, None
        self.frozen_aggregated_data = None
        self.generator, self.discriminator = None, None
        self.index = index
        self.optimizer_ex, self.optimizer_eh = None, None
        self.optimizer_g, self.optimizer_d = None, None
        self.prev_data = None, None
        self.server = server

    def init(self):
        if self.args.source_dataset[self.index] == 'pemsd4':
            self.dataset = PeMSD4(args=self.args)
            self.args.__setattr__('input_size', 307)
            self.args.__setattr__('output_size', 307)
        elif self.args.source_dataset[self.index] == 'pemsd8':
            self.dataset = PeMSD8(args=self.args)
            self.args.__setattr__('input_size', 170)
            self.args.__setattr__('output_size', 170)
        elif self.args.source_dataset[self.index] == 'ft_aed':
            self.dataset = PeMSD8(args=self.args)
            self.args.__setattr__('input_size', 196)
            self.args.__setattr__('output_size', 196)
        elif self.args.source_dataset[self.index] == 'hk_traffic':
            self.dataset = PeMSD8(args=self.args)
            self.args.__setattr__('input_size', 411)
            self.args.__setattr__('output_size', 411)
        else:
            raise Exception('Unsupported source dataset.')
        self.dataset.load_dataset()

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
            raise Exception('Unsupported source dataset.')

        self.args.__setattr__('hidden_size', [1024, 1024])
        self.generator = Generator(args=self.args, road1=torch.tensor(self.dataset.road_network, dtype=torch.float32),
                                   road2=torch.tensor(self.server.dataset.road_network, dtype=torch.float32),
                                   prototype1=self.dataset.prototype, prototype2=self.server.dataset.prototype)
        self.discriminator = Discriminator(args=self.args)
        self.optimizer_g = optim.Adam(params=self.generator.parameters(), lr=self.args.learning_rate)
        self.optimizer_d = optim.Adam(params=self.discriminator.parameters(), lr=self.args.learning_rate)

    def impute(self):
        if self.args.source_dataset == 'pemsd4':
            self.dataset = PeMSD4(args=self.args)
            self.args.__setattr__('input_dim', 307)
            self.args.__setattr__('output_size', 307)
        elif self.args.source_dataset == 'pemsd8':
            self.dataset = PeMSD8(args=self.args)
            self.args.__setattr__('input_dim', 170)
            self.args.__setattr__('output_size', 170)
        elif self.args.target_dataset == 'ft_aed':
            self.dataset = PeMSD8(args=self.args)
            self.args.__setattr__('input_dim', 196)
            self.args.__setattr__('output_size', 196)
        elif self.args.target_dataset == 'hk_traffic':
            self.dataset = PeMSD8(args=self.args)
            self.args.__setattr__('input_dim', 411)
            self.args.__setattr__('output_size', 411)
        else:
            raise Exception('Unsupported source dataset.')
        self.dataset.load_dataset()

        self.extender = Extender(args=self.args)
        self.enhancer = Enhancer(args=self.args)
        self.optimizer_ex = optim.Adam(params=self.extender.parameters(), lr=self.args.learning_rate)
        self.optimizer_eh = optim.Adam(params=self.enhancer.parameters(), lr=self.args.learning_rate)
        road = torch.tensor(self.dataset.road_network, dtype=torch.float32).to(self.args.device)
        road = torch.where(torch.isnan(road), torch.zeros_like(road), road).tolist()
        shortest_paths = torch.tensor(shortest_distance_matrix(road))
        shortest_paths[torch.isinf(shortest_paths)] = 0

        missing = list()
        averaged_loss = 0
        for epoch in range(self.args.epoch):
            while True:
                if self.data_size >= len(self.dataset.train_loader):
                    self.dataset.reset_train()
                    self.data_size = 0
                data = next(self.dataset.train_iter)
                data = data[0][0, 0, :].to(self.args.device)
                self.data_size += 1
                if torch.isnan(data).any():
                    break

            criterion_ex = nn.L1Loss()
            self.extender.train()
            self.optimizer_ex.zero_grad()
            output = self.extender(data, shortest_paths)
            mask = ~torch.isnan(data)
            missing.append((self.data_size, mask, data))
            loss = criterion_ex(
                output[mask] * (self.dataset.scalar['train']['max'] - self.dataset.scalar['train']['min']) +
                self.dataset.scalar['train']['min'],
                data[mask] * (self.dataset.scalar['train']['max'] - self.dataset.scalar['train']['min']) +
                self.dataset.scalar['train']['min'])

            loss.backward()
            self.optimizer_ex.step()

            averaged_loss += loss.item()
            if epoch % 10 == 0:
                if epoch != 0:
                    averaged_loss /= 10
                print(f'Epoch: {epoch}, loss: {averaged_loss}')
                averaged_loss = 0

        self.data_size = 0
        missing_iter = iter(missing)
        for epoch in range(self.args.epoch):
            while True:
                if self.data_size >= len(missing):
                    missing_iter = iter(missing)
                    self.data_size = 0
                i, mask, data = next(missing_iter)
                self.data_size += 1
                if 12 <= i <= len(self.dataset.train_loader) - 12:
                    break

            prev_data = torch.stack([self.dataset.train_loader.dataset[j][0] for j in range(i - 12, i)], dim=0).to(
                self.args.device)
            late_data = torch.stack([self.dataset.train_loader.dataset[j] for j in range(i + 12, i, -1)], dim=0).to(
                self.args.device)
            criterion_eh = nn.L1Loss()
            self.enhancer.train()
            self.optimizer_eh.zero_grad()
            output = self.enhancer(prev_data, late_data)
            loss = criterion_eh(
                output[mask] * (self.dataset.scalar['train']['max'] - self.dataset.scalar['train']['min']) +
                self.dataset.scalar['train']['min'],
                data[mask] * (self.dataset.scalar['train']['max'] - self.dataset.scalar['train']['min']) +
                self.dataset.scalar['train']['min'])

            loss.backward()
            self.optimizer_eh.step()

            averaged_loss += loss.item()
            if epoch % 10 == 0:
                if epoch != 0:
                    averaged_loss /= 10
                print(f'Epoch: {epoch}, loss: {averaged_loss}')
                averaged_loss = 0

    def train(self, epoch, prototype, aggregated_data):
        if epoch % self.args.frozen == 0:
            self.frozen_aggregated_data = aggregated_data

        if self.data_size >= len(self.dataset.train_loader):
            self.dataset.reset_train()
            self.data_size = 0
        batch_x, batch_y = next(self.dataset.train_iter)
        batch_x = batch_x.to(self.args.device)
        batch_y = batch_y.to(self.args.device)
        self.data_size += 1

        criterion_g = nn.L1Loss()
        self.generator.train()
        self.optimizer_g.zero_grad()
        converted_data = self.generator.convert(data=(batch_x, batch_y))
        loss = criterion_g(torch.cat(converted_data, dim=1), prototype)

        if epoch == 0:
            mask_data = converted_data
        else:
            labels = torch.cat([
                torch.zeros(converted_data[0].numel() // converted_data[0].shape[-1], device=self.args.device),
                torch.zeros(converted_data[1].numel() // converted_data[1].shape[-1], device=self.args.device),
                torch.ones(self.frozen_aggregated_data[0].numel() // self.frozen_aggregated_data[0].shape[-1],
                           device=self.args.device),
                torch.ones(self.frozen_aggregated_data[1].numel() // self.frozen_aggregated_data[1].shape[-1],
                           device=self.args.device)
            ]).unsqueeze(1)
            criterion_d = nn.BCELoss()
            self.discriminator.train()
            self.optimizer_d.zero_grad()
            output = self.discriminator.classify(data=converted_data, aggregated_data=self.frozen_aggregated_data)
            loss -= 0.4 * criterion_d(output, labels)

            mask_data = converted_data[0] + self.frozen_aggregated_data[0] - self.prev_data[0], \
                        converted_data[1] + self.frozen_aggregated_data[1] - self.prev_data[1]
        self.prev_data = converted_data[0].detach(), converted_data[1].detach()

        if epoch % self.args.frozen == 0:
            return mask_data, loss
        else:
            return None, None

    def step(self):
        self.optimizer_g.step()
        self.optimizer_d.step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--dataset', type=str, default='pemsd4')
    parser.add_argument('--imputation', type=bool, default=True)
    parser.add_argument('--model', type=str, default='mlp')
    parser.add_argument('--task', type=str, default='flow')
    parser.add_argument('--test_split', type=float, default=0.1)
    parser.add_argument('--time_step', type=float, default=1)
    parser.add_argument('--train_split', type=float, default=0.8)
    parser.add_argument('--val_split', type=float, default=0.1)
    args = parser.parse_args()

    client = Client(args=args, index=0, party=None)
    client.impute()
