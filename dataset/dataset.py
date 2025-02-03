import math
import numpy as np
from torch import tensor
from torch.utils.data import DataLoader


class Dataset(object):
    def __init__(self, args):
        self.args = args
        self.data = None
        self.prototype = None
        self.road_network = None
        self.scalar = None
        self.train_iter, self.test_iter, self.val_iter = None, None, None
        self.train_loader, self.test_loader, self.val_loader = None, None, None

    def load_dataset(self):
        if self.args.task == 'flow':
            self.data = self.data[..., 0]
        elif self.args.task == 'speed':
            self.data = self.data[..., 1]
        elif self.args.task == 'occupancy':
            self.data = self.data[..., 2]
        else:
            raise Exception('Unsupported task.')

        dataset_size = self.data.shape[0]
        dataset_end_train = math.ceil(dataset_size * self.args.train_split)
        dataset_end_test = math.ceil(dataset_size * (self.args.train_split + self.args.test_split))
        dataset_train = self.data[:dataset_end_train, ...]
        dataset_test = self.data[dataset_end_train:dataset_end_test, ...]
        dataset_val = self.data[dataset_end_test:, ...]
        dataset_size_train = dataset_train.shape[0]
        dataset_size_test = dataset_test.shape[0]
        dataset_size_val = dataset_val.shape[0]
        if self.args.time_step > 1:
            dataset_train = dataset_train[
                            :-(dataset_size_train % np.lcm(self.args.batch_size, self.args.time_step + 1))]
            dataset_test = dataset_test[:-(dataset_size_test % np.lcm(self.args.batch_size, self.args.time_step + 1))]
            dataset_val = dataset_val[:-(dataset_size_val % np.lcm(self.args.batch_size, self.args.time_step + 1))]
            dataset_size_train = dataset_train.shape[0]
            dataset_size_test = dataset_test.shape[0]
            dataset_size_val = dataset_val.shape[0]

        # scalar
        max_train = np.nanmax(dataset_train)
        max_test = np.nanmax(dataset_test)
        max_val = np.nanmax(dataset_val)
        min_train = np.nanmin(dataset_train)
        min_test = np.nanmin(dataset_test)
        min_val = np.nanmin(dataset_val)
        dataset_train = (dataset_train - min_train) / (max_train - min_train)
        dataset_test = (dataset_test - min_test) / (max_test - min_test)
        dataset_val = (dataset_val - min_val) / (max_val - min_val)

        # dataloader
        self.train_loader = DataLoader(
            [(dataset_train[i:i + self.args.time_step, ...],
              dataset_train[i + self.args.time_step:i + self.args.time_step + 1, ...])
             for i in range(0, dataset_size_train - self.args.time_step, self.args.time_step + 1)],
            self.args.batch_size, shuffle=False)
        self.test_loader = DataLoader(
            [(dataset_test[i:i + self.args.time_step, ...],
              dataset_test[i + self.args.time_step:i + self.args.time_step + 1, ...])
             for i in range(0, dataset_size_test - self.args.time_step, self.args.time_step + 1)],
            self.args.batch_size, shuffle=False)
        self.val_loader = DataLoader(
            [(dataset_val[i:i + self.args.time_step, ...],
              dataset_val[i + self.args.time_step:i + self.args.time_step + 1, ...])
             for i in range(0, dataset_size_val - self.args.time_step, self.args.time_step + 1)],
            self.args.batch_size, shuffle=False)
        self.scalar = {'train': {'max': max_train, 'min': min_train},
                       'test': {'max': max_test, 'min': min_test},
                       'val': {'max': max_val, 'min': min_val}}

        self.train_iter = iter(self.train_loader)
        self.test_iter = iter(self.test_loader)
        self.val_iter = iter(self.val_loader)

        self.prototype = tensor(sum(dataset_train) / len(dataset_train)).to(self.args.device)

        matrix = np.full((307, 307), np.nan)
        for _, row in self.road_network.iterrows():
            from_node = int(row["from"]) - 1
            to_node = int(row["to"]) - 1
            cost = row["cost"]
            matrix[from_node][to_node] = cost
        for i in range(matrix.shape[0]):
            matrix[i][i] = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                if matrix[i][j] != matrix[j][i] and (np.isnan(matrix[i][j]) or np.isnan(matrix[j][i])):
                    if (matrix[i][j] == 0 or np.isnan(matrix[i][j])) and matrix[j][i] != 0:
                        matrix[i][j] = matrix[j][i]
                    elif (matrix[j][i] == 0 or np.isnan(matrix[j][i])) and matrix[i][j] != 0:
                        matrix[j][i] = matrix[i][j]
                    else:
                        raise Exception('Error')
        self.road_network = matrix

    def reset_train(self):
        self.train_iter = iter(self.train_loader)

    def reset_test(self):
        self.test_iter = iter(self.test_loader)

    def reset_val(self):
        self.val_iter = iter(self.val_loader)
