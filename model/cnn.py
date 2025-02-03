from model.model import Model
import torch


class CNN(Model):
    def __init__(self, args):
        super().__init__(args=args)
        num_features = 1
        lags = 12
        out_dim = 1
        in_channels = [1, 16, 16, 32]
        out_channels = [16, 16, 32, 32]
        kernel_sizes = [(16, 3), (3, 5), (8, 3), (4, 3)]
        pool_kernel_sizes = [(2, 1)]
        assert len(in_channels) == len(out_channels) == len(kernel_sizes)
        self.activation = torch.nn.ReLU()
        self.num_lags = lags
        self.nodes = self.args.input_size
        self.num_features = num_features
        self.conv1 = torch.nn.Conv2d(in_channels=in_channels[0], out_channels=out_channels[0],
                                     kernel_size=kernel_sizes[0], padding="same").to(self.args.device)
        self.conv2 = torch.nn.Conv2d(in_channels=in_channels[1], out_channels=out_channels[1],
                                     kernel_size=kernel_sizes[1], padding="same").to(self.args.device)
        self.conv3 = torch.nn.Conv2d(in_channels=in_channels[2], out_channels=out_channels[2],
                                     kernel_size=kernel_sizes[3], padding="same").to(self.args.device)
        self.conv4 = torch.nn.Conv2d(in_channels=in_channels[3], out_channels=out_channels[3],
                                     kernel_size=kernel_sizes[3], padding="same").to(self.args.device)
        self.pool = torch.nn.AvgPool2d(kernel_size=pool_kernel_sizes[0]).to(self.args.device)
        # num_outs = out_channels(conv) * (time_lags / avg_pool[0]) *  (num_features / avg_pool[1])
        kernel0, kernel1 = pool_kernel_sizes[-1][0], pool_kernel_sizes[-1][1]
        self.fc = torch.nn.Linear(
            # in_features=(out_channels[3] * int(lags / kernel0) * int(num_features / kernel1)) + exogenous_dim,
            in_features=32 * 6,
            out_features=out_dim).to(self.args.device)

    def forward(self, x):
        x = x.transpose(1, 2).transpose(1, 3)
        x = self.conv1(x)
        x = self.activation(x)
        x = self.conv2(x)
        x = self.activation(x)
        x = self.conv3(x)
        x = self.activation(x)
        x = self.conv4(x)
        x = self.activation(x)
        x = self.pool(x)
        x = x.transpose(1, 3)
        x = x.reshape(self.args.batch_size, self.nodes, 6 * 32)
        x = self.fc(x)
        x = x.transpose(1, 2).unsqueeze(-1)
        return x
