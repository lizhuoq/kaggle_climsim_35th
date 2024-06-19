import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):  #@save
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv1d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.ln1 = nn.LayerNorm(num_channels)
        self.ln2 = nn.LayerNorm(num_channels)

    def forward(self, X: torch.Tensor):
        """
        Input:
            X shape: batch_size, in_channel, seq_len
        Return:
            shape: batch_size, out_channel, seq_len
        """
        Y = F.relu(self.ln1(self.conv1(X).transpose(1, 2))).transpose(1, 2)
        Y = self.ln2(self.conv2(Y).transpose(1, 2)).transpose(1, 2)
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    

def resnet_block(input_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(input_channels, num_channels,
                                use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(num_channels, num_channels))
    return blk
    

class ResNet18(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.conv1 = nn.Conv1d(configs.in_channel, 64, kernel_size=7, stride=2, padding=3)
        self.ln1 = nn.LayerNorm(64)
        self.ac1 = nn.Sequential(nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block(64, 64, 2, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 2))
        b4 = nn.Sequential(*resnet_block(128, 256, 2))
        b5 = nn.Sequential(*resnet_block(256, 512, 2))
        self.net = nn.Sequential(b2, b3, b4, b5,
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(), nn.Linear(512, configs.out_channel))

    def forward(self, x: torch.Tensor):
        """
        Input shape:
            x: (batch_size, seq_len, in_channel)
        Return shape:
            (batch_size, out_channel)
        """
        x = self.ln1(self.conv1(x.transpose(1, 2)).transpose(1, 2))
        x = self.ac1(x.transpose(1, 2))
        return self.net(x)


class ResNet34(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.conv1 = nn.Conv1d(configs.in_channel, 64, kernel_size=7, stride=2, padding=3)
        self.ln1 = nn.LayerNorm(64)
        self.ac1 = nn.Sequential(nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block(64, 64, 3, first_block=True))
        b3 = nn.Sequential(*resnet_block(64, 128, 4))
        b4 = nn.Sequential(*resnet_block(128, 256, 6))
        b5 = nn.Sequential(*resnet_block(256, 512, 3))
        self.net = nn.Sequential(b2, b3, b4, b5,
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(), nn.Linear(512, configs.out_channel))

    def forward(self, x: torch.Tensor):
        """
        Input shape:
            x: (batch_size, seq_len, in_channel)
        Return shape:
            (batch_size, out_channel)
        """
        x = self.ln1(self.conv1(x.transpose(1, 2)).transpose(1, 2))
        x = self.ac1(x.transpose(1, 2))
        return self.net(x)
    

class Residual_1x1(nn.Module):  #@save
    def __init__(self, input_channels, hidden_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv1d(input_channels, hidden_channels,
                               kernel_size=1)
        self.conv2 = nn.Conv1d(hidden_channels, hidden_channels,
                               kernel_size=3, padding=1, stride=strides)
        if use_1x1conv:
            self.conv3 = nn.Conv1d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.ln1 = nn.LayerNorm(hidden_channels)
        self.ln2 = nn.LayerNorm(hidden_channels)

        self.conv4 = nn.Conv1d(hidden_channels, num_channels, kernel_size=1)
        self.ln3 = nn.LayerNorm(num_channels)

    def forward(self, X: torch.Tensor):
        """
        Input:
            X shape: batch_size, in_channel, seq_len
        Return:
            shape: batch_size, out_channel, seq_len
        """
        Y = F.relu(self.ln1(self.conv1(X).transpose(1, 2))).transpose(1, 2)
        Y = F.relu(self.ln2(self.conv2(Y).transpose(1, 2))).transpose(1, 2)
        Y = self.ln3(self.conv4(Y).transpose(1, 2)).transpose(1, 2)
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)
    

def resnet_block_1x1(input_channels, hidden_channels, num_channels, num_residuals,
                 first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual_1x1(input_channels, hidden_channels, num_channels,
                                use_1x1conv=True, strides=2))
        elif i == 0 and first_block:
            blk.append(Residual_1x1(input_channels, hidden_channels, num_channels, use_1x1conv=True))
        else:
            blk.append(Residual_1x1(num_channels, hidden_channels, num_channels))
    return blk
    

class ResNet50(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.conv1 = nn.Conv1d(configs.in_channel, 64, kernel_size=7, stride=2, padding=3)
        self.ln1 = nn.LayerNorm(64)
        self.ac1 = nn.Sequential(nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block_1x1(64, 64, 256, 3, first_block=True))
        b3 = nn.Sequential(*resnet_block_1x1(256, 128, 512, 4))
        b4 = nn.Sequential(*resnet_block_1x1(512, 256, 1024, 6))
        b5 = nn.Sequential(*resnet_block_1x1(1024, 512, 2048, 3))
        self.net = nn.Sequential(b2, b3, b4, b5,
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(), nn.Linear(2048, configs.out_channel))

    def forward(self, x: torch.Tensor):
        """
        Input shape:
            x: (batch_size, seq_len, in_channel)
        Return shape:
            (batch_size, out_channel)
        """
        x = self.ln1(self.conv1(x.transpose(1, 2)).transpose(1, 2))
        x = self.ac1(x.transpose(1, 2))
        return self.net(x)
    

class ResNet101(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.conv1 = nn.Conv1d(configs.in_channel, 64, kernel_size=7, stride=2, padding=3)
        self.ln1 = nn.LayerNorm(64)
        self.ac1 = nn.Sequential(nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block_1x1(64, 64, 256, 3, first_block=True))
        b3 = nn.Sequential(*resnet_block_1x1(256, 128, 512, 4))
        b4 = nn.Sequential(*resnet_block_1x1(512, 256, 1024, 23))
        b5 = nn.Sequential(*resnet_block_1x1(1024, 512, 2048, 3))
        self.net = nn.Sequential(b2, b3, b4, b5,
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(), nn.Linear(2048, configs.out_channel))

    def forward(self, x: torch.Tensor):
        """
        Input shape:
            x: (batch_size, seq_len, in_channel)
        Return shape:
            (batch_size, out_channel)
        """
        x = self.ln1(self.conv1(x.transpose(1, 2)).transpose(1, 2))
        x = self.ac1(x.transpose(1, 2))
        return self.net(x)
    

class ResNet152(nn.Module):
    def __init__(self, configs):
        super().__init__()
        self.conv1 = nn.Conv1d(configs.in_channel, 64, kernel_size=7, stride=2, padding=3)
        self.ln1 = nn.LayerNorm(64)
        self.ac1 = nn.Sequential(nn.ReLU(), nn.MaxPool1d(kernel_size=3, stride=2, padding=1))
        b2 = nn.Sequential(*resnet_block_1x1(64, 64, 256, 3, first_block=True))
        b3 = nn.Sequential(*resnet_block_1x1(256, 128, 512, 8))
        b4 = nn.Sequential(*resnet_block_1x1(512, 256, 1024, 36))
        b5 = nn.Sequential(*resnet_block_1x1(1024, 512, 2048, 3))
        self.net = nn.Sequential(b2, b3, b4, b5,
                    nn.AdaptiveAvgPool1d(1),
                    nn.Flatten(), nn.Linear(2048, configs.out_channel))

    def forward(self, x: torch.Tensor):
        """
        Input shape:
            x: (batch_size, seq_len, in_channel)
        Return shape:
            (batch_size, out_channel)
        """
        x = self.ln1(self.conv1(x.transpose(1, 2)).transpose(1, 2))
        x = self.ac1(x.transpose(1, 2))
        return self.net(x)
    

class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        if configs.n_layers == 18:
            self.model = ResNet18(configs)
        elif configs.n_layers == 34:
            self.model = ResNet34(configs)
        elif configs.n_layers == 50:
            self.model = ResNet50(configs)
        elif configs.n_layers == 101:
            self.model = ResNet101(configs)
        elif configs.n_layers == 152:
            self.model = ResNet152(configs)

    def forward(self, x):
        return self.model(x)
