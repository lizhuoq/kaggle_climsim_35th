from torch import nn
import torch


class ConvBlock(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.blk = nn.Sequential(
            nn.Conv1d(d_model, d_model * 4, kernel_size=3, padding=1), 
            nn.BatchNorm1d(d_model * 4), 
            nn.ReLU(), 
            nn.Conv1d(d_model * 4, d_model, kernel_size=3, padding=1), 
            nn.BatchNorm1d(d_model)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        y = self.blk(x)
        return self.relu(y + x)


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.embedding = nn.Linear(configs.in_channel, configs.d_model)
        blks = []
        for _ in range(configs.n_layers):
            blks.append(ConvBlock(configs.d_model))
        self.blks = nn.Sequential(*blks)
        self.projection = nn.Linear(configs.d_model, configs.out_channel)

    def forward(self, X: torch.Tensor):
        """
        Input:
            X shape: batch_size, seq_len, in_channel
        Return:
            shape: batch_size, seq_len, in_channel
        """
        X = self.embedding(X).transpose(1, 2)
        X = self.blks(X)
        return self.projection(X.transpose(1, 2))