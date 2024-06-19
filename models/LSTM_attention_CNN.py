from torch import nn
import torch


class TextCNN(nn.Module):
    def __init__(self, in_channel, d_model, kernel_sizes: list[int], num_channels: list[int], dropout):
        super(TextCNN, self).__init__()
        self.embedding = nn.Linear(in_channel, d_model)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(sum(num_channels), d_model)
        # 最大时间汇聚层没有参数，因此可以共享此实例
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.relu = nn.ReLU()
        # 创建多个一维卷积层
        self.convs = nn.ModuleList()
        for c, k in zip(num_channels, kernel_sizes):
            self.convs.append(nn.Conv1d(d_model, c, k))

    def forward(self, x: torch.Tensor):
        """
        Input shape:
            x: (batch_size, seq_len, in_channel)
        Return shape:
            (batch_size, d_model)
        """
        # 沿着向量维度将两个嵌入层连结起来，
        # 每个嵌入层的输出形状都是（批量大小，词元数量，词元向量维度）连结起来
        embeddings = self.embedding(x).transpose(1, 2)
        # 根据一维卷积层的输入格式，重新排列张量，以便通道作为第2维
        # 每个一维卷积层在最大时间汇聚层合并后，获得的张量形状是（批量大小，通道数，1）
        # 删除最后一个维度并沿通道维度连结
        encoding = torch.cat([
            torch.squeeze(self.relu(self.pool(conv(embeddings))), dim=-1)
            for conv in self.convs], dim=1)
        outputs = self.decoder(self.dropout(encoding))
        return outputs


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.pre_attention = nn.MultiheadAttention(
            configs.in_channel, 
            num_heads=1, 
            dropout=configs.dropout, 
            batch_first=True
        )
        self.cnn = TextCNN(
            configs.in_channel, 
            configs.d_model, 
            kernel_sizes=[3, 4, 5], 
            num_channels=[configs.d_model for _ in range(3)], 
            dropout=configs.dropout
        )
        self.lstm = nn.LSTM(
            input_size=configs.in_channel, 
            hidden_size=configs.d_model, 
            num_layers=configs.n_layers, 
            batch_first=True, 
            dropout=configs.dropout, 
            bidirectional=configs.bidirectional
        )
        self.end_attention = nn.MultiheadAttention(
            embed_dim=configs.d_model * 2 + configs.d_model if configs.bidirectional else configs.d_model + configs.d_model, 
            num_heads=1, 
            dropout=configs.dropout, 
            batch_first=True
        )
        self.projection = nn.Linear(
            in_features=configs.d_model * 2 + configs.d_model if configs.bidirectional else configs.d_model + configs.d_model, 
            out_features=configs.out_channel
        )

    def forward(self, x):
        """
        Input shape:
            x: (batch_size, seq_len, in_channel)
        Return shape:
            (batch_size, seq_len, out_channel)
        """
        x, _ = self.pre_attention(x, x, x)
        cnn_out = self.cnn(x).unsqueeze(1).repeat(1, 60, 1)
        x, _ = self.lstm(x)
        x = torch.cat([x, cnn_out], dim=2)
        x, _ = self.end_attention(x, x, x)
        return self.projection(x)