from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.embedding = nn.Linear(configs.in_channel, configs.d_model)
        self.position_embedding = nn.Embedding(60, configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=configs.d_model, 
                nhead=configs.nhead, 
                dim_feedforward=configs.d_ff, 
                dropout=configs.dropout, 
                batch_first=True
            ), 
            num_layers=configs.n_layers, 
        )
        self.projection = nn.Linear(configs.d_model, configs.out_channel)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Input:
            x: [batch_size, seq_len, in_channel]
        Return:
            [batch_size, seq_len, out_channel]
        """
        pos = self.position_embedding(torch.arange(x.shape[1]).to(x.device)).unsqueeze(0)
        x = self.dropout(self.embedding(x) + pos)
        x = self.encoder(x)
        return self.projection(x)
      