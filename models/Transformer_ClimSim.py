from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.levels = configs.levels
        self.vertical_out_channel = configs.vertical_out_channel
        self.embedding = nn.Linear(configs.vertical_in_channel + configs.scalar_in_channel, configs.d_model)
        self.dropout = nn.Dropout(configs.dropout)
        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=configs.d_model, 
                nhead=configs.nhead, 
                dim_feedforward=configs.d_ff, 
                dropout=configs.dropout, 
                activation=configs.activation, 
                batch_first=True
            ), 
            num_layers=configs.n_layers, 
        )
        self.projection = nn.Linear(configs.d_model, configs.vertical_out_channel + configs.scalar_out_channel)

    def forward(self, in_vertical: torch.Tensor, in_scalar: torch.Tensor) -> torch.Tensor:
        """
        Input:
            in_vertical shape: batch_size, vertical_in_channel, level
            in_scalar shape: batch_size, scalar_in_channel
        Return:
            vertical_out shape: batch_size, vertical_out_channel, level
            out_scalar shape: batch_size, scalar_out_channel
        """
        in_scalar = in_scalar.unsqueeze(1).repeat(1, self.levels, 1)
        enc_in = torch.cat([in_vertical.transpose(1, 2), in_scalar], dim=2)
        enc_in = self.dropout(self.embedding(enc_in))
        enc_out = self.encoder(enc_in)
        enc_out = self.projection(enc_out)
        vertical_out = enc_out[:, :, :self.vertical_out_channel].transpose(1, 2)
        scalar_out = enc_out[:, 0, self.vertical_out_channel:]
        return vertical_out, scalar_out