from torch import nn
import torch


class Model(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.src_embedding = nn.Linear(configs.in_channel, configs.d_model)
        self.tgt_embedding = nn.Linear(configs.out_channel, configs.d_model)
        self.position_embedding = nn.Embedding(60, configs.d_model)
        self.transformer = nn.Transformer(
            d_model=configs.d_model, 
            nhead=configs.nhead, 
            num_encoder_layers=configs.e_layers, 
            num_decoder_layers=configs.d_layers, 
            dim_feedforward=configs.d_ff, 
            dropout=configs.dropout, 
            batch_first=True
        )
        self.projection = nn.Linear(configs.d_model, configs.out_channel)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, tgt_is_causal=None) -> torch.Tensor:
        """
        Input:
            src: [batch_size, seq_len, in_channel]
            tgt: [batch_size, seq_len, out_channel]
        Return:
            [batch_size, seq_len, out_channel]
        """
        pos_embedding = self.position_embedding(torch.arange(60).to(src.device)).unsqueeze(0)
        src = self.src_embedding(src) + pos_embedding
        tgt = self.tgt_embedding(tgt) + self.position_embedding(torch.arange(tgt.shape[1]).to(tgt.device)).unsqueeze(0)
        if tgt_is_causal is None:
            output = self.transformer(src, tgt)
        else:
            output = self.transformer(src, tgt, tgt_is_causal=tgt_is_causal, tgt_mask=self.transformer.generate_square_subsequent_mask(tgt.shape[1]))
        return self.projection(output)
    
    def decode(self, src, tgt):
        """
        Input:
            src: [batch_size, seq_len, in_channel]
            tgt: [batch_size, 1, out_channel]
        Return:
            [batch_size, seq_len, out_channel]
        """
        pos_embedding = self.position_embedding(torch.arange(60).to(src.device)).unsqueeze(0)
        src = self.src_embedding(src) + pos_embedding
        memory = self.transformer.encoder(src)
        for i in range(60):
            dec_in = self.tgt_embedding(tgt) + self.position_embedding(torch.arange(tgt.shape[1]).to(tgt.device)).unsqueeze(0)
            step_out = self.transformer.decoder(dec_in, memory)
            step_out = self.projection(step_out)
            tgt = torch.cat([tgt, step_out[:, -1:, :]], dim=1)
        return tgt[:, 1:, :]
