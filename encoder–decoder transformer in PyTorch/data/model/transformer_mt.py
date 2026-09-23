import math
import torch
import torch.nn as nn
from positional_encoding import PositionalEncoding

class TransformerMT(nn.Module):
    def __init__(self, en_vocab_size, hi_vocab_size, d_model, nhead, num_layers, pad_idx):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx

        self.src_embed = nn.Embedding(en_vocab_size, d_model, padding_idx=pad_idx)
        self.tgt_embed = nn.Embedding(hi_vocab_size, d_model, padding_idx=pad_idx)
        self.pos = PositionalEncoding(d_model)

        self.tr = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dim_feedforward=4 * d_model,
            batch_first=True,
        )

        self.out = nn.Linear(d_model, hi_vocab_size)

    def make_src_padding_mask(self, src):
        return src == self.pad_idx

    def make_tgt_mask(self, tgt):
        L = tgt.size(1)
        mask = torch.triu(torch.ones(L, L), diagonal=1).bool()
        return mask.to(tgt.device)

    def forward(self, src, tgt):
        src_emb = self.pos(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        src_pad_mask = self.make_src_padding_mask(src)
        tgt_pad_mask = (tgt == self.pad_idx)
        tgt_mask = self.make_tgt_mask(tgt)

        h = self.tr(
            src=src_emb,
            tgt=tgt_emb,
            src_key_padding_mask=src_pad_mask,
            tgt_key_padding_mask=tgt_pad_mask,
            tgt_mask=tgt_mask
        )
        return self.out(h)
