# tokenizer/models/transformer_mt.py

import math
import torch
import torch.nn as nn

from .positional_encoding import PositionalEncoding


class TransformerMT(nn.Module):
    def __init__(self, en_vocab_size, hi_vocab_size,
                 d_model, nhead, num_layers, pad_id: int):
        super().__init__()
        self.d_model = d_model
        self.pad_id = pad_id

        self.src_embed = nn.Embedding(en_vocab_size, d_model, padding_idx=pad_id)
        self.tgt_embed = nn.Embedding(hi_vocab_size, d_model, padding_idx=pad_id)
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
        return src == self.pad_id  # [B, S]

    def make_tgt_mask(self, tgt):
        L = tgt.size(1)
        mask = torch.triu(torch.ones(L, L, device=tgt.device), diagonal=1).bool()
        return mask  # [L, L]

    def forward(self, src, tgt, return_states: bool = False):
        """
        src: [B, S]
        tgt: [B, T]
        return_states:
            False -> return logits only
            True  -> return (logits, encoder_memory, decoder_out)
        """
        src_emb = self.pos(self.src_embed(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos(self.tgt_embed(tgt) * math.sqrt(self.d_model))

        src_pad_mask = self.make_src_padding_mask(src)          # [B, S]
        tgt_pad_mask = (tgt == self.pad_id)                     # [B, T]
        tgt_mask = self.make_tgt_mask(tgt)                      # [T, T]

        # Manually run encoder + decoder so we can grab states
        memory = self.tr.encoder(
            src=src_emb,
            src_key_padding_mask=src_pad_mask,
        )  # [B, S, D]

        dec_out = self.tr.decoder(
            tgt=tgt_emb,
            memory=memory,
            tgt_key_padding_mask=tgt_pad_mask,
            memory_key_padding_mask=src_pad_mask,
            tgt_mask=tgt_mask,
        )  # [B, T, D]

        logits = self.out(dec_out)

        if return_states:
            return logits, memory, dec_out
        return logits