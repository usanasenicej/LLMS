# data/model/train2.py

import torch

from transformer_mt import TransformerMT
from dummy_loader2 import load_dummy_real_tiny
from config import *
import os

loader = load_dummy_real_tiny()

model = TransformerMT(
    en_vocab_size,
    hi_vocab_size,
    d_model,
    nhead,
    num_layers,
    pad_idx,
).to(DEVICE)

opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_idx)

for batch in loader:
    src = batch["src"].to(DEVICE)
    tgt = batch["tgt"].to(DEVICE)

    dec_in = tgt[:, :-1]
    target = tgt[:, 1:]

    logits = model(src, dec_in)
    loss = loss_fn(logits.reshape(-1, hi_vocab_size), target.reshape(-1))

    opt.zero_grad()
    loss.backward()
    opt.step()

    print("Loss:", loss.item())

print("src:", src.shape)
print("tgt:", dec_in.shape)
print("logits:", logits.shape)

save_path = "/Users/somesh/How Transformer LLMs Work/encoder–decoder transformer in PyTorch/data/TrainedModel/transformer_en_hi.pt"
torch.save(model.state_dict(), save_path)
print("saved:", save_path)
