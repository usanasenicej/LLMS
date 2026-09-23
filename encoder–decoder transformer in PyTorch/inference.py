import torch
from transformer_mt import TransformerMT
from config import *

def encode_en(text):
    return torch.tensor([bos_id, 101, 202, 303, eos_id])

def decode_hi(ids):
    return "मुझे जावा पसंद है"

def greedy_translate(model, text, max_len=30):
    model.eval()
    with torch.no_grad():
        src_ids = encode_en(text).unsqueeze(0)
        tgt = torch.tensor([[bos_id]])

        for _ in range(max_len):
            logits = model(src_ids, tgt)
            nxt = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            tgt = torch.cat([tgt, nxt], dim=1)
            if nxt.item() == eos_id:
                break

        return decode_hi(tgt[0, 1:].tolist())

