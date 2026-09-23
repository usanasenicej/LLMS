import torch
from config import en_vocab_size, hi_vocab_size

def load_dummy():
    src = torch.randint(0, en_vocab_size, (2, 5))
    tgt = torch.randint(0, hi_vocab_size, (2, 6))
    return [{"src": src, "tgt": tgt}]
