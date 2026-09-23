import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

d_model = 256
nhead = 8
num_layers = 3
en_vocab_size = 8000
hi_vocab_size = 8000
pad_idx = 0

bos_id = 1
eos_id = 2
