# data/model/dummy_loader2.py

import os
import json
import torch

# /.../encoder–decoder transformer in PyTorch/data/model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# /.../encoder–decoder transformer in PyTorch/data/en_hi_tiny.json
DATA_PATH = os.path.join(BASE_DIR, "..", "en_hi_tiny.json")


def load_dummy_real_tiny():
    # load JSON list: [{"en": "...", "hi": "..."}, ...]
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    loader = []

    for item in data:
        # fake "tokenization": char → id (just to make tensors)
        src_ids = torch.tensor(
            [ord(c) % 255 for c in item["en"]],
            dtype=torch.long,
        )
        tgt_ids = torch.tensor(
            [ord(c) % 255 for c in item["hi"]],
            dtype=torch.long,
        )

        loader.append(
            {
                "src": src_ids.unsqueeze(0),  # shape (1, src_len)
                "tgt": tgt_ids.unsqueeze(0),  # shape (1, tgt_len)
            }
        )

    return loader
