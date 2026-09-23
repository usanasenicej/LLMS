import torch
from torch.utils.data import Dataset

class ParallelDataset(Dataset):
    def __init__(self, src_ids, tgt_ids, pad_id):
        assert len(src_ids) == len(tgt_ids)
        self.src_ids = src_ids
        self.tgt_ids = tgt_ids
        self.pad_id = pad_id

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, idx):
        return {
            "src": torch.tensor(self.src_ids[idx], dtype=torch.long),
            "tgt": torch.tensor(self.tgt_ids[idx], dtype=torch.long),
        }