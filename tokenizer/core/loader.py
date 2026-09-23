import torch
import sentencepiece as spm

def load_parallel_corpus(en_path, hi_path, en_tok, hi_tok, bos, eos):
    en_lines = open(en_path).read().strip().split("\n")
    hi_lines = open(hi_path).read().strip().split("\n")

    src_ids, tgt_ids = [], []

    for en, hi in zip(en_lines, hi_lines):
        s = [bos] + en_tok.encode(en, out_type=int) + [eos]
        t = [bos] + hi_tok.encode(hi, out_type=int) + [eos]
        src_ids.append(s)
        tgt_ids.append(t)

    return src_ids, tgt_ids


def collate_batch(batch):
    pad = 0

    src = [b["src"] for b in batch]
    tgt = [b["tgt"] for b in batch]

    max_src = max(len(s) for s in src)
    max_tgt = max(len(t) for t in tgt)

    padded_src, padded_tgt_in, padded_tgt_out = [], [], []

    for s, t in zip(src, tgt):
        s_pad = torch.nn.functional.pad(s, (0, max_src - len(s)), value=pad)
        padded_src.append(s_pad)

        t_in = t[:-1]
        t_out = t[1:]

        t_in_pad = torch.nn.functional.pad(t_in, (0, max_tgt - 1 - len(t_in)), value=pad)
        t_out_pad = torch.nn.functional.pad(t_out, (0, max_tgt - 1 - len(t_out)), value=pad)

        padded_tgt_in.append(t_in_pad)
        padded_tgt_out.append(t_out_pad)

    return torch.stack(padded_src), torch.stack(padded_tgt_in), torch.stack(padded_tgt_out)