# tokenizer/core/mt_translate_core.py

import os
from typing import List, Tuple, Dict, Any

import torch
import torch.nn.functional as F
import sentencepiece as spm

from tokenizer.models.transformer_mt import TransformerMT
from tokenizer.core.config import d_model, nhead, num_layers, pad_id, bos_id, eos_id

# --------------------------------------------------------------------------------------
# Paths / device
# --------------------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")

TOKENIZER_DIR = os.path.join(DATA_DIR, "tokenizers")
MODEL_DIR = os.path.join(DATA_DIR, "trained_models")
MODEL_PATH = os.path.join(MODEL_DIR, "en_hi_spm400_d256_v1.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------------------------------------------------------------
# Tokenizers
# --------------------------------------------------------------------------------------
en_tok = spm.SentencePieceProcessor(
    model_file=os.path.join(TOKENIZER_DIR, "en_spm.model")
)
hi_tok = spm.SentencePieceProcessor(
    model_file=os.path.join(TOKENIZER_DIR, "hi_spm.model")
)

EN_VOCAB = en_tok.get_piece_size()
HI_VOCAB = hi_tok.get_piece_size()

# --------------------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------------------
model = TransformerMT(
    en_vocab_size=EN_VOCAB,
    hi_vocab_size=HI_VOCAB,
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    pad_id=pad_id,
).to(DEVICE)

state = torch.load(MODEL_PATH, map_location=DEVICE)
model.load_state_dict(state)
model.eval()


def _encode_src(text: str) -> torch.Tensor:
    ids = [bos_id] + en_tok.encode(text, out_type=int) + [eos_id]
    return torch.tensor(ids, dtype=torch.long, device=DEVICE)


def _decode_tgt(ids: List[int]) -> str:
    return hi_tok.decode(ids)


# --------------------------------------------------------------------------------------
# 1. Greedy single-sentence translate (old behavior)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def translate(text: str, max_len: int = 40) -> str:
    src = _encode_src(text).unsqueeze(0)  # [1, S]
    tgt = torch.tensor([[bos_id]], dtype=torch.long, device=DEVICE)

    for _ in range(max_len):
        logits = model(src, tgt)  # [1, T, V]
        next_id = int(logits[0, -1].argmax(-1))
        tgt = torch.cat(
            [tgt, torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)], dim=1
        )
        if next_id == eos_id:
            break

    pred = tgt[0, 1:]
    if len(pred) > 0 and pred[-1].item() == eos_id:
        pred = pred[:-1]
    return _decode_tgt(pred.tolist())


# --------------------------------------------------------------------------------------
# 2. Batch greedy decode
# --------------------------------------------------------------------------------------
@torch.no_grad()
def translate_batch(texts: List[str], max_len: int = 40) -> List[str]:
    src_seqs = [ _encode_src(t) for t in texts ]
    max_s = max(x.size(0) for x in src_seqs)

    padded = []
    for s in src_seqs:
        pad_len = max_s - s.size(0)
        padded.append(F.pad(s, (0, pad_len), value=pad_id))
    src = torch.stack(padded, dim=0)  # [B, S]

    B = src.size(0)
    tgt = torch.full((B, 1), bos_id, dtype=torch.long, device=DEVICE)

    finished = [False] * B
    outputs = [[] for _ in range(B)]

    for _ in range(max_len):
        logits = model(src, tgt)  # [B, T, V]
        next_ids = logits[:, -1, :].argmax(-1)  # [B]

        tgt = torch.cat(
            [tgt, next_ids.unsqueeze(1)],
            dim=1,
        )

        for i in range(B):
            if finished[i]:
                continue
            nid = int(next_ids[i])
            if nid == eos_id:
                finished[i] = True
            else:
                outputs[i].append(nid)

        if all(finished):
            break

    return [_decode_tgt(ids) for ids in outputs]


# --------------------------------------------------------------------------------------
# 3. Beam search decode (per sentence)
# --------------------------------------------------------------------------------------
@torch.no_grad()
def translate_beam(text: str, max_len: int = 40, beam_size: int = 4) -> str:
    src = _encode_src(text).unsqueeze(0)  # [1, S]

    # Each beam = (log_prob, tensor_ids)
    beams: List[Tuple[float, torch.Tensor]] = [
        (0.0, torch.tensor([[bos_id]], dtype=torch.long, device=DEVICE))
    ]

    for _ in range(max_len):
        new_beams: List[Tuple[float, torch.Tensor]] = []
        for log_p, seq in beams:
            if int(seq[0, -1]) == eos_id:
                # already finished, keep as-is
                new_beams.append((log_p, seq))
                continue

            logits = model(src, seq)  # [1, T, V]
            next_logits = logits[0, -1]  # [V]
            probs = F.log_softmax(next_logits, dim=-1)  # log-probs

            topk = torch.topk(probs, beam_size)
            for score, idx in zip(topk.values, topk.indices):
                nid = int(idx)
                new_seq = torch.cat(
                    [seq, torch.tensor([[nid]], dtype=torch.long, device=DEVICE)],
                    dim=1,
                )
                new_beams.append((log_p + float(score), new_seq))

        # prune
        new_beams.sort(key=lambda x: x[0], reverse=True)
        beams = new_beams[:beam_size]

        # early stop if all ended
        if all(int(b[1][0, -1]) == eos_id for b in beams):
            break

    # pick best
    best_log_p, best_seq = max(beams, key=lambda x: x[0])
    pred = best_seq[0, 1:]
    if len(pred) > 0 and pred[-1].item() == eos_id:
        pred = pred[:-1]
    return _decode_tgt(pred.tolist())


# --------------------------------------------------------------------------------------
# 4. Attention-style heatmap: encoder vs decoder similarity
# --------------------------------------------------------------------------------------
@torch.no_grad()
def translate_with_attention(text: str, max_len: int = 40) -> Dict[str, Any]:
    """
    Returns:
      {
        "src_tokens": [...],
        "tgt_tokens": [...],
        "translation": str,
        "heatmap": 2D tensor [tgt_len, src_len] in CPU
      }
    """
    src_ids = [bos_id] + en_tok.encode(text, out_type=int) + [eos_id]
    src = torch.tensor([src_ids], dtype=torch.long, device=DEVICE)

    tgt = torch.tensor([[bos_id]], dtype=torch.long, device=DEVICE)

    for _ in range(max_len):
        logits, memory, dec_out = model(src, tgt, return_states=True)
        next_id = int(logits[0, -1].argmax(-1))
        tgt = torch.cat(
            [tgt, torch.tensor([[next_id]], dtype=torch.long, device=DEVICE)],
            dim=1,
        )
        if next_id == eos_id:
            break

    # remove BOS/EOS
    tgt_ids = tgt[0, 1:]
    if len(tgt_ids) > 0 and tgt_ids[-1].item() == eos_id:
        tgt_ids = tgt_ids[:-1]

    # memory: [1, S, D], dec_out: [1, T, D]
    mem = memory[0]          # [S, D]
    dec = dec_out[0, 1:1+len(tgt_ids)]  # [T, D] (skip BOS state)

    # cosine similarity => attention-like matrix [T, S]
    mem_norm = F.normalize(mem, dim=-1)
    dec_norm = F.normalize(dec, dim=-1)
    heatmap = dec_norm @ mem_norm.t()   # [T, S]

    src_pieces = [en_tok.id_to_piece(i) for i in src_ids]
    tgt_pieces = [hi_tok.id_to_piece(i) for i in tgt_ids.tolist()]

    return {
        "src_tokens": src_pieces,
        "tgt_tokens": tgt_pieces,
        "translation": hi_tok.decode(tgt_ids.tolist()),
        "heatmap": heatmap.cpu(),
    }