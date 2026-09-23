import torch
from transformer_mt import TransformerMT
from mt_translate_core import translate
import sentencepiece as spm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    model = TransformerMT(
        en_vocab_size=300,
        hi_vocab_size=300,
        d_model=256,
        nhead=8,
        num_layers=3,
        pad_id=0
    )
    model.load_state_dict(torch.load("pytorch_model.bin", map_location=DEVICE))
    model.eval()
    return model.to(DEVICE)

model = load_model()
en_tok = spm.SentencePieceProcessor(model_file="en_spm.model")
hi_tok = spm.SentencePieceProcessor(model_file="hi_spm.model")

def run(text, max_len=40):
    return translate(text, model, en_tok, hi_tok, max_len=max_len)
