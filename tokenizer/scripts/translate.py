from tokenizer.core.mt_translate_core import translate
from tokenizer.core.mt_translate_core import translate
def main():
    while True:
        text = input("\nEN: ").strip()
        if text.lower() in {"q", "quit"}:
            break
        print("HI:", translate(text))

if __name__ == "__main__":
    main()

# import os, sys
# import torch
# import sentencepiece as spm
#
# ROOT = os.path.dirname(os.path.abspath(__file__))
# sys.path.append(ROOT)
#
# from transformer_mt import TransformerMT
# from config import *
#
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
#
# en_tok = spm.SentencePieceProcessor(model_file=os.path.join(ROOT, "en_spm.model"))
# hi_tok = spm.SentencePieceProcessor(model_file=os.path.join(ROOT, "hi_spm.model"))
#
# model = TransformerMT(
#     en_vocab_size=en_tok.get_piece_size(),
#     hi_vocab_size=hi_tok.get_piece_size(),
#     d_model=d_model,
#     nhead=nhead,
#     num_layers=num_layers,
#     pad_idx=pad_id,     # must come from config.py
# ).to(DEVICE)
#
# state_path = os.path.join(ROOT, "trained_models", "en_hi_spm400_d256_v1.pt")
# model.load_state_dict(torch.load(state_path, map_location=DEVICE))
# model.eval()
#
# bos = 1
# eos = 2
#
# def translate(text, max_len=40):
#     src_ids = [bos] + en_tok.encode(text, out_type=int) + [eos]
#     src = torch.tensor([src_ids], dtype=torch.long).to(DEVICE)
#
#     tgt = torch.tensor([[bos]], dtype=torch.long).to(DEVICE)
#
#     for _ in range(max_len):
#         logits = model(src, tgt)
#         next_id = int(logits[0, -1].argmax())
#         tgt = torch.cat([tgt, torch.tensor([[next_id]], device=DEVICE)], dim=1)
#         if next_id == eos:
#             break
#
#     pred = tgt[0, 1:-1].tolist()
#     return hi_tok.decode(pred)
#
# print(translate("I love Java"))
# print(translate("How are you?"))
# print(translate("Hello"))
#
#
