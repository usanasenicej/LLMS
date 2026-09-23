# tokenizer/train/train_real.py

import os
import json
import time

import torch
import sentencepiece as spm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from tokenizer.core.dataset import ParallelDataset
from tokenizer.core.loader import collate_batch
from tokenizer.models.transformer_mt import TransformerMT
from tokenizer.core.config import d_model, nhead, num_layers, pad_id, bos_id, eos_id


# ---------------------------------------------------------------------
# Paths & tokenizers
# ---------------------------------------------------------------------
# .../How Transformer LLMs Work/tokenizer
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")

CORPUS_DIR = os.path.join(DATA_DIR, "corpus")
JSON_DIR = os.path.join(DATA_DIR, "json")
TOK_DIR = os.path.join(DATA_DIR, "tokenizers")
MODEL_DIR = os.path.join(DATA_DIR, "trained_models")
RUNS_DIR = os.path.join(DATA_DIR, "runs")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)

# SentencePiece tokenizers
en_tok = spm.SentencePieceProcessor(
    model_file=os.path.join(TOK_DIR, "en_spm.model")
)
hi_tok = spm.SentencePieceProcessor(
    model_file=os.path.join(TOK_DIR, "hi_spm.model")
)

# Parallel corpus (JSON)
with open(os.path.join(JSON_DIR, "parallel_en_hi.json"), "r", encoding="utf-8") as f:
    parallel = json.load(f)

# ---------------------------------------------------------------------
# Build token ID pairs
# ---------------------------------------------------------------------
src_ids, tgt_ids = [], []
for item in parallel:
    en = item["en"].strip()
    hi = item["hi"].strip()
    src_ids.append([bos_id] + en_tok.encode(en, out_type=int) + [eos_id])
    tgt_ids.append([bos_id] + hi_tok.encode(hi, out_type=int) + [eos_id])

dataset = ParallelDataset(src_ids, tgt_ids, pad_id=pad_id)
loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_batch)

# ---------------------------------------------------------------------
# Model / optimizer / loss
# ---------------------------------------------------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = TransformerMT(
    en_vocab_size=en_tok.get_piece_size(),
    hi_vocab_size=hi_tok.get_piece_size(),
    d_model=d_model,
    nhead=nhead,
    num_layers=num_layers,
    pad_id=pad_id,           # <-- fixed here
).to(DEVICE)

opt = torch.optim.Adam(model.parameters(), lr=1e-4)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=pad_id)

# ---------------------------------------------------------------------
# Run naming + logging dirs
# ---------------------------------------------------------------------
timestamp = time.strftime("%Y%m%d_%H%M%S")
run_name = f"enhi_d{d_model}_L{num_layers}_V{en_tok.get_piece_size()}_{timestamp}"

run_dir = os.path.join(RUNS_DIR, run_name)
os.makedirs(run_dir, exist_ok=True)

writer = SummaryWriter(log_dir=run_dir)

metrics_path = os.path.join(run_dir, "metrics.jsonl")
meta_path = os.path.join(run_dir, "metadata.json")

meta = {
    "run_name": run_name,
    "d_model": d_model,
    "nhead": nhead,
    "num_layers": num_layers,
    "pad_id": pad_id,
    "bos_id": bos_id,
    "eos_id": eos_id,
    "en_vocab_size": en_tok.get_piece_size(),
    "hi_vocab_size": hi_tok.get_piece_size(),
    "num_examples": len(dataset),
    "device": DEVICE,
}
with open(meta_path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, ensure_ascii=False)

print(f"[RUN] {run_name}")
print(f"[LOG] TensorBoard dir: {run_dir}")
print(f"[LOG] Metrics file    : {metrics_path}")

# ---------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------
EPOCHS = 20

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    total_tokens = 0

    for batch in loader:
        src, tgt_in, tgt_out = batch
        src = src.to(DEVICE)
        tgt_in = tgt_in.to(DEVICE)
        tgt_out = tgt_out.to(DEVICE)

        logits = model(src, tgt_in)  # [B, T, V]

        loss = loss_fn(
            logits.reshape(-1, logits.size(-1)),
            tgt_out.reshape(-1),
        )

        opt.zero_grad()
        loss.backward()
        opt.step()

        # for average loss
        total_loss += loss.item() * tgt_out.numel()
        total_tokens += tgt_out.numel()

    avg_loss = total_loss / max(total_tokens, 1)
    print(f"Epoch {epoch:02d}  Loss: {avg_loss:.4f}")

    # TensorBoard scalar
    writer.add_scalar("train/loss", avg_loss, epoch)

    # JSONL metrics
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"epoch": epoch, "loss": avg_loss}) + "\n")

writer.close()

# ---------------------------------------------------------------------
# After training loop — generate dashboard JSON files
# ---------------------------------------------------------------------

# Convert metrics.jsonl → list for loss_curve.json and metrics.json
loss_history = []
with open(metrics_path, "r", encoding="utf-8") as f:
    for line in f:
        obj = json.loads(line)
        loss_history.append(obj["loss"])

# 1. loss_curve.json
json.dump(
    {"loss_curve": loss_history},
    open(os.path.join(run_dir, "loss_curve.json"), "w"),
    indent=2
)

# 2. metadata.json
metadata = {
    "run_id": run_name,
    "d_model": d_model,
    "layers": num_layers,
    "nhead": nhead,
    "en_vocab": en_tok.get_piece_size(),
    "hi_vocab": hi_tok.get_piece_size(),
    "timestamp": timestamp,
    "device": DEVICE,
}
json.dump(
    metadata,
    open(os.path.join(run_dir, "metadata.json"), "w"),
    indent=2
)

# 3. metrics.json (list form)
json.dump(
    [{"epoch": i, "loss": float(v)} for i, v in enumerate(loss_history)],
    open(os.path.join(run_dir, "metrics.json"), "w"),
    indent=2
)

print(f"[DASHBOARD] Exported loss_curve.json, metadata.json, metrics.json → {run_dir}")

# ---------------------------------------------------------------------
# Save checkpoint
# ---------------------------------------------------------------------
ckpt_name = f"{run_name}.pt"
save_path = os.path.join(MODEL_DIR, ckpt_name)
torch.save(model.state_dict(), save_path)
print("Saved checkpoint:", save_path)

# Also update a "latest" symlink / copy for inference if you like:
latest_path = os.path.join(MODEL_DIR, "en_hi_latest.pt")
try:
    if os.path.islink(latest_path) or os.path.exists(latest_path):
        os.remove(latest_path)
    # best-effort: on Windows this will just copy
    os.symlink(ckpt_name, latest_path)
except OSError:
    # fallback: copy the file
    import shutil
    shutil.copy2(save_path, latest_path)
    print("Copied latest model to:", latest_path)
