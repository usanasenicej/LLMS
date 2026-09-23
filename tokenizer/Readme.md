```
tokenizer/
│
├── app/                               # UI + API layer
│   ├── __init__.py
│   ├── main_app.py                    # Streamlit UI (Translator + Heatmap + Dashboard)
│   └── api.py                         # FastAPI server (REST translation)
│
├── core/                              # Core model + data pipeline
│   ├── __init__.py
│   ├── config.py                      # Hyperparameters + global constants
│   ├── dataset.py                     # ParallelDataset + utilities
│   ├── loader.py                      # collate_batch (padding, BOS/EOS handling)
│   └── mt_translate_core.py           # Greedy, batch, beam, attention decode
│
├── data/                              # All generated assets
│   ├── corpus/
│   │   ├── __init__.py
│   │   ├── en_corpus.txt              # Raw English text
│   │   └── hi_corpus.txt              # Raw Hindi text
│   │
│   ├── json/
│   │   ├── __init__.py
│   │   └── parallel_en_hi.json        # Paired EN–HI dataset after preprocessing
│   │
│   ├── runs/                          # Training runs → loss/metadata/logs
│   │   ├── <run_id>/
│   │   │    ├── loss_curve.json
│   │   │    ├── metadata.json
│   │   │    └── metrics.jsonl
│   │   └── ...
│   │
│   ├── tokenizers/                    # SentencePiece artifacts
│   │   ├── en_spm.model
│   │   ├── en_spm.vocab
│   │   ├── hi_spm.model
│   │   └── hi_spm.vocab
│   │
│   └── trained_models/                # Transformer checkpoints
│       ├── __init__.py
│       ├── en_hi_latest.pt
│       └── enhi_d256_L3_V300_*.pt
│
├── models/                            # Transformer internals
│   ├── __init__.py
│   ├── positional_encoding.py         # Sinusoidal encoding
│   └── transformer_mt.py              # Full Encoder–Decoder model
│
├── scripts/                           # Utils / CLI tools
│   ├── __init__.py
│   └── translate.py                   # Command-line translation tool
│
└── train/                             # Training utilities
    ├── __init__.py
    ├── make_corpora.py                # Corpus building from raw text
    ├── train_spm.py                   # SentencePiece tokenizer training
    └── train_real.py                  # Transformer model training loop
```
