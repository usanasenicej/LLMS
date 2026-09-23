```
tokenizer/
│
├── app/
│   ├── __init__.py
│   ├── main_app.py
│       Streamlit UI containing:
│         - Translator (single, batch, beam search)
│         - Attention heatmap with token alignment
│         - Training dashboard (loss curves + metadata)
│         - REST API tester (calls FastAPI backend)
│         - Model upload/download system
│       Internally uses reusable components + robust JSON loaders.
│
│   └── api.py
│       FastAPI server exposing:
│         POST /translate      → returns Hindi translation
│         POST /beam           → beam search inference
│         GET /health          → health check
│       Loads SentencePiece + latest Transformer weights.
│
├── core/
│   ├── __init__.py
│   ├── config.py
│       Central configuration registry:
│         - d_model, num_layers, nhead
│         - BOS/EOS/PAD token IDs
│         - directory structure for model/data/runs
│         - device selection (cpu/cuda)
│
│   ├── dataset.py
│       Two dataset implementations:
│         1. EnHiDataset:
│              - Loads JSON parallel_en_hi.json
│              - Encodes using SentencePiece
│              - Builds src_ids and tgt_ids with BOS/EOS
│
│         2. ParallelDataset:
│              - Stores pre-tokenized integer sequences
│              - Used in final training loop (train_real.py)
│
│   ├── loader.py
│       collate_batch(batch):
│         - Finds max lengths in src/tgt
│         - Pads sequences with PAD_ID
│         - Builds:
│             src     → [B, S]
│             tgt_in  → [B, T] (shifted right)
│             tgt_out → [B, T] (next-token labels)
│         - Produces tensors + attention masks.
│
│   └── mt_translate_core.py
│       Unified inference module containing:
│         - greedy_decode(text, max_len)
│         - translate_batch(list_of_sentences)
│         - beam_search_decode(text, k)
│         - translate_with_attention(text) → tokens + similarity heatmap
│       This is reused by both CLI, Streamlit, and FastAPI.
│
├── models/
│   ├── __init__.py
│   ├── positional_encoding.py
│       Implements sinusoidal PE matrix:
│         PE[pos, 2i]   = sin(pos / 10000^(2i/d))
│         PE[pos, 2i+1] = cos(pos / 10000^(2i/d))
│       Added to embeddings before Transformer layers.
│
│   └── transformer_mt.py
│       Full encoder–decoder Transformer:
│         - Embedding + positional encoding
│         - Encoder:
│             Multi-head attention + FFN × num_layers
│         - Decoder:
│             Masked self-attention + cross-attention + FFN
│         - Final linear projection to target vocabulary
│       forward(src, tgt_in) returns:
│         logits shape [batch, seq_len, vocab_size]
│
├── scripts/
│   ├── __init__.py
│   └── translate.py
│       CLI interface:
│         $ python translate.py "Hello"
│       Pipeline:
│         - Loads tokenizers + model checkpoint
│         - Runs greedy decode
│         - Prints Hindi text
│
├── train/
│   ├── __init__.py
│   ├── make_corpora.py
│       Converts raw JSON dataset into SentencePiece corpora:
│         data/corpus/en_corpus.txt
│         data/corpus/hi_corpus.txt
│       Ensures tokenizers train on pure language data.
│
│   ├── train_spm.py
│       Trains subword tokenizers on the corpus:
│         - English SPM → en_spm.model, en_spm.vocab
│         - Hindi SPM → hi_spm.model, hi_spm.vocab
│       Uses sentencepiece.Unigram/BPE with fixed vocab size.
│
│   └── train_real.py
│       Full training loop:
│         - Loads SPM tokenizers
│         - Loads parallel_en_hi.json
│         - Builds token-ID pairs
│         - Creates ParallelDataset + DataLoader
│         - Initializes TransformerMT(d_model, nhead, layers)
│         - Teacher-forced training (CrossEntropyLoss)
│         - Logging:
│             loss_curve.json         → clean list OR dict list
│             metrics.jsonl           → per-epoch structured log
│             metadata.json           → d_model, layers, vocab sizes, timestamp
│             TensorBoard events      → for visualization
│         - Saves checkpoint to:
│             data/trained_models/<run>.pt
│         - Maintains en_hi_latest.pt symlink/copy.
│
├── data/
│   ├── corpus/
│   │   ├── __init__.py
│   │   ├── en_corpus.txt
│   │   └── hi_corpus.txt
│
│   ├── json/
│   │   ├── __init__.py
│   │   └── parallel_en_hi.json      # Raw aligned dataset
│
│   ├── tokenizers/
│   │   ├── en_spm.model             # binary SentencePiece model
│   │   ├── en_spm.vocab
│   │   ├── hi_spm.model
│   │   └── hi_spm.vocab
│
│   ├── runs/
│   │   ├── <run_id_1>/
│   │   │     ├── loss_curve.json    # numeric or dict-based loss
│   │   │     ├── metadata.json      # full run info
│   │   │     ├── metrics.jsonl      # per-epoch logs
│   │   │     └── events.out.tfevents...
│   │   └── <run_id_2>/ ...
│
│   └── trained_models/
│         ├── __init__.py
│         ├── en_hi_latest.pt
│         └── enhi_d256_L3_V300_*.pt
│
└── README.md


Full documentation:
- Installation / venv setup
- Training workflow
- Tokenizer generation
- Model architecture
- Logging + Streamlit dashboard
- REST API usage
```
