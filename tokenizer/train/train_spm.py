import os
import sentencepiece as spm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "corpus")
OUT = os.path.join(ROOT, "data", "tokenizers")

VOCAB = 300

spm.SentencePieceTrainer.train(
    input=os.path.join(DATA, "en_corpus.txt"),
    model_prefix=os.path.join(OUT, "en_spm"),
    vocab_size=VOCAB,
    model_type="unigram",
    character_coverage=1.0,
)

spm.SentencePieceTrainer.train(
    input=os.path.join(DATA, "hi_corpus.txt"),
    model_prefix=os.path.join(OUT, "hi_spm"),
    vocab_size=VOCAB,
    model_type="unigram",
    character_coverage=0.9995,
)