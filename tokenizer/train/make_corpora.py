import json

data = json.load(open("/tokenizer/data/parallel_en_hi.json", "r", encoding="utf-8"))

with open("/tokenizer/data/en_corpus.txt", "w", encoding="utf-8") as f_en, \
        open("/tokenizer/data/hi_corpus.txt", "w", encoding="utf-8") as f_hi:
    for item in data:
        f_en.write(item["en"].strip() + "\n")
        f_hi.write(item["hi"].strip() + "\n")
