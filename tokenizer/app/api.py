from fastapi import FastAPI
from pydantic import BaseModel
import os, sys

from tokenizer import app

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(ROOT)

from tokenizer.core.mt_translate_core import translate, model

@app.on_event("startup")
def load_once():
    print("Model loaded:", model is not None)

app = FastAPI()

class Req(BaseModel):
    text: str
    max_len: int = 40

class Res(BaseModel):
    translation: str

@app.get("/")
def root():
    return {"status": "ok", "message": "FastAPI running"}

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/translate", response_model=Res)
def t(req: Req):
    out = translate(req.text, req.max_len)
    return Res(translation=out)
