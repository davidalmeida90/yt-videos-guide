"""Kronos helpers: load the pretrained tokenizer and predictor from the Hugging Face Hub, and the per-window z-score
(clipped at 5) that Kronos's own predictor applies, so the fine-tune sees inputs scaled exactly as in pretraining.
The model code is the vendored copy in kronos_model/ (github.com/shiyu-coder/Kronos, MIT, commit 67b630e).
"""
import numpy as np
import torch

from kronos_model import Kronos, KronosTokenizer

COLS = ["open", "high", "low", "close", "volume", "amount"]
DEV = "cuda:0" if torch.cuda.is_available() else "cpu"


def load(name="NeoQuasar/Kronos-small", tok="NeoQuasar/Kronos-Tokenizer-base"):
    t = KronosTokenizer.from_pretrained(tok).to(DEV).eval()
    m = Kronos.from_pretrained(name).to(DEV).eval()
    return t, m


def norm(x, clip=5.0):
    mu, sd = x.mean(1, keepdims=True), x.std(1, keepdims=True)
    return np.clip((x - mu) / (sd + 1e-5), -clip, clip), mu, sd
