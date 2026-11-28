"""Load and split the Sentiment140 dataset (fixed, reproducible).

Sentiment140: 1.6M tweets labeled by emoticon (0 = negative, 4 = positive).
Source: http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip
Run scripts/fetch_data.py once to download it into data/ (gitignored).

We work on a deterministic 40k subsample (20k per class) so experiments run
in seconds-to-minutes on a laptop. Split is stratified 70/15/15 train/val/test
with a fixed seed. The autoresearch runner sees train+val only; the test split
is touched once by scripts/final_test.py.
"""
import html
import os
import pickle
import re

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
SAMPLE_PER_CLASS = 20000
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
RAW_CSV = os.path.join(DATA_DIR, "training.1600000.processed.noemoticon.csv")
CACHE = os.path.join(DATA_DIR, "cache.pkl")

_URL = re.compile(r"https?://\S+|www\.\S+")
_MENTION = re.compile(r"@\w+")
_WS = re.compile(r"\s+")


def clean(text):
    text = html.unescape(text).lower()
    text = _URL.sub(" ", text)
    text = _MENTION.sub(" ", text)
    text = text.replace("#", "")
    return _WS.sub(" ", text).strip()


def load_raw():
    df = pd.read_csv(
        RAW_CSV,
        header=None,
        usecols=[0, 5],
        names=["label", "text"],
        encoding="latin-1",
    )
    df = df[df["label"].isin([0, 4])]
    df["label"] = (df["label"] == 4).astype(int)
    return df


def load_split():
    """Returns (train, val, test) as (texts, labels) tuples. Cached on disk."""
    if os.path.exists(CACHE):
        with open(CACHE, "rb") as f:
            return pickle.load(f)
    df = load_raw()
    parts = [
        df[df["label"] == c].sample(n=SAMPLE_PER_CLASS, random_state=SEED)
        for c in (0, 1)
    ]
    df = pd.concat(parts).sample(frac=1.0, random_state=SEED)  # shuffle
    texts = [clean(t) for t in df["text"]]
    labels = df["label"].to_numpy()
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(
        texts, labels, test_size=0.30, random_state=SEED, stratify=labels
    )
    X_val, X_te, y_val, y_te = train_test_split(
        X_tmp, y_tmp, test_size=0.50, random_state=SEED, stratify=y_tmp
    )
    split = ((X_tr, y_tr), (X_val, y_val), (X_te, y_te))
    with open(CACHE, "wb") as f:
        pickle.dump(split, f)
    return split
