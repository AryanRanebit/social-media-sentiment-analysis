#!/usr/bin/env python3
"""Download Sentiment140 into data/ (gitignored). Run once before any experiment.

Sentiment140 (Go, Bhayani, Huang 2009): 1.6M tweets with distant-supervision
labels (0 = negative, 4 = positive), collected via emoticons.
"""
import os
import urllib.request
import zipfile

URL = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(HERE, "..", "data"))
CSV = "training.1600000.processed.noemoticon.csv"

os.makedirs(DATA_DIR, exist_ok=True)
dest = os.path.join(DATA_DIR, CSV)
if os.path.exists(dest):
    print(f"already present: {dest}")
else:
    zpath = os.path.join(DATA_DIR, "sentiment140.zip")
    print(f"downloading {URL} ...")
    urllib.request.urlretrieve(URL, zpath)
    with zipfile.ZipFile(zpath) as z:
        z.extract(CSV, DATA_DIR)
    os.remove(zpath)
    print(f"extracted {dest}")
