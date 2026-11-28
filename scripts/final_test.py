#!/usr/bin/env python3
"""One-shot held-out TEST evaluation of the best kept config.

Run this once, after tuning stops. The autoresearch runner never touches the
test split; this is the unbiased estimate of the kept config, reported next to
the untouched logistic-regression baseline for comparison.
"""
import os
import sys

from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "autoresearch"))
import data
import models
from candidate import CFG

BASELINE = dict(CFG, model="logreg", max_features=20000, ngram_max=1,
                min_df=1, sublinear_tf=False, C=1.0)


def score(cfg, split):
    (X_tr, y_tr), _, (X_te, y_te) = split
    pipe = models.build_pipeline(cfg)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_te)
    return f1_score(y_te, pred), accuracy_score(y_te, pred)


if __name__ == "__main__":
    split = data.load_split()
    for name, cfg in [("baseline logreg", BASELINE), ("best kept", CFG)]:
        f1, acc = score(cfg, split)
        print(f"{name:16s} test_f1={f1:.4f} test_acc={acc:.4f}  cfg={cfg}")
