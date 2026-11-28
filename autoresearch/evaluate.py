"""LOCKED evaluation for the autoresearch-style experiments. Experiments must not edit this file.

Protocol: train on the fixed train split, score on the fixed validation split.
Metric: validation F1 on the positive class (higher is better); accuracy is
reported alongside. The test split is never loaded here - it is scored once by
scripts/final_test.py after tuning stops.
"""
import os
import sys
import time

from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import data
import models


def evaluate(cfg):
    (X_tr, y_tr), (X_val, y_val), _ = data.load_split()
    pipe = models.build_pipeline(cfg)
    pipe.fit(X_tr, y_tr)
    pred = pipe.predict(X_val)
    return {
        "val_f1": float(f1_score(y_val, pred)),
        "val_acc": float(accuracy_score(y_val, pred)),
    }


if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from candidate import CFG

    t = time.time()
    r = evaluate(CFG)
    print({k: round(v, 4) for k, v in r.items()}, f"{time.time() - t:.0f}s")
