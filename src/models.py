"""Model construction for the sentiment experiments. Shared by evaluate.py.

Two model families, both on TF-IDF features:
  logreg - LogisticRegression baseline (strong on this dataset).
  mlp    - one-hidden-layer neural net (sklearn MLPClassifier), the challenger.
"""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline


def build_pipeline(cfg):
    vec = TfidfVectorizer(
        max_features=cfg["max_features"],
        ngram_range=(1, cfg["ngram_max"]),
        min_df=cfg["min_df"],
        sublinear_tf=cfg["sublinear_tf"],
    )
    if cfg["model"] == "logreg":
        clf = LogisticRegression(C=cfg["C"], max_iter=1000, random_state=42)
    elif cfg["model"] == "mlp":
        clf = MLPClassifier(
            hidden_layer_sizes=(cfg["hidden"],),
            alpha=cfg["alpha"],
            learning_rate_init=cfg["lr"],
            max_iter=cfg["max_iter"],
            early_stopping=True,
            n_iter_no_change=4,
            random_state=42,
        )
    else:
        raise ValueError(f"unknown model {cfg['model']!r}")
    return Pipeline([("tfidf", vec), ("clf", clf)])
