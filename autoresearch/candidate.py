"""The ONE file an experiment may change: model/vectorizer settings.

evaluate.py (locked) reads CFG. Change values here only through
`python3 autoresearch/run_experiment.py "<what you tried>" key=value ...`.
"""
CFG = {
    'model': 'logreg',
    'max_features': 50000,
    'ngram_max': 2,
    'min_df': 1,
    'sublinear_tf': False,
    'C': 1.0,
    'hidden': 64,
    'alpha': 0.0001,
    'lr': 0.001,
    'max_iter': 60,
}
