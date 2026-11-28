# social-media-sentiment-analysis

NLP sentiment analysis on social-media posts (Sentiment140 tweets), with a
bounded autoresearch-style experiment runner: one config change at a time,
locked evaluation, keep-if-better, every run logged.

## Dataset

Sentiment140 (Go, Bhayani & Huang, 2009): 1.6M tweets labeled via emoticons
(negative / positive). We use a deterministic 40k subsample (20k per class),
stratified 70/15/15 train/val/test, fixed seed. The test split is scored once
by `scripts/final_test.py` and is never used for tuning decisions.

## Setup

```
pip install -r requirements.txt
python3 scripts/fetch_data.py   # downloads ~80MB into data/ (gitignored)
```

## Layout

- `src/data.py` — load, clean (URLs/mentions out, lowercase), subsample, split
- `src/models.py` — TF-IDF + LogisticRegression baseline, TF-IDF + one-hidden-layer
  MLP (the neural-net challenger)
- `autoresearch/candidate.py` — the one file experiments may change (current best config)
- `autoresearch/evaluate.py` — locked protocol: train on train split, score val F1
- `autoresearch/run_experiment.py` — applies whitelisted `key=value` overrides,
  keeps a config only if val F1 improves by >= 0.002, logs every run to `results.tsv`
- `autoresearch/locked.sha256` — hash guard on the evaluation files
- `scripts/final_test.py` — one-shot held-out test score of the best kept config
- `tests/test_autoresearch.py` — runner logic tests (no dataset needed)

## Running an experiment

```
python3 autoresearch/run_experiment.py "try bigrams" ngram_max=2 max_features=50000
python3 autoresearch/run_experiment.py "mlp challenger" model="'mlp'" hidden=64
```

This is a config runner, not an autonomous code-writing loop: a person or agent
chooses each change, one command per experiment.

## First experiment series (11 runs, see autoresearch/results.tsv)

Validation F1 0.7735 (baseline) → 0.7822 (kept: bigrams, 50k features,
sublinear TF). Three MLP configs were tried; none beat tuned logistic
regression, so the neural net stays a challenger, not the winner. Held-out test
(run once, after tuning): baseline 0.7723, best kept 0.7805.

## Next steps

- VADER lexicon baseline for comparison
- Trend visualizations over time
- Richer NN input (embeddings) as a new experiment series
