"""Tests for the autoresearch config runner. Fast: no dataset, stub evaluation.

  python3 -m pytest tests/ -q   (or)   python3 tests/test_autoresearch.py
"""
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "autoresearch"))
import run_experiment as R

BASE_CFG = {
    "model": "logreg", "max_features": 20000, "ngram_max": 1, "min_df": 1,
    "sublinear_tf": False, "C": 1.0, "hidden": 64, "alpha": 0.0001,
    "lr": 0.001, "max_iter": 60,
}


def write_candidate(path, cfg):
    with open(path, "w") as f:
        f.write('"""test candidate"""\nCFG = {\n')
        for k, v in cfg.items():
            f.write(f"    {k!r}: {v!r},\n")
        f.write("}\n")


def test_overrides_accept_valid():
    cfg = R.apply_overrides(BASE_CFG, ["C=2.5", "ngram_max=2", "sublinear_tf=True", "model='mlp'"])
    assert cfg["C"] == 2.5 and cfg["ngram_max"] == 2 and cfg["sublinear_tf"] is True
    assert cfg["model"] == "mlp" and cfg["max_features"] == 20000  # untouched keys stay


def test_overrides_reject_bad_input():
    for bad in ["bogus=1", "C='x'", "max_features=10", "hidden=100000",
                "ngram_max=3", "sublinear_tf=1", "model='svm'", "noequals"]:
        try:
            R.apply_overrides(BASE_CFG, [bad])
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted invalid override: {bad}")


def test_decide_margin():
    assert R.decide(0.8000, 0.8020) == "keep"        # improvement >= MARGIN
    assert R.decide(0.8000, 0.8019) == "discard"     # below MARGIN
    assert R.decide(0.8000, 0.8100) == "keep"
    assert R.decide(0.8000, 0.7999) == "discard"     # worse never keeps


def test_end_to_end_keep_then_discard():
    """Baseline -> keep (better) -> discard (worse); candidate holds the kept config."""
    with tempfile.TemporaryDirectory() as d:
        cand, log = os.path.join(d, "candidate.py"), os.path.join(d, "results.tsv")
        write_candidate(cand, BASE_CFG)
        scores = iter([0.80, 0.85, 0.82])  # baseline, improvement, regression

        def fake_evaluate(cfg):
            return {"val_f1": next(scores), "val_acc": 0.80}

        st, _ = R.run("baseline", [], fake_evaluate, cand, log)
        assert st == "baseline"
        st, _ = R.run("better bigrams", ["ngram_max=2"], fake_evaluate, cand, log)
        assert st == "keep"
        st, _ = R.run("worse", ["C=9.0"], fake_evaluate, cand, log)
        assert st == "discard"

        ns = {}
        exec(open(cand).read(), ns)
        assert ns["CFG"]["ngram_max"] == 2, "kept override must persist in candidate.py"
        assert ns["CFG"]["C"] == 1.0, "discarded override must not persist"

        rows = R.read_log(log)
        assert len(rows) == 3 and [r[5] for r in rows] == ["baseline", "keep", "discard"]


def test_crash_does_not_touch_candidate():
    with tempfile.TemporaryDirectory() as d:
        cand, log = os.path.join(d, "candidate.py"), os.path.join(d, "results.tsv")
        write_candidate(cand, BASE_CFG)

        def boom(cfg):
            raise RuntimeError("fake crash")

        st, r = R.run("crash run", ["C=5.0"], boom, cand, log)
        assert st == "crash" and r is None
        ns = {}
        exec(open(cand).read(), ns)
        assert ns["CFG"]["C"] == 1.0
        assert R.read_log(log)[0][5] == "crash"


if __name__ == "__main__":
    for name, fn in sorted({k: v for k, v in globals().items() if k.startswith("test_")}.items()):
        fn()
        print(f"ok {name}")
    print("all tests passed")
