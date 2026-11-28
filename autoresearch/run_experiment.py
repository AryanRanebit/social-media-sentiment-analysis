#!/usr/bin/env python3
"""One experiment = one candidate.py change -> locked evaluation -> keep or discard -> log row.

  python3 autoresearch/run_experiment.py "short description" key=value [key=value ...]

This is a plain config runner. It is NOT an autonomous LLM proposer: a person or
agent chooses each change, one command per experiment.

Applies the overrides to the current best CFG in candidate.py (whitelisted keys,
type and range checked), evaluates on the fixed validation split, and appends a
row to results.tsv. Keep rule: validation F1 improves by at least MARGIN over
the current best kept run. A discarded or crashed run leaves candidate.py
untouched, so it always holds the best kept config. The locked files
(evaluate.py, src/models.py, src/data.py) are hash-checked against locked.sha256.
"""
import ast
import hashlib
import json
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
MARGIN = 0.002
SPEC = {  # key: (type, lo, hi) allowed range, inclusive
    "model": (str, None, None),
    "max_features": (int, 2000, 150000),
    "ngram_max": (int, 1, 2),
    "min_df": (int, 1, 10),
    "sublinear_tf": (bool, None, None),
    "C": (float, 0.01, 100.0),
    "hidden": (int, 8, 512),
    "alpha": (float, 1e-6, 1.0),
    "lr": (float, 1e-5, 1.0),
    "max_iter": (int, 10, 300),
}
MODELS = {"logreg", "mlp"}
LOCKED = ["autoresearch/evaluate.py", "src/models.py", "src/data.py"]
HEAD = "id\tgit\tcfg_sha\tval_f1\tval_acc\tstatus\tseconds\tdescription\tconfig\n"


def apply_overrides(cfg, sets):
    cfg = dict(cfg)
    for kv in sets:
        if "=" not in kv:
            raise ValueError(f"expected key=value, got {kv!r}")
        k, v = kv.split("=", 1)
        if k not in SPEC:
            raise ValueError(f"unknown key {k!r}; allowed: {sorted(SPEC)}")
        t, lo, hi = SPEC[k]
        val = ast.literal_eval(v)
        if t is bool:
            if not isinstance(val, bool):
                raise ValueError(f"{k} must be True/False")
        elif t is str:
            if not isinstance(val, str):
                raise ValueError(f"{k} must be a string")
            if k == "model" and val not in MODELS:
                raise ValueError(f"model must be one of {sorted(MODELS)}")
        else:
            if isinstance(val, bool) or not isinstance(val, (int, float)) or (t is int and not isinstance(val, int)):
                raise ValueError(f"{k} must be {t.__name__}")
            val = t(val)
            if not (lo <= val <= hi):
                raise ValueError(f"{k}={val} outside [{lo}, {hi}]")
        cfg[k] = val
    return cfg


def decide(best_f1, new_f1, margin=MARGIN):
    return "keep" if (new_f1 - best_f1) >= margin - 1e-12 else "discard"


def cfg_sha(cfg):
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:8]


def write_cfg(path, cfg):
    src = open(path).read().split("CFG = ")[0]
    open(path, "w").write(src + "CFG = {\n" + "".join(f"    {k!r}: {v!r},\n" for k, v in cfg.items()) + "}\n")


def read_log(path):
    if not os.path.exists(path):
        return []
    return [l.split("\t") for l in open(path).read().splitlines()[1:]]


def check_locked(root):
    want = dict(l.split()[::-1] for l in open(os.path.join(root, "autoresearch", "locked.sha256")).read().splitlines())
    for f in LOCKED:
        got = hashlib.sha256(open(os.path.join(root, f), "rb").read()).hexdigest()
        if want.get(f) != got:
            raise RuntimeError(f"locked file changed: {f}")


def git_id(root):
    r = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=root).stdout.strip() or "nogit"
    d = subprocess.run(["git", "status", "--porcelain", "--", "src", "autoresearch/evaluate.py"], capture_output=True, text=True, cwd=root).stdout.strip()
    return r + ("-dirty" if d else "")


def run(desc, sets, evaluate_fn, cand_path, log_path, git="nogit"):
    ns = {}
    exec(open(cand_path).read(), ns)
    cfg = apply_overrides(ns["CFG"], sets)
    prev = read_log(log_path)
    kept = [x for x in prev if x[5] in ("baseline", "keep")]
    t = time.time()
    try:
        r = evaluate_fn(cfg)
    except Exception as e:
        status, r = "crash", None
        err = f"{type(e).__name__}: {e}"[:200].replace("\t", " ").replace("\n", " ")
    else:
        if not kept:
            status = "baseline"
        else:
            status = decide(float(kept[-1][3]), r["val_f1"])
    sec = time.time() - t
    rec = json.dumps(cfg if r else dict(cfg, _error=err), sort_keys=False)
    cells = [str(len(prev)), git, cfg_sha(cfg)]
    cells += [f"{r['val_f1']:.4f}", f"{r['val_acc']:.4f}"] if r else ["nan", "nan"]
    cells += [status, f"{sec:.0f}", desc.replace("\t", " "), rec]
    if not os.path.exists(log_path):
        open(log_path, "w").write(HEAD)
    open(log_path, "a").write("\t".join(cells) + "\n")
    if status in ("baseline", "keep"):
        write_cfg(cand_path, cfg)
    return status, r


if __name__ == "__main__":
    root = os.path.abspath(os.path.join(HERE, ".."))
    check_locked(root)
    import evaluate as EV

    st, r = run(sys.argv[1], sys.argv[2:], EV.evaluate, os.path.join(HERE, "candidate.py"),
                os.path.join(HERE, "results.tsv"), git_id(root))
    print(st, f"val_f1={r['val_f1']:.4f} val_acc={r['val_acc']:.4f}" if r else "(run failed, see log)")
