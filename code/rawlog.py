"""Collect per-instance raw data and write it to code/data/raw/<name>.csv."""
import csv, os
HERE = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(HERE, "data", "raw")
SUMMARY = os.path.join(HERE, "data", "summary")
os.makedirs(SUMMARY, exist_ok=True)


def summary(fname):
    """Path of a summary JSON file in code/data/summary."""
    return os.path.join(SUMMARY, fname)
_buf = {}


def log(name, **row):
    _buf.setdefault(name, []).append(row)


def flush(name):
    rows = _buf.pop(name, [])
    if not rows:
        return
    os.makedirs(RAW, exist_ok=True)
    keys = list(rows[0].keys())
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(os.path.join(RAW, name + ".csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
