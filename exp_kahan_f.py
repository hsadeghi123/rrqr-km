#!/usr/bin/env python3
"""
Gu--Eisenstat Algorithm 5 audit: Kahan, extended Kahan, and nine UCI
matrices, for both RRQR-FS (columns of standardized X) and RRQR-Init
(columns of Zc.T, i.e. samples as centres).

Usage:
    python exp_kahan_f.py

Writes:
    kahan_f_results.csv
    kahan_f_results.txt
"""
from __future__ import annotations

import csv
import os
import urllib.request

import numpy as np
from scipy.linalg import qr

SEED = 20240601
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uci_data")
OUT_CSV = "kahan_f_results.csv"
OUT_TXT = "kahan_f_results.txt"


def select_cols(M, k, f, max_swaps=None):
    """Return (indices, n_swaps, certified). Cap = 10*n, not 200."""
    M = np.asarray(M, dtype=float)
    m, n = M.shape
    k = int(min(k, m, n))
    if k <= 0:
        return [], 0, True
    if max_swaps is None:
        max_swaps = 10 * n

    _, _, piv = qr(M, mode="economic", pivoting=True)
    order = [int(p) for p in np.atleast_1d(piv)]
    n_swaps = 0
    certified = False

    for _ in range(max_swaps):
        _, R = qr(M[:, order], mode="economic")
        A = R[:k, :k]
        B = R[:k, k:]
        C = R[k:, k:]
        if B.size == 0:
            certified = True
            break
        W = np.linalg.solve(A, B)
        Ainv = np.linalg.inv(A)
        omega = np.sqrt(np.sum(Ainv ** 2, axis=1))
        if C.size:
            gamma = np.sqrt(np.sum(C ** 2, axis=0))
        else:
            gamma = np.zeros(B.shape[1], dtype=float)
        score = np.maximum(np.abs(W), np.outer(omega, gamma))
        i, j = np.unravel_index(np.argmax(score), score.shape)
        if score[i, j] <= f:
            certified = True
            break
        order[i], order[k + j] = order[k + j], order[i]
        n_swaps += 1

    return order[:k], n_swaps, certified


def cpqr_cols(M, k):
    M = np.asarray(M, dtype=float)
    k = int(min(k, min(M.shape)))
    _, _, piv = qr(M, mode="economic", pivoting=True)
    return [int(p) for p in np.atleast_1d(piv)[:k]]


def log_vol_and_smin(M, idx):
    S = np.asarray(M, dtype=float)[:, list(idx)]
    if S.size == 0:
        return float("-inf"), float("nan")
    s = np.linalg.svd(S, compute_uv=False)
    s = np.maximum(s, np.finfo(float).tiny)
    return float(np.sum(np.log(s))), float(s.min())


def kahan(n, theta=1.2):
    c, s = np.cos(theta), np.sin(theta)
    A = np.zeros((n, n))
    for i in range(n):
        A[i, i:] = -s * (c ** i)
        A[i, i] = c ** i
    return A


def extended_kahan(n, theta=1.2):
    h = n // 2
    K = kahan(h, theta)
    A = np.zeros((n, n))
    A[:h, :h] = K
    A[h:, h:] = K
    return A


def standardize(X):
    X = np.asarray(X, dtype=float)
    mu = X.mean(axis=0)
    sig = X.std(axis=0, ddof=0)
    sig[sig == 0.0] = 1.0
    return (X - mu) / sig


def center_columns(Z):
    return Z - Z.mean(axis=0)


UCI = {
    "iris": {
        "file": "iris.data",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
        "k": 3,
        "load": lambda p: _load_delimited(p, skip_last=True, delim=","),
    },
    "wine": {
        "file": "wine.data",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data",
        "k": 3,
        "load": lambda p: _load_delimited(p, skip_first=True, delim=","),
    },
    "seeds": {
        "file": "seeds_dataset.txt",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt",
        "k": 3,
        "load": lambda p: _load_delimited(p, skip_last=True, delim=None),
    },
    "glass": {
        "file": "glass.data",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data",
        "k": 6,
        "load": lambda p: _load_delimited(p, skip_first=True, skip_last=True, delim=","),
    },
    "wdbc": {
        "file": "wdbc.data",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data",
        "k": 2,
        "load": lambda p: _load_delimited(p, skip_first=True, skip_second=True, delim=","),
    },
    "ionosphere": {
        "file": "ionosphere.data",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data",
        "k": 2,
        "load": lambda p: _load_delimited(p, skip_last=True, delim=","),
    },
    "banknote": {
        "file": "data_banknote_authentication.txt",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt",
        "k": 2,
        "load": lambda p: _load_delimited(p, skip_last=True, delim=","),
    },
    "yeast": {
        "file": "yeast.data",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data",
        "k": 10,
        "load": lambda p: _load_delimited(p, skip_first=True, skip_last=True, delim=None),
    },
    "wholesale": {
        "file": "wholesale.csv",
        "url": "https://archive.ics.uci.edu/ml/machine-learning-databases/00247/data/Wholesale%20customers%20data.csv",
        "k": 3,
        "load": lambda p: _load_wholesale(p),
    },
}

EXPECTED = {
    "iris": (150, 4, 3),
    "wine": (178, 13, 3),
    "seeds": (210, 7, 3),
    "glass": (214, 9, 6),
    "wdbc": (569, 30, 2),
    "ionosphere": (351, 34, 2),
    "banknote": (1372, 4, 2),
    "yeast": (1484, 8, 10),
    "wholesale": (440, 6, 3),
}


def _fetch(path, url):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.isfile(path) and os.path.getsize(path) > 0:
        return
    print(f"[dl] {os.path.basename(path)}")
    with urllib.request.urlopen(url, timeout=60) as r:
        data = r.read()
    with open(path, "wb") as f:
        f.write(data)


def _rows_from_file(path, delim):
    text = open(path, "r", encoding="utf-8", errors="ignore").read()
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split() if delim is None else [t.strip() for t in line.split(delim)]
        rows.append(parts)
    return rows


def _load_delimited(path, skip_first=False, skip_second=False, skip_last=False, delim=","):
    rows = _rows_from_file(path, delim)
    out = []
    for parts in rows:
        if skip_first:
            parts = parts[1:]
        if skip_second:
            parts = parts[1:]
        if skip_last:
            parts = parts[:-1]
        if not parts:
            continue
        try:
            out.append([float(x) for x in parts])
        except ValueError:
            continue
    X = np.asarray(out, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        raise RuntimeError(f"failed to parse {path}")
    return X


def _load_wholesale(path):
    rows = _rows_from_file(path, ",")
    if rows and not _is_float(rows[0][-1]):
        rows = rows[1:]
    return np.asarray([[float(x) for x in r[2:]] for r in rows], dtype=float)


def _is_float(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_uci(name):
    meta = UCI[name]
    path = os.path.join(DATA_DIR, meta["file"])
    _fetch(path, meta["url"])
    X = meta["load"](path)
    k = meta["k"]
    N, d = X.shape
    exp = EXPECTED[name]
    if (N, d, k) != exp:
        raise RuntimeError(f"{name}: expected {exp}, got {(N, d, k)}")
    return X, k


def f_grid(n):
    return [1.1, 2.0, float(np.sqrt(n)), float(n)]


def run_one(name, stage, M, k, f_list=None):
    M = np.asarray(M, dtype=float)
    m, n = M.shape
    k_eff = int(min(k, m, n))
    tautology = k_eff == n
    if f_list is None:
        f_list = f_grid(n)
    base = cpqr_cols(M, k_eff)
    rows = []
    for f in f_list:
        idx, nsw, ok = select_cols(M, k_eff, f)
        same = set(idx) == set(base)
        log_ge, smin_ge = log_vol_and_smin(M, idx)
        log_cp, smin_cp = log_vol_and_smin(M, base)
        rows.append(
            dict(
                matrix=name,
                stage=stage,
                m=m,
                n=n,
                k=k_eff,
                k_requested=k,
                f=float(f),
                swaps=int(nsw),
                certified=bool(ok),
                same_as_cpqr=bool(same),
                tautology=bool(tautology),
                log_vol_diff=log_ge - log_cp,
                smin_ge=smin_ge,
                smin_cpqr=smin_cp,
            )
        )
    return rows


def fmt_row(r):
    return (
        f"{r['matrix']:28s} {r['stage']:4s}  "
        f"n={r['n']:4d} k={r['k']:2d}  f={r['f']:8.3f}  "
        f"swaps={r['swaps']:3d}  cert={str(r['certified']):5s}  "
        f"same_cpqr={str(r['same_as_cpqr']):5s}  "
        f"dlogV={r['log_vol_diff']:9.3e}  taut={r['tautology']}"
    )


def main():
    np.random.seed(SEED)
    os.makedirs(DATA_DIR, exist_ok=True)
    all_rows = []

    print("=== adversarial matrices ===")
    for name, A, k in [
        ("Kahan(n=64,theta=1.2)", kahan(64, 1.2), 32),
        ("ExtKahan(n=64,theta=1.2)", extended_kahan(64, 1.2), 32),
    ]:
        rows = run_one(name, "ADV", A, k)
        all_rows.extend(rows)
        for r in rows:
            print(fmt_row(r))

    print("\n=== FS = columns of standardized X ===")
    print("=== INIT = columns of Zc.T (this is the missing test) ===")
    for name in [
        "banknote", "glass", "ionosphere", "iris", "seeds",
        "wdbc", "wholesale", "wine", "yeast",
    ]:
        X, k = load_uci(name)
        Z = standardize(X)
        Zc = center_columns(Z)
        N, d = Z.shape
        print(f"\n[data] {name:12s} N={N:4d} d={d:2d} k={k:2d}")

        fs_rows = run_one(f"{name}[FS]", "FS", Z, k)
        all_rows.extend(fs_rows)
        for r in fs_rows:
            print("  " + fmt_row(r))

        init_rows = run_one(f"{name}[INIT]", "INIT", Zc.T, k)
        all_rows.extend(init_rows)
        for r in init_rows:
            print("  " + fmt_row(r))

    fields = list(all_rows[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(all_rows)

    with open(OUT_TXT, "w", encoding="utf-8") as f:
        for r in all_rows:
            f.write(fmt_row(r) + "\n")

    n_fs = sum(r["swaps"] for r in all_rows if r["stage"] == "FS")
    n_init = sum(r["swaps"] for r in all_rows if r["stage"] == "INIT")
    print(f"\nSaved {len(all_rows)} rows -> {OUT_CSV}")
    print(f"FS total swaps:   {n_fs}")
    print(f"INIT total swaps: {n_init}")
    print("Look at stage=INIT, especially iris / wine / yeast.")
    print("Yeast FS is tautology (k>d). Yeast INIT is a real test (n=1484).")


if __name__ == "__main__":
    main()
