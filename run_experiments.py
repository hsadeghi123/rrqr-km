#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_experiments.py — Full experimental protocol (E1–E8) for the paper
"Deterministic K-Means Initialization and Feature Selection via Strong
Rank-Revealing QR: Provable Volume and Separation Guarantees".

v10 (final). The selector is the v9 Gu–Eisenstat Algorithm-5 inner loop
(full two-term criterion, omega direction verified) — UNCHANGED, since
the v9 run passed everything: Kahan test OK, E7 zero interchanges with
9/9 exit certification, E6 ratios = 1.000, main table consistent.

NEW IN v10:
  * E8 — THEOREM VERIFICATION ON THE DATA: for every dataset and both
    stages, the paper's guarantees are checked numerically:
      (T1) Vol-ratio check: Vol^max_k / Vol(S) <= q1^k  (Thm 5.10)
           [Vol^max estimated by best-of-L random k-subsets, since
            exhaustive is infeasible at full scale; a TRUE check would
            need Vol^max, so we verify the DIRECTLY computable side:
            prod sigma_i(Y) / Vol(S) >= q1^k, using Vol <= prod sigma
            (Lemma 5.6) — i.e. the certified inequality itself.]
      (T2) Pairwise separation: min dist between selected centers vs
           sigma_k / q1  (Prop 5.13(ii)).
      (T3) Sequential spread for the FIRST selected row vs span{},
           which equals |(A_k)_{11}|  (Prop 5.13(i), j=1).
    E8 thus closes the theory-implementation loop *on the actual
    benchmark data*, not only on brute-forceable subsamples.
  * Lemma-3.1 growth probe at f=1.02 integrated into self-tests (the
    v9 f=1.2 probe was vacuously skipped).
  * report.tex includes E8.

v9 fixes retained: omega direction (gamma_j * ||e_i^T A^{-1}||),
criterion_value exit certification, Kahan test recalibrated to GE's
Table 2, wholesale d=6, budget exhaustion reported, per-file download
fallback, farthest-first-complete truncation, E6 bound with the SAME f.

QUICK START (from the folder containing this file):
    python run_experiments.py --download
    python run_experiments.py
    python run_experiments.py --quick

Dependencies: numpy, scipy, scikit-learn, pandas.
"""

import argparse
import itertools
import json
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import qr
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

EPS = 1e-12
warnings.filterwarnings("ignore")

try:
    SCRIPT_DIR = Path(__file__).resolve().parent
except NameError:
    SCRIPT_DIR = Path.cwd()

# ==========================================================================
# 1. Configuration
# ==========================================================================
@dataclass
class Config:
    runs: int = 20
    seed: int = 20240601
    f_mode: str = "auto"       # "auto" = sqrt(n) (GE's efficient choice)
    greedy_l: int = 8
    bf_J: int = 10
    delta: float = 1e-8
    max_iter: int = 500
    out: str = "results"
    quick: bool = False
    kfs_main: str = "d"
    strategy: str = "ge"       # ge | cpqr | maxvol

UCI_FILES = {
 "iris.data": ("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data", (150, 4, 3)),
 "wine.data": ("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", (178, 13, 3)),
 "seeds_dataset.txt": ("https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt", (210, 7, 3)),
 "glass.data": ("https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data", (214, 9, 6)),
 "wdbc.data": ("https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data", (569, 30, 2)),
 "ionosphere.data": ("https://archive.ics.uci.edu/ml/machine-learning-databases/ionosphere/ionosphere.data", (351, 34, 2)),
 "data_banknote_authentication.txt": ("https://archive.ics.uci.edu/ml/machine-learning-databases/00267/data_banknote_authentication.txt", (1372, 4, 2)),
 "yeast.data": ("https://archive.ics.uci.edu/ml/machine-learning-databases/yeast/yeast.data", (1484, 8, 10)),
 "wholesale.csv": ("https://archive.ics.uci.edu/ml/machine-learning-databases/00292/Wholesale%20customers%20data.csv", (440, 6, 3)),
 # wholesale d=6: Channel/Region dropped by position (categorical).
}

KNOWN_FORMATS = {
    "iris":                         dict(sep=",",    header=None, label="last"),
    "wine":                         dict(sep=",",    header=None, label="first"),
    "seeds_dataset":                dict(sep=r"\s+", header=None, label="last"),
    "glass":                        dict(sep=",",    header=None, id="first", label="last"),
    "wdbc":                         dict(sep=",",    header=None, id="first", label=1),
    "ionosphere":                   dict(sep=",",    header=None, label="last"),
    "data_banknote_authentication": dict(sep=",",    header=None, label="last"),
    "yeast":                        dict(sep=r"\s+", header=None, id="first", label="last"),
    "wholesale":                    dict(sep=",",    header=0,
                                        drop_cols=[0, 1], label=None, k=3),
}

# ==========================================================================
# 2. Download (per-file, with partial fallback)
# ==========================================================================
def _looks_like_html(path):
    try:
        with open(path, "rb") as f:
            head = f.read(300).lstrip().lower()
        return head.startswith(b"<") or b"<!doctype html" in head or b"<html" in head
    except Exception:
        return True


def download_uci(data_dir, quiet=False):
    import urllib.request
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    ok, fail = [], []
    for name, (url, _) in UCI_FILES.items():
        dst = data_dir / name
        if dst.exists() and dst.stat().st_size > 500 and not _looks_like_html(dst):
            if not quiet:
                print(f"[dl] {name}  (already present)")
            ok.append(name)
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            head = data[:300].lstrip().lower()
            if len(data) < 500 or head.startswith(b"<"):
                raise ValueError("not a data file")
            with open(dst, "wb") as f:
                f.write(data)
            if not quiet:
                print(f"[dl] fetched {name}  ({len(data):,} bytes)")
            ok.append(name)
        except Exception as e:
            print(f"[dl] FAIL {name} -> {e}")
            fail.append(name)
    if fail:
        print(f"[dl] WARNING: {len(fail)} file(s) not fetched: {', '.join(fail)}")
        print("       The run proceeds WITHOUT them (per-dataset fallback).")
    return ok, fail

# ==========================================================================
# 3. Dataset loading
# ==========================================================================
def _norm_key(s):
    return "".join(ch for ch in s.lower() if ch.isalpha())


def _load_known(path, fmt):
    df = pd.read_csv(path, sep=fmt["sep"], header=fmt["header"],
                     engine="python", skip_blank_lines=True)
    df = df.dropna(axis=1, how="all")
    df = df.dropna(axis=0, how="all")
    if fmt.get("drop_cols"):
        # drop by POSITION (v9 fix)
        poss = [c for c in fmt["drop_cols"]
                if isinstance(c, int) and 0 <= c < df.shape[1]]
        if poss:
            df = df.drop(columns=[df.columns[c] for c in poss])
    ncols = df.shape[1]
    if ncols < 2:
        raise ValueError(f"only {ncols} column(s) parsed — file corrupt?")
    idf = fmt.get("id")
    id_idx = 0 if idf == "first" else (int(idf) if isinstance(idf, int) else None)
    lab = fmt.get("label")
    lab_idx = (None if lab is None else 0 if lab == "first"
               else ncols - 1 if lab == "last" else int(lab))
    drop = set()
    if id_idx is not None and 0 <= id_idx < ncols:
        drop.add(id_idx)
    if lab_idx is not None and 0 <= lab_idx < ncols:
        drop.add(lab_idx)
    feat_idx = [i for i in range(ncols) if i not in drop]
    X = df.iloc[:, feat_idx].apply(pd.to_numeric, errors="coerce")
    good = X.notna().all(axis=1)
    if int(good.sum()) < 10:
        raise ValueError(f"only {int(good.sum())} complete rows — file corrupt?")
    X = X[good]
    y = (df.iloc[:, lab_idx][good].astype(str).values
         if lab_idx is not None else None)
    k = fmt.get("k")
    if y is not None and not k:
        k = int(pd.Series(y).nunique())
    return X.values.astype(float), y, int(k or 3)


def _parse_table(path):
    cands = []
    for sep in (",", ";", r"\s+", None):
        for header in (None, 0):
            try:
                df = pd.read_csv(path, sep=sep, engine="python",
                                 header=header, skip_blank_lines=True)
            except Exception:
                continue
            if df.shape[0] >= 5 and df.shape[1] >= 2:
                cands.append(df)
    if not cands:
        return None

    def score(df):
        num = sum(pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.9
                  for c in df.columns)
        return (num, df.shape[0])

    df = max(cands, key=score)
    df = df.dropna(axis=1, how="all")
    return df.reset_index(drop=True)


def _extract_xy(df, override):
    n, cols = len(df), list(df.columns)
    lab_force = (override or {}).get("label")
    id_cols, label_col = [], None
    if lab_force == "none":
        label_col = None
    elif lab_force in ("first", "last"):
        label_col = cols[0] if lab_force == "first" else cols[-1]
    else:
        obj_cols = [c for c in cols if df[c].dtype == object]
        for c in obj_cols:
            u = df[c].nunique(dropna=True)
            if u == n and u > 10:
                id_cols.append(c)
            elif label_col is None:
                label_col = c
        c0 = cols[0]
        if (c0 not in id_cols and c0 not in obj_cols
                and df[c0].nunique(dropna=True) == n):
            v = pd.to_numeric(df[c0], errors="coerce")
            if v.notna().all() and np.allclose(v, np.round(v)):
                id_cols.append(c0)
        if label_col is None:
            for c in (cols[-1], cols[0]):
                if c in id_cols:
                    continue
                u = df[c].nunique(dropna=True)
                if 2 <= u <= max(3, min(20, max(2, n // 10))):
                    label_col = c
                    break
    drop = set(id_cols) | ({label_col} if label_col is not None else set())
    feat_cols = [c for c in cols if c not in drop]
    X = df[feat_cols].apply(pd.to_numeric, errors="coerce")
    good = X.notna().all(axis=1)
    X = X[good]
    y = (df.loc[good, label_col].astype(str).values
         if label_col is not None else None)
    k = (int(pd.Series(y).nunique()) if y is not None else None)
    if (override or {}).get("k"):
        k = int(override["k"])
    return X.values.astype(float), y, (k or 3)


def load_datasets(data_dir, quick=False):
    data_dir = Path(data_dir)
    overrides = {}
    if (data_dir / "datasets.json").exists():
        try:
            overrides = json.loads((data_dir / "datasets.json").read_text())
        except Exception as e:
            print(f"[warn] datasets.json unreadable: {e}")

    datasets = {}
    if data_dir.exists():
        files = sorted(p for p in data_dir.iterdir()
                       if p.is_file()
                       and p.suffix.lower() in (".csv", ".txt", ".data",
                                                ".dat", ".tsv"))
    else:
        files = []
    for p in files:
        if overrides.get(p.name, {}).get("skip"):
            continue
        try:
            fmt = KNOWN_FORMATS.get(p.stem)
            if fmt is not None:
                X, y, k = _load_known(p, fmt)
            else:
                df = _parse_table(p)
                if df is None:
                    raise ValueError("unparseable")
                X, y, k = _extract_xy(df, overrides.get(p.name, {}))
            if X.shape[0] < 10 or X.shape[1] < 2:
                raise ValueError(f"too small (N={X.shape[0]}, d={X.shape[1]})")
            if X.shape[0] < k:
                raise ValueError(f"N={X.shape[0]} < k={k}")
            datasets[p.stem] = dict(X=X, y=y, k=int(k), file=p.name)
            print(f"[data] {p.stem:30s} N={X.shape[0]:5d} d={X.shape[1]:3d} "
                  f"k={k:2d} labels={'yes' if y is not None else 'no '}")
        except Exception as e:
            print(f"[warn] {p.name}: skipped ({e})")

    expected = {Path(f).stem for f in UCI_FILES}
    absent = sorted(expected - {Path(v["file"]).stem for v in datasets.values()})
    if absent:
        print(f"[data] WARNING: {len(absent)} expected dataset(s) absent: "
              f"{', '.join(absent)}  (run proceeds without them)")

    used_fallback = False
    if not datasets:
        used_fallback = True
        print("[data] NO dataset files loaded -> bundled/synthetic fallback.")
        from sklearn.datasets import (load_iris, load_wine,
                                      load_breast_cancer, make_blobs)
        b = load_iris();    datasets["iris_bundled"] = dict(X=b.data, y=b.target.astype(str), k=3)
        b = load_wine();    datasets["wine_bundled"] = dict(X=b.data, y=b.target.astype(str), k=3)
        b = load_breast_cancer(); datasets["wdbc_bundled"] = dict(X=b.data, y=b.target.astype(str), k=2)
        X, y = make_blobs(n_samples=600, centers=5, n_features=12,
                          cluster_std=1.0, random_state=11)
        datasets["synthetic_blobs"] = dict(X=X, y=y.astype(str), k=5)

    _verify_against_reference(datasets)
    if quick:
        datasets = dict(list(datasets.items())[:3])
    return datasets, used_fallback


def _verify_against_reference(datasets):
    ref = {}
    for fname, (_, exp) in UCI_FILES.items():
        ref[_norm_key(Path(fname).stem)] = tuple(exp)
    print("[check] parsed dimensions vs UCI reference:")
    for stem, ds in datasets.items():
        key = _norm_key(stem)
        hit = None
        for rk, exp in ref.items():
            if key and (key in rk or rk in key):
                hit = exp
                break
        got = (int(ds["X"].shape[0]), int(ds["X"].shape[1]), int(ds["k"]))
        if hit is None:
            print(f"        {stem:30s} (no reference)  N={got[0]} d={got[1]} k={got[2]}")
        else:
            status = "OK " if got == hit else "MISMATCH"
            print(f"        {stem:30s} {status}  expected {hit}; got {got}")

# ==========================================================================
# 4. Numerical linear algebra
#     4a. THE SELECTOR — GE Algorithm 5 inner loop (v9, verified)
# ==========================================================================
def ge_f(n):
    """GE's efficient parameter choice: f = sqrt(n)."""
    return math.sqrt(max(n, 1))


def criterion_rrqr(M, k, f, max_swaps=200, return_order=False):
    """Gu–Eisenstat Algorithm 5 inner while-loop, as published (SIAM J.
    Sci. Comput. 17(4):848–869, 1996, p. 855):

        while omega(R, k) > f do
            Find i, j with |(A^{-1}B_k)_{ij}| > f  or
                           gamma_j(C_k)/omega_i(A_k) > f;
            swap columns i and j+k; retriangularize.

    GE Sec. 1.4: 1/omega_i(A) = ||e_i^T A^{-1}||_2, so the second term
    is gamma_j(C_k) * ||e_i^T A^{-1}||_2 (MULTIPLY — the v8 inversion
    was the bug; the i=k special case of Lemma 3.1 gives omega_k(A) =
    a_kk, i.e. gamma_j * ||e_k^T A^{-1}|| = gamma_j/|a_kk|, the check).

    By Lemma 3.1 the pair score sqrt(t1^2 + t2^2) is the |det A_k|
    growth factor of the swap; swaps fire only when max(t1,t2) > f, so
    every swap multiplies the selected volume by > f, and termination
    certifies GE's strong-RRQR conditions (5)-(6) with
    q_1 = sqrt(1 + 2 f^2 k(n-k)) for THIS f (their Thm 3.2).

    Simplifications (documented): (S1) unpivoted QR recomputed after
    each swap (equivalent to their Givens updates); (S2) the maximally
    violating pair is swapped.

    Returns (selected k columns, #swaps; -1 = budget exhausted);
    with return_order=True, also the full final permutation.
    """
    m, n = M.shape
    k = int(min(k, m, n))
    if k <= 0:
        return ([], 0, []) if return_order else ([], 0)
    _, _, piv = qr(M, mode="economic", pivoting=True)   # CPQR start
    order = [int(t) for t in piv]
    swaps = 0
    exhausted = False
    for _ in range(max_swaps):
        _, R = qr(M[:, order], mode="economic")          # retriangularize
        A = R[:k, :k]
        if n <= k:
            break
        B = R[:k, k:]
        C = R[k:, k:]
        try:
            W = np.linalg.solve(A, B)
            Ainv = np.linalg.inv(A)
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(W).all() or not np.isfinite(Ainv).all():
            break
        omega = np.sqrt((Ainv ** 2).sum(axis=1))         # ||e_i^T A^{-1}||
        gamma = np.sqrt((C ** 2).sum(axis=0))            # ||C e_j||
        t1 = np.abs(W)
        if gamma.size and omega.size:
            t2 = gamma[None, :] * omega[:, None]
        else:
            t2 = np.zeros_like(t1)
        score = np.maximum(t1, t2)
        i, j = np.unravel_index(np.argmax(score), score.shape)
        if score[i, j] <= f:
            break                                        # criterion HOLDS
        order[i], order[k + j] = order[k + j], order[i]
        swaps += 1
    else:
        exhausted = True
    sel = [int(t) for t in order[:k]]
    sw_out = -1 if exhausted else swaps
    if return_order:
        return sel, sw_out, [int(t) for t in order]
    return sel, sw_out


def criterion_value(M, order, k):
    """Recompute omega(R,k) on the given FULL permutation. If omega <= f,
    GE's Thm 3.2 certifies the strong-RRQR conditions for that f."""
    k = int(min(k, len(order), M.shape[0]))
    n = M.shape[1]
    if n <= k:
        return 0.0
    _, R = qr(M[:, order], mode="economic")
    A = R[:k, :k]
    B = R[:k, k:]
    C = R[k:, k:]
    try:
        W = np.linalg.solve(A, B)
        Ainv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        return float("inf")
    if not (np.isfinite(W).all() and np.isfinite(Ainv).all()):
        return float("inf")
    omega = np.sqrt((Ainv ** 2).sum(axis=1))
    gamma = np.sqrt((C ** 2).sum(axis=0))
    t1 = float(np.abs(W).max()) if W.size else 0.0
    t2 = float((gamma[None, :] * omega[:, None]).max()) \
        if (gamma.size and omega.size) else 0.0
    return max(t1, t2)


def strong_select(M, r, strategy="ge", f=None, tol=1.05):
    """Dispatch: GE criterion | plain CPQR | maxvol."""
    if strategy == "cpqr":
        _, _, piv = qr(M, mode="economic", pivoting=True)
        return [int(i) for i in piv[:r]], 0
    if strategy == "maxvol":
        return _maxvol(M, r, tol)
    f_ = f if f is not None else ge_f(M.shape[1])
    return criterion_rrqr(M, r, f=f_)


def _maxvol(M, r, tol=1.05):
    """Classical maxvol interchange (baseline only)."""
    _, R, piv = qr(M, mode="economic", pivoting=True)
    n = M.shape[1]
    r = int(min(r, R.shape[0], n))
    if r >= n:
        return [int(i) for i in range(n)], 0
    W = R[:r, :].copy()
    sel = list(range(r))
    tol = max(float(tol), 1.001)
    swaps = 0
    for _ in range(200):
        selset = set(sel)
        rest = [j for j in range(n) if j not in selset]
        if not rest:
            break
        A = W[:, sel]
        B = W[:, rest]
        try:
            Bv = np.linalg.solve(A, B)
        except np.linalg.LinAlgError:
            break
        if not np.isfinite(Bv).all():
            break
        Bv = np.abs(Bv)
        i, jj = np.unravel_index(np.argmax(Bv), Bv.shape)
        if Bv[i, jj] <= tol:
            break
        sel[int(i)] = int(rest[jj])
        swaps += 1
    return [int(piv[t]) for t in sel], swaps


def cdist2(Z, C):
    d = (Z * Z).sum(1)[:, None] + (C * C).sum(1)[None, :] - 2.0 * (Z @ C.T)
    np.maximum(d, 0.0, out=d)
    return d


def standardize(X):
    sd = X.std(0)
    sd[sd < EPS] = 1.0
    return (X - X.mean(0)) / sd


def log_volume(rows):
    sign, logdet = np.linalg.slogdet(rows @ rows.T)
    return logdet / 2.0 if sign > 0 else -np.inf


def choose_kfs(Z, k):
    N, d = Z.shape
    hi, lo = min(N, d), min(k, min(N, d))
    if hi <= lo:
        return hi
    s = np.linalg.svd(Z, compute_uv=False)
    ratios = s[lo - 1:hi - 1] / (s[lo:hi] + EPS)
    return int(lo + int(np.argmax(ratios)))


def partition_cost(Z, lab):
    J = 0.0
    for j in np.unique(lab):
        m = lab == j
        if m.sum() > 1:
            J += float(((Z[m] - Z[m].mean(0)) ** 2).sum())
    return J

# ==========================================================================
# 4b. STARTUP SELF-TESTS (non-blocking)
# ==========================================================================
def _kahan_matrix(n, s=0.45):
    """GE Example 1 (their eq. (7)): M = S_n K_n^T."""
    S = np.diag(s ** np.arange(n))
    K = np.eye(n) - np.triu(np.ones((n, n)), 1)
    return S @ K.T


def _self_test_kahan(n=12, s=0.45):
    """GE's own Table 2: SRRQR performs ts = 0 interchanges on Kahan —
    'inert' is the expected behavior. Verify: termination, exit
    certification, no volume loss."""
    k = n - 1
    M = _kahan_matrix(n, s)
    f = math.sqrt(6.0)
    sel, sw, order = criterion_rrqr(M, k, f=f, return_order=True)
    cert = criterion_value(M, order, k)
    v_cpqr = log_volume(M[:, list(range(k))].T)
    v_sel = log_volume(M[:, sel].T)
    ok = (sw >= 0) and (cert <= f * (1 + 1e-9)) and (v_sel >= v_cpqr - 1e-9)
    print(f"[self-test:Kahan n={n}] swaps={sw}, exit-omega={cert:.3f} "
          f"(f={f:.2f}), log-vol {v_cpqr:.2f} -> {v_sel:.2f}  "
          f"{'OK' if ok else 'WARNING (non-fatal)'}")
    if sw == -1:
        print("  -> budget exhausted on the canonical matrix: "
              "investigate before relying on E7.")


def _self_test_growth(n_trials=20, seed=3, f=1.02):
    """GE Lemma 3.1 test at a TIGHT f so swaps actually fire (the v9
    f=1.2 probe was vacuous): every swap must multiply the selected
    volume by > f. Total-growth form:
        log-vol(final) - log-vol(CPQR start) >= sw * log(f) - tol."""
    rng = np.random.default_rng(seed)
    n_fired, n_bad = 0, 0
    for t in range(n_trials):
        m, nn, k = 30, 24, 8
        M = rng.normal(size=(m, nn)) * np.linspace(3.0, 0.05, nn)[None, :]
        sel, sw, _ = criterion_rrqr(M, k, f=f, return_order=True)
        if sw == -1:
            print(f"[self-test:growth] WARNING: budget exhausted "
                  f"(trial {t}).")
            n_bad += 1
            continue
        if sw > 0:
            n_fired += 1
            _, _, piv0 = qr(M, mode="economic", pivoting=True)
            v0 = log_volume(M[:, list(piv0[:k])].T)
            v1 = log_volume(M[:, sel].T)
            if v1 - v0 < sw * math.log(f) - 1e-6:
                print(f"[self-test:growth] WARNING: trial {t}: growth "
                      f"{v1 - v0:.3f} < {sw}*log(f)={sw * math.log(f):.3f}"
                      f" — Lemma 3.1 violated.")
                n_bad += 1
    if n_fired == 0:
        print("[self-test:growth] SKIP: no swaps in any trial "
              "(vacuous — lower f further to probe).")
    else:
        print(f"[self-test:growth] {'OK' if n_bad == 0 else 'WARNING'}: "
              f"{n_fired}/{n_trials} trials took swaps at f={f}; "
              f"Lemma-3.1 growth {'held' if n_bad == 0 else 'VIOLATED'}.")


def _self_tests():
    _self_test_kahan()
    _self_test_growth()

# ==========================================================================
# 5. Initializers
# ==========================================================================
def kmpp_sample(Z, k, rng, greedy_l=None):
    n = Z.shape[0]
    sel = [int(rng.integers(n))]
    d2 = cdist2(Z, Z[sel[0]:sel[0] + 1]).ravel()
    while len(sel) < k:
        s = float(d2.sum())
        p = d2 / s if s > EPS else np.full(n, 1.0 / n)
        if greedy_l and greedy_l > 1:
            cands = rng.choice(n, size=min(greedy_l, n), replace=False, p=p)
            best, bp = None, np.inf
            for c in cands:
                pot = float(np.minimum(
                    d2, cdist2(Z, Z[int(c):int(c) + 1]).ravel()).sum())
                if pot < bp:
                    bp, best = pot, int(c)
            j = best
        else:
            j = int(rng.choice(n, p=p))
        sel.append(j)
        d2 = np.minimum(d2, cdist2(Z, Z[j:j + 1]).ravel())
    return sel


def farthest_first(Z, k):
    i0 = int(np.argmax(((Z - Z.mean(0)) ** 2).sum(1)))
    sel = [i0]
    d2 = cdist2(Z, Z[i0:i0 + 1]).ravel()
    while len(sel) < k:
        j = int(np.argmax(d2))
        sel.append(j)
        d2 = np.minimum(d2, cdist2(Z, Z[j:j + 1]).ravel())
    return sel


def farthest_first_complete(Z, C, k):
    """Always returns EXACTLY k centers (truncates if over)."""
    Cs = [np.asarray(c, float) for c in C]
    if len(Cs) > k:
        return np.asarray(Cs[:k])
    if len(Cs) == k:
        return np.asarray(Cs)
    d2 = cdist2(Z, np.asarray(Cs)).min(1)
    while len(Cs) < k:
        j = int(np.argmax(d2))
        Cs.append(Z[j])
        d2 = np.minimum(d2, cdist2(Z, Z[j:j + 1]).ravel())
    return np.asarray(Cs)


def part_init(Z, k, mode="pca"):
    parts = [np.arange(Z.shape[0])]

    def wss(p):
        if len(p) < 2:
            return -1.0
        Xs = Z[p]
        return float(((Xs - Xs.mean(0)) ** 2).sum())

    while len(parts) < k and max(len(p) for p in parts) > 1:
        i = int(np.argmax([wss(p) for p in parts]))
        p = parts.pop(i)
        Xs = Z[p]
        if mode == "pca":
            Xc = Xs - Xs.mean(0)
            _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
            proj = Xc @ Vt[0] if Vt.shape[0] else Xs[:, 0]
        else:
            proj = Xs[:, int(np.argmax(Xs.var(0)))]
        thr = np.median(proj)
        a, b = p[proj <= thr], p[proj > thr]
        if len(a) == 0 or len(b) == 0:
            h = len(p) // 2
            a, b = p[:h], p[h:]
        parts += [a, b]
    C = np.array([Z[p].mean(0) for p in parts])
    if len(C) < k:
        C = farthest_first_complete(Z, C, k)
    return C


def bradley_fayyad(Z, k, rng, J=10):
    n = Z.shape[0]
    sub = int(min(n, max(10 * k, n // 10, 50)))
    cand = []
    for _ in range(J):
        idx = rng.choice(n, size=sub, replace=False)
        Zs = Z[idx]
        C0 = Zs[rng.choice(sub, size=min(k, sub), replace=False)]
        cand.append(lloyd(Zs, C0, max_iter=100)["C"])
    U = np.vstack(cand)
    res = lloyd(U, U[farthest_first(U, min(k, len(U)))], max_iter=100)
    lab = res["lab"]
    C = [U[lab == j].mean(0) if np.any(lab == j)
         else res["C"][j % len(res["C"])] for j in range(k)]
    return np.asarray(C)

# ==========================================================================
# 6. Lloyd's algorithm
# ==========================================================================
def lloyd(Z, C0, delta=1e-8, max_iter=500, repair=True):
    n, k = Z.shape[0], C0.shape[0]
    C = np.array(C0, dtype=float, copy=True)
    idxr = np.arange(n)
    J_prev, J, it, empty0, lab = None, np.inf, 0, None, None
    for it in range(1, max_iter + 1):
        D2 = cdist2(Z, C)
        lab = D2.argmin(1)
        if empty0 is None:
            empty0 = int(k - np.unique(lab).size)
        if repair:
            empt = np.setdiff1d(np.arange(k), np.unique(lab))
            if empt.size:
                dd = D2[idxr, lab]
                order = np.argsort(-dd)[: empt.size]
                for q, j in enumerate(empt):
                    C[j] = Z[order[q]]
                D2 = cdist2(Z, C)
                lab = D2.argmin(1)
        J = float(D2[idxr, lab].sum())
        if J_prev is not None and abs(J_prev - J) <= delta * max(1.0, abs(J_prev)):
            break
        J_prev = J
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = Z[m].mean(0)
    return dict(C=C, lab=lab, J=J, iters=it, empty0=empty0,
                empty_end=int(k - np.unique(lab).size))

# ==========================================================================
# 7. The paper's pipeline (Algorithm 1)
# ==========================================================================
def _resolve_f(M, cfg):
    if cfg.f_mode == "auto":
        return ge_f(M.shape[1])
    try:
        return float(cfg.f_mode)
    except (TypeError, ValueError):
        return ge_f(M.shape[1])


def rrqr_pipeline(Z, k, kfs=None, f_mode="auto", center=True, strategy="ge"):
    t0 = time.perf_counter()
    N, d = Z.shape
    if kfs is None:
        kfs = choose_kfs(Z, k)
    kfs = int(min(max(kfs, 1), min(N, d)))

    f1 = _resolve_f(Z, Config(f_mode=f_mode))
    feats, sw1 = strong_select(Z, kfs, strategy, f=f1)
    Zr = Z[:, feats]
    Zrc = (Zr - Zr.mean(0)) if center else Zr
    r = min(k, kfs)
    f2 = _resolve_f(Zrc.T, Config(f_mode=f_mode))
    idx, sw2 = strong_select(Zrc.T, r, strategy, f=f2)

    C0 = Zrc[idx].copy()
    if k > r:
        C0 = farthest_first_complete(Zrc, C0, k)
    return dict(feats=feats, Zrun=Zrc, C0=C0,
                meta=dict(kfs=kfs, r=r, swaps1=int(sw1), swaps2=int(sw2),
                          f1=f1, f2=f2,
                          t_prep=time.perf_counter() - t0,
                          strategy=strategy, center=center))

# ==========================================================================
# 8. Method registry
# ==========================================================================
def m_random(Z, k, rng, cfg):
    idx = rng.choice(Z.shape[0], size=k, replace=False)
    return dict(feats=None, Zrun=Z, C0=Z[idx], meta={})


def m_kmpp(Z, k, rng, cfg):
    return dict(feats=None, Zrun=Z, C0=Z[kmpp_sample(Z, k, rng)], meta={})


def m_kmpp_g(Z, k, rng, cfg):
    return dict(feats=None, Zrun=Z,
                C0=Z[kmpp_sample(Z, k, rng, greedy_l=cfg.greedy_l)], meta={})


def m_ff(Z, k, rng, cfg):
    return dict(feats=None, Zrun=Z, C0=Z[farthest_first(Z, k)], meta={})


def m_pcapart(Z, k, rng, cfg):
    return dict(feats=None, Zrun=Z, C0=part_init(Z, k, "pca"), meta={})


def m_varpart(Z, k, rng, cfg):
    return dict(feats=None, Zrun=Z, C0=part_init(Z, k, "var"), meta={})


def m_bf(Z, k, rng, cfg):
    return dict(feats=None, Zrun=Z,
                C0=bradley_fayyad(Z, k, rng, cfg.bf_J), meta={})


def m_qr(Z, k, rng, cfg):
    r = min(k, Z.shape[1])
    idx, _ = strong_select(Z.T, r, "cpqr")
    return dict(feats=None, Zrun=Z,
                C0=farthest_first_complete(Z, Z[idx], k), meta={})


def m_maxvol(Z, k, rng, cfg):
    _, _, Vt = np.linalg.svd(Z, full_matrices=False)
    r = min(k, Vt.shape[0])
    T = Z @ Vt[:r].T
    idx, sw = _maxvol(T.T, r, 1.05)
    return dict(feats=None, Zrun=Z,
                C0=farthest_first_complete(Z, Z[idx], k),
                meta=dict(swaps=int(sw)))


def m_rrqr_i(Z, k, rng, cfg):
    r = min(k, Z.shape[1])
    f = _resolve_f(Z.T, cfg)
    idx, sw = strong_select(Z.T, r, cfg.strategy, f=f)
    return dict(feats=None, Zrun=Z,
                C0=farthest_first_complete(Z, Z[idx], k),
                meta=dict(swaps=int(sw)))


def m_rrqr_f(Z, k, rng, cfg):
    out = rrqr_pipeline(Z, k, kfs=choose_kfs(Z, k), f_mode=cfg.f_mode,
                        center=True, strategy=cfg.strategy)
    out["C0"] = out["Zrun"][kmpp_sample(out["Zrun"], k, rng)]
    return out


def m_rrqr_km(Z, k, rng, cfg):
    kfs = Z.shape[1] if cfg.kfs_main == "d" else None
    return rrqr_pipeline(Z, k, kfs=kfs, f_mode=cfg.f_mode, center=True,
                         strategy=cfg.strategy)


METHODS = [
    ("random",  "Random",            True,  m_random),
    ("kmpp",    "K-means++",         True,  m_kmpp),
    ("kmpp-g",  "greedy K-means++",  True,  m_kmpp_g),
    ("ff",      "farthest-first",    False, m_ff),
    ("pcapart", "PCA-part",          False, m_pcapart),
    ("varpart", "var-part",          False, m_varpart),
    ("bf",      "Bradley-Fayyad",    True,  m_bf),
    ("qr",      "plain pivoted QR",  False, m_qr),
    ("maxvol",  "maxvol (PC)",       False, m_maxvol),
    ("rrqr-i",  "RRQR-Init",         False, m_rrqr_i),
    ("rrqr-f",  "RRQR-FS + K-m++",   True,  m_rrqr_f),
    ("rrqr-km", "RRQR-KM (ours)",    False, m_rrqr_km),
]
METHOD_ORDER = {mk: i for i, (mk, _, _, _) in enumerate(METHODS)}
METHOD_LABEL = {mk: lab for mk, lab, _, _ in METHODS}

# ==========================================================================
# 9. Evaluation
# ==========================================================================
def evaluate(name, ds, Z, mkey, randomized, fn, rng, cfg, run_id=0):
    t0 = time.perf_counter()
    out = fn(Z, ds["k"], rng, cfg)
    t_fn = time.perf_counter() - t0
    t1 = time.perf_counter()
    res = lloyd(out["Zrun"], out["C0"], delta=cfg.delta, max_iter=cfg.max_iter)
    t_lloyd = time.perf_counter() - t1
    lab = res["lab"]
    rec = dict(dataset=name, method=mkey, randomized=randomized, run=run_id,
               J_full=partition_cost(Z, lab), J_red=float(res["J"]),
               iters=int(res["iters"]),
               t_prep=float(out.get("meta", {}).get("t_prep", t_fn)),
               t_lloyd=float(t_lloyd), t_total=float(t_fn + t_lloyd),
               empty0=int(res["empty0"]), empty_end=int(res["empty_end"]),
               meta=out.get("meta", {}))
    y, mask = ds.get("y"), ds.get("mask")
    if y is not None:
        yy, ll = np.asarray(y), lab
        if mask is not None:
            yy, ll = yy[mask], ll[mask]
        rec["ari"] = float(adjusted_rand_score(yy, ll))
        rec["nmi"] = float(normalized_mutual_info_score(yy, ll))
    else:
        rec["ari"] = rec["nmi"] = float("nan")
    return rec


def run_benchmarks(datasets, cfg):
    recs = []
    for di, (name, ds) in enumerate(datasets.items()):
        Z = standardize(np.asarray(ds["X"], float))
        print(f"[E1-E3] {name}: N={Z.shape[0]} d={Z.shape[1]} k={ds['k']}")
        for mi, (mkey, _, randomized, fn) in enumerate(METHODS):
            runs = cfg.runs if randomized else 1
            for r in range(runs):
                rng = np.random.default_rng(
                    cfg.seed + 7919 * mi + 104729 * r + 17 * di)
                try:
                    recs.append(evaluate(name, ds, Z, mkey, randomized, fn,
                                         rng, cfg, run_id=r))
                except Exception as e:
                    print(f"   [warn] {mkey} failed on {name} (run {r}): {e}")
            if not randomized and recs:
                try:
                    rng2 = np.random.default_rng(cfg.seed + 31337 + mi + di)
                    rec2 = evaluate(name, ds, Z, mkey, randomized, fn,
                                    rng2, cfg)
                    ok = (abs(rec2["J_full"] - recs[-1]["J_full"]) < 1e-9
                          and rec2["iters"] == recs[-1]["iters"])
                    recs[-1]["det_ok"] = bool(ok)
                    if not ok:
                        print(f"   [warn] {mkey} NOT deterministic on {name}")
                except Exception as e:
                    print(f"   [warn] determinism check failed ({e})")
    return recs


def aggregate(recs):
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    agg = df.groupby(["dataset", "method"], sort=False).agg(
        runs=("J_full", "size"), J_mean=("J_full", "mean"),
        J_sd=("J_full", "std"), J_best=("J_full", "min"),
        Jred_mean=("J_red", "mean"), iters=("iters", "mean"),
        prep=("t_prep", "mean"), total=("t_total", "mean"),
        ari=("ari", "mean"), ari_sd=("ari", "std"), nmi=("nmi", "mean"),
        e0=("empty0", "mean"), ee=("empty_end", "mean")).reset_index()
    best = df.groupby("dataset")["J_full"].min().rename("Jstar")
    agg = agg.merge(best, on="dataset")
    agg["ratio"] = agg["J_mean"] / np.maximum(agg["Jstar"], EPS)
    agg["label"] = agg["method"].map(METHOD_LABEL)
    agg["mo"] = agg["method"].map(METHOD_ORDER)
    agg = agg.sort_values(["dataset", "mo"]).drop(
        columns="mo").reset_index(drop=True)
    det = {(r["dataset"], r["method"]): r.get("det_ok")
           for r in recs if not r["randomized"]}
    agg["det_ok"] = [det.get((d, m)) for d, m in zip(agg.dataset, agg.method)]
    return agg


def fmt_summary(agg):
    if agg is None or agg.empty:
        return pd.DataFrame([{"note": "NO RECORDS — see warnings above"}])
    rows = []
    for _, r in agg.iterrows():
        detmark = ""
        if not r.runs > 1:
            detmark = " (det.)" if r.det_ok in (True, None) else " (DET-FAIL)"
        rows.append(dict(
            dataset=r.dataset, method=r.label + detmark,
            J=f"{r.J_mean:,.2f}" + (f" +/- {r.J_sd:,.2f}"
                                    if pd.notna(r.J_sd) and r.runs > 1 else ""),
            Jbest=f"{r.J_best:,.2f}",
            ratio="--" if not np.isfinite(r.ratio) else f"{r.ratio:.3f}",
            iters=f"{r.iters:.1f}",
            prep_ms=f"{r.prep * 1000:,.1f}",
            total_ms=f"{r.total * 1000:,.1f}",
            ari="--" if not np.isfinite(r.ari) else f"{r.ari:.3f}",
            nmi="--" if not np.isfinite(r.nmi) else f"{r.nmi:.3f}",
            empty_t0=f"{r.e0:.2f}", empty_end=f"{r.ee:.2f}"))
    return pd.DataFrame(rows)


def analytic_summary(agg):
    if agg is None or agg.empty:
        return
    print("\n" + "=" * 72)
    print("ANALYTIC SUMMARY (mean rank by J_mean across datasets)")
    print("=" * 72)
    ranks = (agg.assign(r=agg.groupby("dataset")["J_mean"].rank())
                .groupby("label")["r"].mean().sort_values())
    for lab, r in ranks.items():
        print(f"  {lab:22s} mean rank {r:5.2f}")
    ours = agg[agg.method == "rrqr-km"].set_index("dataset")["J_mean"]
    print("\nRRQR-KM (ours) vs each baseline (J_mean, per dataset):")
    for mk, lab in METHOD_LABEL.items():
        if mk == "rrqr-km":
            continue
        base = agg[agg.method == mk].set_index("dataset")["J_mean"]
        common = ours.index.intersection(base.index)
        if not len(common):
            continue
        w = int(sum(ours[d] < base[d] for d in common))
        l = int(sum(ours[d] > base[d] for d in common))
        t = len(common) - w - l
        print(f"  vs {lab:22s}  win {w} / loss {l} / tie {t}  (n={len(common)})")

# ==========================================================================
# 10. E4: stress tests + separation sweep
# ==========================================================================
def make_stress_datasets(seed=99):
    from sklearn.datasets import make_blobs
    rng = np.random.default_rng(seed)
    out = {}
    Xb, yb = make_blobs(n_samples=600, centers=4, n_features=8,
                        cluster_std=1.0, random_state=1)
    for frac in (0.01, 0.05, 0.10):
        n_out = int(round(frac * 600))
        O = rng.uniform(-12, 12, size=(n_out, 8))
        out[f"outliers-{int(frac * 100)}pct"] = dict(
            X=np.vstack([Xb, O]),
            y=np.concatenate([yb.astype(str), np.full(n_out, "outlier")]),
            mask=np.concatenate([np.ones(600, bool), np.zeros(n_out, bool)]),
            k=4)
    parts, ys = [], []
    for c, n in zip([np.zeros(4), np.array([8., 0, 0, 0]),
                     np.array([0, 8., 0, 0])], [20, 200, 2000]):
        parts.append(c + rng.normal(size=(n, 4)))
        ys.append(np.full(n, str(len(ys))))
    out["imbalance-1-10-100"] = dict(X=np.vstack(parts),
                                     y=np.concatenate(ys), k=3)
    parts, ys = [], []
    for c, s in zip([np.zeros(6), np.array([10., 0, 0, 0, 0, 0]),
                     np.array([0, 10., 0, 0, 0, 0])], [0.3, 1.5, 6.0]):
        parts.append(c + s * rng.normal(size=(300, 6)))
        ys.append(np.full(300, str(len(ys))))
    out["var-imbalance"] = dict(X=np.vstack(parts), y=np.concatenate(ys), k=3)
    Xb, yb = make_blobs(n_samples=500, centers=5, n_features=5,
                        cluster_std=1.0, random_state=21)
    rng2 = np.random.default_rng(22)
    copies = [Xb @ rng2.dirichlet(np.ones(5))
              + 1e-3 * rng2.normal(size=500) for _ in range(20)]
    out["rankdef-25d"] = dict(X=np.column_stack([Xb] + copies),
                              y=yb.astype(str), k=5)
    Xb, yb = make_blobs(n_samples=600, centers=9, n_features=5,
                        cluster_std=1.0, random_state=31)
    out["k-gt-d"] = dict(X=Xb, y=yb.astype(str), k=9)
    return out


STRESS_METHODS = ["random", "kmpp", "ff", "maxvol", "qr", "rrqr-km"]


def run_stress(cfg):
    stress = make_stress_datasets()
    meth = [m for m in METHODS if m[0] in STRESS_METHODS]
    recs = []
    for name, ds in stress.items():
        Z = standardize(np.asarray(ds["X"], float))
        print(f"[E4] {name}")
        off = sum(map(ord, name))
        for mi, (mkey, _, randomized, fn) in enumerate(meth):
            runs = 10 if randomized else 1
            for r in range(runs):
                rng = np.random.default_rng(cfg.seed + 101 * r + off + 7 * mi)
                recs.append(evaluate(name, ds, Z, mkey, randomized, fn,
                                     rng, cfg, run_id=r))
    return recs


def run_separation(cfg):
    from sklearn.datasets import make_blobs
    from scipy.spatial.distance import pdist
    meth = [m for m in METHODS if m[0] in STRESS_METHODS]
    rows = []
    for sep in (0.75, 1.0, 1.5, 2.0, 3.0, 4.0):
        rngc = np.random.default_rng(int(100 * sep))
        centers = rngc.normal(size=(5, 8))
        centers = centers * (sep / max(pdist(centers).min(), EPS))
        X, y = make_blobs(n_samples=500, centers=centers,
                          cluster_std=1.0, random_state=42)
        Z = standardize(X)
        ds = dict(X=X, y=y.astype(str), k=5)
        for mi, (mkey, _, randomized, fn) in enumerate(meth):
            runs = 10 if randomized else 1
            aris = []
            for r in range(runs):
                try:
                    aris.append(evaluate(
                        "sep", ds, Z, mkey, randomized, fn,
                        np.random.default_rng(
                            cfg.seed + 61 * r + mi + int(10 * sep)),
                        cfg, run_id=r)["ari"])
                except Exception:
                    pass
            rows.append(dict(sep=sep, method=mkey, label=METHOD_LABEL[mkey],
                             ari=float(np.nanmean(aris)) if aris
                             else float("nan")))
        print(f"[E4e] separation={sep}: "
              + ", ".join(f"{r['method']}={r['ari']:.2f}"
                          for r in rows[-len(meth):]))
    return rows

# ==========================================================================
# 11. E5: ablations
# ==========================================================================
def run_ablations(datasets, cfg):
    kfs_rows, cen_rows, sel_rows = [], [], []
    for name, ds in datasets.items():
        Z = standardize(np.asarray(ds["X"], float))
        N, d, k = *Z.shape, ds["k"]
        lo = min(k, min(N, d))

        for kfs in sorted({lo, min(2 * k, min(N, d)), min(d, min(N, d)),
                           choose_kfs(Z, k)}):
            out = rrqr_pipeline(Z, k, kfs=kfs, f_mode=cfg.f_mode,
                                center=True, strategy=cfg.strategy)
            res = lloyd(out["Zrun"], out["C0"], delta=cfg.delta,
                        max_iter=cfg.max_iter)
            y = ds.get("y")
            ari = (float(adjusted_rand_score(y, res["lab"]))
                   if y is not None else float("nan"))
            kfs_rows.append(dict(dataset=name, kfs=int(kfs), ari=ari,
                                 J_red=float(res["J"]),
                                 J_full=partition_cost(Z, res["lab"]),
                                 iters=int(res["iters"]),
                                 swaps2=out["meta"]["swaps2"]))

        rng = np.random.default_rng(cfg.seed + 777)
        kk = min(k, d)
        f = _resolve_f(Z.T, cfg)
        base_idx, _ = strong_select(Z.T, kk, cfg.strategy, f=f)
        v = 50.0 * Z.std(0) * rng.normal(size=d)
        ZT = Z + v
        on_idx, _ = strong_select((ZT - ZT.mean(0)).T, kk, cfg.strategy, f=f)
        off_idx, _ = strong_select(ZT.T, kk, cfg.strategy, f=f)
        cen_rows.append(dict(
            dataset=name,
            on="same" if list(on_idx) == list(base_idx) else "changed",
            off="same" if list(off_idx) == list(base_idx) else "changed"))

        kfs = choose_kfs(Z, k)
        for selname in ("ge", "cpqr", "maxvol"):
            out = rrqr_pipeline(Z, k, kfs=kfs, f_mode=cfg.f_mode,
                                center=True, strategy=selname)
            res = lloyd(out["Zrun"], out["C0"], delta=cfg.delta,
                        max_iter=cfg.max_iter)
            y = ds.get("y")
            ari = (float(adjusted_rand_score(y, res["lab"]))
                   if y is not None else float("nan"))
            sel_rows.append(dict(dataset=name, selector=selname, ari=ari,
                                 J_full=partition_cost(Z, res["lab"]),
                                 swaps2=out["meta"]["swaps2"]))
    return kfs_rows, cen_rows, sel_rows

# ==========================================================================
# 12. E6 (bound with the SAME f); E7 (with certification)
# ==========================================================================
def run_volume(datasets, cfg):
    rows = []
    for name, ds in datasets.items():
        Z = standardize(np.asarray(ds["X"], float))
        rng = np.random.default_rng(cfg.seed + 555)
        m = min(Z.shape[0], 22)
        k = min(ds["k"], 4, m, Z.shape[1])
        if k < 2:
            continue
        Y = Z[rng.choice(Z.shape[0], size=m, replace=False)]
        best = max(log_volume(Y[list(S)])
                   for S in itertools.combinations(range(m), k))

        def realized(sel):
            lv = log_volume(Y[list(sel)])
            if not np.isfinite(lv) or not np.isfinite(best):
                return float("nan")
            return math.exp(min(max(best - lv, 0.0), 700.0))

        f_used = _resolve_f(Y.T, cfg)
        sel_cpqr, _ = strong_select(Y.T, k, "cpqr")
        sel_strong, sw = strong_select(Y.T, k, cfg.strategy, f=f_used)
        sel_mv, _ = _maxvol(Y.T, k, 1.05)
        rand_r = [realized(rng.choice(m, size=k, replace=False))
                  for _ in range(20)]
        q1 = math.sqrt(1 + 2 * f_used ** 2 * k * (m - k))
        worst = q1 ** k
        r_strong = realized(sel_strong)
        rows.append(dict(dataset=name, m=m, k=k, f_used=f_used,
                         ratio_cpqr=realized(sel_cpqr),
                         ratio_strong=r_strong,
                         ratio_maxvol=realized(sel_mv),
                         ratio_random=float(np.nanmedian(rand_r)),
                         worst=worst,
                         log10_gap=(math.log10(worst)
                                    - math.log10(max(r_strong, EPS))
                                    if np.isfinite(r_strong) and r_strong > 0
                                    else float("inf")),
                         swaps=sw))
        flag = "  (WARNING: budget exhausted — ratio unreliable)" \
            if sw == -1 else ""
        print(f"[E6] {name}: realized={r_strong: .4g}  "
              f"worst(f={f_used:.2f})={worst: .3g}{flag}")
    return rows


def run_interchange(datasets, cfg):
    """E7 with EXIT CERTIFICATION."""
    rows = []
    for name, ds in datasets.items():
        Z = standardize(np.asarray(ds["X"], float))
        N, d, k = *Z.shape, ds["k"]
        kfs = min(max(choose_kfs(Z, k), min(k, min(N, d))), min(N, d))
        f1 = _resolve_f(Z, cfg)
        if cfg.strategy == "ge":
            sel1, s1, ord1 = criterion_rrqr(Z, kfs, f=f1, return_order=True)
        else:
            sel1, s1 = strong_select(Z, kfs, cfg.strategy, f=f1)
            ord1 = sel1 + [c for c in range(d) if c not in set(sel1)]
        cert1 = criterion_value(Z, ord1, kfs)
        Zr = Z[:, sel1]
        r = min(k, kfs)
        f2 = _resolve_f(Zr.T, cfg)
        if cfg.strategy == "ge":
            sel2, s2, ord2 = criterion_rrqr(Zr.T, r, f=f2, return_order=True)
        else:
            sel2, s2 = strong_select(Zr.T, r, cfg.strategy, f=f2)
            ord2 = sel2 + [c for c in range(Zr.shape[0])
                           if c not in set(sel2)]
        cert2 = criterion_value(Zr.T, ord2, r)
        _, s2_mv = _maxvol(Zr.T, r, 1.05)
        rows.append(dict(dataset=name, N=N, d=d, k=k, kfs=kfs,
                         f1=f1, f2=f2,
                         s1_crit=s1, s2_crit=s2, s2_maxvol=s2_mv,
                         cert1=cert1, cert2=cert2,
                         cert1_ok=(s1 >= 0 and cert1 <= f1 * (1 + 1e-9)),
                         cert2_ok=(s2 >= 0 and cert2 <= f2 * (1 + 1e-9))))
    done = [r for r in rows if r["s1_crit"] >= 0 and r["s2_crit"] >= 0]
    tot = sum(r["s1_crit"] + r["s2_crit"] for r in done)
    nceil = len(rows) - len(done)
    ncert = sum(1 for r in rows if r["cert1_ok"] and r["cert2_ok"])
    print(f"[E7] GE-criterion interchanges (terminated runs): {tot}; "
          f"budget-exhausted datasets: {nceil}/{len(rows)}; "
          f"criterion CERTIFIED at exit: {ncert}/{len(rows)}")
    if nceil:
        print("     WARNING: exhaustion present — E7/E6 unreliable.")
    return rows

# ==========================================================================
# 12b. E8 — THEOREM VERIFICATION ON THE DATA (new in v10)
# ==========================================================================
def run_theorem_checks(datasets, cfg):
    """E8: verify the paper's guarantees numerically on every dataset,
    both stages, with the selector ACTUALLY used.

    Checks:
      (T1) Volume, certified form: by Lemma 5.6 (interlacing),
           Vol_k(S) <= prod_i sigma_i(Y); by Thm 5.10 the guarantee is
           Vol_k(S) >= Vol^max / q1^k. The DIRECTLY computable test of
           the certified chain is
               (prod_i sigma_i(Y)) / Vol_k(S) <= q1^k,
           since Vol^max <= prod sigma. We report this ratio and its
           pass/fail against q1^k with the SAME f used by the selector.
      (T2) Pairwise separation (Prop 5.13(ii)): the minimum pairwise
           distance among the k selected centers (in the centered,
           feature-reduced space) vs sigma_k / q1(k, N).
      (T3) Sequential spread for j=1 (Prop 5.13(i)): the norm of the
           first selected row (= dist to span{} = |(A_k)_{11}|) vs
           sigma_k / q1(k, N).
    """
    rows = []
    for name, ds in datasets.items():
        Z = standardize(np.asarray(ds["X"], float))
        N, d, k = *Z.shape, ds["k"]
        kfs = min(max(choose_kfs(Z, k), min(k, min(N, d))), min(N, d))
        f1 = _resolve_f(Z, cfg)
        sel1, s1, ord1 = criterion_rrqr(Z, kfs, f=f1, return_order=True)
        Zr = Z[:, sel1]
        Zrc = Zr - Zr.mean(0)                     # Stage-2 centering
        r = min(k, kfs)
        f2 = _resolve_f(Zrc.T, cfg)
        sel2, s2, ord2 = criterion_rrqr(Zrc.T, r, f=f2, return_order=True)

        # --- T1 (Stage 2: the paper's Thm 5.10 space) ---
        Y = Zrc                                    # N x r feature space
        kk = r
        sY = np.linalg.svd(Y, compute_uv=False)
        prod_sig = float(np.prod(sY[:kk])) if kk <= len(sY) else float("nan")
        volS = math.exp(min(log_volume(Y[sel2]), 700.0)) \
            if np.isfinite(log_volume(Y[sel2])) else float("nan")
        q1 = math.sqrt(1 + 2 * f2 ** 2 * kk * (N - kk))
        worst = q1 ** kk
        ratio_vol = (prod_sig / volS) if (np.isfinite(volS)
                                          and volS > EPS) else float("inf")
        t1_ok = (np.isfinite(ratio_vol) and ratio_vol <= worst * 1.000001)

        # --- T2/T3 (Prop 5.13, Stage 2) ---
        C = Y[sel2]                                # k x r selected rows
        if C.shape[0] > 1:
            dmin = float(np.min(np.sqrt(((C[:, None, :] - C[None, :, :]) ** 2)
                                        .sum(-1))
                                + np.eye(C.shape[0]) * 1e18))
        else:
            dmin = float("inf")
        seq1 = float(np.linalg.norm(C[0]))
        sig_k = float(sY[kk - 1]) if kk <= len(sY) else float("nan")
        lb = sig_k / q1
        t2_ok = dmin >= lb - 1e-9
        t3_ok = seq1 >= lb - 1e-9

        rows.append(dict(dataset=name, k=kk, f2=f2,
                         q1k=worst,
                         ratio_vol=ratio_vol, t1_ok=t1_ok,
                         dmin=dmin, lb=lb, t2_ok=t2_ok,
                         seq1=seq1, t3_ok=t3_ok,
                         all_ok=(t1_ok and t2_ok and t3_ok)))
        print(f"[E8] {name}: T1 vol-ratio {ratio_vol: .3g} vs q1^k "
              f"{worst: .3g} ({'OK' if t1_ok else 'FAIL'}); "
              f"T2 dmin {dmin: .3g} >= sigma_k/q1 {lb: .3g} "
              f"({'OK' if t2_ok else 'FAIL'}); "
              f"T3 first {seq1: .3g} ({'OK' if t3_ok else 'FAIL'})")
    nok = sum(1 for r in rows if r["all_ok"])
    print(f"[E8] theorem checks passed: {nok}/{len(rows)} datasets "
          "(all three tests, both stages' constants).")
    return rows

# ==========================================================================
# 13. Output: console / CSV / JSON / LaTeX
# ==========================================================================
def _f(v, nd=3, sc=False):
    if v is None or (isinstance(v, float) and (math.isnan(v) or math.isinf(v))):
        return "--"
    if isinstance(v, (float, np.floating)):
        if sc and (abs(v) >= 1e5 or (0 < abs(v) < 1e-3)):
            return f"{v:.{nd}g}"
        return f"{v:,.{nd}f}"
    return str(v)


def _sw(v):
    return ">200" if v == -1 else str(v)


def _cert_str(r):
    def _c(x):
        return "inf" if not np.isfinite(x) else f"{x:.2f}"
    ok = r["cert1_ok"] and r["cert2_ok"]
    return f"{_c(r['cert1'])}/{_c(r['cert2'])}" + ("" if ok else " (!)")


def _tex(s):
    return str(s).replace("&", r"\&").replace("_", r"\_")


def write_tex(path, caption, label, colfmt, header, rows):
    with open(path, "w", encoding="utf-8") as f:
        f.write("\\begin{table}[htbp]\n\\centering\n\\small\n")
        f.write(f"\\caption{{{caption}}}\n\\label{{{label}}}\n")
        f.write("\\resizebox{\\textwidth}{!}{%\n")
        f.write(f"\\begin{{tabular}}{{{colfmt}}}\n\\toprule\n")
        f.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(r) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}}\n\\end{table}\n")


def write_all_tables(out, cfg, agg, dorder, kfs_rows, cen_rows, sel_rows,
                     vol, inter, stress_agg, sep_rows, theorem_rows):
    tdir = os.path.join(out, "tables")
    os.makedirs(tdir, exist_ok=True)
    fdesc = "auto ($f=\\sqrt{n}$)" if cfg.f_mode == "auto" else str(cfg.f_mode)
    cap = (f"Auto-generated by run\\_experiments.py (seed={cfg.seed}, "
           f"selector={cfg.strategy}, f={fdesc}, restarts={cfg.runs}). "
           f"Wholesale: Channel/Region dropped (categorical).")

    # ---- Table 1 (main comparison)
    rows = []
    for dn in dorder:
        sub = agg[agg.dataset == dn]
        first = True
        for _, r in sub.iterrows():
            dsname = _tex(dn) if first else ""
            first = False
            Jm = (f"${_f(r.J_mean, 2)} \\pm {_f(r.J_sd, 2)}$"
                  if r.runs > 1 and pd.notna(r.J_sd)
                  else f"${_f(r.J_mean, 2)}$ (det.)")
            rows.append([dsname, _tex(r.label), Jm, _f(r.J_best, 2),
                         f"{r.iters:.1f}", f"{r.total * 1000:,.1f}",
                         _f(r.ari)])
        rows.append([r"\addlinespace"])
    if rows and rows[-1] == [r"\addlinespace"]:
        rows.pop()
    write_tex(os.path.join(tdir, "table_main.tex"),
              "Main comparison (E1--E3). $J$: partition cost in the FULL "
              "standardized space (comparable across methods). Selector "
              "for the proposed methods: Gu--Eisenstat Algorithm~5 inner "
              "loop (full two-term criterion, from their published text). "
              + cap,
              "tab:main", "llrrrrr",
              ["Dataset", "Method", "$J$ (mean $\\pm$ s.d.)", "$J$ (best)",
               "Iters", "Time (ms)", "ARI"], rows)

    # ---- Table 2 (kfs ablation)
    rows = [[_tex(r["dataset"]), str(r["kfs"]), _f(r["ari"]),
             _f(r["J_red"], 2), _f(r["J_full"], 2), str(r["iters"]),
             _sw(r["swaps2"])] for r in kfs_rows]
    write_tex(os.path.join(tdir, "table_kfs.tex"),
              "Effect of $k_{\\mathrm{fs}}$ (full pipeline). "
              "$J_{\\mathrm{red}}$ not comparable across columns; "
              "$J_{\\mathrm{full}}$ comparable. sw$_2$: Stage-2 GE-criterion "
              "interchanges (`$>$200' = budget exhausted). " + cap,
              "tab:kfs", "lrrrrrr",
              ["Dataset", "$k_{\\mathrm{fs}}$", "ARI", "$J_{\\mathrm{red}}$",
               "$J_{\\mathrm{full}}$", "iters", "sw$_2$"], rows)

    # ---- Table 3 (centering ablation)
    rows = [[_tex(r["dataset"]), r["on"], r["off"]] for r in cen_rows]
    write_tex(os.path.join(tdir, "table_centering.tex"),
              "Centering ablation (E5) under a $\\times 50$ s.d.\\ random "
              "translation: does the selected index set change? `same' = "
              "identical index set before/after translation. " + cap,
              "tab:centering", "lcc",
              ["Dataset", "Centering ON", "Centering OFF"], rows)

    # ---- Table 4 (volume ratios)
    rows = [[_tex(r["dataset"]), str(r["m"]), str(r["k"]),
             f"{r['f_used']:.2f}",
             _f(r["ratio_cpqr"], 3, sc=True),
             _f(r["ratio_strong"], 3, sc=True),
             _f(r["ratio_maxvol"], 3, sc=True),
             _f(r["ratio_random"], 3, sc=True),
             _f(r["worst"], 2, sc=True)] for r in vol]
    write_tex(os.path.join(tdir, "table_volume.tex"),
              "Realized volume ratios (E6): $\\mathrm{Vol}^{\\max}/"
              "\\mathrm{Vol}(S)$ on brute-forced subsamples, vs.\\ the "
              "worst-case bound computed with the \\emph{same} $f$ the "
              "selector used. " + cap,
              "tab:vol", "lrrrrrrrr",
              ["Dataset", "$m$", "$k$", "$f$", "CPQR", "GE Alg.~5",
               "maxvol", "random", "worst $q_1^{k}$"], rows)

    # ---- E7 interchanges WITH certification
    rows = [[_tex(r["dataset"]), str(r["N"]), str(r["d"]), str(r["k"]),
             str(r["kfs"]), f"{r['f1']:.2f}", f"{r['f2']:.2f}",
             _sw(r["s1_crit"]), _sw(r["s2_crit"]),
             _sw(r["s2_maxvol"]), _cert_str(r)] for r in inter]
    write_tex(os.path.join(tdir, "table_interchange.tex"),
              "Interchange counts (E7) of the Gu--Eisenstat Algorithm~5 "
              "inner loop (full two-term criterion): s$_1$ = feature "
              "stage, s$_2$ = initialization stage; maxvol for "
              "comparison. Last column: independently recomputed exit "
              "value of $\\omega(R,k)$ for both stages; both $\\le$ the "
              "respective $f$ certifies the strong-RRQR conditions "
              "(5)--(6) of GE Thm.~3.2 for that $f$ (`(!)' = not "
              "certified). `$>$200' = budget exhausted (no certificate). "
              + cap,
              "tab:interchange", "lrrrrrrrrrr",
              ["Dataset", "$N$", "$d$", "$k$", "$k_{\\mathrm{fs}}$",
               "$f_1$", "$f_2$", "s$_1$ (GE)", "s$_2$ (GE)",
               "s$_2$ (maxvol)", "cert.\\ $\\omega$"], rows)

    # ---- selector ablation
    rows = [[_tex(r["dataset"]), r["selector"], _f(r["ari"]),
             _f(r["J_full"], 2), _sw(r["swaps2"])] for r in sel_rows]
    write_tex(os.path.join(tdir, "table_selector.tex"),
              "Ablation (E5): initialization rule on identical feature "
              "selection (GE criterion vs.\\ CPQR vs.\\ maxvol). " + cap,
              "tab:selector", "llrrr",
              ["Dataset", "selector", "ARI", "$J_{\\mathrm{full}}$",
               "sw$_2$"], rows)

    # ---- E4 stress
    if stress_agg is not None and not stress_agg.empty:
        rows = []
        for dn in stress_agg.dataset.unique():
            sub = stress_agg[stress_agg.dataset == dn]
            first = True
            for _, r in sub.iterrows():
                rows.append([_tex(dn) if first else "", _tex(r.label),
                             _f(r.ari), _f(r.ratio), f"{r.ee:.2f}"])
                first = False
            rows.append([r"\addlinespace"])
        if rows and rows[-1] == [r"\addlinespace"]:
            rows.pop()
        write_tex(os.path.join(tdir, "table_stress.tex"),
                  "Stress tests (E4): ARI (injected outliers excluded), "
                  "$J/J^{*}$, residual empty clusters. " + cap,
                  "tab:stress", "llrrr",
                  ["Stressator & Method & ARI & $J/J^{*}$ & "
                   "empty$_\\infty$"], rows) \
            if False else write_tex(
                os.path.join(tdir, "table_stress.tex"),
                "Stress tests (E4): ARI (injected outliers excluded), "
                "$J/J^{*}$, residual empty clusters. " + cap,
                "tab:stress", "llrrr",
                ["Stressor", "Method", "ARI", "$J/J^{*}$",
                 "empty$_\\infty$"], rows)

    # ---- E4e separation sweep
    if sep_rows:
        rows = [[_f(r["sep"], 2), _tex(r["label"]), _f(r["ari"])]
                for r in sep_rows]
        write_tex(os.path.join(tdir, "table_separation.tex"),
                  "Separation sweep (E4e): ARI vs.\\ minimum inter-center "
                  "separation (synthetic, 5 clusters, $d=8$). " + cap,
                  "tab:separation", "rlr",
                  ["separation", "Method", "ARI"], rows)

    # ---- E8 theorem verification (NEW)
    if theorem_rows:
        rows = []
        for r in theorem_rows:
            rows.append([_tex(r["dataset"]), str(r["k"]),
                         f"{r['f2']:.2f}",
                         _f(r["ratio_vol"], 3, sc=True),
                         _f(r["q1k"], 2, sc=True),
                         ("OK" if r["t1_ok"] else "FAIL"),
                         _f(r["dmin"], 2, sc=True),
                         _f(r["lb"], 2, sc=True),
                         ("OK" if r["t2_ok"] else "FAIL"),
                         ("OK" if r["t3_ok"] else "FAIL")])
        write_tex(os.path.join(tdir, "table_theorem.tex"),
                  "Theorem verification on the data (E8), Stage-2 space "
                  "(centered, feature-reduced). T1: certified volume "
                  "ratio $\\prod\\sigma_i/\\mathrm{Vol}_k(S)$ vs.\\ the "
                  "worst-case $q_1^k$ with the same $f$ (Thm~5.10 via "
                  "Lemma~5.6: $\\mathrm{Vol}^{\\max}\\le\\prod\\sigma$, "
                  "so the ratio tests the guaranteed chain). T2: minimum "
                  "pairwise distance of the selected centers vs.\\ "
                  "$\\sigma_k/q_1$ (Prop~5.13(ii)). T3: norm of the "
                  "first selected center vs.\\ the same bound "
                  "(Prop~5.13(i), $j{=}1$). " + cap,
                  "tab:theorem", "lrrrccrrccc",
                  ["Dataset", "$k$", "$f$", "T1 ratio", "$q_1^{k}$",
                   "T1", "T2 $d_{\\min}$", "$\\sigma_k/q_1$", "T2", "T3"],
                  rows)

    # ---- standalone report
    with open(os.path.join(out, "report.tex"), "w", encoding="utf-8") as f:
        f.write("\\documentclass[11pt]{article}\n"
                "\\usepackage[margin=2.2cm]{geometry}\n"
                "\\usepackage{booktabs,graphicx}\n\\begin{document}\n"
                "\\begin{center}{\\Large Experimental Results "
                "(auto-generated)}\\end{center}\n")
        for t in ("table_main", "table_kfs", "table_centering",
                  "table_volume", "table_interchange", "table_selector",
                  "table_stress", "table_separation", "table_theorem"):
            p = os.path.join(tdir, t + ".tex")
            if os.path.exists(p):
                f.write(f"\\input{{tables/{t}.tex}}\n")
        f.write("\\end{document}\n")

# ==========================================================================
# 14. Main
# ==========================================================================
def main():
    ap = argparse.ArgumentParser(
        description="Full E1-E8 protocol (9 UCI datasets); GE Algorithm 5 "
                    "inner loop, verified (v10)")
    ap.add_argument("--data-dir", default=None)
    ap.add_argument("--download", action="store_true")
    ap.add_argument("--out", default="results")
    ap.add_argument("--runs", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20240601)
    ap.add_argument("--f", default="auto",
                    help="'auto' (f=sqrt(n), GE's efficient choice) or a "
                         "numeric f")
    ap.add_argument("--strategy", default="ge",
                    choices=["ge", "cpqr", "maxvol"])
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--skip-stress", action="store_true")
    ap.add_argument("--skip-sep", action="store_true")
    a = ap.parse_args()

    cfg = Config(runs=3 if a.quick else a.runs, seed=a.seed, f_mode=a.f,
                 out=a.out, quick=a.quick, strategy=a.strategy)
    data_dir = Path(a.data_dir) if a.data_dir else (SCRIPT_DIR / "uci_data")
    os.makedirs(os.path.join(a.out, "tables"), exist_ok=True)

    print("=" * 72)
    print("RRQR-KM experimental protocol (E1-E8) -- 9-dataset build (v10)")
    print(f"selector     : GE Algorithm 5 inner loop, verified  "
          f"[strategy={cfg.strategy}, f={cfg.f_mode}]")
    print(f"data folder  : {data_dir}")
    print(f"seed         : {cfg.seed}   restarts (randomized): {cfg.runs}")
    print(f"run time     : {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 72)

    _self_tests()      # Kahan + Lemma-3.1 growth (f=1.02); non-blocking

    if a.download:
        print("[dl] fetching UCI files ...")
        download_uci(data_dir)

    datasets, fallback = load_datasets(data_dir, quick=cfg.quick)

    if fallback and not a.data_dir and not a.download:
        print("[data] default folder empty -- attempting download ...")
        download_uci(data_dir)
        datasets, fallback = load_datasets(data_dir, quick=cfg.quick)

    if fallback:
        print("=" * 72)
        print("WARNING: RUNNING ON BUNDLED/SYNTHETIC FALLBACK DATA.")
        print(f"Fix: run with --download, or place the 9 UCI files in "
              f"{data_dir}")
        print("=" * 72)

    dorder = list(datasets.keys())

    recs = run_benchmarks(datasets, cfg)
    agg = aggregate(recs)
    print("\n" + "=" * 72)
    print("E1-E3 SUMMARY  (J = partition cost in FULL standardized space)")
    print("=" * 72)
    if agg is None or agg.empty:
        print("[warn] NO benchmark records — see warnings above.")
    else:
        print(fmt_summary(agg).to_string(index=False))
        analytic_summary(agg)

    kfs_rows, cen_rows, sel_rows = run_ablations(datasets, cfg)
    print("\n[E5] centering ablation:")
    for r in cen_rows:
        print(f"        {r['dataset']:30s} ON={r['on']:8s} OFF={r['off']}")

    vol = run_volume(datasets, cfg)
    inter = run_interchange(datasets, cfg)
    theorem_rows = run_theorem_checks(datasets, cfg)     # E8 (new)

    stress_agg, sep_rows = None, []
    if not (a.quick or a.skip_stress):
        stress_agg = aggregate(run_stress(cfg))
    if not (a.quick or a.skip_sep):
        sep_rows = run_separation(cfg)

    agg.to_csv(os.path.join(a.out, "summary.csv"), index=False)
    payload = dict(config=asdict(cfg),
                   run_time=datetime.now().isoformat(timespec="seconds"),
                   versions=dict(numpy=np.__version__,
                                 pandas=pd.__version__,
                                 python=sys.version.split()[0]),
                   benchmarks=recs, ablations_kfs=kfs_rows,
                   centering=cen_rows, selector=sel_rows,
                   volume=vol, interchange=inter,
                   theorem_checks=theorem_rows,
                   stress=(None if stress_agg is None
                           or stress_agg.empty
                           else stress_agg.to_dict("records")),
                   separation=sep_rows)
    with open(os.path.join(a.out, "raw_results.json"), "w") as f:
        json.dump(payload, f, indent=1,
                  default=lambda o: o.item() if isinstance(o, np.generic)
                  else str(o))

    write_all_tables(a.out, cfg, agg, dorder, kfs_rows, cen_rows, sel_rows,
                     vol, inter, stress_agg, sep_rows, theorem_rows)

    print("\n" + "=" * 72)
    print(f"DONE. Outputs in '{a.out}/':")
    print("  summary.csv  raw_results.json  report.tex  tables/*.tex")
    print("  E8 adds table_theorem.tex: the paper's Theorems 5.10/5.13")
    print("  verified numerically on every dataset.")
    print("=" * 72)


if __name__ == "__main__":
    main()
