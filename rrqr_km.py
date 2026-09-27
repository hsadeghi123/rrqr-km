"""
RRQR-KM core (Gu–Eisenstat Algorithm 5 + two-stage pipeline).

Companion to:
  H. Sadeghi, Volume bounds for strong rank-revealing QR of arbitrary
  shape, and a deterministic K-means initialization.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import qr


def select_cols(M, k, f):
    """Column selection realizing Gu–Eisenstat Algorithm 5."""
    M = np.asarray(M, float)
    p, q = M.shape
    k = int(k)
    if k <= 0:
        return []
    k = min(k, p, q)

    _, _, piv = qr(M, mode="economic", pivoting=True)
    order = [int(t) for t in piv]

    if k >= q:
        return order[:k]

    for _ in range(200):
        _, R = qr(M[:, order], mode="economic")
        A = R[:k, :k]
        B = R[:k, k:]
        C = R[k:, k:]
        W = np.linalg.solve(A, B)
        Ainv = np.linalg.inv(A)
        omega = np.sqrt((Ainv**2).sum(axis=1))
        if C.size == 0:
            gamma = np.zeros(B.shape[1])
        else:
            gamma = np.sqrt((C**2).sum(axis=0))
        score = np.maximum(np.abs(W), gamma[None, :] * omega[:, None])
        i, j = np.unravel_index(int(np.argmax(score)), score.shape)
        if score[i, j] <= f:
            break
        order[i], order[k + j] = order[k + j], order[i]
    return [int(t) for t in order[:k]]


def rrqr_km(X, k, kfs=None, f=None, delta=1e-8, maxit=500):
    """Standardize → RRQR-FS → center → RRQR-Init → Lloyd."""
    X = np.asarray(X, float)
    N, d = X.shape
    k = int(k)
    kfs = int(min(kfs if kfs is not None else d, N, d))
    f = f if f is not None else np.sqrt(max(N, d))

    Z = (X - X.mean(0)) / (X.std(0) + 1e-12)
    feats = select_cols(Z, kfs, f)
    Zc = Z[:, feats]
    Zc = Zc - Zc.mean(0)
    r = min(k, kfs)
    idx = list(select_cols(Zc.T, r, f))
    while len(idx) < k:
        S = Zc[idx]
        d2 = ((Zc[:, None, :] - S[None, :, :]) ** 2).sum(-1).min(1)
        d2[idx] = -np.inf
        idx.append(int(np.argmax(d2)))
    C = Zc[idx].copy()
    Jold = np.inf
    lab = np.zeros(N, dtype=int)
    J = np.inf
    t = 0
    for t in range(maxit):
        D2 = ((Zc[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        lab = D2.argmin(1)
        J = D2[np.arange(N), lab].sum()
        if abs(Jold - J) <= delta * max(J, 1e-12):
            break
        Jold = J
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = Zc[m].mean(0)
    return dict(centers=C, labels=lab, J=J, iters=t, feats=feats, init=idx)


if __name__ == "__main__":
    rng = np.random.default_rng(20240601)
    X = rng.normal(size=(80, 5))
    out = rrqr_km(X, k=3)
    print("feats:", out["feats"])
    print("init: ", out["init"])
    print("J:    ", float(out["J"]))
    print("iters:", int(out["iters"]))
