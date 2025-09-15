from pathlib import Path
import logging
import numpy as np


def reconstruct_sparse(path: Path) -> np.ndarray:
    z = np.load(path, allow_pickle=False)
    L = int(np.max(z["idx_j"]) + 1)
    blk = z["block"]  # (K, Q, Q)
    if blk.ndim < 2:
        raise ValueError("Invalid 'block' shape in sparse_J.npz")
    Q_saved = blk.shape[-1]
    J = np.zeros((L, L, Q_saved, Q_saved), np.float32)
    ii, jj = z["idx_i"], z["idx_j"]
    J[ii, jj] = blk
    J[jj, ii] = np.transpose(blk, (0, 2, 1))
    return J


def apc(mat: np.ndarray) -> np.ndarray:
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        raise ValueError("apc expects a square 2D matrix")
    m = np.asarray(mat, dtype=np.float32)
    row = m.mean(axis=1, keepdims=True)
    col = m.mean(axis=0, keepdims=True)
    glob = float(m.mean())
    eps = 1e-8
    corr = (row @ col) / (glob + eps)
    return m - corr


def score_from_J(J: np.ndarray, apc_correct: bool = True):
    Jsym = 0.5 * (J + np.transpose(J, (1, 0, 3, 2)))
    L = Jsym.shape[0]
    for i in range(L):
        Jsym[i, i] = 0.0
    for i in range(L):
        for j in range(L):
            if i == j:
                continue
            B = Jsym[i, j]
            row_m = B.mean(axis=1, keepdims=True)
            col_m = B.mean(axis=0, keepdims=True)
            glob_m = B.mean()
            Jsym[i, j] = B - row_m - col_m + glob_m
    Q_local = J.shape[2]
    f = np.sqrt(np.sum(Jsym[:, :, : Q_local - 1, : Q_local - 1] ** 2, axis=(2, 3)))
    score = apc(f) if apc_correct else f
    return Jsym, score


def save_score(out: Path, apc_correct: bool = True, save: bool = True):
    J = reconstruct_sparse(out / "sparse_J.npz")
    Jsym, score = score_from_J(J, apc_correct=apc_correct)
    if save:
        np.save(out / "Jsym.npy", Jsym.astype(np.float32))
        np.save(out / "score.npy", score.astype(np.float32))
        logging.info("Saved: Jsym.npy, score.npy")
    return Jsym, score
