from pathlib import Path
from typing import List
import numpy as np
from . import alphabet


def read_a3m(path: Path) -> List[str]:
    seqs, buf = [], []
    with path.open() as f:
        for line in f:
            l = line.strip()
            if not l:
                continue
            if l.startswith(">"):
                if buf:
                    seqs.append("".join(buf))
                    buf = []
            else:
                buf.append("".join([c for c in l if not c.islower()]))
        if buf:
            seqs.append("".join(buf))
    if not seqs:
        raise ValueError("No sequences.")
    L = len(seqs[0])
    if any(len(s) != L for s in seqs):
        raise ValueError("Inconsistent lengths.")
    return seqs


def filter_msa(seqs, max_id, max_gap_frac, col_gap_frac) -> List[str]:

    L = len(seqs[0])
    arr = (
        np.frombuffer("".join(seqs).encode("ascii"), dtype="S1")
        .reshape(len(seqs), L)
        .astype("U1")
    )
    gaps = (arr == "-").mean(0)
    keep_cols = np.where(gaps <= col_gap_frac)[0]
    if keep_cols.size == 0:
        raise ValueError("All columns filtered.")
    arr = arr[:, keep_cols]
    seq_gap = (arr == "-").mean(1)
    keep_seq = np.where(seq_gap <= max_gap_frac)[0]
    arr = arr[keep_seq]
    QQ = int(alphabet.Q)
    ATOI = alphabet.AA_TO_IDX
    idx_map = np.vectorize(lambda c: ATOI.get(c, QQ - 1))(arr)
    keep, taken = [], np.zeros(idx_map.shape[0], dtype=bool)
    for i in range(idx_map.shape[0]):
        if taken[i]:
            continue
        keep.append(i)
        same = (idx_map == idx_map[i]).mean(1) >= max_id
        taken |= same
    return ["".join(arr[i].tolist()) for i in keep]


def to_onehot(seqs: List[str]) -> np.ndarray:
    N, L = len(seqs), len(seqs[0])
    b = np.frombuffer("".join(seqs).encode("ascii"), dtype=np.uint8).reshape(N, L)
    lut = np.full(256, alphabet.AA_TO_IDX["-"], dtype=np.int32)
    for k, v in alphabet.AA_TO_IDX.items():
        if len(k) != 1:
            continue
        lut[ord(k)] = v
        lut[ord(k.lower())] = v
    idx = lut[b]
    oh = np.zeros((N, L, int(alphabet.Q)), np.float32)
    oh[np.arange(N)[:, None], np.arange(L)[None, :], idx] = 1.0
    return oh
