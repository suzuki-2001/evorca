from pathlib import Path
from typing import Dict, Any, Union
import numpy as np
import jax.numpy as jnp

from .alphabet import setup_alphabet
from . import alphabet
from .io_utils import read_a3m, filter_msa, to_onehot
from .model import henikoff_w_ignore_gap, train_core
from .post import save_score as _save_score
from .viz import visualize as _visualize, visualize_array as _visualize_array

__all__ = ["fit", "visualize", "visualize_array", "setup_alphabet"]


def fit(
    a3m: Union[str, Path],
    out: Union[str, Path, None] = Path("output"),
    *,
    seq_type: str = "protein",
    epochs: int = 30,
    batch: int = 256,
    lr: float = 0.02,
    l2_h: float = 0.01,
    l2_J: float = 0.2,
    dtype: str = "fp32",
    seed: int = 0,
    nested: bool = False,
    msa_subsample: bool = False,
    max_id: float = 0.9,
    max_gap_frac: float = 0.9,
    col_gap_frac: float = 1.0,
    apc: bool = True,
) -> Dict[str, Any]:
    setup_alphabet(seq_type)
    out_path = None if out is None else Path(out)
    if out_path is not None:
        out_path.mkdir(parents=True, exist_ok=True)
    seqs = read_a3m(Path(a3m))
    if msa_subsample:
        seqs = filter_msa(seqs, max_id, max_gap_frac, col_gap_frac)
    X_cpu = to_onehot(seqs)
    w_cpu = np.array(henikoff_w_ignore_gap(jnp.asarray(X_cpu)))
    jdtype = {"fp16": jnp.float16, "bf16": jnp.bfloat16}.get(dtype, jnp.float32)
    meta = dict(
        dtype=dtype,
        seed=seed,
        seq_type=seq_type,
        alphabet=alphabet.AA,
        Q=alphabet.Q,
        nested=nested,
        lr=lr,
        l2_h=l2_h,
        l2_J=l2_J,
    )
    train_core(
        X_cpu,
        w_cpu,
        epochs,
        batch,
        lr,
        l2_h,
        l2_J,
        jdtype,
        seed,
        out_path if out_path is not None else Path("."),
        meta,
    )
    Jsym, score = _save_score(
        out_path if out_path is not None else Path("."),
        apc_correct=apc,
        save=(out_path is not None),
    )
    h = np.load((out_path if out_path is not None else Path(".")) / "h.npy")
    return {
        "meta": meta,
        "h": h,
        "Jsym": Jsym,
        "score": score,
    }


def visualize(
    out: Union[str, Path] = Path("output"), topk: int = 50, *, seq_type: str = "protein"
) -> None:
    out = Path(out)
    _visualize(out, topk, seq_type=seq_type)


def visualize_array(
    score: np.ndarray, *, topk: int = 50, seq_type: str = "protein"
) -> None:
    _visualize_array(score, topk=topk, seq_type=seq_type)
