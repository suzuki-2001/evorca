import os, json, logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import jax, jax.numpy as jnp
from jax import random, jit, value_and_grad
import optax

from . import alphabet

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
os.environ.setdefault("XLA_PYTHON_CLIENT_ALLOCATOR", "platform")


@jit
def henikoff_w_ignore_gap(X: jnp.ndarray) -> jnp.ndarray:
    Qv = int(alphabet.Q)
    Xng = X[:, :, : Qv - 1]
    col = Xng.sum(0)
    pres = (col > 0).astype(Xng.dtype)
    r = pres.sum(1)
    inv = 1.0 / (jnp.maximum(r, 1e-8)[:, None] * jnp.maximum(col, 1.0))
    w = (Xng * inv).sum((1, 2))
    return w * X.shape[0] / jnp.maximum(w.sum(), 1.0)


def init_params(L: int, dtype) -> Dict[str, jnp.ndarray]:
    Qv = int(alphabet.Q)
    return {"h": jnp.zeros((L, Qv), dtype), "J": jnp.zeros((L, L, Qv, Qv), dtype)}


@jit
def project_J(J: jnp.ndarray) -> jnp.ndarray:
    Jsym = 0.5 * (J + jnp.transpose(J, (1, 0, 3, 2)))
    L = Jsym.shape[0]
    Qloc = Jsym.shape[2]
    zero = jnp.zeros((L, Qloc, Qloc), J.dtype)
    idx = jnp.arange(L)
    Jsym = Jsym.at[idx, idx].set(zero)
    return Jsym


@jit
def site_ll_full(params, X, i, Y):
    h = params["h"][i]
    J = params["J"][i]
    z = jnp.einsum("njq,jqa->na", X, J) - jnp.einsum("nq,qa->na", X[:, i, :], J[i])
    z = z + h[None, :]
    logp = jax.nn.log_softmax(z, axis=-1)
    return jnp.take_along_axis(logp, Y[:, None], axis=1).squeeze()


def npll_full(params, X, Y, w, l2h, l2J):
    def li(i):
        return (w * site_ll_full(params, X, i, Y[:, i])).sum()

    L_all = params["h"].shape[0]
    cols = jnp.arange(L_all, dtype=jnp.int32)
    ll = jax.vmap(li)(cols).sum()
    reg_J = jnp.sum(params["J"] ** 2)
    reg = l2h * jnp.sum(params["h"] ** 2) + l2J * reg_J
    return -(ll / jnp.sum(w)) + reg


def train_core(
    X_cpu: np.ndarray,
    w_cpu: np.ndarray,
    epochs: int,
    batch: int,
    lr: float,
    l2h: float,
    l2J: float,
    dtype,
    seed: int,
    outdir: Path,
    meta: Dict[str, Any],
) -> Tuple[dict, float]:
    N, L, _ = X_cpu.shape
    Y_cpu = X_cpu.argmax(2).astype(np.int32)
    params = init_params(L, dtype)
    key = random.PRNGKey(seed)
    X_all = jnp.asarray(X_cpu, dtype=dtype)
    Y_all = jnp.asarray(Y_cpu, dtype=jnp.int32)
    w_all = jnp.asarray(w_cpu, dtype=dtype)

    loss_fn = npll_full
    grad_fn_local = value_and_grad(loss_fn)

    optimizer = optax.adamw(learning_rate=lr)
    opt_state = optimizer.init(params)

    @jit
    def step(params, opt_state, xb, yb, wb):
        loss, grads = grad_fn_local(params, xb, yb, wb, l2h, l2J)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = {**params, "J": project_J(params["J"])}
        return params, opt_state, loss

    last_loss = None
    for ep in range(epochs):
        key, sub_s = random.split(key, 2)
        perm = np.asarray(random.permutation(sub_s, N))
        for s in range(0, N, batch):
            sel = perm[s : s + batch]
            xb = X_all[sel]
            yb = Y_all[sel]
            wb = w_all[sel]
            params, opt_state, loss = step(params, opt_state, xb, yb, wb)
            last_loss = loss
        if ep % max(1, epochs // 5) == 0:
            logging.info(f"epoch {ep}/{epochs} loss={float(loss):.4f}")

    np.save(outdir / "h.npy", np.array(params["h"], np.float32))
    tri = np.triu_indices(L, k=1)
    np.savez_compressed(
        outdir / "sparse_J.npz",
        idx_i=tri[0].astype(np.int32),
        idx_j=tri[1].astype(np.int32),
        block=np.array(params["J"][tri], np.float32),
    )
    meta["L"] = L
    meta["Q"] = int(alphabet.Q)
    meta["epochs"] = epochs
    with (outdir / "meta.json").open("w") as f:
        json.dump(meta, f)
    return params, (float(last_loss) if last_loss is not None else None)
