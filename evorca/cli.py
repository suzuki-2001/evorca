#!/usr/bin/env python3
import argparse, logging
from pathlib import Path
import numpy as np
import jax.numpy as jnp

from .alphabet import setup_alphabet
from . import alphabet
from .io_utils import read_a3m, filter_msa, to_onehot
from .model import henikoff_w_ignore_gap, train_core
from .post import save_score
from .viz import visualize


logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
)


def _add_fit_parser(sub):
    sp = sub.add_parser(
        "fit",
        help="Fit Potts model from A3M and save outputs",
        description="Fit a Potts (plmDCA) model using an input A3M MSA",
    )
    sp.add_argument("a3m", type=Path, help="Input A3M file path")
    sp.add_argument("--out", type=Path, default=Path("output"))
    sp.add_argument("--seq-type", choices=["protein", "rna"], default="protein")
    sp.add_argument("--epochs", type=int, default=10)
    sp.add_argument("--batch", type=int, default=256)
    sp.add_argument("--lr", type=float, default=0.02)
    sp.add_argument("--l2-h", dest="l2_h", type=float, default=0.01)
    sp.add_argument("--l2-J", dest="l2_J", type=float, default=0.2)
    sp.add_argument("--dtype", choices=["fp32", "fp16", "bf16"], default="fp32")
    sp.add_argument("--seed", type=int, default=0)
    sp.add_argument(
        "--nested", "--nested-coevolution", dest="nested", action="store_true"
    )
    sp.add_argument(
        "--msa-subsample",
        dest="msa_subsample",
        action="store_true",
        help="Subsample MSA by identity and gaps",
    )
    sp.add_argument("--max-id", type=float, default=0.9)
    sp.add_argument("--max-gap-frac", type=float, default=0.9)
    sp.add_argument("--col-gap-frac", type=float, default=1.0)
    sp.add_argument(
        "--no-apc",
        dest="apc",
        action="store_false",
        help="Disable APC correction for score",
    )
    sp.add_argument(
        "--viz", action="store_true", help="Render contact map after training"
    )
    sp.add_argument("--topk", type=int, default=50, help="Top-k contacts for viz")
    return sp

    # score subcommand removed; scoring happens automatically after fit


def _add_viz_parser(sub):
    sp = sub.add_parser(
        "viz",
        help="Visualize score.npy as contact map",
        description="Plot contact map and top-k contacts",
    )
    sp.add_argument("--out", type=Path, default=Path("output"))
    sp.add_argument("--topk", type=int, default=50)
    sp.add_argument("--seq-type", choices=["protein", "rna"], default="protein")
    return sp


def main():
    p = argparse.ArgumentParser(
        prog="evorca",
        description="evorca: plmDCA coevolution tool (JAX+optax)",
    )
    sub = p.add_subparsers(dest="command")
    _add_fit_parser(sub)
    _add_viz_parser(sub)
    sub.add_parser("help", help="Show this help message").set_defaults(help_cmd=True)
    args = p.parse_args()

    # Help subcommand
    if getattr(args, "help_cmd", False):
        p.print_help()
        return

    # Subcommand: fit
    if args.command == "fit":
        setup_alphabet(args.seq_type)
        args.out.mkdir(parents=True, exist_ok=True)
        seqs = read_a3m(args.a3m)
        if args.msa_subsample:
            seqs = filter_msa(seqs, args.max_id, args.max_gap_frac, args.col_gap_frac)
        X_cpu = to_onehot(seqs)
        w_cpu = np.array(henikoff_w_ignore_gap(jnp.asarray(X_cpu)))
        jdtype = {"fp16": jnp.float16, "bf16": jnp.bfloat16}.get(
            args.dtype, jnp.float32
        )
        meta = dict(
            dtype=args.dtype,
            seed=args.seed,
            seq_type=args.seq_type,
            alphabet=alphabet.AA,
            Q=alphabet.Q,
            nested=args.nested,
            lr=args.lr,
            l2_h=args.l2_h,
            l2_J=args.l2_J,
        )
        train_core(
            X_cpu,
            w_cpu,
            args.epochs,
            args.batch,
            args.lr,
            args.l2_h,
            args.l2_J,
            jdtype,
            args.seed,
            args.out,
            meta,
        )
        save_score(args.out, apc_correct=getattr(args, "apc", True))
        if args.viz:
            visualize(args.out, args.topk, seq_type=args.seq_type)
        return

    if args.command == "viz":
        visualize(args.out, args.topk, seq_type=args.seq_type)
        return

    # If no subcommand matched, show help
    p.print_help()


if __name__ == "__main__":
    main()
