import os
import subprocess
from pathlib import Path

import numpy as np
import pytest

import evorca as ev


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"
A3M = DATA / "tiny.a3m"


def run(cmd, **kwargs):
    return subprocess.run(cmd, text=True, capture_output=True, **kwargs)


def test_help_cli():
    r = run(["evorca", "help"])
    assert r.returncode == 0
    assert "fit" in r.stdout and "viz" in r.stdout


@pytest.mark.parametrize("apc", [True, False])
def test_fit_viz_cli(tmp_path: Path, apc: bool):
    outdir = tmp_path / "output"
    r = run(
        [
            "evorca",
            "fit",
            str(A3M),
            "--out",
            str(outdir),
            "--epochs",
            "1",
            "--batch",
            "2",
            "--dtype",
            "fp32",
            "--msa-subsample",
        ]
    )
    assert r.returncode == 0, r.stderr
    assert (outdir / "sparse_J.npz").exists()

    # scores are generated automatically after fit
    assert (outdir / "score.npy").exists()

    r3 = run(["evorca", "viz", "--out", str(outdir)])
    assert r3.returncode == 0
    assert (outdir / "contact_map.png").exists()


def test_python_api_fit_and_visualize(tmp_path: Path):
    outdir = tmp_path / "output"
    out = ev.fit(
        A3M,
        outdir,
        seq_type="rna",
        epochs=1,
        batch=2,
        dtype="fp32",
        msa_subsample=True,
        apc=False,
    )
    assert "meta" in out and "h" in out and "Jsym" in out and "score" in out
    assert isinstance(out["h"], np.ndarray)
    assert isinstance(out["Jsym"], np.ndarray)
    assert isinstance(out["score"], np.ndarray)
    assert (outdir / "sparse_J.npz").exists()
    ev.visualize_array(out["score"], topk=10, seq_type="rna")
