import sys
from pathlib import Path
import numpy as np

try:
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes
except Exception:
    plt = None


def _render_score_plot(score: np.ndarray, topk: int, seq_type: str = "protein"):
    """Render the coupling matrix and top-k contacts plot and return (fig, (ax1, ax2))."""
    L = score.shape[0]
    iu = np.triu_indices(L, k=5)
    vals = score[iu]
    idx = np.argsort(vals)[::-1][:topk]
    i, j, s = iu[0][idx], iu[1][idx], vals[idx]
    x_all = np.concatenate([i, j])
    y_all = np.concatenate([j, i])
    c_all = np.concatenate([s, s])
    size = 40 + 100 * (np.abs(c_all) - np.abs(c_all).min()) / (
        np.abs(c_all).max() - np.abs(c_all).min() + 1e-8
    )
    vmax = float(np.abs(score).max())
    norm = Normalize(vmin=-vmax, vmax=vmax)
    cmap = "cividis"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 6))
    fig.patch.set_facecolor("#f3f3f3")
    ax1.imshow(
        score,
        origin="lower",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
        extent=[-0.5, L - 0.5, -0.5, L - 0.5],
    )
    ax1.plot([0, L - 1], [0, L - 1], color="black", lw=0.8, ls="--", alpha=0.5)
    ax1.set_aspect("equal")
    ax1.set_title("Predicted Coupling Matrix")
    label = "Nucleotide index" if seq_type == "rna" else "Residue index"
    ax1.set_xlabel(label)
    ax1.set_ylabel(label)
    ax2.scatter(
        x_all,
        y_all,
        c=c_all,
        cmap=cmap,
        norm=norm,
        s=size,
        alpha=0.9,
        edgecolors="k",
        linewidths=0.3,
    )
    ax2.plot([0, L - 1], [0, L - 1], color="gray", lw=0.8, ls="--", alpha=0.5)
    ax2.set_aspect("equal")
    ax2.set_title("Top Predicted Contacts")
    ax2.set_xlabel(label)
    ax2.set_ylabel(label)

    for ax in (ax1, ax2):
        step = max(1, L // 6)
        ax.set_xticks(np.arange(0, L, step))
        ax.set_yticks(np.arange(0, L, step))
        ax.grid(True, which="major", linestyle=":", linewidth=0.4, alpha=0.3)
    cax = inset_axes(
        ax2,
        width=0.18,
        height=2.8,
        loc="center left",
        bbox_to_anchor=(1.1, 0.5),
        bbox_transform=ax2.transAxes,
        borderpad=0,
    )
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=cmap), cax=cax)
    cbar.ax.set_title("Coupling\nScore", fontsize=10, pad=6)
    cbar.ax.tick_params(labelsize=9)
    plt.subplots_adjust(wspace=0.18)
    return fig, (ax1, ax2)


def visualize(out: Path, topk: int, seq_type: str = "protein") -> None:
    if plt is None:
        sys.exit("matplotlib is required for --viz")
    score = np.load(out / "score.npy")
    fig, _ = _render_score_plot(score, topk=topk, seq_type=seq_type)
    plt.savefig(out / "contact_map.png", dpi=300)


def visualize_array(score: np.ndarray, topk: int, seq_type: str = "protein") -> None:
    if plt is None:
        sys.exit("matplotlib is required for visualization")
    _render_score_plot(score, topk=topk, seq_type=seq_type)
    plt.show()
