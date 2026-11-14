"""Generate the Gaussian-TE explanation figures (docs/figures/*.png).

Figures (all article-ready, 150 dpi):

  1. fig_te_geometry.png  - the partial-correlation geometry: why conditioning
                            on the target history z^B_{t-1} removes the
                            self-correlation part of the raw correlation, and
                            the residual correlation IS the TE (a monotone
                            function of it).  Synthetic VAR: X -> Y (b=0.5).
  2. fig_te_mapping.png   - TE = -1/2 log2(1 - rho^2) as a function of the
                            partial correlation rho.
  3. fig_te_real_pair.png - the SAME geometry on a real pair of GVRNN latent
                            chains (artifacts/m1p1/latents.npz, node 0 -> node 1
                            if present).
  4. fig_te_pipeline.png  - the data pipeline: 32-dim chain -> PCA -> standardise
                            -> lagged triples -> Gaussian closed form.

Run from anywhere:

    python D:/MarkovSoccerTE/scripts/plot_te_gaussian.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from mte import conditional_gaussian_mte, pca_fit, pca_transform, standardize
from mte.synthetic import simulate_chain

FIGS = ROOT / "docs" / "figures"


def _resid(a, b):
    """Residual of a after OLS on b (both 1-d)."""
    beta = np.dot(b, a) / np.dot(b, b)
    return a - beta * b


def partial_corr(x, y, z):
    """Partial correlation of x and y given z (all 1-d)."""
    return float(np.corrcoef(_resid(x, z), _resid(y, z))[0, 1])


def te_from_rho(rho):
    """TE (bits) from a partial correlation, Gaussian 1-dim closed form."""
    rho = np.clip(rho, -1 + 1e-9, 1 - 1e-9)
    return -0.5 * np.log2(1.0 - rho ** 2)


def _scatter(ax, x, y, color, xlab, ylab, label):
    ax.scatter(x, y, s=4, alpha=0.35, color=color, linewidths=0)
    ax.set_xlabel(xlab, fontsize=9)
    ax.set_ylabel(ylab, fontsize=9)
    ax.set_title(label, fontsize=9)
    ax.tick_params(labelsize=7)


def make_geometry(x, y, z, out_png, suptitle):
    """4-panel geometry figure for one (x, y, z) triple."""
    r_xy = float(np.corrcoef(x, y)[0, 1])
    r_xz = float(np.corrcoef(x, z)[0, 1])
    r_yz = float(np.corrcoef(y, z)[0, 1])
    rho = partial_corr(x, y, z)
    te = te_from_rho(rho)

    fig, axes = plt.subplots(2, 2, figsize=(9, 7))
    _scatter(axes[0, 0], x, y, "C0", r"$j_n$ (source past)",
             r"$i_{n+1}$ (target future)",
             rf"(a) raw: $\rho(j_n,i_{{n+1}})$ = {r_xy:+.3f}")
    _scatter(axes[0, 1], x, z, "C1", r"$j_n$", r"$i_n$",
             rf"(b) $\rho(j_n,i_n)$ = {r_xz:+.3f}")
    _scatter(axes[1, 0], z, y, "C2", r"$i_n$", r"$i_{n+1}$",
             rf"(c) $\rho(i_n,i_{{n+1}})$ = {r_yz:+.3f}")
    _scatter(axes[1, 1], _resid(x, z), _resid(y, z), "C3",
             r"$j_n - \beta\,i_n$", r"$i_{n+1} - \gamma\,i_n$",
             rf"(d) partial $\rho_{{XY\cdot Z}}$ = {rho:+.3f}"
             rf"  $\Rightarrow$ TE = {te:.3f} bits")
    fig.suptitle(suptitle, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return r_xy, r_xz, r_yz, rho, te


def fig_geometry_synthetic():
    ch = simulate_chain(2000, b_xy=0.5, seed=0)
    X, Y = ch["X"], ch["Y"]
    x = standardize(X[:-1])[0]
    y = standardize(Y[1:])[0]
    z = standardize(Y[:-1])[0]
    make_geometry(x, y, z, FIGS / "fig_te_geometry.png",
                  "Gaussian TE as a partial correlation (synthetic VAR, X -> Y, b=0.5)")


def fig_mapping():
    rho = np.linspace(-0.99, 0.99, 800)
    te = te_from_rho(rho)
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(rho, te, color="C0", lw=2)
    for r in (0.0, 0.3, 0.5, 0.8, 0.9, 0.95):
        ax.plot([r], [te_from_rho(r)], "o", color="C3")
        ax.annotate(rf"$\rho$={r}\nTE={te_from_rho(r):.3f}",
                    (r, te_from_rho(r)), textcoords="offset points",
                    xytext=(8, 6), fontsize=8)
    ax.set_xlabel(r"partial correlation $\rho_{XY\cdot Z}$")
    ax.set_ylabel(r"TE = $-\frac{1}{2}\log_2(1-\rho^2)$  (bits)")
    ax.set_title("Gaussian TE is a monotone function of the partial correlation")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_te_mapping.png", dpi=150)
    plt.close(fig)


def fig_real_pair():
    lat_path = ROOT / "artifacts" / "m1p1" / "latents.npz"
    if not lat_path.exists():
        print(f"[plot_te_gaussian] skip real-pair figure: {lat_path} not found")
        return
    from run_player_te import frame_numbers, mask_and_stride
    lat = dict(np.load(lat_path, allow_pickle=False))
    z = np.asarray(lat["z_post"], dtype=np.float32)
    periods = np.asarray(lat["periods"], dtype=int)
    ft = np.asarray(lat["frame_times"], dtype=float)
    frame = frame_numbers(lat)
    zs, ps, fts, fs, pm = mask_and_stride(z, periods, ft, frame,
                                          stride=10, mask_gaps=True)

    A_ch = zs[:, 0]      # node 0 -> node 1 (see player_ids for ids)
    B_ch = zs[:, 1]
    ids = lat.get("player_ids", None)
    label = "node 0 -> node 1"
    if ids is not None:
        label = f"player {int(ids[0])} -> {int(ids[1])}"

    A_t = standardize(pca_transform(pca_fit(A_ch, 1), A_ch))[0]
    B_t = standardize(pca_transform(pca_fit(B_ch, 1), B_ch))[0]
    x = A_t[:-1][pm].ravel()
    y = B_t[1:][pm].ravel()
    zcol = B_t[:-1][pm].ravel()
    te = conditional_gaussian_mte(x, y, zcol[:, None])
    make_geometry(x, y, zcol, FIGS / "fig_te_real_pair.png",
                  f"Real GVRNN latent chains: TE({label}) = {te:.3f} bits")


def fig_pipeline():
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 2.4)
    ax.axis("off")
    steps = [
        (0.0, 1.6, r"$z^k_t\in\mathbb{R}^{32}$\n(10 Hz latent chain)"),
        (2.0, 1.6, "PCA\n$\\to$ 1st PC"),
        (4.0, 1.6, "standardise\n(unit variance)"),
        (6.2, 1.6, "lagged triples\n$X=z^A_{t-1},\\ Y=z^B_t,\\ Z=z^B_{t-1}$"),
        (8.8, 1.6, r"Gaussian closed form\n$I(X;Y|Z)=\frac{1}{2}\log_2\frac{|S_{XZ}||S_{YZ}|}{|S_Z||S_{XYZ}|}$"),
    ]
    for (x0, y0, txt) in steps:
        box = FancyBboxPatch((x0, y0), 1.7, 0.8,
                             boxstyle="round,pad=0.02", fc="#eef3fb", ec="#1f4e79")
        ax.add_patch(box)
        ax.text(x0 + 0.85, y0 + 0.4, txt, ha="center", va="center", fontsize=8)
    for x0 in (1.7, 3.7, 5.7, 7.9):
        ax.add_patch(FancyArrowPatch((x0, 2.0), (x0 + 0.28, 2.0),
                                     arrowstyle="-|>", mutation_scale=14, color="0.3"))
    ax.text(0.85, 0.55, "1-dim chains  =>  TE = $-\\frac{1}{2}\\log_2(1-\\rho^2_{XY\\cdot Z})$",
            ha="center", fontsize=10)
    ax.set_title("From latent chain to Gaussian TE", fontsize=11)
    fig.tight_layout()
    fig.savefig(FIGS / "fig_te_pipeline.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    FIGS.mkdir(parents=True, exist_ok=True)
    fig_geometry_synthetic()
    fig_mapping()
    fig_real_pair()
    fig_pipeline()
    print(f"[plot_te_gaussian] wrote figures to {FIGS}")
