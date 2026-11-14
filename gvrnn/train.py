"""
Training the GVRNN (Yeh et al. CVPR 2019 protocol adapted for latent
extraction rather than forecasting).

Data: 10 Hz tracking of the 10 offensive outfield players + the ball
(goalkeeper excluded).  Windows of 50 frames with 50% overlap; each
window is an independent episode (h reset per window).  Node features:
[x, y, vx, vy] per node.

Preprocessing (paper): center by the play centre, then globally scale so
all coordinates lie in [-1, 1]; the centre/scale are stored in the
checkpoint and re-applied at extraction time.  The paper's exact maximum
is replaced by a robust 99.9% quantile of the centred max (an outlier-
robust variant of the same normalization) - see fit_normalization.

Optimiser (paper): Adam lr=5e-4, exponential decay 0.999 per epoch,
Xavier init, no KL annealing, single reparameterization sample,
teacher forcing, model selection by validation NLL.

Adapted from markov-te's gvrnn/train.py: the CLI loads a tracking npz
directly (produced by data_loading.export_gvrnn_npz) instead of the
markov_te.data module, keeping this directory independent of the git
repository D:\\MarkovSoccer.
"""
import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import config
from .model import GVRNN, make_adjacency


@dataclass
class TrackingData:
    """One match of tracking data (numpy arrays)."""
    pos: np.ndarray          # (T, N, 2) player positions [m]
    vel: np.ndarray          # (T, N, 2) player velocities [m/s]
    ball_pos: np.ndarray     # (T, 2)
    ball_vel: np.ndarray     # (T, 2)
    team_ids: np.ndarray     # (N,) int
    period_of_frame: np.ndarray  # (T,) int
    frame_times: np.ndarray  # (T,) seconds
    # optional (present when the npz was exported with the substitute-
    # analysis extensions; None otherwise)
    frame: np.ndarray = None          # (T,) int original 10 fps frame numbers
    player_ids: np.ndarray = None     # (N,) int node index -> player id
    live_mask: np.ndarray = None      # (T, N) bool real position (pre-fill)

    def node_features(self) -> np.ndarray:
        """(T, N+1, 4): [x, y, vx, vy] of the players, then the ball."""
        players = np.concatenate([self.pos, self.vel], axis=-1)
        ball = np.concatenate([self.ball_pos, self.ball_vel], axis=-1)
        return np.concatenate([players, ball[:, None, :]], axis=1)


def fit_normalization(*datasets: TrackingData):
    """Play-centre centering + global scaling (the paper's [-1, 1]
    normalization, with the exact max replaced by a robust 99.9% quantile),
    fitted on the given (training) matches.  NaN-robust: centre/scale use
    nan-ignoring statistics so a single NaN cannot poison the whole
    normalization; an invalid scale raises loudly instead of training on
    NaNs."""
    xs = [d.node_features() for d in datasets]
    allx = np.concatenate([x.reshape(-1, x.shape[-1])[:, :2] for x in xs], axis=0)
    centre = np.nanmean(allx, axis=0)                 # play centre (x, y)
    centred = np.abs(allx - centre)
    scale = float(np.nanquantile(centred.max(axis=1), 0.999))  # robust max
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(
            "normalization scale is invalid - the tracking data still "
            "contains NaN? re-export it with data_loading.py")
    return {"centre": centre.astype(np.float32), "scale": np.float32(scale)}


def normalize(data: TrackingData, norm: dict) -> torch.Tensor:
    x = data.node_features().astype(np.float32)
    # positions (cols 0,1 of each node's 4 features): centre + scale
    for node in range(x.shape[1]):
        x[:, node, 0] = (x[:, node, 0] - norm["centre"][0]) / norm["scale"]
        x[:, node, 1] = (x[:, node, 1] - norm["centre"][1]) / norm["scale"]
        x[:, node, 2:] /= norm["scale"]              # velocities
    return torch.from_numpy(x)


class WindowDataset(Dataset):
    """Sliding 50-frame windows (stride 25) over one match.

    Windows that STRADDLE a dead-ball gap are skipped: at a gap boundary the
    finite-differenced velocity is a spurious spike (positions jump tens of
    metres between the two kept frames), and those outliers drive the
    decoder logvar to overflow (exp -> inf -> NaN loss)."""

    def __init__(self, feat: torch.Tensor, frame_times=None):
        super().__init__()
        self.feat = feat                               # (T, N+1, 4)
        self.T = len(feat)
        self.window = config.GVRNN_WINDOW_FRAMES
        self.stride = self.window // 2
        starts = list(range(0, max(1, self.T - self.window + 1), self.stride))
        if starts and starts[-1] + self.window < self.T:
            starts.append(self.T - self.window)
        if frame_times is not None:
            ft = np.asarray(frame_times, dtype=float)
            keep = []
            for s in starts:
                seg = ft[s:s + self.window]
                # normal frame step is 0.1 s (10 Hz); a larger jump means the
                # window crosses a dead-ball gap or a period break
                if len(seg) >= 2 and (np.diff(seg) > 0.11).any():
                    continue
                keep.append(s)
            starts = keep
        self.starts = starts

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = self.starts[i]
        return self.feat[s:s + self.window]


def _batch_window_elbo(model, xw_batch, adj, node_types, beta=1.0):
    """ELBO over a batch of windows (B, W, N, F): the GRU recurrence is
    still sequential in time, but the B windows of the batch run IN PARALLEL
    at every frame (model.elbo handles B > 1).  This removes the per-window
    Python loop and ~B-fold reduces the GPU kernel launches and Python
    overhead - the main cost of the previous 10-min/epoch implementation.

    Returns (elbo_mean, kl_mean, recon_mean); elbo stays a TENSOR so
    gradients flow through every GRU step of every window (full BPTT).
    kl/recon sync to Python ONCE per batch."""
    B, W = xw_batch.shape[0], xw_batch.shape[1]
    h = None
    elbo_sum = 0.0
    kl_sum = 0.0
    recon_sum = 0.0
    xp = torch.cat([xw_batch[:, :1], xw_batch[:, :-1]], dim=1)  # (B, W, N, F)
    for t in range(W):
        elbo, out = model.elbo(xw_batch[:, t], adj, node_types, h, xp[:, t], beta)
        h = out["h_new"]
        elbo_sum = elbo_sum + elbo
        kl = model._kl(out["mu_enc"], out["logvar_enc"],
                       out["mu_prior"], out["logvar_prior"]).sum(-1).mean()
        kl_sum = kl_sum + kl
        # recon = ELBO + beta * KL (beta=1 throughout), for the history log
        recon_sum = recon_sum + (-(elbo + beta * kl))
    return elbo_sum / W, float((kl_sum / W).item()), float((recon_sum / W).item())


def train_gvrnn(
    train_data: TrackingData,
    val_data: TrackingData,
    out_dir: Path,
    latent_dim: int = config.GVRNN_LATENT_DIM,
    hidden_dim: int = config.GVRNN_HIDDEN_DIM,
    epochs: int = 30,
    batch_size: int = 32,
    lr: float = 5e-4,
    decay: float = 0.999,
    patience: int = 5,
    seed: int = 42,
    device: str = "cpu",
):
    """Train and save the best model (by validation NLL) to out_dir."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    out_dir.mkdir(parents=True, exist_ok=True)

    norm = fit_normalization(train_data)
    train_feat = normalize(train_data, norm)
    val_feat = normalize(val_data, norm)

    n_nodes = train_data.pos.shape[1] + 1              # players + ball
    ball_idx = n_nodes - 1
    adj_dict = make_adjacency(n_nodes, ball_idx, device=device)
    adj, node_types = adj_dict["adj"], adj_dict["node_types"]

    model = GVRNN(n_nodes, feat_dim=4, latent_dim=latent_dim, hidden_dim=hidden_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, gamma=decay)

    train_dl = DataLoader(WindowDataset(train_feat, train_data.frame_times),
                          batch_size=batch_size, shuffle=True)
    val_dl = DataLoader(WindowDataset(val_feat, val_data.frame_times),
                        batch_size=batch_size, shuffle=False)

    history = {"train_nll": [], "val_nll": [], "train_recon": [], "train_kl": []}
    best_val, best_epoch, stale = np.inf, 0, 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        nlls, kls, recons = [], [], []
        for xb in train_dl:
            xb = xb.to(device)
            opt.zero_grad()
            elbo, kl, recon = _batch_window_elbo(model, xb, adj, node_types)
            total = -elbo                    # loss = -mean ELBO over windows
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            nlls.append(-total.item())
            kls.append(kl)
            recons.append(recon)
        sched.step()
        train_nll = float(np.mean(nlls))

        model.eval()
        val_nlls = []
        with torch.no_grad():
            for xb in val_dl:
                xb = xb.to(device)
                elbo, _, _ = _batch_window_elbo(model, xb, adj, node_types)
                val_nlls.append((-elbo).item())
        val_nll = float(np.mean(val_nlls))

        history["train_nll"].append(train_nll)
        history["val_nll"].append(val_nll)
        history["train_recon"].append(float(np.mean(recons)))
        history["train_kl"].append(float(np.mean(kls)))
        print(f"[epoch {epoch:3d}] train NLL {train_nll:9.3f}  val NLL {val_nll:9.3f}  "
              f"KL {np.mean(kls):7.3f}  ({time.time() - t0:.0f}s)", flush=True)

        if val_nll < best_val:
            best_val, best_epoch, stale = val_nll, epoch, 0
            torch.save({
                "state_dict": model.state_dict(),
                "n_nodes": n_nodes, "feat_dim": 4,
                "latent_dim": latent_dim, "hidden_dim": hidden_dim,
                "norm": norm,
                "n_offensive_players": train_data.pos.shape[1],
            }, out_dir / "best_model.pt")
        else:
            stale += 1
            if stale >= patience:
                print(f"early stop at epoch {epoch} (best {best_epoch})", flush=True)
                break

    with open(out_dir / "train_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"saved model to {out_dir / 'best_model.pt'}", flush=True)
    return out_dir / "best_model.pt"


def _split(data: TrackingData, frac: float = 0.7) -> tuple:
    """Split one match into train/val along the frame axis, per period
    (first `frac` of each period for training, the rest for validation)."""
    keep = np.zeros(len(data.frame_times), dtype=bool)
    for p in np.unique(data.period_of_frame):
        m = np.where(data.period_of_frame == p)[0]
        keep[m[: int(len(m) * frac)]] = True

    def take(mask):
        return TrackingData(
            pos=data.pos[mask], vel=data.vel[mask],
            ball_pos=data.ball_pos[mask], ball_vel=data.ball_vel[mask],
            team_ids=data.team_ids,
            period_of_frame=data.period_of_frame[mask],
            frame_times=data.frame_times[mask],
            frame=data.frame[mask] if data.frame is not None else None,
            player_ids=data.player_ids,
            live_mask=(data.live_mask[mask]
                       if data.live_mask is not None else None),
        )

    return take(keep), take(~keep)


def load_tracking_npz(npz_path: Path) -> TrackingData:
    """Rebuild TrackingData from the npz written by
    data_loading.export_gvrnn_npz (keys: pos, vel, ball_pos, ball_vel,
    team_ids, period_of_frame, frame_times)."""
    d = np.load(npz_path, allow_pickle=False)
    return TrackingData(
        pos=d["pos"], vel=d["vel"],
        ball_pos=d["ball_pos"], ball_vel=d["ball_vel"],
        team_ids=d["team_ids"],
        period_of_frame=d["period_of_frame"],
        frame_times=d["frame_times"],
        frame=d["frame"] if "frame" in d else None,
        player_ids=d["player_ids"] if "player_ids" in d else None,
        live_mask=d["live_mask"] if "live_mask" in d else None,
    )


def main(argv=None):
    """Train one GVRNN on the offensive view of one match:

        python -m gvrnn.train --npz artifacts/match1/tracking_team0.npz \\
            --out-dir artifacts/match1/model_team0
    """
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--npz", required=True, help="tracking npz from data_loading.py")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--val-frac", type=float, default=0.7)
    ap.add_argument("--device", default="cpu",
                    help="'cpu', 'cuda', 'cuda:0', or 'auto' (cuda if available)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    device = config.resolve_device(args.device)
    print(f"[gvrnn.train] device = {device}", flush=True)
    data = load_tracking_npz(config.abs_path(args.npz))
    train, val = _split(data, args.val_frac)
    train_gvrnn(train, val, config.abs_path(args.out_dir), epochs=args.epochs,
                seed=args.seed, device=device)


if __name__ == "__main__":
    main()
