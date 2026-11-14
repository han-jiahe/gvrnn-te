"""
Per-player latent-chain extraction: run a trained GVRNN over a full match,
frame by frame, and write the PER-PLAYER implicit chains z^k_t.

z^k_t = posterior mean E_q[z^k_t | x_t, h_{t-1}] of node k (10 outfield
players + the ball, K = 11) - the deterministic point estimate the MTE
protocol consumes (PROTOCOL.md sec. 4).  The prior mean (the latent state
PREDICTABLE from history alone) and the per-node GRU states h^k_t are
saved alongside for diagnostics.

The filtering pass mirrors model.forward's encoder + GRU recurrence
exactly (prior from h = 0 at episode start, encoder on [x_t, h_{t-1},
x_{t-1}], deterministic posterior mean fed to the GRU - no sampling).
The GRU state h is carried across the whole match and reset at period
breaks (same reset rule as the paper's windowed training).

Adapted from markov-te's gvrnn/extract.py: this version keeps the FULL
per-node latent array (T, K, L) instead of pooling over agents, because
player-to-player MTE needs the individual chains.
"""
from pathlib import Path

import numpy as np
import torch

import config
from .model import GVRNN, make_adjacency
from .train import TrackingData, normalize


@torch.no_grad()
def extract_latent(
    data: TrackingData,
    model_path: Path,
    device: str = "cpu",
) -> dict:
    """Returns dict with arrays:
    z_post   (T, K, L)  per-player posterior means   (node k at row-k dim)
    z_prior  (T, K, L)  per-player prior means
    h        (T, K, H)  per-player GRU states (diagnostics)
    frame_times (T,), periods (T,)
    """
    # weights_only=False: the checkpoint stores our own numpy norm dict
    # (centre/scale) alongside the state_dict; trusted file, no pickle risk
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    model = GVRNN(ckpt["n_nodes"], ckpt["feat_dim"],
                  ckpt["latent_dim"], ckpt["hidden_dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    n_nodes = ckpt["n_nodes"]
    ball_idx = n_nodes - 1
    adj = {et: idx.to(device) for et, idx in make_adjacency(n_nodes, ball_idx).items()}

    # re-apply the training normalization: the checkpoint stores centre/scale
    # and the model was trained on [-1, 1]-scaled inputs
    feat = normalize(data, ckpt["norm"])                    # (T, N+1, 4) tensor
    T = len(feat)
    K, L, H = n_nodes, ckpt["latent_dim"], ckpt["hidden_dim"]
    h = None
    x_prev = None
    z_post = np.zeros((T, K, L), dtype=np.float32)
    z_prior = np.zeros((T, K, L), dtype=np.float32)
    h_out = np.zeros((T, K, H), dtype=np.float32)

    for t in range(T):
        gap = (t > 0 and data.frame_times[t] - data.frame_times[t - 1] > 0.11)
        if t > 0 and (data.period_of_frame[t] != data.period_of_frame[t - 1] or gap):
            h = None       # period break or dead-ball gap: reset the state
            x_prev = None
        x = feat[t:t + 1].to(device)                        # (1, N+1, 4)
        if x_prev is None:
            x_prev = x     # teacher forcing at window starts, as in training
        if h is None:
            h = torch.zeros(1, K, H, device=device)

        # prior: GNN_pri(h_{t-1}) - the paper's single-argument spec
        # NOTE the GNN signature is forward(v, node_types, adj) - unlike
        # model.forward's (x, adj, node_types) - keep the order straight
        mu_prior = model.pri_mu(model.gnn_pri(h, adj["node_types"], adj["adj"]))
        # posterior: GNN_enc([x_t, h_{t-1}], skip x_{t-1}); deterministic mean
        enc_in = torch.cat([x, h, x_prev], dim=-1)
        mu_q = model.enc_mu(model.gnn_enc(enc_in, adj["node_types"], adj["adj"]))
        # GRU recurrence with the deterministic posterior mean (no sampling)
        rnn_in = torch.cat([x, mu_q], dim=-1)
        h = model.rnn(rnn_in.reshape(K, -1), h.reshape(K, -1)).reshape(1, K, H)
        x_prev = x

        z_post[t] = mu_q[0].cpu().numpy()
        z_prior[t] = mu_prior[0].cpu().numpy()
        h_out[t] = h[0].cpu().numpy()

    frame = (data.frame if data.frame is not None
             else np.rint(data.frame_times * 10.0).astype(np.int64))
    out = {"z_post": z_post, "z_prior": z_prior, "h": h_out,
           "frame_times": data.frame_times,
           "periods": data.period_of_frame,
           "frame": frame}
    if data.player_ids is not None:
        out["player_ids"] = data.player_ids
    return out


def save_latent(out_npz: Path, latent: dict) -> None:
    np.savez_compressed(out_npz, **latent)


def load_latent(npz_path: Path) -> dict:
    return dict(np.load(npz_path, allow_pickle=False))


def main(argv=None):
    """Extract per-player latent chains over one match:

        python -m gvrnn.extract --npz artifacts/match1/tracking_team0.npz \\
            --model artifacts/match1/model_team0/best_model.pt \\
            --out artifacts/match1/latents_team0.npz
    """
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--npz", required=True, help="tracking npz (data_loading.py)")
    ap.add_argument("--model", required=True, help="best_model.pt from gvrnn.train")
    ap.add_argument("--out", required=True, help="output .npz path")
    ap.add_argument("--device", default="cpu",
                    help="'cpu', 'cuda', 'cuda:0', or 'auto' (cuda if available)")
    args = ap.parse_args(argv)

    from .train import load_tracking_npz
    data = load_tracking_npz(config.abs_path(args.npz))
    latent = extract_latent(data, config.abs_path(args.model),
                            device=config.resolve_device(args.device))
    save_latent(config.abs_path(args.out), latent)
    print(f"latents -> {args.out}: z_post {latent['z_post'].shape}, "
          f"frames {len(latent['frame_times'])}")


if __name__ == "__main__":
    main()
