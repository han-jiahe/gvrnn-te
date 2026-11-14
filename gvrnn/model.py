"""
GVRNN model (Yeh et al., CVPR 2019) - VRNN with type-aware edge GNNs.

Architecture as extracted from the paper (Sec. 2, Eqs. 1-5; Sec. 4.1):

Nodes: K = 11 for soccer - 10 offensive outfield players + the ball
(goalkeeper excluded; the ball has its own node type).  The graph is
FULLY connected; edges carry one of the 4 directed type pairs
(player-player, player-ball, ball-player, ball-ball) as a one-hot
embedding concatenated to the message input (single shared edge MLP).

Per timestep t and node k:

    prior    [mu_pri^t, sig_pri^t] = GNN_pri(h_{1:K}^{t-1})  (h=0 at t=0)
    encoder  [mu_enc^t, sig_enc^t] = GNN_enc([x_t, h_{t-1}], skip x_{t-1})
    z_t^k ~ N(mu_enc,k^t, diag(sig_enc,k^t)^2)       (reparam, 1 sample)
    decoder  [Delta_xhat_t, sig_dec^t] = GNN_dec([z_t, h_{t-1}], skip x_{t-1})
    xhat_t = x_{t-1} + Delta_xhat_t                   (coordinate CHANGE)
    h_t^k  = GRU(h_{t-1}^k, [x_t^k, z_t^k])           (shared GRU)

NOTE on the decoder conditioning (settled from the primary paper text):
Eq. 1 gives "[mu_t^dec, sig_t^dec] = f_dec(z_t, h_{t-1}; theta)" and
Sec. 4.1 restates it as "GNN_dec([z_t^{1:K}, h_{t-1}^{1:K}])" - the
decoder conditions on the PRE-GRU recurrent state h_{t-1}, exactly like
the encoder (f_enc(x_t, h_{t-1})) and the prior (f_pri(h_{t-1})).  The
forward() therefore feeds the input h (h_{t-1}) - not h_new - into
GNN_dec; h_new is still returned and used for recurrence only.  The
x_{t-1} "skip connection" (the paper's "we use as input to the GNNs the
previous observations x_{t-1}") is applied to the encoder and decoder;
the prior keeps its single-argument spec GNN_pri(h_{t-1}).

Training (paper's details): ELBO with analytic KL and Gaussian
reconstruction, teacher forcing, Adam lr=5e-4 with 0.999/epoch decay,
Xavier init, no KL annealing, validation-NLL model selection.  The paper
does not state the latent dimension; this project uses 32 per agent
(config.GVRNN_LATENT_DIM), hidden 128 (config.GVRNN_HIDDEN_DIM).

This module is an exact copy of markov-te's gvrnn/model.py with the
config import adapted for this standalone project; it is NOT a git
worktree of D:\\MarkovSoccer.
"""
import torch
import torch.nn as nn

import config


class TypeAwareEdgeGNN(nn.Module):
    """Message-passing layer with node- and edge-type embeddings.

    v_i, v_j, t_ij -> e(i,j) = f_e([v_i, v_j, onehot(t_ij)])   (edge MLP)
    o_i = f_v([sum_{j in N(i)} e(i,j), onehot(node_type_i)])   (node MLP)
    """

    def __init__(self, in_dim: int, hidden_dim: int, n_edge_types: int = 4,
                 n_node_types: int = 2):
        super().__init__()
        self.f_e = nn.Sequential(
            nn.Linear(2 * in_dim + n_edge_types, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.f_v = nn.Sequential(
            nn.Linear(hidden_dim + n_node_types, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),   # out dim = hidden_dim (head-ready)
        )
        self.n_node_types = n_node_types

    def forward(self, v, node_types, adj):
        """v: (B, N, in_dim); node_types: (N,) int; adj: (E, 2) long.

        Returns updated node embeddings (B, N, hidden_dim) - the small
        parameter heads (mu/logvar/dx) are applied by the caller."""
        B, N, D = v.shape
        src, dst = adj[:, 0], adj[:, 1]

        # edge type one-hot: pp=0, pb=1, bp=2, bb=3  (players first, ball last)
        nt_src = node_types[src]
        nt_dst = node_types[dst]
        etype = nt_src * 2 + nt_dst
        t_ij = torch.zeros(len(src), 4, device=v.device)
        t_ij.scatter_(1, etype[:, None], 1.0)

        e = self.f_e(torch.cat([v[:, src], v[:, dst], t_ij[None].expand(B, -1, -1)], dim=-1))
        # aggregate per batch-dst node
        dst_all = dst[None, :].expand(B, -1).contiguous()
        agg = torch.zeros(B * N, e.shape[-1], device=v.device)
        batch_off = (torch.arange(B, device=v.device) * N)[:, None]
        agg.index_add_(0, (dst_all + batch_off).reshape(-1), e.reshape(-1, e.shape[-1]))
        agg = agg.reshape(B, N, -1)

        nt_onehot = torch.zeros(B * N, self.n_node_types, device=v.device)
        nt_onehot.scatter_(1, node_types[None, :].expand(B, -1).reshape(-1, 1), 1.0)
        o = self.f_v(torch.cat([agg, nt_onehot.reshape(B, N, -1)], dim=-1))
        return o


class GVRNN(nn.Module):
    def __init__(
        self,
        n_nodes: int,
        feat_dim: int,
        latent_dim: int = config.GVRNN_LATENT_DIM,
        hidden_dim: int = config.GVRNN_HIDDEN_DIM,
        n_edge_types: int = 4,
        n_node_types: int = 2,
    ):
        super().__init__()
        self.n_nodes = n_nodes
        self.feat_dim = feat_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        self.gnn_pri = TypeAwareEdgeGNN(hidden_dim, hidden_dim, n_edge_types, n_node_types)
        self.gnn_enc = TypeAwareEdgeGNN(feat_dim + hidden_dim + feat_dim, hidden_dim,
                                        n_edge_types, n_node_types)
        self.gnn_dec = TypeAwareEdgeGNN(latent_dim + hidden_dim + feat_dim, hidden_dim,
                                        n_edge_types, n_node_types)

        self.pri_mu = nn.Linear(hidden_dim, latent_dim)
        self.pri_logvar = nn.Linear(hidden_dim, latent_dim)
        self.enc_mu = nn.Linear(hidden_dim, latent_dim)
        self.enc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec_dx = nn.Linear(hidden_dim, feat_dim)
        self.dec_logvar = nn.Linear(hidden_dim, feat_dim)

        self.rnn = nn.GRUCell(feat_dim + latent_dim, hidden_dim)   # shared across agents

        self._init_xavier()

    def _init_xavier(self):
        for p in self.parameters():
            if p.dim() >= 2:
                nn.init.xavier_uniform_(p)
        # logvar heads start at log(0.01) ~ -4.6 with zero weights: the
        # untrained encoder's sum aggregation over 121 edges has high gain
        # and drives logvar to ~25 along the recurrent chain (exp(12.5) ~
        # 3e5), which NaNs the first batches; a small FIXED initial
        # variance keeps the latent chain bounded until the optimizer takes
        # over (standard VAE initialisation practice)
        for head in (self.pri_logvar, self.enc_logvar, self.dec_logvar):
            nn.init.zeros_(head.weight)
            nn.init.constant_(head.bias, -4.6)

    @staticmethod
    def _reparam(mu, logvar):
        # clamp before exp: at random init the encoder logvar drifts to ~25
        # along the recurrent chain, and exp(0.5*25) ~ 3e5 blows z up and
        # turns the ELBO into -inf/NaN within the first batches
        std = torch.exp(0.5 * logvar.clamp(-10, 10))
        eps = torch.randn_like(std)
        return mu + std * eps

    @staticmethod
    def _kl(mu_q, lv_q, mu_p, lv_p):
        # clamp the logvars before exp: exp overflow/underflow here turns a
        # finite ELBO into -inf/NaN and poisons the whole training run
        var_q, var_p = lv_q.clamp(-20, 20).exp(), lv_p.clamp(-20, 20).exp()
        return 0.5 * (lv_p - lv_q + (var_q + (mu_q - mu_p) ** 2) / var_p - 1)

    def forward(self, x, adj, node_types, h=None, x_prev=None):
        """One VRNN step.  x: (B, N, F); h: (B, N, H) or None;
        x_prev: (B, N, F) previous observations for the GNN skip input
        (defaults to x when not given).  When h is None (episode start)
        the zero state is fed through GNN_pri, i.e. the first-step prior
        is a LEARNED constant Gaussian - the paper and the VRNN reference
        implementation apply phi_prior(h=0) uniformly at t=1 (no N(0,I)
        special case).

        Returns dict with xhat (absolute positions), dx (the predicted
        coordinate change, xhat = x_prev + dx), mu_enc, logvar_enc,
        mu_prior, logvar_prior, z, h_new."""
        B, N, F = x.shape
        device = x.device
        if h is None:
            h = torch.zeros(B, N, self.hidden_dim, device=device)
        pri_out = self.gnn_pri(h, node_types, adj)
        mu_p = self.pri_mu(pri_out)
        lv_p = self.pri_logvar(pri_out)
        if x_prev is None:
            x_prev = x

        # encoder (per-node Gaussians through the GNN)
        enc_in = torch.cat([x, h, x_prev], dim=-1)
        enc_out = self.gnn_enc(enc_in, node_types, adj)
        mu_q = self.enc_mu(enc_out)
        lv_q = self.enc_logvar(enc_out)
        z = self._reparam(mu_q, lv_q)                       # (B, N, L)

        # recurrence with the shared GRU
        rnn_in = torch.cat([x, z], dim=-1)                  # (B, N, F+L)
        h_new = self.rnn(rnn_in.reshape(B * N, -1), h.reshape(B * N, -1))
        h_new = h_new.reshape(B, N, self.hidden_dim)

        # decoder: coordinate CHANGE, then absolute position; conditioned on
        # the PRE-GRU state h (h_{t-1}), per the paper's f_dec(z_t, h_{t-1})
        dec_in = torch.cat([z, h, x_prev], dim=-1)
        dec_out = self.gnn_dec(dec_in, node_types, adj)
        dx = self.dec_dx(dec_out)
        lv_dec = self.dec_logvar(dec_out)
        xhat = x_prev + dx

        return {"xhat": xhat, "dx": dx, "z": z,
                "mu_prior": mu_p, "logvar_prior": lv_p,
                "mu_enc": mu_q, "logvar_enc": lv_q,
                "logvar_dec": lv_dec, "h_new": h_new}

    def elbo(self, x, adj, node_types, h=None, x_prev=None, beta: float = 1.0):
        """ELBO per sample: Gaussian recon of x (diagonal var) - beta * KL."""
        out = self.forward(x, adj, node_types, h, x_prev)
        lv_dec = out["logvar_dec"].clamp(-20, 20)   # numerical guard (see _kl)
        var_dec = lv_dec.exp()
        recon = -0.5 * ((x - out["xhat"]) ** 2 / var_dec
                        + lv_dec + torch.log(torch.full_like(var_dec, 2 * torch.pi))).sum(-1).mean()
        kl = self._kl(out["mu_enc"], out["logvar_enc"],
                      out["mu_prior"], out["logvar_prior"]).sum(-1).mean()
        return recon - beta * kl, out


def make_adjacency(n_nodes: int, ball_idx: int, device=None) -> dict:
    """Complete directed graph over all nodes (players and ball).

    Returns {"adj": (E, 2) long tensor with ALL ordered pairs i->j
    INCLUDING self-edges i->i (the paper's fully connected graph, so
    E = n_nodes^2; the edge type of a self-edge is still the directed
    pair type), "node_types": (N,) long tensor with 0=player, 1=ball}."""
    pairs = [(i, j) for i in range(n_nodes) for j in range(n_nodes)]
    adj = torch.tensor(pairs, dtype=torch.long, device=device)
    node_types = torch.zeros(n_nodes, dtype=torch.long, device=device)
    node_types[ball_idx] = 1
    return {"adj": adj, "node_types": node_types}
