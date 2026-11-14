"""
Central configuration for the player-level transfer entropy (TE) project
(MarkovSoccerTE).

Pairwise transfer entropy between per-player GVRNN latent chains:

    TE(A -> B) = I( z^A_{t-1} ; z^B_t | z^B_{t-1} )

i.e. the transfer entropy from player A's implicit chain to player B's,
conditioned ONLY on B's own history (order-1, k = l = 1; no third-party
channels).  This directory is deliberately INDEPENDENT of D:\\MarkovSoccer
(a GitHub clone): no work is done inside the git repository.

All tunable parameters of the three TE estimators and of the per-player
latent extraction live here, so every script reads from one source of truth.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

# --------------------------------------------------------------------------- #
# GVRNN (Yeh et al., CVPR 2019) - per-player latent chains                   #
# The paper does not state the latent/hidden dimensions; 32 / 128 are        #
# project choices, same values as D:\\MarkovSoccer\\markov-te.                 #
# --------------------------------------------------------------------------- #
GVRNN_LATENT_DIM = 32
GVRNN_HIDDEN_DIM = 128
GVRNN_WINDOW_FRAMES = 50     # 5 s at 10 Hz (paper's training windows)
GVRNN_FRAME_RATE_HZ = 10.0

# node layout of the extracted latent array (T, K, L):
# K = 10 offensive outfield players + 1 ball; the ball node is LAST.
GVRNN_N_PLAYERS = 10
GVRNN_BALL_IDX = GVRNN_N_PLAYERS       # ball is node index 10 (0-based)


# --------------------------------------------------------------------------- #
# TE estimation parameters                                                    #
# --------------------------------------------------------------------------- #
@dataclass
class TEConfig:
    """Parameters shared by the three TE estimators."""
    # history length - FIXED at 1 (order-1 TE, k=l=1, see PROTOCOL.md sec. 3)
    embedding: int = 1

    # preprocessing
    pca_dims: Tuple[int, ...] = (1, 2)   # PCA-reduced dims of each player's
                                         # 32-dim latent chain
    stride: Tuple[int, ...] = (5, 10)    # downsample the 10 Hz chain by
                                         # stride -> 2 Hz / 1 Hz (autocorrelation)

    # KSG (PROTOCOL.md sec. 5)
    k_neighbors: Tuple[int, ...] = (3, 5, 8)

    # discrete plug-in (PROTOCOL.md sec. 7, small systems only)
    n_clusters: Tuple[int, ...] = (3, 5)
    min_samples_per_cell: int = 10       # skip a pair if cells > n / this

    # Gaussian (PROTOCOL.md sec. 6)
    gaussian_ridge: float = 1e-3         # trace-scaled ridge on covariances

    # significance testing (PROTOCOL.md sec. 9)
    n_surrogates: int = 200
    block_len: int = 25                  # block-shuffle block length (frames at
                                         # the downsampled rate)
    alpha: float = 0.05

    # base of log: 2 -> bits
    base: float = 2.0

    # random seed for reproducibility
    seed: int = 42


DEFAULT_TE_CONFIG = TEConfig()


# --------------------------------------------------------------------------- #
# Paths                                                                       #
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent
# The SkillCorner Open Data checkout lives OUTSIDE this project; it does NOT
# need to be copied here - only the small artifacts (trained models, per-player
# latents) and the results live in this directory.  Override with --data-dir.
SKILLCORNER_DATA_DIR = Path(r"D:\SkillCorner_opendata\data")
# Artifacts/results are ABSOLUTE (anchored to PROJECT_ROOT) so the project can
# be run from any working directory (e.g. C:\ cmd); relative CLI args are also
# anchored to PROJECT_ROOT via abs_path(), never to the caller's cwd.
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"     # trained models, per-player latents
RESULTS_DIR = PROJECT_ROOT / "results"         # TE matrices / JSON per match


def abs_path(p) -> Path:
    """Resolve a possibly-relative path against PROJECT_ROOT.

    The project is runnable from any working directory: relative --out/--npz/
    --latents/--model arguments resolve here, NOT against the caller's cwd.
    """
    p = Path(p)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def resolve_device(device: str) -> str:
    """Resolve a --device argument: 'auto' -> 'cuda' if torch sees a GPU,
    else 'cpu'; any other value ('cpu'/'cuda'/'cuda:0') passes through."""
    if device != "auto":
        return device
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"
