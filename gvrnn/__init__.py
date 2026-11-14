"""
gvrnn - GVRNN (Yeh et al., CVPR 2019) reimplementation for per-player latent
chain extraction.  Self-contained copy of D:\\MarkovSoccer\\markov-te's
gvrnn module (identical architecture, including the decoder-conditioning
fix: all three GNNs condition on the PRE-GRU state h_{t-1}, per the paper's
Eq. 1), adapted to import from this project's top-level config.py.

  * model.py    GVRNN architecture (type-aware edge GNN + per-agent latent)
  * train.py    training protocol (paper's): 50-frame windows @10 Hz, Adam
                lr=5e-4 x 0.999/epoch, teacher forcing, Xavier init
  * extract.py  per-player posterior-mean extraction z^k_t for k = 1..K
"""
