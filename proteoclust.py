"""
ProteoCLUST — Unified Spectral Clustering Pipeline for Proteomics MGF Files
=============================================================================
Three-stage pipeline:
  Stage 1  Universal Spectrum Encoder    (USE)  — dense embeddings
  Stage 2  Hierarchical Graph Clustering (HGCE) — spectral + DBSCAN
  Stage 3  Biologically-Aware Reranker   (BAR)  — quality scoring + PTM flags

Memory-efficient design targets Intel Core i7 without HPC infrastructure.
Handles datasets from ~100 MB to several GB via chunk-streaming, sparse
matrices, and approximate nearest-neighbour search.

Requirements (install once):
    pip install numpy scipy scikit-learn pyteomics tqdm torch

Usage (Spyder console or terminal):
    python proteoclust.py --mgf your_file.mgf --out results/
    python proteoclust.py --mgf coread.mgf --out results/ --batch 4000
"""

# ─── stdlib ──────────────────────────────────────────────────────────────────
import os
import sys
import time
import argparse
import logging
import warnings
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Iterator
import math
import gc

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─── third-party ─────────────────────────────────────────────────────────────
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import normalize
from sklearn.neighbors import NearestNeighbors, BallTree
from tqdm import tqdm

# Torch is used for the lightweight transformer encoder only; it gracefully
# degrades to a pure-NumPy fallback if not installed.
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[WARN] PyTorch not found — using NumPy-based encoder (slightly lower accuracy)")

try:
    from pyteomics import mgf as pyteomics_mgf
    PYTEOMICS_AVAILABLE = True
except ImportError:
    PYTEOMICS_AVAILABLE = False
    print("[WARN] pyteomics not found — using built-in MGF parser")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Config:
    # ── I/O ──────────────────────────────────────────────────────────────────
    mgf_path: str = ""
    output_dir: str = "proteoclust_out"
    resume: bool = False  # cache/reuse Stage 1 & 2 results across interrupted runs

    # ── Stage 1 — Encoder ────────────────────────────────────────────────────
    mz_min: float = 50.0          # m/z range start
    mz_max: float = 1500.0        # m/z range end
    bin_width: float = 0.02       # Da per bin → 72 500 bins total
    top_k_peaks: int = 150        # retain only top-K intense peaks
    embed_dim: int = 128          # output embedding dimension
    n_heads: int = 4              # transformer attention heads
    n_layers: int = 2             # transformer encoder layers
    ff_dim: int = 256             # feed-forward inner dimension
    batch_size: int = 2048        # spectra per encoding batch

    # ── Stage 2 — Clustering ─────────────────────────────────────────────────
    precursor_tol_da: float = 1.5 # hard mass tolerance for candidate pairs
    edge_threshold: float = 0.70  # cosine similarity pruning threshold
    n_eigenvectors: int = 30      # spectral embedding dimensions
    dbscan_eps: float = 0.15      # DBSCAN epsilon on spectral space
    dbscan_min_samples: int = 2   # minimum cluster size
    split_var_threshold: float = 0.04   # dynamic split threshold
    merge_dist_threshold: float = 0.05  # dynamic merge threshold
    max_edges_per_node: int = 50  # cap edges to keep graph sparse
    ann_n_neighbours: int = 100   # approximate NN search width
    ann_query_chunk_size: int = 2000  # rows per kneighbors() call (memory cap)
    large_n_threshold: int = 100_000  # above this use graph connected-components instead of spectral DBSCAN

    # ── Stage 3 — Reranker ───────────────────────────────────────────────────
    sigma_q: float = 0.10         # quality score temperature
    high_q_threshold: float = 0.80
    medium_q_threshold: float = 0.50

    # ── Runtime ──────────────────────────────────────────────────────────────
    n_jobs: int = -1              # -1 = all cores
    device: str = "cpu"           # "cpu" or "cuda"
    seed: int = 42
    verbose: bool = True
    max_spectra: Optional[int] = None   # truncate for testing


# ─────────────────────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Spectrum:
    scan_id: str
    precursor_mz: float
    precursor_charge: int
    mz_array: np.ndarray
    intensity_array: np.ndarray
    rt: float = 0.0

@dataclass
class Cluster:
    cluster_id: int
    member_ids: List[int]          # indices into the global spectrum list
    centroid: Optional[np.ndarray] = None
    quality_score: float = 0.0
    confidence: str = "LOW"
    ptm_flag: bool = False
    within_var: float = 0.0
    mean_similarity: float = 0.0
    consensus_mz: Optional[np.ndarray] = None
    consensus_intensity: Optional[np.ndarray] = None


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str, verbose: bool) -> logging.Logger:
    os.makedirs(output_dir, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-7s  %(message)s"
    handlers = [logging.FileHandler(os.path.join(output_dir, "proteoclust.log"))]
    if verbose:
        handlers.append(logging.StreamHandler(sys.stdout))
    logging.basicConfig(level=logging.INFO, format=fmt, handlers=handlers)
    return logging.getLogger("ProteoCLUST")


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 0 — MGF PARSER (streaming, memory-efficient)
# ─────────────────────────────────────────────────────────────────────────────

def _parse_mgf_builtin(path: str) -> Iterator[Spectrum]:
    """Pure-Python streaming MGF parser. Handles large files without loading
    the entire file into RAM."""
    scan_id = ""
    pmz = 0.0
    pcharge = 0
    rt = 0.0
    mz_list: List[float] = []
    int_list: List[float] = []
    inside = False
    scan_counter = 0

    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            upper = line.upper()
            if upper == "BEGIN IONS":
                inside = True
                mz_list, int_list = [], []
                scan_id, pmz, pcharge, rt = "", 0.0, 0, 0.0
                continue
            if upper == "END IONS":
                inside = False
                if mz_list:
                    scan_counter += 1
                    yield Spectrum(
                        scan_id=scan_id or f"scan_{scan_counter}",
                        precursor_mz=pmz,
                        precursor_charge=pcharge,
                        mz_array=np.array(mz_list, dtype=np.float32),
                        intensity_array=np.array(int_list, dtype=np.float32),
                        rt=rt,
                    )
                continue
            if not inside:
                continue
            if "=" in line:
                key, _, val = line.partition("=")
                key = key.strip().upper()
                val = val.strip()
                if key == "TITLE":
                    scan_id = val
                elif key == "PEPMASS":
                    parts = val.split()
                    pmz = float(parts[0])
                elif key == "CHARGE":
                    pcharge = int(val.replace("+", "").replace("-", ""))
                elif key in ("RTINSECONDS", "RETENTION_TIME", "RT"):
                    try:
                        rt = float(val)
                    except ValueError:
                        pass
            else:
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        mz_list.append(float(parts[0]))
                        int_list.append(float(parts[1]))
                    except ValueError:
                        pass


def _parse_mgf_pyteomics(path: str) -> Iterator[Spectrum]:
    scan_counter = 0
    with pyteomics_mgf.read(path) as reader:
        for spec in reader:
            scan_counter += 1
            params = spec.get("params", {})
            pmz_raw = params.get("pepmass", (0.0,))
            pmz = float(pmz_raw[0]) if pmz_raw else 0.0
            charge_raw = params.get("charge", [0])
            pcharge = int(charge_raw[0]) if charge_raw else 0
            rt = float(params.get("rtinseconds", 0.0))
            title = params.get("title", f"scan_{scan_counter}")
            mz_arr = np.asarray(spec.get("m/z array", []), dtype=np.float32)
            int_arr = np.asarray(spec.get("intensity array", []), dtype=np.float32)
            if len(mz_arr) > 0:
                yield Spectrum(
                    scan_id=title,
                    precursor_mz=pmz,
                    precursor_charge=pcharge,
                    mz_array=mz_arr,
                    intensity_array=int_arr,
                    rt=rt,
                )


def load_spectra(path: str, max_spectra: Optional[int] = None,
                 verbose: bool = True) -> List[Spectrum]:
    """Stream-parse MGF and return a list of Spectrum objects. Uses pyteomics
    when available for robustness, otherwise falls back to built-in parser."""
    parse_fn = _parse_mgf_pyteomics if PYTEOMICS_AVAILABLE else _parse_mgf_builtin
    spectra: List[Spectrum] = []
    desc = f"Parsing {Path(path).name}"
    iterator = parse_fn(path)
    if verbose:
        iterator = tqdm(iterator, desc=desc, unit=" spec")
    for s in iterator:
        spectra.append(s)
        if max_spectra and len(spectra) >= max_spectra:
            break
    return spectra


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 1 — UNIVERSAL SPECTRUM ENCODER (USE)
# ─────────────────────────────────────────────────────────────────────────────

class SpectrumBinner:
    """Converts a Spectrum into a fixed-length intensity vector via binning."""

    def __init__(self, cfg: Config):
        self.mz_min = cfg.mz_min
        self.mz_max = cfg.mz_max
        self.bin_width = cfg.bin_width
        self.n_bins = int(math.ceil((cfg.mz_max - cfg.mz_min) / cfg.bin_width))
        self.top_k = cfg.top_k_peaks

    def bin_spectrum(self, spec: Spectrum) -> np.ndarray:
        vec = np.zeros(self.n_bins, dtype=np.float32)
        mz = spec.mz_array
        intensity = spec.intensity_array
        if len(mz) == 0:
            return vec

        # Keep only top-K peaks by intensity
        if len(mz) > self.top_k:
            idx = np.argpartition(intensity, -self.top_k)[-self.top_k:]
            mz = mz[idx]
            intensity = intensity[idx]

        # Sqrt-normalise intensities
        max_i = intensity.max()
        if max_i > 0:
            intensity = np.sqrt(intensity / max_i)

        # Bin
        bin_idx = ((mz - self.mz_min) / self.bin_width).astype(np.int32)
        mask = (bin_idx >= 0) & (bin_idx < self.n_bins)
        np.maximum.at(vec, bin_idx[mask], intensity[mask])
        return vec

    def bin_batch(self, spectra: List[Spectrum]) -> np.ndarray:
        return np.stack([self.bin_spectrum(s) for s in spectra])


# ── PyTorch transformer encoder ───────────────────────────────────────────────
# Encodes each spectrum as a SET of peaks: each token = (norm_mz, norm_intensity).
# Sequence length = top_k (≤150), making attention matrices tiny (150×150).

if TORCH_AVAILABLE:
    class LightTransformerEncoder(nn.Module):
        """Peak-token transformer encoder.
        Input : (B, K, 2)  — K = top_k peaks, features = [norm_mz, norm_intensity]
        Output: (B, embed_dim) L2-normalised embedding
        """
        def __init__(self, cfg: Config):
            super().__init__()
            d = cfg.embed_dim
            self.input_proj = nn.Linear(2, d)
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d, nhead=cfg.n_heads, dim_feedforward=cfg.ff_dim,
                dropout=0.0, batch_first=True, norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(enc_layer,
                                                 num_layers=cfg.n_layers)
            self.proj_norm = nn.Sequential(
                nn.Linear(d, d),
                nn.LayerNorm(d),
            )

        def forward(self, x: torch.Tensor,
                    pad_mask: torch.Tensor) -> torch.Tensor:
            # x: (B, K, 2)   pad_mask: (B, K) True = padding token
            x = self.input_proj(x)                    # (B, K, d)
            x = self.encoder(x, src_key_padding_mask=pad_mask)
            # Mean pool over non-padding tokens
            valid = (~pad_mask).unsqueeze(-1).float()  # (B, K, 1)
            x = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
            x = self.proj_norm(x)
            x = nn.functional.normalize(x, dim=-1)
            return x


def _spectrum_to_peak_tokens(spec: Spectrum, top_k: int,
                              mz_min: float, mz_max: float) -> np.ndarray:
    """Return (top_k, 2) float32 array of [norm_mz, norm_intensity].
    Rows beyond actual peak count are zero-padded."""
    tokens = np.zeros((top_k, 2), dtype=np.float32)
    mz = spec.mz_array
    intensity = spec.intensity_array
    if len(mz) == 0:
        return tokens

    # Top-K by intensity
    if len(mz) > top_k:
        idx = np.argpartition(intensity, -top_k)[-top_k:]
        mz = mz[idx]
        intensity = intensity[idx]

    # Sort by mz for reproducibility
    order = np.argsort(mz)
    mz = mz[order]
    intensity = intensity[order]

    # Normalise both axes to [0, 1]
    norm_mz = np.clip((mz - mz_min) / (mz_max - mz_min), 0.0, 1.0)
    max_i = intensity.max()
    norm_int = np.sqrt(intensity / max_i) if max_i > 0 else intensity

    n = len(mz)
    tokens[:n, 0] = norm_mz
    tokens[:n, 1] = norm_int
    return tokens


class NumpyFallbackEncoder:
    """NumPy encoder when PyTorch is unavailable — projects binned spectrum."""

    def __init__(self, cfg: Config):
        n_bins = SpectrumBinner(cfg).n_bins
        rng = np.random.RandomState(cfg.seed)
        self.proj = rng.randn(n_bins, cfg.embed_dim).astype(np.float32)
        self.proj /= np.linalg.norm(self.proj, axis=0, keepdims=True)
        self.embed_dim = cfg.embed_dim
        self.binner = SpectrumBinner(cfg)

    def encode_batch(self, spectra) -> np.ndarray:
        binned = self.binner.bin_batch(spectra)
        emb = binned @ self.proj
        norms = np.linalg.norm(emb, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return (emb / norms).astype(np.float32)


class UniversalSpectrumEncoder:
    """Stage 1 encoder. Peak-token transformer (Torch) or binned projection (NumPy)."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.device = torch.device(cfg.device) if TORCH_AVAILABLE else None

        if TORCH_AVAILABLE:
            self.model = LightTransformerEncoder(cfg).to(self.device)
            self.model.eval()
            self._torch = True
        else:
            self.model = NumpyFallbackEncoder(cfg)
            self._torch = False

    def _make_peak_token_batch(self, spectra: List[Spectrum]):
        """Return tokens (B, K, 2) and padding mask (B, K)."""
        K = self.cfg.top_k_peaks
        tokens = np.stack([
            _spectrum_to_peak_tokens(s, K, self.cfg.mz_min, self.cfg.mz_max)
            for s in spectra
        ])  # (B, K, 2)
        # A token is padding if both features are zero
        pad_mask = (tokens.sum(axis=-1) == 0)   # (B, K)
        return tokens, pad_mask

    def encode(self, spectra: List[Spectrum], logger: logging.Logger) -> np.ndarray:
        N = len(spectra)
        out = np.zeros((N, self.cfg.embed_dim), dtype=np.float32)
        bs = self.cfg.batch_size
        n_batches = math.ceil(N / bs)
        for b in tqdm(range(n_batches), desc="Stage1 encoding", unit="batch"):
            sl = slice(b * bs, (b + 1) * bs)
            batch = spectra[sl]
            if self._torch:
                tokens, pad_mask = self._make_peak_token_batch(batch)
                t_tok = torch.from_numpy(tokens).to(self.device)
                t_pad = torch.from_numpy(pad_mask).to(self.device)
                with torch.no_grad():
                    emb = self.model(t_tok, t_pad).cpu().numpy()
            else:
                emb = self.model.encode_batch(batch)
            out[sl] = emb
        logger.info(f"Stage 1 complete: {N} embeddings, dim={self.cfg.embed_dim}")
        return out


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 2 — HIERARCHICAL GRAPH CLUSTERING ENGINE (HGCE)
# ─────────────────────────────────────────────────────────────────────────────

def _precursor_score(pmz_i: float, pmz_j: float, charge_i: int, charge_j: int,
                     tol: float) -> float:
    """Returns a [0,1] precursor compatibility score."""
    if charge_i != charge_j and min(charge_i, charge_j) > 0:
        return 0.0
    delta = abs(pmz_i - pmz_j)
    return max(0.0, 1.0 - delta / tol)


def build_sparse_similarity_graph(
        embeddings: np.ndarray,
        spectra: List[Spectrum],
        cfg: Config,
        logger: logging.Logger,
) -> csr_matrix:
    """Build a sparse cosine similarity graph with precursor mass hard-filter.

    Memory strategy:
      - Sort spectra by precursor m/z to enable fast range queries.
      - Use sklearn NearestNeighbors in cosine space to limit comparisons.
      - Assemble as COO → CSR sparse matrix.
    """
    N = len(spectra)
    pmz = np.array([s.precursor_mz for s in spectra], dtype=np.float64)
    pcharge = np.array([s.precursor_charge for s in spectra], dtype=np.int32)

    # ANN search in embedding space (cosine via normalised L2)
    logger.info("Building approximate neighbour index …")
    k = min(cfg.ann_n_neighbours, N - 1)
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine",
                          algorithm="brute", n_jobs=cfg.n_jobs)
    nn.fit(embeddings)

    # Build COO lists
    rows, cols, data = [], [], []

    # Query neighbours in row-chunks to bound peak memory usage — a single
    # call to kneighbors(embeddings) materialises an (N, N) pairwise-distance
    # block internally for brute-force cosine search, which can blow up to
    # many GB for large N. Chunking keeps each block at (chunk, N).
    chunk_size = max(1, min(N, cfg.ann_query_chunk_size))

    for start in tqdm(range(0, N, chunk_size), desc="Stage2 ANN query", unit=" chunk"):
        end = min(start + chunk_size, N)
        dists, indices = nn.kneighbors(embeddings[start:end])   # (chunk, k+1)

        for offset in range(end - start):
            i = start + offset
            for rank in range(1, k + 1):           # skip self (rank=0)
                j = int(indices[offset, rank])
                cos_dist = float(dists[offset, rank])
                cos_sim = max(0.0, 1.0 - cos_dist)

                # Hard precursor mass filter
                if abs(pmz[i] - pmz[j]) > cfg.precursor_tol_da:
                    continue

                p_score = _precursor_score(pmz[i], pmz[j], pcharge[i], pcharge[j],
                                           cfg.precursor_tol_da)
                w = cos_sim * p_score

                if w < cfg.edge_threshold:
                    continue

                rows.append(i)
                cols.append(j)
                data.append(w)

    if not data:
        logger.warning("No edges above threshold — lowering edge_threshold may help.")
        return csr_matrix((N, N), dtype=np.float32)

    W = csr_matrix((data, (rows, cols)), shape=(N, N), dtype=np.float32)
    # Symmetrize
    W = (W + W.T) / 2.0
    logger.info(f"Sparse graph: {N} nodes, {W.nnz} edges")
    return W


def compute_spectral_embedding(W: csr_matrix, k: int,
                                logger: logging.Logger) -> np.ndarray:
    """Normalised graph Laplacian → top-k eigenvectors."""
    N = W.shape[0]
    from scipy.sparse import diags, eye

    # Degree vector
    degrees = np.asarray(W.sum(axis=1)).flatten()
    # Avoid division by zero for isolated nodes
    d_inv_sqrt = np.where(degrees > 0, 1.0 / np.sqrt(degrees), 0.0)
    D_inv_sqrt = diags(d_inv_sqrt)

    # Normalised Laplacian  L_sym = I - D^{-1/2} W D^{-1/2}
    L_sym = eye(N, format="csr") - D_inv_sqrt @ W @ D_inv_sqrt

    k_eff = min(k + 1, N - 2)
    logger.info(f"Computing {k_eff} eigenvectors of L_sym ({N}×{N}) …")
    try:
        # ARPACK — memory-efficient for sparse matrices
        eigenvalues, eigenvectors = eigsh(L_sym, k=k_eff, which="SM",
                                          tol=1e-4, maxiter=2000)
    except Exception as e:
        logger.warning(f"eigsh failed ({e}); falling back to dense eigh on small N.")
        from scipy.linalg import eigh
        Ldense = L_sym.toarray()
        eigenvalues, eigenvectors = eigh(Ldense, subset_by_index=[0, k_eff])

    # Sort by eigenvalue, drop the trivial zero eigenvalue (index 0)
    order = np.argsort(eigenvalues)
    Z = eigenvectors[:, order[1:k_eff]]   # skip Fiedler-0
    Z = normalize(Z, norm="l2")            # row normalise
    logger.info(f"Spectral embedding: {Z.shape}")
    return Z.astype(np.float32)


def _dbscan_on_spectral(Z: np.ndarray, cfg: Config) -> np.ndarray:
    """Exact DBSCAN via a chunked BallTree radius search + union-find.

    sklearn's DBSCAN materialises *all* point neighbourhoods from a single
    `radius_neighbors` call before clustering. For N≈500K, if even a modest
    fraction of points sit in dense regions, the combined neighbour-index
    arrays can run into the GB range and trigger MemoryError (independent of
    n_jobs — the allocation happens inside `query_radius` itself). Querying
    the tree in row-chunks keeps at most `chunk` neighbourhoods resident at
    once, and a union-find over (core-point, neighbour) pairs reproduces
    sklearn's exact DBSCAN semantics — same labels, bounded memory.
    """
    N = Z.shape[0]
    eps = cfg.dbscan_eps
    min_samples = cfg.dbscan_min_samples

    tree = BallTree(Z, metric="euclidean")

    chunk = max(1, min(N, cfg.ann_query_chunk_size))
    n_neighbors = np.empty(N, dtype=np.int64)
    parent = np.arange(N, dtype=np.int64)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    is_core = np.zeros(N, dtype=bool)

    for start in tqdm(range(0, N, chunk), desc="Stage2 DBSCAN radius query", unit=" chunk"):
        end = min(start + chunk, N)
        neigh = tree.query_radius(Z[start:end], r=eps, return_distance=False)
        for offset, idxs in enumerate(neigh):
            i = start + offset
            n_neighbors[i] = len(idxs)
            if len(idxs) >= min_samples:
                is_core[i] = True
                for j in idxs:
                    j = int(j)
                    if j != i:
                        union(i, j)

    labels = np.full(N, -1, dtype=np.int64)
    root_to_label: Dict[int, int] = {}
    next_label = 0
    for i in range(N):
        if not is_core[i]:
            continue
        root = find(i)
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        labels[i] = root_to_label[root]

    # Border points: assign to the cluster of any core neighbour (sklearn
    # behaviour — first-found cluster wins, ties broken by scan order).
    for start in tqdm(range(0, N, chunk), desc="Stage2 DBSCAN border assign", unit=" chunk"):
        end = min(start + chunk, N)
        neigh = tree.query_radius(Z[start:end], r=eps, return_distance=False)
        for offset, idxs in enumerate(neigh):
            i = start + offset
            if is_core[i] or labels[i] != -1:
                continue
            for j in idxs:
                j = int(j)
                if is_core[j]:
                    labels[i] = labels[j]
                    break

    return labels


def _within_cluster_variance(member_embs: np.ndarray) -> float:
    """Mean cosine distance of members to their centroid."""
    if len(member_embs) < 2:
        return 0.0
    centroid = member_embs.mean(axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return 0.0
    centroid /= norm
    sims = member_embs @ centroid
    return float(1.0 - sims.mean())


def dynamic_split(labels: np.ndarray, embeddings: np.ndarray,
                  cfg: Config, logger: logging.Logger) -> np.ndarray:
    """Re-cluster any cluster whose within-cluster variance exceeds threshold."""
    new_labels = labels.copy()
    next_id = int(labels.max()) + 1
    unique = [c for c in np.unique(labels) if c >= 0]
    splits = 0
    for cid in unique:
        mask = new_labels == cid
        idx = np.where(mask)[0]
        if len(idx) < 4:
            continue
        embs = embeddings[idx]
        var = _within_cluster_variance(embs)
        if var > cfg.split_var_threshold:
            sub = _dbscan_on_spectral(embs, cfg)
            for sc in np.unique(sub):
                if sc < 0:
                    new_labels[idx[sub == sc]] = -1
                else:
                    new_labels[idx[sub == sc]] = next_id
                    next_id += 1
            splits += 1
    logger.info(f"Dynamic split: {splits} clusters re-split")
    return new_labels


def dynamic_merge(labels: np.ndarray, embeddings: np.ndarray,
                  cfg: Config, logger: logging.Logger) -> np.ndarray:
    """Merge clusters whose centroids are closer than threshold."""
    unique = sorted([c for c in np.unique(labels) if c >= 0])
    if len(unique) < 2:
        return labels

    centroids = np.array([
        normalize(embeddings[labels == c].mean(axis=0, keepdims=True))[0]
        for c in unique
    ])

    new_labels = labels.copy()
    label_map = {c: c for c in unique}
    merges = 0

    for i, ci in enumerate(unique):
        for j in range(i + 1, len(unique)):
            cj = unique[j]
            dist = float(cdist(centroids[i:i+1], centroids[j:j+1],
                               metric="cosine")[0, 0])
            if dist < cfg.merge_dist_threshold:
                root_i = label_map[ci]
                root_j = label_map[cj]
                if root_i != root_j:
                    target = min(root_i, root_j)
                    source = max(root_i, root_j)
                    for k in label_map:
                        if label_map[k] == source:
                            label_map[k] = target
                    merges += 1

    for c in unique:
        new_labels[labels == c] = label_map[c]

    logger.info(f"Dynamic merge: {merges} cluster pairs merged")
    return new_labels


def run_hgce(embeddings: np.ndarray, spectra: List[Spectrum],
             cfg: Config, logger: logging.Logger) -> np.ndarray:
    """Full Stage 2 pipeline → cluster labels (−1 = noise)."""
    W = build_sparse_similarity_graph(embeddings, spectra, cfg, logger)

    n_nonzero = W.nnz
    if n_nonzero == 0:
        logger.warning("Empty graph — assigning all spectra to noise.")
        return np.full(len(spectra), -1, dtype=np.int32)

    N = len(spectra)

    # For large datasets the spectral-embedding+DBSCAN path is impractical
    # (BallTree radius search degenerates at 30D for N>100K). Instead derive
    # initial labels directly from connected components of the already-filtered
    # similarity graph — O(N+E), memory-bounded, and exact given the graph's
    # strict cosine×precursor threshold.  For small datasets (<= large_n_thresh)
    # the traditional spectral path is retained for finer granularity.
    large_n_thresh = cfg.large_n_threshold
    if N <= large_n_thresh:
        k = min(cfg.n_eigenvectors, N - 2)
        Z = compute_spectral_embedding(W, k, logger)
        logger.info("Running adaptive DBSCAN on spectral embedding …")
        labels = _dbscan_on_spectral(Z, cfg)
        work_embs = Z
    else:
        logger.info(f"N={N:,} > {large_n_thresh:,}: using graph connected-components "
                    "as initial clusters (skips memory-intensive spectral DBSCAN) …")
        from scipy.sparse.csgraph import connected_components
        n_comp, comp_labels = connected_components(W, directed=False,
                                                   connection="weak")
        # Treat singleton components (no edges) as noise
        comp_sizes = np.bincount(comp_labels, minlength=n_comp)
        noise_comps = set(np.where(comp_sizes < cfg.dbscan_min_samples)[0])
        labels = np.where(
            np.isin(comp_labels, list(noise_comps)), -1, comp_labels.astype(np.int64)
        )
        # Re-number cluster ids to be contiguous starting from 0
        unique_ids = np.unique(labels[labels >= 0])
        remap = {old: new for new, old in enumerate(unique_ids)}
        labels = np.array(
            [remap[v] if v >= 0 else -1 for v in labels], dtype=np.int64
        )
        work_embs = embeddings

    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    logger.info(f"Initial clustering: {n_clusters} clusters, {n_noise} noise points")

    labels = dynamic_split(labels, work_embs, cfg, logger)
    labels = dynamic_merge(labels, work_embs, cfg, logger)

    n_final = len(set(labels) - {-1})
    n_noise_final = int((labels == -1).sum())
    logger.info(f"Final clusters: {n_final}, noise: {n_noise_final}")
    return labels.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — BIOLOGICALLY-AWARE RERANKER (BAR)
# ─────────────────────────────────────────────────────────────────────────────

def _sarle_bimodality(values: np.ndarray) -> float:
    """Sarle's bimodality coefficient b = (skew²+1) / (kurtosis+3).
    b > 0.555 suggests bimodality (PTM candidate)."""
    if len(values) < 4:
        return 0.0
    s = skew(values)
    k = kurtosis(values, fisher=True)   # excess kurtosis
    denom = k + 3.0
    if denom == 0:
        return 0.0
    return float((s ** 2 + 1.0) / denom)


def _consensus_spectrum(spectra: List[Spectrum], weights: np.ndarray,
                        binner: SpectrumBinner) -> Tuple[np.ndarray, np.ndarray]:
    """Similarity-weighted mean spectrum in bin space, returned as sparse peaks."""
    weights = weights / weights.sum()
    acc = np.zeros(binner.n_bins, dtype=np.float32)
    for s, w in zip(spectra, weights):
        acc += w * binner.bin_spectrum(s)
    # Convert dense bin vector back to (mz, intensity) sparse peaks
    nonzero = acc > 1e-6
    bin_indices = np.where(nonzero)[0]
    mz_vals = binner.mz_min + (bin_indices + 0.5) * binner.bin_width
    return mz_vals.astype(np.float32), acc[nonzero].astype(np.float32)


def run_bar(labels: np.ndarray, embeddings: np.ndarray,
            spectra: List[Spectrum], cfg: Config,
            logger: logging.Logger) -> List[Cluster]:
    """Stage 3 — score every cluster and flag PTM candidates."""
    binner = SpectrumBinner(cfg)
    unique_labels = sorted([c for c in np.unique(labels) if c >= 0])
    clusters: List[Cluster] = []

    for cid in tqdm(unique_labels, desc="Stage3 scoring", unit="cluster"):
        idx = np.where(labels == cid)[0]
        member_embs = embeddings[idx]           # (n, d)
        member_specs = [spectra[i] for i in idx]

        # Step 1-3: centroid, variance, mean similarity
        centroid = member_embs.mean(axis=0)
        cnorm = np.linalg.norm(centroid)
        if cnorm > 0:
            centroid = centroid / cnorm

        sims = member_embs @ centroid           # cosine similarities (≈ L2 normalised)
        mean_sim = float(sims.mean())
        within_var = float(1.0 - mean_sim)

        # Step 4: quality score
        q = mean_sim * math.exp(-within_var / cfg.sigma_q)

        # Step 5: confidence tier
        if q >= cfg.high_q_threshold:
            conf = "HIGH"
        elif q >= cfg.medium_q_threshold:
            conf = "MEDIUM"
        else:
            conf = "LOW"

        # Step 6: PTM flag via Sarle bimodality on similarity distribution
        b_coeff = _sarle_bimodality(sims)
        ptm_flag = b_coeff > 0.555

        # Step 7: consensus spectrum
        weights = np.clip(sims, 0.01, None)
        cons_mz, cons_int = _consensus_spectrum(member_specs, weights, binner)

        clusters.append(Cluster(
            cluster_id=int(cid),
            member_ids=idx.tolist(),
            centroid=centroid,
            quality_score=round(q, 4),
            confidence=conf,
            ptm_flag=ptm_flag,
            within_var=round(within_var, 4),
            mean_similarity=round(mean_sim, 4),
            consensus_mz=cons_mz,
            consensus_intensity=cons_int,
        ))

    # Sort by quality descending
    clusters.sort(key=lambda c: c.quality_score, reverse=True)
    n_high = sum(1 for c in clusters if c.confidence == "HIGH")
    n_ptm = sum(1 for c in clusters if c.ptm_flag)
    logger.info(f"Stage 3 complete: {len(clusters)} clusters scored, "
                f"{n_high} HIGH confidence, {n_ptm} PTM-flagged")
    return clusters


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def write_cluster_tsv(clusters: List[Cluster], spectra: List[Spectrum],
                      labels: np.ndarray, output_dir: str) -> None:
    path = os.path.join(output_dir, "clusters.tsv")
    with open(path, "w") as fh:
        fh.write("cluster_id\tscan_id\tprecursor_mz\tprecursor_charge\t"
                 "quality_score\tconfidence\tptm_flag\twithin_var\tmean_similarity\n")
        for cl in clusters:
            for idx in cl.member_ids:
                s = spectra[idx]
                fh.write(f"{cl.cluster_id}\t{s.scan_id}\t{s.precursor_mz:.4f}\t"
                         f"{s.precursor_charge}\t{cl.quality_score}\t"
                         f"{cl.confidence}\t{cl.ptm_flag}\t"
                         f"{cl.within_var:.4f}\t{cl.mean_similarity:.4f}\n")

    # Noise / unclustered
    noise_path = os.path.join(output_dir, "noise_spectra.tsv")
    with open(noise_path, "w") as fh:
        fh.write("scan_id\tprecursor_mz\tprecursor_charge\n")
        for i, lbl in enumerate(labels):
            if lbl == -1:
                s = spectra[i]
                fh.write(f"{s.scan_id}\t{s.precursor_mz:.4f}\t{s.precursor_charge}\n")


def write_consensus_mgf(clusters: List[Cluster], output_dir: str) -> None:
    """Write consensus spectra of HIGH/MEDIUM clusters as an MGF file."""
    path = os.path.join(output_dir, "consensus_spectra.mgf")
    written = 0
    with open(path, "w") as fh:
        for cl in clusters:
            if cl.confidence not in ("HIGH", "MEDIUM"):
                continue
            if cl.consensus_mz is None or len(cl.consensus_mz) == 0:
                continue
            fh.write("BEGIN IONS\n")
            fh.write(f"TITLE=cluster_{cl.cluster_id}_n{len(cl.member_ids)}\n")
            fh.write(f"QUALITY={cl.quality_score}\n")
            fh.write(f"CONFIDENCE={cl.confidence}\n")
            fh.write(f"PTM_FLAG={cl.ptm_flag}\n")
            for mz, inten in zip(cl.consensus_mz, cl.consensus_intensity):
                fh.write(f"{mz:.4f} {inten:.4f}\n")
            fh.write("END IONS\n\n")
            written += 1


def write_summary(clusters: List[Cluster], spectra: List[Spectrum],
                  labels: np.ndarray, output_dir: str,
                  elapsed: float) -> None:
    n_total = len(spectra)
    n_noise = int((labels == -1).sum())
    n_clustered = n_total - n_noise
    n_high = sum(1 for c in clusters if c.confidence == "HIGH")
    n_med = sum(1 for c in clusters if c.confidence == "MEDIUM")
    n_low = sum(1 for c in clusters if c.confidence == "LOW")
    n_ptm = sum(1 for c in clusters if c.ptm_flag)
    sizes = [len(c.member_ids) for c in clusters]
    avg_size = np.mean(sizes) if sizes else 0.0
    max_size = max(sizes) if sizes else 0

    path = os.path.join(output_dir, "summary.txt")
    lines = [
        "=" * 60,
        "ProteoCLUST — Run Summary",
        "=" * 60,
        f"Input spectra      : {n_total:>10,}",
        f"Clustered          : {n_clustered:>10,}  ({100*n_clustered/max(n_total,1):.1f}%)",
        f"Noise / unclustered: {n_noise:>10,}",
        f"Total clusters     : {len(clusters):>10,}",
        f"  HIGH confidence  : {n_high:>10,}",
        f"  MEDIUM confidence: {n_med:>10,}",
        f"  LOW confidence   : {n_low:>10,}",
        f"PTM-flagged        : {n_ptm:>10,}",
        f"Avg cluster size   : {avg_size:>10.1f}",
        f"Max cluster size   : {max_size:>10,}",
        f"Wall time (s)      : {elapsed:>10.1f}",
        "=" * 60,
    ]
    text = "\n".join(lines)
    print("\n" + text)
    with open(path, "w") as fh:
        fh.write(text + "\n")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def run_pipeline(cfg: Config) -> List[Cluster]:
    t0 = time.time()
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = setup_logging(cfg.output_dir, cfg.verbose)
    logger.info(f"ProteoCLUST starting — input: {cfg.mgf_path}")
    logger.info(f"Output directory: {cfg.output_dir}")

    np.random.seed(cfg.seed)

    # ── Parse ─────────────────────────────────────────────────────────────────
    logger.info("Loading spectra …")
    spectra = load_spectra(cfg.mgf_path, cfg.max_spectra, cfg.verbose)
    if not spectra:
        logger.error("No spectra loaded — check MGF path and format.")
        return []
    logger.info(f"Loaded {len(spectra):,} spectra")

    # ── Checkpointing ─────────────────────────────────────────────────────────
    # Resuming a long run after a sleep/crash skips already-completed stages by
    # loading cached intermediate arrays from <output_dir>/checkpoints/.
    ckpt_dir = os.path.join(cfg.output_dir, "checkpoints")
    os.makedirs(ckpt_dir, exist_ok=True)
    emb_ckpt = os.path.join(ckpt_dir, f"embeddings_n{len(spectra)}.npy")
    lbl_ckpt = os.path.join(ckpt_dir, f"labels_n{len(spectra)}.npy")

    # ── Stage 1 ───────────────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STAGE 1 — Universal Spectrum Encoder")
    if cfg.resume and os.path.exists(emb_ckpt):
        logger.info(f"Resuming: loading cached embeddings from {emb_ckpt}")
        embeddings = np.load(emb_ckpt)
    else:
        encoder = UniversalSpectrumEncoder(cfg)
        embeddings = encoder.encode(spectra, logger)
        del encoder
        gc.collect()
        if cfg.resume:
            np.save(emb_ckpt, embeddings)
            logger.info(f"Checkpoint saved: {emb_ckpt}")

    # ── Stage 2 ───────────────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STAGE 2 — Hierarchical Graph Clustering Engine")
    if cfg.resume and os.path.exists(lbl_ckpt):
        logger.info(f"Resuming: loading cached cluster labels from {lbl_ckpt}")
        labels = np.load(lbl_ckpt)
    else:
        labels = run_hgce(embeddings, spectra, cfg, logger)
        gc.collect()
        if cfg.resume:
            np.save(lbl_ckpt, labels)
            logger.info(f"Checkpoint saved: {lbl_ckpt}")

    # ── Stage 3 ───────────────────────────────────────────────────────────────
    logger.info("=" * 50)
    logger.info("STAGE 3 — Biologically-Aware Reranker")
    clusters = run_bar(labels, embeddings, spectra, cfg, logger)

    # ── Write outputs ─────────────────────────────────────────────────────────
    logger.info("Writing outputs …")
    write_cluster_tsv(clusters, spectra, labels, cfg.output_dir)
    write_consensus_mgf(clusters, cfg.output_dir)
    elapsed = time.time() - t0
    write_summary(clusters, spectra, labels, cfg.output_dir, elapsed)
    logger.info(f"Done in {elapsed:.1f} s")
    return clusters


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="proteoclust",
        description="ProteoCLUST: Spectral Clustering for MGF Proteomics Data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--mgf", required=True, help="Input MGF file path")
    p.add_argument("--out", default="proteoclust_out", help="Output directory")

    # Encoder
    p.add_argument("--mz-max", type=float, default=1500.0)
    p.add_argument("--bin-width", type=float, default=0.02)
    p.add_argument("--top-k", type=int, default=150)
    p.add_argument("--embed-dim", type=int, default=128)
    p.add_argument("--n-heads", type=int, default=4)
    p.add_argument("--n-layers", type=int, default=2)
    p.add_argument("--batch", type=int, default=2048, dest="batch_size")

    # Clustering
    p.add_argument("--precursor-tol", type=float, default=1.5,
                   help="Precursor mass tolerance (Da)")
    p.add_argument("--edge-threshold", type=float, default=0.70)
    p.add_argument("--n-eigvec", type=int, default=30)
    p.add_argument("--dbscan-eps", type=float, default=0.15)
    p.add_argument("--dbscan-min", type=int, default=2)

    # Runtime
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--max-spectra", type=int, default=None,
                   help="Truncate input for quick testing")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Cache Stage 1/2 results in <out>/checkpoints and "
                        "reuse them on a subsequent run (e.g. after sleep/crash)")
    return p


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    cfg = Config(
        mgf_path=args.mgf,
        output_dir=args.out,
        mz_max=args.mz_max,
        bin_width=args.bin_width,
        top_k_peaks=args.top_k,
        embed_dim=args.embed_dim,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        precursor_tol_da=args.precursor_tol,
        edge_threshold=args.edge_threshold,
        n_eigenvectors=args.n_eigvec,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min,
        n_jobs=args.jobs,
        device=args.device,
        max_spectra=args.max_spectra,
        verbose=not args.quiet,
        resume=args.resume,
    )
    run_pipeline(cfg)


# ─────────────────────────────────────────────────────────────────────────────
# SPYDER / NOTEBOOK ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
# To run directly from Spyder without the terminal, edit the path below and
# call run_pipeline(cfg) in the console, or just run this file with F5.

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        # Called from terminal with arguments
        main()
    else:
        # ── Spyder / direct execution — edit these paths ───────────────────
        cfg = Config(
            mgf_path="D:\\Shraddha\\Original MGFs\\COREAD\\20201022_FS_Choudhary_LMS2_FS03_MS2_16plex.mgf",      # ← change to your MGF path
            output_dir="D:\\Shraddha\\ProteoClust\\proteoclust_out",
			resume=True,
            # Adjust for your dataset size:
            # For COREAD (2.7 GB) use batch_size=1024, ann_n_neighbours=50
            # For UPS (128 MB)    use batch_size=2048, ann_n_neighbours=100
            batch_size=2048,
            ann_n_neighbours=100,
            max_spectra=None,                      # set e.g. 5000 for a quick test
            verbose=True,
        )
        clusters = run_pipeline(cfg)
