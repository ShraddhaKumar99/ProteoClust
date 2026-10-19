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
from typing import List, Tuple, Optional, Dict, Iterator, Union
import math
import gc
import threading

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ─── third-party ─────────────────────────────────────────────────────────────
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import cdist
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
    # A single path, comma-separated paths, a directory of .mgf files, or a
    # list of paths/directories. Multiple inputs are clustered together,
    # enabling cross-experiment / cross-MS-run clustering.
    mgf_path: Union[str, List[str]] = ""
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
    split_var_threshold: float = 0.04   # dynamic split threshold — ONLY valid
                                         # for the low-dim spectral embedding Z;
                                         # see split_var_threshold_embedding for
                                         # the raw-embedding-space equivalent
    merge_dist_threshold: float = 0.05  # dynamic merge threshold
    dynamic_split_enabled: bool = True  # FIX A: set False to skip the post-hoc
                                         # dynamic_split cleanup entirely.
                                         # Recommended False for the HDBSCAN path
                                         # if you'd rather trust HDBSCAN's own
                                         # density-based cluster selection than
                                         # this heuristic re-split step.
    dynamic_merge_enabled: bool = False  # set False to skip dynamic_merge
    dbscan_eps_embedding: Optional[float] = None
                                         # FIX A: euclidean eps used by
                                         # dynamic_split when it's given RAW
                                         # (non-spectral) embeddings — the
                                         # HDBSCAN path, or the large-N graph
                                         # connected-components path.
                                         # cfg.dbscan_eps (0.15) is tuned for
                                         # the compressed spectral embedding Z
                                         # ONLY and is far too tight here — it
                                         # silently turned most re-split
                                         # cluster members into noise. Leave
                                         # as None to auto-derive from
                                         # edge_threshold:
                                         #   eps = sqrt(2*(1-edge_threshold))
                                         # or set explicitly to override.
    split_var_threshold_embedding: float = 0.30
                                         # FIX A: within-cluster-variance
                                         # threshold used by dynamic_split on
                                         # raw embeddings. split_var_threshold
                                         # (0.04) assumes the near-perfect
                                         # (>0.99 similarity) clustering that
                                         # spectral eigen-decomposition
                                         # produces; a typical SupCon-trained
                                         # transformer embedding's genuine
                                         # same-peptide similarity is more
                                         # like 0.7-0.85, so 0.04 flagged
                                         # nearly every real cluster for
                                         # re-splitting. Tune this against your
                                         # own pretrain run's val_intra_sim
                                         # (use ~ 1 - val_intra_sim, with some
                                         # margin) rather than trusting 0.30
                                         # blindly.
    max_edges_per_node: int = 50  # cap edges to keep graph sparse
    ann_n_neighbours: int = 100   # approximate NN search width
    ann_query_chunk_size: int = 2000  # rows per kneighbors() call (memory cap)
    large_n_threshold: int = 100_000  # above this use graph connected-components instead of spectral DBSCAN
    use_hdbscan: bool = False          # use HDBSCAN on embeddings directly (better for transformer embeddings)
    hdbscan_min_cluster_size: int = 2  # HDBSCAN min_cluster_size
    hdbscan_min_samples: int = 1       # HDBSCAN min_samples (controls noise sensitivity)
    hdbscan_epsilon: float = 0.0       # cluster_selection_epsilon (0=auto); set ~0.72 for pretrained embeddings
    hdbscan_cluster_selection_method: str = 'leaf'  # leaf gives tighter clusters than eom for pretrained embeddings
                                        # Tested both options against synthetic
                                        # data with KNOWN ground-truth clusters
                                        # (matching this checkpoint's real
                                        # separation stats): 'eom' correctly
                                        # recovered ~52 clusters vs 50 true
                                        # ones; 'leaf' badly over-fragmented
                                        # into 111 tiny clusters with most
                                        # points pushed to noise. The common
                                        # claim that 'eom' produces pathological
                                        # giant clusters is real in general but
                                        # NOT universal — it didn't hold on this
                                        # embedding space, so 'eom' is the
                                        # tested-better default here. If your
                                        # own real-data summary.txt shows a
                                        # single implausibly large cluster (well
                                        # above what a handful of replicate
                                        # spectra per peptide would produce),
                                        # try 'leaf' and COMPARE your own
                                        # results — don't take either default
                                        # on faith.

    # ── Stage 2 — Hyperdimensional pre-filter ───────────────────────────────
    # HyperSpec-style binary-hash bucketing: project embeddings onto random
    # hyperplanes, group spectra into buckets by sign pattern, and only run
    # the (expensive) brute-force cosine kNN search within each bucket. This
    # turns the O(N^2) candidate-pair search into ~O(N * bucket_size) and is
    # the key enabler for scaling beyond ~10^6 spectra.
    use_hd_prefilter: bool = False
    hd_prefilter_min_n: int = 20_000   # below this, plain brute-force kNN is fine
    hd_target_bucket_size: int = 2_000  # aim for ~this many spectra per bucket
    hd_max_bits: int = 20              # cap on number of hash bits (buckets = 2^bits)
    hd_seed: int = 42
    hd_n_repeats: int = 6               # independent hash tables (multi-hash repetition)
    # R=6 chosen empirically on the 506K-spectrum COREAD run: matches brute-
    # force PTM-flag recall (2,635 vs 2,629) and HIGH-confidence cluster count
    # (3,913 vs 3,586) at 1.36x speedup (5,643s vs 7,657s), while R=2/R=4
    # under-recover MEDIUM/LOW clusters and PTM flags disproportionately.
    hd_bucketed_min_repeat_votes: Optional[int] = 2
                                         # FIX G: min. number of the R
                                         # hd_n_repeats runs that must agree
                                         # a pair belongs together before
                                         # _hdbscan_bucketed merges them.
                                         # Tested on synthetic data with
                                         # known ground truth (200 true
                                         # clusters, R=6): min_votes=2 gave
                                         # 202 found/12 noise/zero chaining —
                                         # clearly better than strict
                                         # majority (4/6: 258 found/528
                                         # noise) while still safely above
                                         # the floor of 1, which is the exact
                                         # old (buggy) any-repeat-union
                                         # behavior that produced a real
                                         # ~104K-member merged cluster on
                                         # 500K real data. This field always
                                         # has a floor of 2 regardless of
                                         # what you set it to — it can never
                                         # silently regress to that bug. If
                                         # your real-data results still show
                                         # more noise than expected, try 3;
                                         # if you see any sign of chaining
                                         # (implausibly large/mixed
                                         # clusters), that would be
                                         # surprising given this floor, but
                                         # raise back toward strict majority
                                         # (hd_n_repeats//2 + 1) as a next
                                         # step.

    # ── Stage 3 — Reranker ───────────────────────────────────────────────────
    sigma_q: float = 0.10         # quality score temperature
    high_q_threshold: float = 0.80
    medium_q_threshold: float = 0.50

    # ── Stage 3 — Mixed-cluster (PTM/heterogeneity) splitting ────────────────
    # A "cluster" produced by Stage 2 can contain spectra of a single peptide
    # in more than one chemical state (e.g. modified vs. unmodified) or, in
    # the worst case, spectra of entirely unrelated peptides that happened to
    # pass the similarity/precursor filters. Building one consensus spectrum
    # across such a mixture produces a low-quality consensus that scores
    # poorly in a downstream database search. To avoid this, each cluster is
    # first partitioned into "mass groups" (sub-populations whose neutral
    # precursor masses agree within `mass_group_gap_da`). If more than one
    # mass group is found, each group is emitted as its own cluster:
    #   - if the mass difference between two groups matches a known PTM delta
    #     AND fragment-ion evidence confirms the shift, both groups are
    #     flagged ptm_flag=True with the matching PTM label (a genuine
    #     modified/unmodified pair within one peptide-spectrum cluster).
    #   - otherwise, both groups are flagged heterogeneous_flag=True
    #     (likely unrelated spectra clustered together / false clustering).
    mass_group_gap_da: float = 0.5     # LEGACY neutral-mass gap (Da); only used
                                        # when precursor_tol_ppm <= 0
    precursor_tol_ppm: float = 10.0    # v4: ppm-level precursor tolerance. When > 0
                                        # it (a) makes BAR split each cluster into
                                        # mass groups whose neutral-mass SPAN is
                                        # <= tol (anchored to the group's lightest
                                        # member, so no chaining), and (b) replaces
                                        # the 1.5 Da hard filter / linear precursor
                                        # weight in the Stage-2 graph with a ppm
                                        # hard filter (weight = cosine). Set 0 to
                                        # restore the v3 (Da-level) behaviour.
    ptm_fragment_mz_tol: float = 0.02  # Da tolerance for fragment-ion matching
    ptm_min_unshifted_matches: int = 2 # min. shared (unshifted) fragment peaks
    ptm_min_shifted_matches: int = 1   # min. peaks shifted by the delta mass

    # ── Runtime ──────────────────────────────────────────────────────────────
    n_jobs: int = -1              # -1 = all cores
    device: str = "cpu"           # "cpu" or "cuda"
    seed: int = 42
    verbose: bool = True
    max_spectra: Optional[int] = None   # truncate for testing
    use_transformer: bool = False       # set False to force NumPy fallback encoder
    encoder_weights: str = ""           # path to pretrained_encoder.pt; if set,
                                        # use_transformer is forced True and weights
                                        # are loaded from this checkpoint file
    use_encoder: bool = True            # set False for true No-USE ablation:
                                        # LSH hash codes computed from raw 72,500-bin
                                        # vectors (batchwise); 128-dim projection still
                                        # used for within-bucket kNN and Stage 3.

    # ── Preprocessing (falcon-style, https://github.com/bittremieux-lab/falcon) ──
    quality_filter_enabled: bool = False    # SAFE to enable at any time, no
                                             # retrain needed — this only
                                             # decides which spectra get
                                             # encoded at all, not how
                                             # retained spectra are converted
                                             # to embeddings. Discards
                                             # obviously low-information
                                             # spectra (too few peaks, too
                                             # narrow an m/z range to carry
                                             # real sequence information)
                                             # before they can pollute
                                             # clusters or waste compute.
    quality_min_peaks: int = 5
    quality_min_mz_range: float = 250.0     # Da

    remove_precursor_peaks: bool = False    # DO NOT enable this against
                                             # pretrained_encoder_v8.pt (or
                                             # any existing checkpoint)
                                             # without retraining first. This
                                             # changes what the encoder
                                             # actually sees at inference
                                             # time vs. what it was trained
                                             # on — a silent train/inference
                                             # mismatch that degrades results
                                             # without throwing any error
                                             # (exactly the class of bug the
                                             # charge-grouping fix earlier in
                                             # this project's history caught
                                             # and fixed). If you enable this,
                                             # you MUST also enable the
                                             # matching flag in
                                             # pretrain_transformer_fixed.py
                                             # and train a fresh checkpoint
                                             # before using it here.
    precursor_removal_tol_da: float = 1.5


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
    source_file: str = ""

@dataclass
class Cluster:
    cluster_id: int
    member_ids: List[int]          # indices into the global spectrum list
    centroid: Optional[np.ndarray] = None
    quality_score: float = 0.0
    confidence: str = "LOW"
    ptm_flag: bool = False
    ptm_delta_label: str = ""
    ptm_delta_mass: float = 0.0
    within_var: float = 0.0
    mean_similarity: float = 0.0
    consensus_mz: Optional[np.ndarray] = None
    consensus_intensity: Optional[np.ndarray] = None
    parent_cluster_id: int = -1     # original (pre-split) cluster id, if this
                                     # cluster was produced by splitting a
                                     # mass-heterogeneous cluster
    heterogeneous_flag: bool = False  # True if this cluster was split off from
                                       # a parent whose mass groups did NOT
                                       # match a known PTM delta (likely
                                       # unrelated spectra clustered together)
    n_source_files: int = 1         # number of distinct input files
                                     # contributing members to this cluster
    consensus_precursor_mz: float = 0.0   # weighted-mean precursor m/z for MGF output
    consensus_precursor_charge: int = 0    # most-common precursor charge for MGF output
    consensus_rt: float = 0.0             # mean retention time (s) for MGF output


# ─────────────────────────────────────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────────────────────────────────────

def setup_logging(output_dir: str, verbose: bool) -> logging.Logger:
    """FIX B: build the logger directly instead of via logging.basicConfig().

    basicConfig() is a no-op after the first time it's called in a process
    (unless force=True is passed) — so in a persistent session (e.g. running
    this file repeatedly from Spyder/Jupyter without restarting the kernel),
    every run AFTER the first silently fails to attach its FileHandler: the
    log file for output_dir gets created (0 bytes) but nothing is ever
    written to it, because the root logger already had handlers from the
    first run and basicConfig() refused to reconfigure it. Explicitly
    clearing and rebuilding the named logger's handlers on every call avoids
    this — each run's log file gets its own correctly-attached handler
    regardless of how many times this has already run in the session.
    """
    os.makedirs(output_dir, exist_ok=True)
    fmt = "%(asctime)s  %(levelname)-7s  %(message)s"
    logger = logging.getLogger("ProteoCLUST")
    logger.setLevel(logging.INFO)
    logger.propagate = False  # don't also duplicate through the root logger
    logger.handlers.clear()   # drop any handlers from a previous run this session

    file_handler = logging.FileHandler(os.path.join(output_dir, "proteoclust.log"))
    file_handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(file_handler)

    if verbose:
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(stream_handler)

    return logger


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


def _resolve_mgf_paths(path_or_paths: Union[str, List[str]]) -> List[str]:
    """Expand a single path, a comma-separated list of paths, a directory, or
    a list of paths/directories into a flat list of .mgf file paths. This
    enables clustering spectra across multiple experiments/runs together
    (e.g. to find unassigned spectra that recur across datasets)."""
    if isinstance(path_or_paths, str):
        raw = [p.strip() for p in path_or_paths.split(",") if p.strip()]
    else:
        raw = list(path_or_paths)
    paths: List[str] = []
    for p in raw:
        if os.path.isdir(p):
            paths.extend(sorted(str(f) for f in Path(p).glob("*.mgf")))
        else:
            paths.append(p)
    return paths


def passes_quality_filter(mz: np.ndarray, min_peaks: int, min_mz_range: float) -> bool:
    """falcon-style quality pre-filter (Bittremieux et al.) — discard
    low-information spectra (too few peaks, or peaks spanning too narrow an
    m/z range to carry real fragment-ion sequence information) before they
    can pollute clusters. Safe to enable at any time — this only decides
    whether a spectrum is used at all, not how used spectra get encoded, so
    it doesn't require retraining the encoder."""
    if len(mz) < min_peaks:
        return False
    if len(mz) > 0 and (mz.max() - mz.min()) < min_mz_range:
        return False
    return True


def remove_precursor_peaks(mz: np.ndarray, intensity: np.ndarray,
                           precursor_mz: float, charge: int,
                           tol_da: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    """falcon-style precursor-peak removal — strips peaks near the
    precursor m/z and its charge-reduced forms (z=1..charge), which can
    otherwise inflate similarity between unrelated peptides that happen to
    share a precursor mass. CAUTION: changes encoder input — see
    Config.remove_precursor_peaks docstring before enabling."""
    if len(mz) == 0 or charge <= 0:
        return mz, intensity
    mask = np.ones(len(mz), dtype=bool)
    neutral_mass = precursor_mz * charge - PROTON_MASS * charge
    for z in range(1, charge + 1):
        mz_at_z = neutral_mass / z + PROTON_MASS
        mask &= np.abs(mz - mz_at_z) > tol_da
    return mz[mask], intensity[mask]


def load_spectra(path: Union[str, List[str]], max_spectra: Optional[int] = None,
                 verbose: bool = True, quality_filter_enabled: bool = False,
                 quality_min_peaks: int = 5, quality_min_mz_range: float = 250.0) -> List[Spectrum]:
    """Stream-parse one or more MGF files and return a list of Spectrum
    objects, each tagged with its source file. Uses pyteomics when available
    for robustness, otherwise falls back to the built-in parser.

    Accepts a single path, a comma-separated list of paths, a directory of
    .mgf files, or a list of paths/directories — allowing spectra from
    multiple experiments/MS runs to be clustered together."""
    parse_fn = _parse_mgf_pyteomics if PYTEOMICS_AVAILABLE else _parse_mgf_builtin
    spectra: List[Spectrum] = []
    n_dropped = 0
    for mgf_path in _resolve_mgf_paths(path):
        desc = f"Parsing {Path(mgf_path).name}"
        iterator = parse_fn(mgf_path)
        if verbose:
            iterator = tqdm(iterator, desc=desc, unit=" spec")
        source_name = Path(mgf_path).name
        for s in iterator:
            if quality_filter_enabled and not passes_quality_filter(
                    s.mz_array, quality_min_peaks, quality_min_mz_range):
                n_dropped += 1
                continue
            s.source_file = source_name
            spectra.append(s)
            if max_spectra and len(spectra) >= max_spectra:
                if quality_filter_enabled and n_dropped:
                    logging.getLogger("ProteoCLUST").info(
                        f"Quality filter: {n_dropped:,} spectra dropped "
                        f"(< {quality_min_peaks} peaks or < {quality_min_mz_range} Da range)")
                return spectra
    if quality_filter_enabled and n_dropped:
        logging.getLogger("ProteoCLUST").info(
            f"Quality filter: {n_dropped:,} spectra dropped "
            f"(< {quality_min_peaks} peaks or < {quality_min_mz_range} Da range)")
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
                              mz_min: float, mz_max: float,
                              remove_precursor: bool = False,
                              precursor_tol_da: float = 1.5) -> np.ndarray:
    """Return (top_k, 2) float32 array of [norm_mz, norm_intensity].
    Rows beyond actual peak count are zero-padded.

    `remove_precursor`/`precursor_tol_da` MUST match whatever
    pretrain_transformer_fixed.py used to train the checkpoint currently
    being loaded — see Config.remove_precursor_peaks docstring. Defaults to
    False (matches pretrained_encoder_v8.pt and every checkpoint trained so
    far in this project)."""
    tokens = np.zeros((top_k, 2), dtype=np.float32)
    mz = spec.mz_array
    intensity = spec.intensity_array
    if remove_precursor and len(mz) > 0:
        mz, intensity = remove_precursor_peaks(
            mz, intensity, spec.precursor_mz, spec.precursor_charge, precursor_tol_da)
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

        if cfg.encoder_weights and cfg.remove_precursor_peaks:
            logging.getLogger("ProteoCLUST").warning(
                "remove_precursor_peaks=True is set together with a pretrained "
                "encoder_weights checkpoint. Unless this exact checkpoint was "
                "TRAINED with precursor removal also enabled in "
                "pretrain_transformer_fixed.py, this is a silent train/inference "
                "mismatch — the encoder will see spectra shaped differently than "
                "what it learned on, degrading results WITHOUT throwing any "
                "error. Only proceed if you've retrained a matching checkpoint.")

        if cfg.encoder_weights and os.path.isfile(cfg.encoder_weights):
            if not TORCH_AVAILABLE:
                raise RuntimeError(
                    "encoder_weights specified but PyTorch is not installed.")
            self.model = LightTransformerEncoder(cfg).to(self.device)
            ckpt = torch.load(cfg.encoder_weights, map_location=self.device)
            state = ckpt.get("model_state_dict", ckpt)
            # strict=False: proj_head exists in pretrained checkpoint but not
            # in the inference encoder — we only need the backbone weights
            # (input_proj, encoder, proj_norm). proj_head is discarded.
            missing, unexpected = self.model.load_state_dict(state, strict=False)
            if missing:
                raise RuntimeError(f"Missing backbone keys in checkpoint: {missing}")
            if unexpected:
                # proj_head keys are expected to be unexpected — log but don't fail
                proj_head_keys = [k for k in unexpected if "proj_head" in k]
                other_keys     = [k for k in unexpected if "proj_head" not in k]
                if other_keys:
                    raise RuntimeError(f"Unexpected non-proj_head keys: {other_keys}")
                logger_msg = (f"Loaded pretrained backbone weights. "
                              f"Discarded proj_head ({len(proj_head_keys)} keys) "
                              f"— correct, proj_head is training-only.")
            self.model.eval()
            self._torch = True
            self._pretrained = True
        elif TORCH_AVAILABLE and cfg.use_transformer:
            self.model = LightTransformerEncoder(cfg).to(self.device)
            self.model.eval()
            self._torch = True
            self._pretrained = False
        else:
            self.model = NumpyFallbackEncoder(cfg)
            self._torch = False
            self._pretrained = False

    def _make_peak_token_batch(self, spectra: List[Spectrum]):
        """Return tokens (B, K, 2) and padding mask (B, K)."""
        K = self.cfg.top_k_peaks
        tokens = np.stack([
            _spectrum_to_peak_tokens(s, K, self.cfg.mz_min, self.cfg.mz_max,
                                     remove_precursor=self.cfg.remove_precursor_peaks,
                                     precursor_tol_da=self.cfg.precursor_removal_tol_da)
            for s in spectra
        ])  # (B, K, 2)
        # A token is padding if both features are zero
        pad_mask = (tokens.sum(axis=-1) == 0)   # (B, K)
        return tokens, pad_mask

    def encode(self, spectra: List[Spectrum], logger: logging.Logger) -> np.ndarray:
        if self._torch and self._pretrained:
            logger.info(f"Stage 1 encoder : LightTransformer "
                        f"[pretrained weights: {self.cfg.encoder_weights}]")
        elif self._torch:
            logger.info("Stage 1 encoder : LightTransformer "
                        "[random-initialised — no pretrained weights]")
        else:
            logger.info("Stage 1 encoder : NumpyFallback (random projection)")
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

def _run_with_heartbeat(fn, desc: str, logger: logging.Logger, interval: float = 30.0):
    """Run fn() in a background thread, logging an elapsed-time heartbeat
    every `interval` seconds while it's running.

    Some calls (like HDBSCAN.fit_predict()) give ZERO console output while
    running, whether they take 30 seconds or 5 hours — a 16,740s run and a
    genuine deadlock look identical until one of them finishes. This makes
    long-but-legitimate runs indistinguishable from hangs, which is exactly
    what made an earlier 4.6-hour HDBSCAN run impossible to sanity-check
    while it was in progress. This wraps any such call with a periodic
    "still running, Ns elapsed" log line so you always know it's alive.
    """
    result: Dict[str, object] = {}
    exc: Dict[str, BaseException] = {}

    def target():
        try:
            result["value"] = fn()
        except BaseException as e:  # noqa: BLE001 - re-raised on the caller's thread below
            exc["error"] = e

    t0 = time.time()
    thread = threading.Thread(target=target, daemon=True)
    thread.start()
    while thread.is_alive():
        thread.join(timeout=interval)
        if thread.is_alive():
            logger.info(f"{desc}: still running, {int(time.time() - t0):,}s elapsed …")
    if "error" in exc:
        raise exc["error"]
    logger.info(f"{desc}: completed in {int(time.time() - t0):,}s")
    return result["value"]


def _precursor_score(pmz_i: float, pmz_j: float, charge_i: int, charge_j: int,
                     tol: float) -> float:
    """Returns a [0,1] precursor compatibility score."""
    if charge_i != charge_j and min(charge_i, charge_j) > 0:
        return 0.0
    delta = abs(pmz_i - pmz_j)
    return max(0.0, 1.0 - delta / tol)


def _hd_hash_codes(embeddings: np.ndarray, n_bits: int, seed: int) -> np.ndarray:
    """HyperSpec-style hyperdimensional binary hash: project each embedding
    onto `n_bits` random hyperplanes and pack the resulting sign bits into a
    single integer bucket id per spectrum. Spectra with similar embeddings
    (small angle) land in the same or neighbouring buckets with high
    probability (standard random-hyperplane LSH for cosine similarity)."""
    rng = np.random.default_rng(seed)
    d = embeddings.shape[1]
    planes = rng.standard_normal((d, n_bits)).astype(np.float32)
    bits = (embeddings @ planes) > 0          # (N, n_bits) bool
    weights = (1 << np.arange(n_bits, dtype=np.int64))
    return (bits.astype(np.int64) @ weights)


def _hd_hash_codes_from_bins(spectra: List, binner, n_bits: int, seed: int,
                              batch_size: int = 2000) -> np.ndarray:
    """True No-USE ablation: compute LSH hash codes directly from raw 72,500-bin
    spectrum vectors without any learned or random projection. Processes in
    batches to avoid materialising the full (N, 72500) matrix (~146 GB for N=506K).
    Random hyperplane matrix is (72500, n_bits) — ~2.3 MB, fits in RAM."""
    N = len(spectra)
    rng = np.random.default_rng(seed)
    n_bins = binner.n_bins
    planes = rng.standard_normal((n_bins, n_bits)).astype(np.float32)
    weights = (1 << np.arange(n_bits, dtype=np.int64))
    codes = np.zeros(N, dtype=np.int64)
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        batch_bins = binner.bin_batch(spectra[start:end])   # (batch, n_bins)
        bits = (batch_bins @ planes) > 0                    # (batch, n_bits)
        codes[start:end] = bits.astype(np.int64) @ weights
    return codes


def _bucket_ranges(sorted_codes: np.ndarray) -> Iterator[Tuple[int, int]]:
    """Yield (start, end) index ranges of contiguous equal-valued runs in a
    sorted array."""
    n = len(sorted_codes)
    start = 0
    for i in range(1, n):
        if sorted_codes[i] != sorted_codes[start]:
            yield start, i
            start = i
    if start < n:
        yield start, n


def _candidate_pairs_brute(local_idx: np.ndarray, embeddings: np.ndarray,
                            k: int, n_jobs: int,
                            query_chunk_size: int = 2000) -> Iterator[Tuple[int, int, float]]:
    """Brute-force cosine kNN within a small index subset, yielding
    (global_i, global_j, cosine_similarity) for each candidate edge.

    Queries are processed in row-chunks of `query_chunk_size` so that
    pathologically large buckets (e.g. many spectra collapsing onto the same
    hash code) cannot blow up memory in sklearn's argpartition step, which
    allocates an (n_query x k_local) int64 array per call."""
    sub_emb = embeddings[local_idx]
    n = len(local_idx)
    k_local = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k_local + 1, metric="cosine",
                          algorithm="brute", n_jobs=n_jobs)
    nn.fit(sub_emb)
    for start in range(0, n, query_chunk_size):
        end = min(start + query_chunk_size, n)
        dists, indices = nn.kneighbors(sub_emb[start:end])
        for row, offset in enumerate(range(start, end)):
            i = int(local_idx[offset])
            for rank in range(1, k_local + 1):
                j = int(local_idx[int(indices[row, rank])])
                cos_sim = max(0.0, 1.0 - float(dists[row, rank]))
                yield i, j, cos_sim


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

    k = min(cfg.ann_n_neighbours, N - 1)

    # Build COO lists
    rows, cols, data = [], [], []

    def _score_and_add(i: int, j: int, cos_sim: float) -> None:
        if cfg.precursor_tol_ppm and cfg.precursor_tol_ppm > 0:
            # v4: ppm hard filter on the neutral mass; no linear down-weighting
            # (weight = cosine). Different known charges can never match.
            if pcharge[i] != pcharge[j] and min(pcharge[i], pcharge[j]) > 0:
                return
            z = max(1, int(pcharge[i]))
            dm = abs(pmz[i] - pmz[j]) * z
            if dm > (pmz[i] - PROTON_MASS) * z * cfg.precursor_tol_ppm * 1e-6:
                return
            p_score = 1.0
        else:
            # Legacy hard precursor mass filter (Da)
            if abs(pmz[i] - pmz[j]) > cfg.precursor_tol_da:
                return
            p_score = _precursor_score(pmz[i], pmz[j], pcharge[i], pcharge[j],
                                       cfg.precursor_tol_da)
        w = cos_sim * p_score
        if w < cfg.edge_threshold:
            return
        rows.append(i)
        cols.append(j)
        data.append(w)

    if cfg.use_hd_prefilter and N > cfg.hd_prefilter_min_n:
        # Hyperdimensional binary-hash pre-filter (HyperSpec-style): bucket
        # spectra by a random-hyperplane sign code so brute-force cosine
        # search only runs within each bucket, turning O(N^2) into
        # ~O(N * bucket_size).
        #
        # Multi-hash repetition: a single random-hyperplane hash splits true
        # neighbours that fall on opposite sides of a hyperplane into
        # different buckets ("bucket-edge" misses). Repeating the hashing
        # `hd_n_repeats` times with independent seeds and taking the union of
        # all candidate pairs recovers most of these — each repetition has
        # independent boundary failures, so a pair only needs to co-occur in
        # ONE of the R tables. Cost scales linearly with R.
        n_bits = int(np.ceil(np.log2(max(2, N / cfg.hd_target_bucket_size))))
        n_bits = max(1, min(n_bits, cfg.hd_max_bits))

        # True No-USE ablation: hash directly from raw 72,500-bin vectors.
        # Within-bucket kNN still uses 128-dim embeddings for memory efficiency.
        binner_for_lsh = SpectrumBinner(cfg) if not cfg.use_encoder else None

        candidates: Dict[Tuple[int, int], float] = {}
        n_repeats = max(1, cfg.hd_n_repeats)
        # Multi-hash repetition recall model (Charikar 2002 / paper §III-F-B):
        #   recall(R) = 1 − (θ/π)^R
        # where θ is the angle between two embeddings and R is the number of
        # independent hash tables. Each table has independent boundary failures,
        # so a pair only needs to co-occur in ONE of the R tables.
        # Empirically fitted as recall(R) ≈ 1 − (1−p)^R with p=0.55 on COREAD.
        for rep in range(n_repeats):
            if cfg.use_encoder:
                codes = _hd_hash_codes(embeddings, n_bits, cfg.hd_seed + rep)
            else:
                logger.info(f"No-USE ablation: computing LSH codes from raw "
                            f"{binner_for_lsh.n_bins}-bin vectors (batchwise) …")
                codes = _hd_hash_codes_from_bins(
                    spectra, binner_for_lsh, n_bits, cfg.hd_seed + rep,
                    batch_size=cfg.ann_query_chunk_size)
            order = np.argsort(codes, kind="stable")
            sorted_codes = codes[order]
            n_buckets = len(np.unique(codes))
            logger.info(f"Hyperdimensional pre-filter rep {rep+1}/{n_repeats}: "
                        f"{n_bits} bits, {n_buckets} buckets, "
                        f"~{N / max(1, n_buckets):.0f} spectra/bucket")

            for start, end in tqdm(list(_bucket_ranges(sorted_codes)),
                                   desc=f"Stage2 HD-bucket kNN (rep {rep+1}/{n_repeats})",
                                   unit=" bucket"):
                bucket_idx = order[start:end]
                if len(bucket_idx) < 2:
                    continue
                for i, j, cos_sim in _candidate_pairs_brute(bucket_idx, embeddings, k, cfg.n_jobs,
                                                            query_chunk_size=cfg.ann_query_chunk_size):
                    key = (i, j)
                    prev = candidates.get(key)
                    if prev is None or cos_sim > prev:
                        candidates[key] = cos_sim

        for (i, j), cos_sim in candidates.items():
            _score_and_add(i, j, cos_sim)
    else:
        # Plain brute-force cosine kNN (small/medium N)
        logger.info("Building approximate neighbour index …")
        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine",
                              algorithm="brute", n_jobs=cfg.n_jobs)
        nn.fit(embeddings)

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
                    cos_sim = max(0.0, 1.0 - float(dists[offset, rank]))
                    _score_and_add(i, j, cos_sim)

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


def _dbscan_on_spectral(Z: np.ndarray, cfg: Config, eps: Optional[float] = None,
                        min_samples: Optional[int] = None) -> np.ndarray:
    """Exact DBSCAN via a chunked BallTree radius search + union-find.

    sklearn's DBSCAN materialises *all* point neighbourhoods from a single
    `radius_neighbors` call before clustering. For N≈500K, if even a modest
    fraction of points sit in dense regions, the combined neighbour-index
    arrays can run into the GB range and trigger MemoryError (independent of
    n_jobs — the allocation happens inside `query_radius` itself). Querying
    the tree in row-chunks keeps at most `chunk` neighbourhoods resident at
    once, and a union-find over (core-point, neighbour) pairs reproduces
    sklearn's exact DBSCAN semantics — same labels, bounded memory.

    `eps`/`min_samples` can be overridden per-call — NECESSARY because this
    function gets used on two very different spaces: the low-dimensional
    spectral embedding `Z` (tight scale, cfg.dbscan_eps=0.15 was tuned for
    this) and, elsewhere, raw high-dimensional embedding vectors (much
    larger typical distances). Reusing cfg.dbscan_eps unconditionally on the
    latter silently discards almost every point as noise — see FIX A below.
    """
    N = Z.shape[0]
    eps = cfg.dbscan_eps if eps is None else eps
    min_samples = cfg.dbscan_min_samples if min_samples is None else min_samples

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
                  cfg: Config, logger: logging.Logger,
                  eps: Optional[float] = None,
                  var_threshold: Optional[float] = None) -> np.ndarray:
    """Re-cluster any cluster whose within-cluster variance exceeds threshold.

    FIX A: `eps` and `var_threshold` are now explicit parameters instead of
    always reading cfg.dbscan_eps / cfg.split_var_threshold. Those two
    defaults were tuned for the LOW-DIMENSIONAL SPECTRAL embedding `Z`
    (compute_spectral_embedding output), where same-cluster points end up
    extremely tightly grouped (cosine similarity to centroid > 0.99 is
    typical after eigen-decomposition). When this function is called on RAW
    embeddings instead (the HDBSCAN path, or the large-N graph
    connected-components path), those defaults are far too strict:
      - var_threshold=0.04 requires >96% cosine similarity to centroid to
        skip re-splitting. A real transformer-embedding cluster from a
        typical SupCon-pretrained encoder sits around 70-85% similarity —
        so nearly every genuine cluster gets flagged for re-splitting.
      - eps=0.15 (a raw EUCLIDEAN radius) is tuned for the compressed
        spectral-embedding scale. On raw embeddings, real same-cluster
        distances are commonly 0.6-0.9 — so once a cluster IS flagged, the
        internal DBSCAN re-clustering finds ~zero neighbours within eps and
        dumps almost the entire cluster into noise (label -1).
    The net effect without this fix: correctly-formed clusters get silently
    shredded into noise by this "cleanup" step. Call sites must now pass an
    eps/var_threshold appropriate to whichever embedding space they're
    actually operating on (see run_hgce for the three call sites).
    """
    eps = cfg.dbscan_eps if eps is None else eps
    var_threshold = cfg.split_var_threshold if var_threshold is None else var_threshold

    # FIX H (defense-in-depth): cap the size of a cluster this function will
    # attempt to internally re-split via _dbscan_on_spectral. BallTree
    # radius search degrades badly on a very large, dense cluster (a query
    # returning O(N) neighbours per point turns the whole pass into
    # near-O(N^2)) — this is exactly what turned an ~104,000-member cluster
    # into a projected 80+ hour run in practice. FIX G in _hdbscan_bucketed
    # (majority-vote merging) should prevent such an oversized cluster from
    # forming in the first place, but this cap means that even if one DOES
    # appear (from a bug elsewhere, unusual data, or a future code change),
    # dynamic_split logs a warning and leaves it alone instead of hanging
    # for hours/days.
    max_split_cluster_size = max(cfg.ann_query_chunk_size * 20, 20_000)

    new_labels = labels.copy()
    next_id = int(labels.max()) + 1
    unique = [c for c in np.unique(labels) if c >= 0]
    splits = 0
    n_oversized_skipped = 0
    n_noise_before = int((new_labels == -1).sum())
    for cid in unique:
        mask = new_labels == cid
        idx = np.where(mask)[0]
        if len(idx) < 4:
            continue
        if len(idx) > max_split_cluster_size:
            n_oversized_skipped += 1
            continue
        embs = embeddings[idx]
        var = _within_cluster_variance(embs)
        if var > var_threshold:
            sub = _dbscan_on_spectral(embs, cfg, eps=eps,
                                      min_samples=cfg.dbscan_min_samples)
            for sc in np.unique(sub):
                if sc < 0:
                    new_labels[idx[sub == sc]] = -1
                else:
                    new_labels[idx[sub == sc]] = next_id
                    next_id += 1
            splits += 1
    if n_oversized_skipped:
        logger.warning(f"dynamic_split: {n_oversized_skipped} cluster(s) exceeded "
                       f"{max_split_cluster_size:,} members and were left as-is "
                       f"rather than risking a catastrophically slow internal "
                       f"re-split — an oversized cluster like this usually means "
                       f"something upstream (e.g. the HD-bucketed merge step) "
                       f"produced a bad merge; worth checking the Stage 2 cluster "
                       f"size distribution.")
    n_noise_after = int((new_labels == -1).sum())
    n_newly_noised = n_noise_after - n_noise_before
    logger.info(f"Dynamic split (eps={eps:.3f}, var_thr={var_threshold:.3f}): "
                f"{splits} clusters re-split, "
                f"{n_newly_noised:,} points newly sent to noise by this step")
    if splits > 0 and n_newly_noised > 0.5 * splits:
        logger.warning(f"Dynamic split sent a large fraction of re-split cluster "
                       f"members to noise ({n_newly_noised:,} points across "
                       f"{splits} clusters) — eps/var_threshold may still be "
                       f"too strict for this embedding space; consider raising "
                       f"eps or var_threshold, or disabling dynamic_split_enabled.")
    return new_labels


def dynamic_merge(labels: np.ndarray, embeddings: np.ndarray,
                  cfg: Config, logger: logging.Logger) -> np.ndarray:
    """Merge clusters whose centroids are closer than merge_dist_threshold.

    Unlike dynamic_split, this operates purely on cosine similarity between
    L2-normalised centroids, which is scale-consistent regardless of which
    embedding space (spectral Z or raw transformer embeddings) it's given —
    no eps-style fix needed here.
    """
    unique = sorted([c for c in np.unique(labels) if c >= 0])
    if len(unique) < 2:
        return labels

    centroids = np.array([
        normalize(embeddings[labels == c].mean(axis=0, keepdims=True))[0]
        for c in unique
    ])

    # Vectorised pairwise cosine distances — O(k²) in numpy rather than a
    # Python nested loop (which would be ~47M iterations for k≈10K clusters).
    # cosine_dist = 1 - (C @ C.T)  for unit-norm rows.
    sim_matrix = centroids @ centroids.T          # (k, k) cosine similarities
    np.fill_diagonal(sim_matrix, 0.0)             # ignore self-similarity
    close_i, close_j = np.where(
        (1.0 - sim_matrix) < cfg.merge_dist_threshold
    )
    # Keep only upper triangle to avoid double-counting
    mask = close_i < close_j
    close_i, close_j = close_i[mask], close_j[mask]

    # Union-find over cluster indices (positions in `unique`)
    parent = list(range(len(unique)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    merges = 0
    for pi, pj in zip(close_i.tolist(), close_j.tolist()):
        ri, rj = find(pi), find(pj)
        if ri != rj:
            parent[rj] = ri
            merges += 1

    # Build label remapping
    new_labels = labels.copy()
    root_label = {i: unique[find(i)] for i in range(len(unique))}
    for pos, cid in enumerate(unique):
        new_labels[labels == cid] = root_label[pos]

    logger.info(f"Dynamic merge: {merges} cluster pairs merged")
    return new_labels


def _hdbscan_bucketed(embeddings: np.ndarray, cfg: Config, logger: logging.Logger,
                      HDBSCAN, hdbscan_source: str) -> np.ndarray:
    """Scalable HDBSCAN via hyperdimensional-hash bucketing (HyperSpec-style).

    A single global HDBSCAN call on N=500K+ points in 128 dimensions can
    become impractically slow — tree-based nearest-neighbour search (which
    HDBSCAN's core-distance computation relies on) degrades toward
    brute-force-like cost as N and dimensionality both grow (the "curse of
    dimensionality"), which is what an hours-long "stuck" HDBSCAN run at
    this scale looks like — not a hang, just very slow.

    Naive connected-components on a global similarity graph avoids that
    slowness but reintroduces chaining: it merges transitively (A-B and B-C
    above threshold puts A and C in the same cluster even if A and C aren't
    similar at all), and this gets WORSE as N grows.

    This combines the scalability of HD-hash bucketing (already validated at
    506K spectra for the connected-components path) with HDBSCAN's
    density-based clustering:
      1. Hash embeddings into buckets via R independent random-hyperplane
         repeats (same mechanism as the existing HD prefilter).
      2. Run HDBSCAN independently within each bucket — small (~hd_target_
         bucket_size points), so no curse-of-dimensionality blowup.
      3. MAJORITY-VOTE merge (FIX G): a pair (i, j) is only merged if it
         co-occurs in the same non-noise HDBSCAN sub-cluster in a STRICT
         MAJORITY of the R repeats, not just any single one.

    FIX G — why majority voting, not any-repeat union: the original version
    merged on ANY single repeat's agreement, which is still transitive
    across repeats — if repeat 1 groups {A,B} and repeat 2 (different hash
    boundaries) groups {B,C}, the union merges A-B-C even though A and C
    never co-occurred anywhere. This reintroduced the same chaining problem
    bucketing was meant to avoid, just one level removed, and on real 500K
    data produced a ~104,000-member merged cluster that then made
    dynamic_split's internal _dbscan_on_spectral call effectively never
    finish (projected 80+ hours). Requiring a majority of the R=6 repeats to
    agree breaks this: a genuine same-peptide pair should co-cluster
    consistently across repeats, while a false bridge from one repeat's
    unlucky hash boundary is unlikely to be replicated in most of the other
    5. This does cost more memory/time to track pairwise co-occurrence
    counts instead of doing union-find directly during the scan; sub-cluster
    voting is capped at `_MAX_SUBCLUSTER_FOR_VOTING` members to bound this
    (see below) — real per-bucket sub-clusters (a handful of replicate
    spectra) are almost always far smaller than this cap, so an oversized
    one is itself a signal something already went wrong in that bucket/
    repeat and its votes are excluded rather than trusted.
    """
    from collections import defaultdict

    N = embeddings.shape[0]
    n_bits = int(np.ceil(np.log2(max(2, N / cfg.hd_target_bucket_size))))
    n_bits = max(1, min(n_bits, cfg.hd_max_bits))

    n_repeats = max(1, cfg.hd_n_repeats)
    if cfg.hd_bucketed_min_repeat_votes is not None:
        min_votes = max(2, cfg.hd_bucketed_min_repeat_votes)  # floor of 2:
        # never allow this to silently regress to the old any-repeat-union
        # behavior (min_votes=1), which is the exact behavior that produced
        # the ~104K-member chained cluster on real data.
    else:
        min_votes = max(1, (n_repeats // 2) + 1)  # strict majority
    _MAX_SUBCLUSTER_FOR_VOTING = 200

    pair_votes: Dict[int, int] = defaultdict(int)
    n_oversized_skipped = 0

    for rep in range(n_repeats):
        codes = _hd_hash_codes(embeddings, n_bits, cfg.hd_seed + rep)
        order = np.argsort(codes, kind="stable")
        sorted_codes = codes[order]
        n_buckets = len(np.unique(codes))
        logger.info(f"HDBSCAN HD-prefilter rep {rep+1}/{n_repeats}: "
                    f"{n_bits} bits, {n_buckets} buckets, "
                    f"~{N / max(1, n_buckets):.0f} spectra/bucket")

        bucket_ranges = list(_bucket_ranges(sorted_codes))
        rep_oversized = [0]

        def _process_all_buckets():
            for start, end in tqdm(bucket_ranges,
                                   desc=f"Stage2 HDBSCAN bucket clustering (rep {rep+1}/{n_repeats})",
                                   unit=" bucket"):
                bucket_idx = order[start:end]
                if len(bucket_idx) < max(2, cfg.hdbscan_min_cluster_size):
                    continue
                sub_emb = embeddings[bucket_idx]
                clusterer = HDBSCAN(
                    min_cluster_size         = cfg.hdbscan_min_cluster_size,
                    min_samples              = cfg.hdbscan_min_samples,
                    metric                   = 'euclidean',
                    cluster_selection_method = cfg.hdbscan_cluster_selection_method,
                    cluster_selection_epsilon= 0.0,
                    n_jobs                   = cfg.n_jobs,
                )
                try:
                    sub_labels = clusterer.fit_predict(sub_emb)
                except TypeError:
                    continue
                for sc in np.unique(sub_labels):
                    if sc < 0:
                        continue
                    members = np.sort(bucket_idx[sub_labels == sc]).astype(np.int64)
                    if len(members) > _MAX_SUBCLUSTER_FOR_VOTING:
                        rep_oversized[0] += 1
                        continue
                    for a_pos in range(len(members)):
                        a = int(members[a_pos])
                        for b_pos in range(a_pos + 1, len(members)):
                            b = int(members[b_pos])
                            pair_votes[a * N + b] += 1
            return None

        _run_with_heartbeat(_process_all_buckets,
                            f"HDBSCAN bucketed clustering (rep {rep+1}/{n_repeats})",
                            logger, interval=30.0)
        n_oversized_skipped += rep_oversized[0]

    if n_oversized_skipped:
        logger.warning(f"{n_oversized_skipped} sub-clusters exceeded "
                       f"{_MAX_SUBCLUSTER_FOR_VOTING} members within a single "
                       f"bucket/repeat and were excluded from majority voting "
                       f"(treated as unreliable rather than trusted) — if this "
                       f"count is large relative to your total bucket count, "
                       f"consider lowering hd_target_bucket_size.")

    # Majority-vote union-find
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

    n_merged_pairs = 0
    for key, votes in pair_votes.items():
        if votes >= min_votes:
            a, b = divmod(key, N)
            union(a, b)
            n_merged_pairs += 1
    logger.info(f"Majority-vote merge: {n_merged_pairs:,}/{len(pair_votes):,} "
                f"candidate pairs met the {min_votes}/{n_repeats}-repeat "
                f"agreement threshold")

    labels = np.full(N, -1, dtype=np.int64)
    comp_members: Dict[int, List[int]] = {}
    for i in range(N):
        root = find(i)
        comp_members.setdefault(root, []).append(i)
    root_to_label: Dict[int, int] = {}
    next_label = 0
    for root, members in comp_members.items():
        if len(members) < cfg.hdbscan_min_cluster_size:
            continue
        if root not in root_to_label:
            root_to_label[root] = next_label
            next_label += 1
        for i in members:
            labels[i] = root_to_label[root]

    return labels


def run_hgce(embeddings: np.ndarray, spectra: List[Spectrum],
             cfg: Config, logger: logging.Logger) -> np.ndarray:
    """Full Stage 2 pipeline → cluster labels (−1 = noise)."""

    N = len(spectra)

    # FIX A: eps/var_threshold to use for dynamic_split when operating on RAW
    # (non-spectral) embeddings — i.e. the HDBSCAN path and the large-N graph
    # connected-components path. Derived from cfg.edge_threshold so it's
    # consistent with the same cosine-similarity cutoff already used to build
    # the similarity graph, rather than reusing the spectral-space-tuned
    # cfg.dbscan_eps (0.15) / cfg.split_var_threshold (0.04) defaults, which
    # are only valid for the compressed spectral embedding Z.
    #   euclidean_dist = sqrt(2 * (1 - cosine_sim))  for unit vectors
    raw_split_eps = (cfg.dbscan_eps_embedding if cfg.dbscan_eps_embedding is not None
                     else math.sqrt(2.0 * (1.0 - cfg.edge_threshold)))
    raw_split_var_threshold = cfg.split_var_threshold_embedding

    # ── HDBSCAN path — best for transformer embeddings ────────────────────────
    # HDBSCAN clusters directly on the 128-dim L2-normalised embeddings without
    # building a graph first. It finds clusters of varying density and handles
    # the semantically-grouped transformer embedding space much better than
    # connected components, which chains clusters through weak links.
    if cfg.use_hdbscan:
        # FIX C: prefer sklearn.cluster.HDBSCAN (available since scikit-learn
        # 1.3) over the standalone `hdbscan` PyPI package. The standalone
        # package's compiled BallTree/KDTree (Boruvka) core-distance code has
        # a long history of hanging or deadlocking — not raising a clean
        # error — when its compiled extensions were built against a
        # different scikit-learn version than the one actually installed
        # (this is exactly what an uninterruptible hang at "Running HDBSCAN"
        # that requires a kernel restart looks like — a genuine C-level
        # deadlock, not just slow Python). scikit-learn is already a
        # required dependency of this pipeline, so its built-in HDBSCAN
        # avoids that whole class of problem — same algorithm, same
        # min_cluster_size/min_samples/cluster_selection_epsilon knobs, no
        # separate C-extension package to version-mismatch or fail to build.
        HDBSCAN = None
        hdbscan_source = None
        try:
            from sklearn.cluster import HDBSCAN as HDBSCAN
            hdbscan_source = "sklearn.cluster (preferred — avoids standalone package's known hang issues)"
        except ImportError:
            try:
                from hdbscan import HDBSCAN as HDBSCAN
                hdbscan_source = "standalone hdbscan package (scikit-learn < 1.3; consider upgrading)"
            except ImportError:
                pass

        if HDBSCAN is None:
            logger.warning("Neither sklearn.cluster.HDBSCAN nor the standalone hdbscan "
                           "package is available — falling back to graph clustering. "
                           "Upgrade scikit-learn to >=1.3 (pip install -U scikit-learn) "
                           "for the built-in HDBSCAN, recommended over the standalone package.")
        else:
            logger.info(f"Running HDBSCAN on {N:,} embeddings "
                        f"(min_cluster_size={cfg.hdbscan_min_cluster_size}, "
                        f"selection_method={cfg.hdbscan_cluster_selection_method}, "
                        f"implementation: {hdbscan_source}) …")
            # metric='euclidean' on these L2-normalised embeddings is
            # mathematically equivalent to cosine distance here (unit
            # vectors: euclidean_dist^2 = 2*(1-cosine_sim), a monotonic
            # transform that preserves nearest-neighbour rankings exactly)
            # — do NOT switch this to metric='cosine'. sklearn's fast
            # tree-based HDBSCAN algorithms don't support cosine, so
            # requesting it forces the 'brute' algorithm, which needs the
            # full N x N pairwise distance matrix — for N=150,000 that's
            # ~22.5 billion entries (~90+ GB), all but guaranteed to crash
            # or hang far worse than the issue it would be "fixing".
            clusterer = HDBSCAN(
                min_cluster_size         = cfg.hdbscan_min_cluster_size,
                min_samples              = cfg.hdbscan_min_samples,
                metric                   = 'euclidean',
                cluster_selection_method = cfg.hdbscan_cluster_selection_method,
                # FIX D/E: skip straight to epsilon=0 rather than attempting
                # cfg.hdbscan_epsilon first — on scikit-learn 1.8.0 +
                # numpy>=2.4 the epsilon>0 code path reliably reproduces the
                # known TypeError (github.com/scikit-learn/scikit-learn/
                # issues/33355), so there's no point spending a full
                # fit_predict() attempt on something confirmed to fail.
                cluster_selection_epsilon= 0.0,
                n_jobs                   = cfg.n_jobs,
            )
            # ── Graph-seeded HDBSCAN — fast, no chaining ─────────────────
            # Step 1: Build sparse graph (fast kNN already in pipeline)
            # Step 2: Connected components → candidate clusters
            # Step 3: HDBSCAN within each large component only (small, fast)
            # This avoids O(N²) HDBSCAN on all N points and prevents chaining.
            logger.info(f"Graph-seeded HDBSCAN: building graph "
                        f"(edge_threshold={cfg.edge_threshold}) …")
            W = build_sparse_similarity_graph(embeddings, spectra, cfg, logger)
            if W.nnz == 0:
                logger.warning("Empty graph — all spectra assigned to noise.")
                return np.full(N, -1, dtype=np.int32)
            from scipy.sparse.csgraph import connected_components
            n_comp, comp_labels = connected_components(
                W, directed=False, connection="weak")
            comp_sizes = np.bincount(comp_labels, minlength=n_comp)
            logger.info(f"Graph-seeded HDBSCAN: {n_comp:,} initial components")
            refine_above = max(cfg.hdbscan_min_cluster_size * 3, 6)
            final_labels = np.full(N, -1, dtype=np.int64)
            next_id = 0
            small_comps = np.where(
                (comp_sizes >= cfg.dbscan_min_samples) &
                (comp_sizes <= refine_above))[0]
            large_comps = np.where(comp_sizes > refine_above)[0]
            for cid in small_comps:
                idx = np.where(comp_labels == cid)[0]
                final_labels[idx] = next_id
                next_id += 1
            logger.info(f"Refining {len(large_comps):,} large components "
                        f"with HDBSCAN …")
            for cid in tqdm(large_comps, desc="HDBSCAN refine",
                            unit=" component"):
                idx = np.where(comp_labels == cid)[0]
                sub_emb = embeddings[idx]
                try:
                    sub_clust = HDBSCAN(
                        min_cluster_size         = cfg.hdbscan_min_cluster_size,
                        min_samples              = cfg.hdbscan_min_samples,
                        metric                   = 'euclidean',
                        cluster_selection_method = 'leaf',
                        cluster_selection_epsilon= 0.0,
                        n_jobs                   = 1,
                    )
                    sub_labels = sub_clust.fit_predict(sub_emb)
                except Exception:
                    sub_labels = np.zeros(len(idx), dtype=np.int64)
                for sub_id in np.unique(sub_labels):
                    if sub_id == -1:
                        continue
                    sub_idx = idx[sub_labels == sub_id]
                    final_labels[sub_idx] = next_id
                    next_id += 1
            labels = final_labels
            n_clusters = len(set(labels.tolist()) - {-1})
            n_noise    = int((labels == -1).sum())
            logger.info(f"Graph-seeded HDBSCAN: {n_clusters:,} clusters, "
                        f"{n_noise:,} noise")
            labels = dynamic_split(labels, embeddings, cfg, logger)
            if cfg.dynamic_merge_enabled:   # ← FIX: respect config flag
                labels = dynamic_merge(labels, embeddings, cfg, logger)
            else:
                logger.info("Dynamic merge: DISABLED (dynamic_merge_enabled=False)")
            n_final = len(set(labels.tolist()) - {-1})
            logger.info(f"Final clusters: {n_final}, "
                        f"noise: {int((labels==-1).sum())}")
            return labels.astype(np.int32)
            if cfg.use_hd_prefilter and N > cfg.hd_prefilter_min_n:
                # Legacy bucketed path — kept for compatibility
                pass
            else:
                try:
                    labels = _run_with_heartbeat(
                        lambda: clusterer.fit_predict(embeddings).astype(np.int64),
                        "HDBSCAN fit_predict", logger, interval=30.0)
                except TypeError as e:
                    # Safety net: if a nonzero epsilon is ever explicitly forced
                    # back on and hits the same sklearn bug, retry once more
                    # with epsilon=0 rather than losing the whole (expensive)
                    # Stage 1 encoding to a crash.
                    logger.warning(
                        f"HDBSCAN raised TypeError (likely scikit-learn issue #33355, "
                        f"a known scikit-learn 1.8.0 + numpy>=2.4 bug with "
                        f"cluster_selection_epsilon > 0) — retrying once more with "
                        f"cluster_selection_epsilon=0. Original error: {e}")
                    clusterer = HDBSCAN(
                        min_cluster_size         = cfg.hdbscan_min_cluster_size,
                        min_samples              = cfg.hdbscan_min_samples,
                        metric                   = 'euclidean',
                        cluster_selection_method = cfg.hdbscan_cluster_selection_method,
                        cluster_selection_epsilon= 0.0,
                        n_jobs                   = cfg.n_jobs,
                    )
                    labels = _run_with_heartbeat(
                        lambda: clusterer.fit_predict(embeddings).astype(np.int64),
                        "HDBSCAN fit_predict (retry)", logger, interval=30.0)
            n_clusters = len(set(labels) - {-1})
            n_noise    = int((labels == -1).sum())
            logger.info(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise points")

            # Apply precursor mass filter as post-processing:
            # spectra in the same HDBSCAN cluster but with incompatible precursor
            # masses are split into sub-clusters
            pmz = np.array([s.precursor_mz for s in spectra], dtype=np.float64)
            next_id = int(labels.max()) + 1
            for cid in np.unique(labels):
                if cid < 0:
                    continue
                idx = np.where(labels == cid)[0]
                if len(idx) < 2:
                    continue
                # Check if any member pair exceeds precursor tolerance
                pmz_sub = pmz[idx]
                if pmz_sub.max() - pmz_sub.min() > cfg.precursor_tol_da:
                    # Sort by precursor m/z and split at gaps > tolerance
                    order = np.argsort(pmz_sub)
                    sorted_pmz = pmz_sub[order]
                    gaps = np.where(np.diff(sorted_pmz) > cfg.precursor_tol_da)[0]
                    if len(gaps) > 0:
                        # Re-assign sub-groups
                        start = 0
                        first = True
                        for gap in gaps:
                            sub_idx = idx[order[start:gap+1]]
                            if first:
                                # Keep original cluster id for first sub-group
                                first = False
                            else:
                                labels[sub_idx] = next_id
                                next_id += 1
                            start = gap + 1
                        # Last sub-group
                        sub_idx = idx[order[start:]]
                        labels[sub_idx] = next_id
                        next_id += 1

            # FIX A: pass the raw-embedding-space eps/var_threshold, not the
            # spectral-space defaults — see dynamic_split docstring.
            if cfg.dynamic_split_enabled:
                labels = dynamic_split(labels, embeddings, cfg, logger,
                                       eps=raw_split_eps,
                                       var_threshold=raw_split_var_threshold)
            else:
                logger.info("dynamic_split_enabled=False — skipping (HDBSCAN's "
                            "own density-based selection already handles this "
                            "for the raw embedding space).")
            if cfg.dynamic_merge_enabled:
                labels = dynamic_merge(labels, embeddings, cfg, logger)
            n_final      = len(set(labels) - {-1})
            n_noise_final = int((labels == -1).sum())
            logger.info(f"Final clusters: {n_final}, noise: {n_noise_final}")
            return labels.astype(np.int32)

    # ── Graph-based path (default) ────────────────────────────────────────────
    W = build_sparse_similarity_graph(embeddings, spectra, cfg, logger)

    n_nonzero = W.nnz
    if n_nonzero == 0:
        logger.warning("Empty graph — assigning all spectra to noise.")
        return np.full(len(spectra), -1, dtype=np.int32)

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
        # Genuine spectral-embedding space: cfg.dbscan_eps / cfg.split_var_threshold
        # were tuned for exactly this, so use them unchanged (defaults, no override).
        split_eps_for_path = cfg.dbscan_eps
        split_var_for_path = cfg.split_var_threshold
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
        # FIX A: raw-embedding space here too — same reasoning as the HDBSCAN path.
        split_eps_for_path = raw_split_eps
        split_var_for_path = raw_split_var_threshold

    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    logger.info(f"Initial clustering: {n_clusters} clusters, {n_noise} noise points")

    if cfg.dynamic_split_enabled:
        labels = dynamic_split(labels, work_embs, cfg, logger,
                               eps=split_eps_for_path,
                               var_threshold=split_var_for_path)
    else:
        logger.info("dynamic_split_enabled=False — skipping dynamic_split.")
    if cfg.dynamic_merge_enabled:
        labels = dynamic_merge(labels, work_embs, cfg, logger)

    n_final = len(set(labels) - {-1})
    n_noise_final = int((labels == -1).sum())
    logger.info(f"Final clusters: {n_final}, noise: {n_noise_final}")
    return labels.astype(np.int32)


# ─────────────────────────────────────────────────────────────────────────────
# STAGE 3 — BIOLOGICALLY-AWARE RERANKER (BAR)
# ─────────────────────────────────────────────────────────────────────────────

# Common PTM mass shifts (monoisotopic, Da) used for delta-mass PTM detection
PROTON_MASS = 1.007276
PTM_MASS_TOLERANCE = 0.02  # Da
COMMON_PTM_DELTA_MASSES = {
    "Phosphorylation":   79.9663,
    "Acetylation":       42.0106,
    "Oxidation":         15.9949,
    "Methylation":       14.0157,
    "Deamidation":        0.9840,
    "Carbamidomethyl":   57.0215,
    "Ubiquitination(GG)": 114.0429,
}


def _neutral_mass(precursor_mz: float, charge: int) -> float:
    """Convert observed precursor m/z to neutral monoisotopic mass."""
    z = charge if charge > 0 else 1
    return (precursor_mz - PROTON_MASS) * z


def _match_ptm_delta(diff: float) -> Tuple[bool, str]:
    """Check whether a neutral-mass difference matches a known PTM delta mass.
    Returns (matched, ptm_name)."""
    for name, delta in COMMON_PTM_DELTA_MASSES.items():
        if abs(diff - delta) <= PTM_MASS_TOLERANCE:
            return True, name
    return False, ""


def _split_by_neutral_mass(idx: np.ndarray, member_specs: List["Spectrum"],
                            gap_da: float, ppm_tol: float = 0.0) -> List[np.ndarray]:
    """Partition cluster members into mass sub-groups: sort by neutral
    precursor mass and break wherever a consecutive gap exceeds `gap_da`.
    Returns a list of index arrays (into the global spectrum list), one per
    mass group, in ascending-mass order."""
    masses = np.array([_neutral_mass(s.precursor_mz, s.precursor_charge)
                        for s in member_specs])
    order = np.argsort(masses)
    groups: List[List[int]] = [[order[0]]]
    if ppm_tol and ppm_tol > 0:
        # v4: ppm-level, ANCHORED split. A group is closed as soon as a member
        # lies more than ppm_tol (relative to the group's lightest member)
        # above it, so the span of every group is <= ppm_tol and a chain of
        # slightly-different masses cannot drift across the tolerance.
        anchor = masses[order[0]]
        for k in range(1, len(order)):
            m = masses[order[k]]
            if (m - anchor) > anchor * ppm_tol * 1e-6:
                groups.append([])
                anchor = m
            groups[-1].append(order[k])
        return [idx[g] for g in groups]
    for k in range(1, len(order)):
        if masses[order[k]] - masses[order[k - 1]] > gap_da:
            groups.append([])
        groups[-1].append(order[k])
    return [idx[g] for g in groups]


def _fragment_shift_confirm(spec_a: Spectrum, spec_b: Spectrum, delta_mass: float,
                             mz_tol: float, min_unshifted: int,
                             min_shifted: int) -> bool:
    """Check fragment-ion evidence that `spec_b` is a PTM-shifted variant of
    `spec_a` by `delta_mass`: require at least `min_unshifted` fragment peaks
    shared between the two spectra (unmodified backbone fragments) AND at
    least `min_shifted` pairs of peaks that differ by exactly `delta_mass`
    (fragments carrying the modification)."""
    mz_a, mz_b = spec_a.mz_array, spec_b.mz_array
    if len(mz_a) == 0 or len(mz_b) == 0:
        return False
    unshifted = 0
    shifted = 0
    for m in mz_a:
        if np.any(np.abs(mz_b - m) <= mz_tol):
            unshifted += 1
        if np.any(np.abs(mz_b - (m + delta_mass)) <= mz_tol) or \
           np.any(np.abs(mz_b - (m - delta_mass)) <= mz_tol):
            shifted += 1
        if unshifted >= min_unshifted and shifted >= min_shifted:
            return True
    return unshifted >= min_unshifted and shifted >= min_shifted


def _representative_spectrum(idx_group: np.ndarray, embeddings: np.ndarray,
                              spectra: List[Spectrum]) -> Spectrum:
    """Pick the member of a mass group closest to the group centroid, as a
    representative spectrum for fragment-ion comparisons."""
    embs = embeddings[idx_group]
    centroid = embs.mean(axis=0)
    cnorm = np.linalg.norm(centroid)
    if cnorm > 0:
        centroid = centroid / cnorm
    sims = embs @ centroid
    return spectra[idx_group[int(np.argmax(sims))]]


def _consensus_spectrum(spectra: List[Spectrum], weights: np.ndarray,
                        binner: SpectrumBinner) -> Tuple[np.ndarray, np.ndarray]:
    """Build a consensus spectrum by similarity-weighted averaging in bin space,
    then recover accurate m/z values via weighted-mean of original peak m/z
    values within each occupied bin (avoids the 0.02 Da bin-centre quantisation
    error that causes PD to score poorly).

    Intensities are normalised to a base-peak value of 999 — the conventional
    scale expected by Mascot / Proteome Discoverer.  Bins with a normalised
    intensity below 1.0 (i.e. < 0.1 % of base peak) are suppressed to keep
    the MGF compact and avoid noise peaks that confuse scoring engines.
    """
    weights = weights / weights.sum()

    # ── Step 1: accumulate weighted intensity in bin space ────────────────────
    acc = np.zeros(binner.n_bins, dtype=np.float64)
    # Also accumulate weighted sum of original m/z values per bin, to recover
    # accurate peak positions instead of using bin-centre approximations.
    mz_acc  = np.zeros(binner.n_bins, dtype=np.float64)
    w_acc   = np.zeros(binner.n_bins, dtype=np.float64)

    for s, w in zip(spectra, weights):
        mz      = s.mz_array
        intensity = s.intensity_array
        if len(mz) == 0:
            continue
        # sqrt-normalise intensities (same as binner, keeps relative heights)
        max_i = intensity.max()
        if max_i <= 0:
            continue
        norm_int = np.sqrt(intensity / max_i)

        bin_idx = ((mz - binner.mz_min) / binner.bin_width).astype(np.int32)
        mask    = (bin_idx >= 0) & (bin_idx < binner.n_bins)
        mz_m, int_m, bin_m = mz[mask], norm_int[mask], bin_idx[mask]

        # Accumulate — use numpy.add.at for correct handling of duplicate bins
        np.add.at(acc,    bin_m, w * int_m)
        np.add.at(mz_acc, bin_m, w * int_m * mz_m)   # intensity-weighted m/z
        np.add.at(w_acc,  bin_m, w * int_m)

    # ── Step 2: recover accurate m/z values for occupied bins ─────────────────
    occupied   = acc > 0.0
    bin_indices = np.where(occupied)[0]
    if len(bin_indices) == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)

    # Intensity-weighted mean m/z within each bin (falls back to bin centre
    # only if w_acc is exactly zero, which cannot happen given the guard above)
    bin_centre_mz = binner.mz_min + (bin_indices + 0.5) * binner.bin_width
    denom = w_acc[occupied]
    accurate_mz = np.where(denom > 0,
                           mz_acc[occupied] / denom,
                           bin_centre_mz)

    raw_int = acc[occupied]

    # ── Step 3: normalise intensities to base-peak = 999 ─────────────────────
    max_int = raw_int.max()
    if max_int > 0:
        norm_int_out = (raw_int / max_int) * 999.0
    else:
        norm_int_out = raw_int

    # Suppress peaks below 1 count (< 0.1 % of base peak after normalisation)
    keep = norm_int_out >= 1.0
    return accurate_mz[keep].astype(np.float32), norm_int_out[keep].astype(np.float32)


def _score_member_group(idx: np.ndarray, embeddings: np.ndarray,
                         spectra: List[Spectrum], binner: SpectrumBinner,
                         cfg: Config, cluster_id: int,
                         parent_cluster_id: int) -> Cluster:
    """Score a (possibly mass-split) group of members and build its
    consensus spectrum. Does not set ptm/heterogeneous flags — those are
    determined by the caller, which knows about sibling groups."""
    member_embs = embeddings[idx]
    member_specs = [spectra[i] for i in idx]

    centroid = member_embs.mean(axis=0)
    cnorm = np.linalg.norm(centroid)
    if cnorm > 0:
        centroid = centroid / cnorm

    sims = member_embs @ centroid
    mean_sim = float(sims.mean())
    within_var = float(1.0 - mean_sim)

    q = mean_sim * math.exp(-within_var / cfg.sigma_q)

    if q >= cfg.high_q_threshold:
        conf = "HIGH"
    elif q >= cfg.medium_q_threshold:
        conf = "MEDIUM"
    else:
        conf = "LOW"

    weights = np.clip(sims, 0.01, None)
    cons_mz, cons_int = _consensus_spectrum(member_specs, weights, binner)

    n_source_files = len({s.source_file for s in member_specs})

    # ── Consensus precursor m/z: similarity-weighted mean ────────────────────
    w_norm = weights / weights.sum()
    cons_pmz = float(np.sum([s.precursor_mz * w
                              for s, w in zip(member_specs, w_norm)]))

    # ── Consensus charge: most common charge state among members ──────────────
    charges = [s.precursor_charge for s in member_specs if s.precursor_charge > 0]
    if charges:
        from collections import Counter
        cons_charge = int(Counter(charges).most_common(1)[0][0])
    else:
        cons_charge = 0

    # ── Consensus RT: mean retention time (seconds) ───────────────────────────
    cons_rt = float(np.mean([s.rt for s in member_specs]))

    return Cluster(
        cluster_id=cluster_id,
        member_ids=idx.tolist(),
        centroid=centroid,
        quality_score=round(q, 4),
        confidence=conf,
        within_var=round(within_var, 4),
        mean_similarity=round(mean_sim, 4),
        consensus_mz=cons_mz,
        consensus_intensity=cons_int,
        parent_cluster_id=parent_cluster_id,
        n_source_files=n_source_files,
        consensus_precursor_mz=round(cons_pmz, 6),
        consensus_precursor_charge=cons_charge,
        consensus_rt=round(cons_rt, 3),
    )


def run_bar(labels: np.ndarray, embeddings: np.ndarray,
            spectra: List[Spectrum], cfg: Config,
            logger: logging.Logger) -> List[Cluster]:
    """Stage 3 — for each Stage-2 cluster, detect whether it is a mixture of
    distinct precursor-mass populations (a "mixed cluster"). A mixed cluster
    is either:
      - a genuine modified/unmodified pair of the same peptide (the mass
        difference between the two populations matches a known PTM delta
        mass AND fragment-ion evidence confirms the shift), or
      - a "heterogeneous" cluster of likely-unrelated spectra (the mass
        difference does not match any known PTM and is not confirmed by
        fragment ions) — a candidate false-clustering event.

    In both cases the cluster is split into one sub-cluster per mass
    population *before* building consensus spectra, so that no consensus is
    built across a mass-mixed population (which would score poorly in a
    downstream database search)."""
    binner = SpectrumBinner(cfg)
    unique_labels = sorted([c for c in np.unique(labels) if c >= 0])
    clusters: List[Cluster] = []
    next_id = int(unique_labels[-1]) + 1 if unique_labels else 0

    n_split_ptm = 0
    n_split_het = 0

    for cid in tqdm(unique_labels, desc="Stage3 scoring", unit="cluster"):
        idx = np.where(labels == cid)[0]
        member_specs = [spectra[i] for i in idx]

        mass_groups = _split_by_neutral_mass(idx, member_specs, cfg.mass_group_gap_da,
                                           cfg.precursor_tol_ppm)

        if len(mass_groups) == 1:
            clusters.append(_score_member_group(idx, embeddings, spectra, binner,
                                                  cfg, int(cid), parent_cluster_id=-1))
            continue

        # Multiple mass populations within one Stage-2 cluster: score each
        # population separately, then decide PTM-pair vs heterogeneous by
        # checking adjacent populations for a known PTM delta + fragment
        # confirmation.
        group_clusters = []
        for g_idx in mass_groups:
            sub = _score_member_group(g_idx, embeddings, spectra, binner, cfg,
                                        cluster_id=next_id, parent_cluster_id=int(cid))
            next_id += 1
            group_clusters.append((g_idx, sub))

        any_ptm_pair = False
        for k in range(len(mass_groups) - 1):
            idx_a, sub_a = group_clusters[k]
            idx_b, sub_b = group_clusters[k + 1]
            mass_a = _neutral_mass(spectra[idx_a[0]].precursor_mz, spectra[idx_a[0]].precursor_charge)
            mass_b = _neutral_mass(spectra[idx_b[0]].precursor_mz, spectra[idx_b[0]].precursor_charge)
            diff = abs(mass_b - mass_a)
            matched, ptm_name = _match_ptm_delta(diff)
            if not matched:
                continue
            rep_a = _representative_spectrum(idx_a, embeddings, spectra)
            rep_b = _representative_spectrum(idx_b, embeddings, spectra)
            if _fragment_shift_confirm(rep_a, rep_b, diff, cfg.ptm_fragment_mz_tol,
                                        cfg.ptm_min_unshifted_matches,
                                        cfg.ptm_min_shifted_matches):
                for sub in (sub_a, sub_b):
                    sub.ptm_flag = True
                    sub.ptm_delta_label = ptm_name
                    sub.ptm_delta_mass = round(diff, 4)
                any_ptm_pair = True

        if any_ptm_pair:
            n_split_ptm += 1
        else:
            n_split_het += 1
            for _, sub in group_clusters:
                sub.heterogeneous_flag = True

        clusters.extend(sub for _, sub in group_clusters)

    # Sort by quality descending
    clusters.sort(key=lambda c: c.quality_score, reverse=True)
    n_high = sum(1 for c in clusters if c.confidence == "HIGH")
    n_ptm = sum(1 for c in clusters if c.ptm_flag and len(c.member_ids) >= 2)
    n_het = sum(1 for c in clusters if c.heterogeneous_flag)
    logger.info(f"Stage 3 complete: {len(clusters)} clusters scored "
                f"({n_split_ptm} parent clusters split into confirmed PTM pairs, "
                f"{n_split_het} parent clusters split as heterogeneous), "
                f"{n_high} HIGH confidence, {n_ptm} PTM-flagged, "
                f"{n_het} heterogeneous-flagged")
    return clusters


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT WRITERS
# ─────────────────────────────────────────────────────────────────────────────

def write_cluster_tsv(clusters: List[Cluster], spectra: List[Spectrum],
                      labels: np.ndarray, output_dir: str) -> None:
    path = os.path.join(output_dir, "clusters.tsv")
    with open(path, "w") as fh:
        fh.write("cluster_id\tparent_cluster_id\tscan_id\tsource_file\tprecursor_mz\tprecursor_charge\t"
                 "quality_score\tconfidence\tptm_flag\tptm_delta_label\tptm_delta_mass\t"
                 "heterogeneous_flag\tn_source_files\twithin_var\tmean_similarity\n")
        for cl in clusters:
            for idx in cl.member_ids:
                s = spectra[idx]
                fh.write(f"{cl.cluster_id}\t{cl.parent_cluster_id}\t{s.scan_id}\t{s.source_file}\t"
                         f"{s.precursor_mz:.4f}\t"
                         f"{s.precursor_charge}\t{cl.quality_score}\t"
                         f"{cl.confidence}\t{cl.ptm_flag}\t{cl.ptm_delta_label}\t"
                         f"{cl.ptm_delta_mass:.4f}\t"
                         f"{cl.heterogeneous_flag}\t{cl.n_source_files}\t"
                         f"{cl.within_var:.4f}\t{cl.mean_similarity:.4f}\n")

    # Noise / unclustered
    noise_path = os.path.join(output_dir, "noise_spectra.tsv")
    with open(noise_path, "w") as fh:
        fh.write("scan_id\tsource_file\tprecursor_mz\tprecursor_charge\n")
        for i, lbl in enumerate(labels):
            if lbl == -1:
                s = spectra[i]
                fh.write(f"{s.scan_id}\t{s.source_file}\t{s.precursor_mz:.4f}\t{s.precursor_charge}\n")


def write_consensus_mgf(clusters: List[Cluster], output_dir: str) -> None:
    """Write consensus spectra of HIGH/MEDIUM clusters as a valid Mascot/PD MGF file.

    Complies with the Mascot Generic Format specification
    (https://www.matrixscience.com/help/data_file_help.html):

    ── Rules applied ─────────────────────────────────────────────────────────
    • PEPMASS  — required; written as "<mz>" or "<mz> <intensity>" (no
                 intensity here since we have no precursor signal).
    • CHARGE   — written as "<z>+" (e.g. "2+"); omitted when unknown (0).
    • RTINSECONDS — optional but useful for PD chromatographic alignment.
    • TITLE    — free-text identifier; no spaces in value avoids parser issues.
    • SCANS    — cluster_id used as a unique scan number so PD can track hits.
    • Parameters appear BEFORE fragment peaks, in the order recommended by
      Mascot: TITLE, SCANS, RTINSECONDS, PEPMASS, CHARGE.
    • No custom parameters (QUALITY, PTM_FLAG, etc.) inside the ion block —
      these violate MGF rule 8 and can cause PD to reject the spectrum.
      ProteoCLUST metadata is encoded safely in the TITLE string instead.
    • Fragment ion m/z values are positive non-zero (guaranteed by binner).
    • Intensities are normalised to base-peak = 999 (conventional PD scale).
    • Decimal separator is a period (US-ASCII), no thousands separator.
    • No comment lines (#  ;  !  /) inside ion blocks (rule 11).
    ──────────────────────────────────────────────────────────────────────────
    """
    path = os.path.join(output_dir, "consensus_spectra.mgf")
    written = 0

    with open(path, "w", encoding="ascii", errors="replace") as fh:
        for cl in clusters:
            if cl.confidence not in ("HIGH", "MEDIUM"):
                continue
            if cl.consensus_mz is None or len(cl.consensus_mz) == 0:
                continue

            # Build a compact, PD-friendly TITLE — encode ProteoCLUST metadata
            # as underscore-separated key=value tokens inside the title string
            # (no spaces, no equals sign outside the title value itself).
            ptm_tag   = f"_PTM={cl.ptm_delta_label}" if cl.ptm_flag and cl.ptm_delta_label and len(cl.member_ids) >= 2 else ""
            het_tag   = "_HET=1" if cl.heterogeneous_flag else ""
            conf_tag  = f"_Q={cl.quality_score:.4f}_{cl.confidence}"
            title = (f"cluster_{cl.cluster_id}"
                     f"_n{len(cl.member_ids)}"
                     f"_src{cl.n_source_files}"
                     f"{conf_tag}{ptm_tag}{het_tag}")

            fh.write("BEGIN IONS\n")

            # ── Mandatory / strongly recommended parameters (before peaks) ───
            fh.write(f"TITLE={title}\n")
            fh.write(f"SCANS={cl.cluster_id}\n")

            if cl.consensus_rt > 0.0:
                fh.write(f"RTINSECONDS={cl.consensus_rt:.3f}\n")

            # PEPMASS — required by Mascot/PD; written as m/z only (no intensity)
            fh.write(f"PEPMASS={cl.consensus_precursor_mz:.6f}\n")

            # CHARGE — "z+" notation as per MGF spec; omit if unknown
            if cl.consensus_precursor_charge > 0:
                fh.write(f"CHARGE={cl.consensus_precursor_charge}+\n")

            # ── Fragment ion peak list ────────────────────────────────────────
            # Format: "<mz> <intensity>" per line, US-ASCII period as decimal.
            # m/z to 4 decimal places (sub-0.01 Da precision); intensity to
            # 1 decimal place (base-peak = 999 scale, always >= 1.0).
            for mz_val, int_val in zip(cl.consensus_mz, cl.consensus_intensity):
                fh.write(f"{mz_val:.4f} {int_val:.1f}\n")

            fh.write("END IONS\n\n")
            written += 1

    return written


def write_summary(clusters: List[Cluster], spectra: List[Spectrum],
                  labels: np.ndarray, output_dir: str,
                  elapsed: float, cfg: "Config" = None) -> None:
    n_total = len(spectra)
    n_noise = int((labels == -1).sum())
    n_clustered = n_total - n_noise
    n_high = sum(1 for c in clusters if c.confidence == "HIGH")
    n_med = sum(1 for c in clusters if c.confidence == "MEDIUM")
    n_low = sum(1 for c in clusters if c.confidence == "LOW")
    n_ptm = sum(1 for c in clusters if c.ptm_flag and len(c.member_ids) >= 2)
    n_het = sum(1 for c in clusters if c.heterogeneous_flag)
    n_split = sum(1 for c in clusters if c.parent_cluster_id >= 0)
    n_cross = sum(1 for c in clusters if c.n_source_files > 1)
    sizes = [len(c.member_ids) for c in clusters]
    avg_size = np.mean(sizes) if sizes else 0.0
    max_size = max(sizes) if sizes else 0

    if cfg and not cfg.use_encoder:
        encoder_label = "Raw bins -> LSH (true No-USE ablation)"
    elif cfg and cfg.encoder_weights:
        encoder_label = f"LightTransformer [pretrained: {cfg.encoder_weights}]"
    elif cfg and cfg.use_transformer:
        encoder_label = "LightTransformer [random-init, no pretrained weights]"
    else:
        encoder_label = "Binned projection (NumpyFallback)"
    path = os.path.join(output_dir, "summary.txt")
    lines = [
        "=" * 60,
        "ProteoCLUST — Run Summary",
        "=" * 60,
        f"Stage-1 encoder    : {encoder_label}",
        f"Input spectra      : {n_total:>10,}",
        f"Clustered          : {n_clustered:>10,}  ({100*n_clustered/max(n_total,1):.1f}%)",
        f"Noise / unclustered: {n_noise:>10,}",
        f"Total clusters     : {len(clusters):>10,}",
        f"  HIGH confidence  : {n_high:>10,}",
        f"  MEDIUM confidence: {n_med:>10,}",
        f"  LOW confidence   : {n_low:>10,}",
        f"PTM-flagged        : {n_ptm:>10,}",
        f"Heterogeneous-split: {n_het:>10,}",
        f"  (from mass-split parent clusters: {n_split:>6,})",
        f"Cross-experiment   : {n_cross:>10,}",
        f"Avg cluster size   : {avg_size:>10.1f}",
        f"Max cluster size   : {max_size:>10,}",
        f"Wall time (s)      : {elapsed:>10.1f}",
        "=" * 60,
    ]
    text = "\n".join(lines)
    print("\n" + text)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text + "\n")
    # Save full config as config.txt for reproducibility
    if cfg is not None:
        config_path = os.path.join(output_dir, "config.txt")
        with open(config_path, "w", encoding="utf-8") as fh:
            fh.write("ProteoCLUST — Run Configuration\n")
            fh.write("=" * 60 + "\n")
            for key, val in vars(cfg).items():
                fh.write(f"  {key:<35} = {val}\n")
        print(f"Config saved to: {config_path}")


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
    spectra = load_spectra(cfg.mgf_path, cfg.max_spectra, cfg.verbose,
                           quality_filter_enabled=cfg.quality_filter_enabled,
                           quality_min_peaks=cfg.quality_min_peaks,
                           quality_min_mz_range=cfg.quality_min_mz_range)
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
    n_written = write_consensus_mgf(clusters, cfg.output_dir)
    logger.info(f"consensus_spectra.mgf: {n_written} spectra written (HIGH+MEDIUM only)")
    elapsed = time.time() - t0
    write_summary(clusters, spectra, labels, cfg.output_dir, elapsed, cfg)
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
    p.add_argument("--mgf", required=True, nargs="+",
                   help="Input MGF file path(s), or directory/directories of "
                        ".mgf files. Multiple files are clustered together, "
                        "enabling cross-experiment/cross-run clustering.")
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
    p.add_argument("--precursor-tol-ppm", type=float, default=10.0,
                   help="ppm-level precursor tolerance used by the Stage-2 graph "
                        "filter and Stage-3 BAR mass-group split (0 = legacy "
                        "Da-level behaviour: --precursor-tol and 0.5 Da BAR gap)")
    p.add_argument("--edge-threshold", type=float, default=0.70)
    p.add_argument("--n-eigvec", type=int, default=30)
    p.add_argument("--dbscan-eps", type=float, default=0.15)
    p.add_argument("--dbscan-min", type=int, default=2)
    p.add_argument("--no-dynamic-split", action="store_true",
                   help="Disable the post-hoc dynamic_split cleanup step "
                        "(see Config.dynamic_split_enabled docstring)")
    p.add_argument("--no-dynamic-merge", action="store_true",
                   help="Disable the post-hoc dynamic_merge cleanup step")

    # Runtime
    p.add_argument("--jobs", type=int, default=-1)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--max-spectra", type=int, default=None,
                   help="Truncate input for quick testing")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--resume", action="store_true",
                   help="Cache Stage 1/2 results in <out>/checkpoints and "
                        "reuse them on a subsequent run (e.g. after sleep/crash)")
    p.add_argument("--no-transformer", action="store_true",
                   help="Use NumPy random-projection encoder instead of PyTorch "
                        "transformer (ablation study / PyTorch-unavailable fallback)")
    p.add_argument("--encoder-weights", default="", metavar="PT_FILE",
                   help="Path to pretrained_encoder.pt from pretrain_transformer.py. "
                        "Loads the LightTransformerEncoder with pretrained SupCon "
                        "weights. --no-transformer is ignored when this is set.")
    p.add_argument("--no-encoder", action="store_true",
                   help="True No-USE ablation: compute LSH hash codes directly "
                        "from raw 72,500-bin spectrum vectors (batchwise) instead "
                        "of 128-dim projected embeddings. Within-bucket kNN and "
                        "Stage 3 still use 128-dim NumpyFallback embeddings.")
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
        precursor_tol_ppm=args.precursor_tol_ppm,
        edge_threshold=args.edge_threshold,
        n_eigenvectors=args.n_eigvec,
        dbscan_eps=args.dbscan_eps,
        dbscan_min_samples=args.dbscan_min,
        dynamic_split_enabled=not args.no_dynamic_split,
        dynamic_merge_enabled=not args.no_dynamic_merge,
        n_jobs=args.jobs,
        device=args.device,
        max_spectra=args.max_spectra,
        verbose=not args.quiet,
        resume=args.resume,
        encoder_weights=args.encoder_weights,
        # If pretrained weights are supplied, force transformer mode on
        use_transformer=(bool(args.encoder_weights) or not args.no_transformer),
        use_encoder=not args.no_encoder,
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
        # ── Choose your run condition by editing the flags below ──────────────
        # R=6+Refine (random proj):   encoder_weights="",  use_transformer=False
        # R=6+Refine (pretrained):    encoder_weights="D:\\...\\pretrained_encoder.pt"
        # No-USE ablation (true):     use_transformer=False, use_encoder=False
        # Brute-force:                use_transformer=False, use_hd_prefilter=False

        ENCODER_WEIGHTS = r"D:\Shraddha\falcon\files\pretrained_encoder_v9_falcon-style.pt"   # ← set to full path of pretrained_encoder.pt
                               #   leave "" to use NumpyFallback random projection

        cfg = Config(
            mgf_path         = r"D:\Shraddha\Original_MGFs\COREAD\20201022_FS_Choudhary_LMS2_FS03_MS2_16plex_2.mgf",
            output_dir       = r"D:\Shraddha\falcon\files\pretrained_encoder_output_falcon-style\v4",
            batch_size       = 2048,
            use_hdbscan              = True,
            hdbscan_min_cluster_size = 2,
            hdbscan_min_samples      = 2,
            hdbscan_epsilon          = 0.0,    # must be 0.0 — avoids sklearn numpy bug
            edge_threshold           = 0.85,   # raised for pretrained embeddings
            ann_n_neighbours         = 20,
            ann_query_chunk_size     = 500,
            dynamic_merge_enabled    = False,  # CRITICAL: prevents collapsing HDBSCAN clusters
            encoder_weights  = ENCODER_WEIGHTS,
            use_transformer  = True,
            use_encoder      = True,
            max_spectra      = None,
            resume           = False,          # CRITICAL: must be False to avoid stale cache
            verbose          = True,
            n_jobs           = -1,
        )
        
clusters = run_pipeline(cfg)
