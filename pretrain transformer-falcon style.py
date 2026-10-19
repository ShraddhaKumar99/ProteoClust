# -*- coding: utf-8 -*-
"""
Created on Fri Sep 11 02:51:50 2026

@author: HP
"""

"""
pretrain_transformer.py — Supervised Contrastive Pre-training for ProteoCLUST
==============================================================================
Trains the LightTransformerEncoder so that spectra of the same peptide get
similar embeddings and spectra of different peptides get dissimilar embeddings,
using the Supervised Contrastive (SupCon) loss (Khosla et al., NeurIPS 2020).

THIS VERSION fixes the "no improvement after epoch 1 / plateau" problem seen
in the previous script. Root cause + fixes:

  [FIX 12] The uniformity loss (Fix 4 in the earlier version) was pushing
           EVERY pair of embeddings apart, including same-peptide (positive)
           pairs — directly opposing SupCon's alignment objective and capping
           val_intra_sim almost immediately. Now the uniformity term only
           acts on pairs from DIFFERENT peptides (true negatives), and
           lambda_uniform is warmed up from 0 so alignment gets a head start.

  [FIX 13] Learning rate in the direct-run config was 2e-5, which with a
           3-epoch warmup means epoch 1 trains at ~6.6e-6 — far too small to
           move a randomly-initialized transformer. Raised to 1e-4.

  [FIX 14] Added per-epoch logging of the SupCon term and uniformity term
           SEPARATELY (previously only the combined scalar was logged), plus
           both intra-class and inter-class cosine similarity on validation.
           This lets you see which force is actually driving (or blocking)
           learning instead of guessing.

  [FIX 15] Checkpoint/early-stopping criterion changed from raw val_intra_sim
           to (val_intra_sim - val_inter_sim), i.e. class separation. Raw
           intra-class similarity alone doesn't tell you whether the model is
           actually separating classes or just uniformly increasing all
           pairwise similarity.

  [FIX 16] THE BIG ONE: replaced WeightedRandomSampler (which draws each of
           the 512 batch slots independently) with a proper P x K batch
           sampler. With thousands of unique peptides and a per-sample
           weighted sampler, the expected number of draws for any one
           peptide per batch is ~batch_size/num_classes << 2, so most
           classes present in a batch never get a second draw and
           contribute NO gradient (silently dropped by `has_pos.any()`).
           The new PKBatchSampler picks P peptides and K spectra of each
           per batch, so every included class is GUARANTEED a positive
           pair. This is standard practice in metric learning (e.g. PK
           sampling in person re-ID / face embedding models) and matters
           far more than any loss-term or LR tuning.

  [FIX 17] PSM loader now keeps only the single best-confidence hit per
           scan_id. Previously duplicate (scan_id, peptide) rows for the
           SAME scan but DIFFERENT peptides could survive dedup, pulling
           one physical spectrum toward two different classes — pure label
           noise for a metric-learning objective.

  [FIX 18] Weight decay no longer applied to LayerNorm/bias parameters
           (1-D params), only to matrix weights, following standard
           transformer training practice.

  [FIX 19] Positive-pair classes are now grouped by (peptide, charge), not
           peptide alone. Diagnosed via calibrate_thresholds.py /
           diagnose_intra_pairs.py on a real 506K-spectrum run: charge
           mismatch was present in 42% of same-peptide pairs overall but
           99.9% of the catastrophically-low-similarity tail — i.e. it
           explained almost the entire pathological failure mode. A peptide
           at 2+ vs 3+ produces a genuinely different fragment-ion spectrum;
           forcing both into one SupCon class was asking the encoder to
           align spectra that are legitimately different. See
           SpectrumContrastiveDataset.__init__ for details.

All 11 fixes from the previous revision (MGF SCANS= keying, projection head,
temperature, val metric definition, early stopping, weight decay, min spectra
per peptide, augmentation, O(1) PSM matching, val split) are unchanged and
still applied below.

Requirements:
    pip install torch numpy scipy scikit-learn pyteomics tqdm pandas

Usage:
    python pretrain_transformer.py \
        --mgf  coread.mgf \
        --psm  pd_psm_export.csv \
        --out  pretrained_encoder.pt \
        --epochs 30 --device cuda
"""

import os, sys, math, argparse, logging, random, re
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
except ImportError:
    print("ERROR: pip install torch --index-url https://download.pytorch.org/whl/cu128")
    sys.exit(1)

try:
    from pyteomics import mgf as pyteomics_mgf
    PYTEOMICS_AVAILABLE = True
except ImportError:
    PYTEOMICS_AVAILABLE = False

# ──────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class PretrainConfig:
    mgf_path:   str = ""
    psm_path:   str = ""
    psm_format: str = "proteome_discoverer"
    output_path: str = "pretrained_encoder.pt"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # Encoder architecture — must EXACTLY match proteoclust_fixed.py Config
    mz_min:     float = 50.0
    mz_max:     float = 1500.0
    bin_width:  float = 0.02
    top_k_peaks: int  = 150
    embed_dim:  int   = 128
    n_heads:    int   = 4
    n_layers:   int   = 2
    ff_dim:     int   = 256

    # Training
    epochs:     int   = 30
    batch_size: int   = 512           # only used for val_loader now; train
                                       # batch size is pk_classes_per_batch *
                                       # pk_samples_per_class (FIX 16)
    pk_classes_per_batch: int = 64    # FIX 16: P — unique peptides per batch
    pk_samples_per_class: int = 8     # FIX 16: K — spectra per peptide per
                                       # batch (guarantees positive pairs).
                                       # P*K should be close to batch_size.
    lr:         float = 5e-5          # FIX 13: raised from 2e-5/5e-5 — too
                                       # small a LR was the main reason
                                       # training looked frozen after epoch 1
    weight_decay: float = 1e-2
    warmup_epochs: int  = 5
    temperature: float  = 0.10
    lambda_uniform: float = 0.15      # FIX 12: lowered from 0.5 — 0.5 was
                                       # strong enough to cap intra-class
                                       # similarity almost immediately
    lambda_uniform_warmup_epochs: int = 5   # FIX 12: ramp uniformity in
                                             # from 0 so SupCon can establish
                                             # class structure first
    min_spectra_per_peptide: int = 3
    max_peptides: Optional[int] = None
    fdr_threshold: float = 0.01
    seed: int = 42
    val_split: float = 0.15
    early_stopping_patience: int = 8

    # Data augmentation
    augment:            bool  = True
    aug_peak_drop_prob: float = 0.15
    aug_intensity_noise: float = 0.08
    aug_mz_jitter:      float = 0.01

    # Preprocessing (falcon-style, https://github.com/bittremieux-lab/falcon)
    # MUST be kept in sync with proteoclust_fixed.py's Config — training and
    # inference must see the same spectrum preprocessing, or the encoder
    # sees a different input distribution than what it was trained on.
    quality_filter_enabled: bool = True    # safe to enable independently of
                                             # proteoclust's copy — this only
                                             # decides which spectra get
                                             # trained on, not how retained
                                             # spectra get tokenized.
    quality_min_peaks: int = 5
    quality_min_mz_range: float = 250.0     # Da

    remove_precursor_peaks: bool = True    # DANGER: if you flip this on for
                                             # a new training run, the
                                             # resulting checkpoint is ONLY
                                             # valid for inference if
                                             # proteoclust_fixed.py's Config
                                             # ALSO has remove_precursor_peaks
                                             # =True with a matching
                                             # precursor_removal_tol_da. Not
                                             # compatible with
                                             # pretrained_encoder_v8.pt or any
                                             # earlier checkpoint.
    precursor_removal_tol_da: float = 1.5


# ──────────────────────────────────────────────────────────────────────────
# MGF PARSER — key on SCANS= field, not TITLE string
# ──────────────────────────────────────────────────────────────────────────

def parse_mgf(path: str) -> Dict[str, Tuple[np.ndarray, np.ndarray, float, int]]:
    """Parse MGF -> dict keyed by SCANS= value (matches PD First Scan exactly)."""
    spectra: Dict[str, Tuple] = {}

    if PYTEOMICS_AVAILABLE:
        with pyteomics_mgf.read(path) as reader:
            for spec in tqdm(reader, desc="Parsing MGF"):
                params   = spec.get("params", {})
                pmz_raw  = params.get("pepmass", (0.0,))
                pmz      = float(pmz_raw[0]) if pmz_raw else 0.0
                charge   = int(params.get("charge", [0])[0]) if params.get("charge") else 0
                mz       = np.asarray(spec.get("m/z array", []), dtype=np.float32)
                intensity= np.asarray(spec.get("intensity array", []), dtype=np.float32)
                if len(mz) == 0:
                    continue

                scan_num = None
                if "scans" in params:
                    raw = str(params["scans"]).strip()
                    scan_num = raw.split("|")[0].split(";")[0].strip()

                if not scan_num:
                    title = params.get("title", "")
                    m = re.search(r'scans:\s*["\']?(\d+)', title, re.IGNORECASE)
                    if m:
                        scan_num = m.group(1)

                if not scan_num:
                    title = params.get("title", "")
                    m = re.search(r'NewScan:(\d+)', title, re.IGNORECASE)
                    if m:
                        scan_num = m.group(1)

                if not scan_num:
                    title = params.get("title", "").strip()
                    if title.isdigit():
                        scan_num = title

                if scan_num:
                    spectra[scan_num] = (mz, intensity, pmz, charge)

    else:
        title = pmz = charge = scan_num = None
        mz_list: List[float] = []
        int_list: List[float] = []
        inside = False
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in tqdm(fh, desc="Parsing MGF"):
                line = line.strip()
                if not line:
                    continue
                upper = line.upper()
                if upper == "BEGIN IONS":
                    inside = True
                    mz_list, int_list = [], []
                    title = pmz = charge = scan_num = None
                elif upper == "END IONS":
                    inside = False
                    if mz_list and scan_num:
                        spectra[scan_num] = (
                            np.array(mz_list, dtype=np.float32),
                            np.array(int_list, dtype=np.float32),
                            pmz or 0.0, charge or 0,
                        )
                elif inside and "=" in line:
                    key, _, val = line.partition("=")
                    key = key.strip().upper()
                    val = val.strip()
                    if key == "SCANS":
                        scan_num = val.split("|")[0].strip()
                    elif key == "TITLE":
                        title = val
                        if not scan_num:
                            m = re.search(r'scans:\s*["\']?(\d+)', val, re.IGNORECASE)
                            if m:
                                scan_num = m.group(1)
                    elif key == "PEPMASS":
                        pmz = float(val.split()[0])
                    elif key == "CHARGE":
                        charge = int(val.replace("+","").replace("-",""))
                elif inside and line and line[0].isdigit():
                    parts = line.split()
                    if len(parts) >= 2:
                        try:
                            mz_list.append(float(parts[0]))
                            int_list.append(float(parts[1]))
                        except ValueError:
                            pass

    logging.info(f"Parsed {len(spectra):,} spectra from {Path(path).name}")
    return spectra


# ──────────────────────────────────────────────────────────────────────────
# PSM LOADERS
# ──────────────────────────────────────────────────────────────────────────

def load_psm_proteome_discoverer(path: str, fdr: float) -> pd.DataFrame:
    sep = "," if path.lower().endswith(".csv") else "\t"
    df  = pd.read_csv(path, sep=sep, low_memory=False)
    df.columns = df.columns.str.strip()
    qval_col = next((c for c in ["Percolator q-Value","q-Value","q_value","FDR"]
                     if c in df.columns), None)
    pep_col  = next((c for c in ["Annotated Sequence","Sequence","Peptide","peptide"]
                     if c in df.columns), None)
    scan_col = next((c for c in ["First Scan","Spectrum File","ScanNum","scan","TITLE"]
                     if c in df.columns), None)
    if pep_col  is None: raise ValueError(f"No peptide column. Cols: {list(df.columns)}")
    if scan_col is None: raise ValueError(f"No scan column.    Cols: {list(df.columns)}")
    df = df.rename(columns={pep_col:"peptide", scan_col:"scan_id"})
    if qval_col:
        df = df[pd.to_numeric(df[qval_col], errors="coerce") <= fdr]
    df["scan_id"] = df["scan_id"].astype(str).str.strip()
    df["peptide"] = df["peptide"].astype(str).str.strip().str.upper()

    # FIX 17: keep only the single best-confidence hit per scan_id. Without
    # this, a scan with two ambiguous PSM rows for DIFFERENT peptides
    # survives as two rows after drop_duplicates() (since the pair
    # (scan_id, peptide) is unique even though scan_id is not) — one
    # physical spectrum then gets pulled toward two different classes,
    # which is pure label noise for a metric-learning objective.
    conf_col = next((c for c in ["Confidence","XCorr","Ions Score",
                                  "Percolator SVM Score","Score"]
                     if c in df.columns), None)
    df = df[["scan_id","peptide"] + ([conf_col] if conf_col else [])].dropna(
        subset=["scan_id","peptide"])
    if conf_col:
        df[conf_col] = pd.to_numeric(df[conf_col], errors="coerce")
        df = df.sort_values(conf_col, ascending=False)
        n_before = len(df)
        df = df.drop_duplicates(subset="scan_id", keep="first")
        if n_before != len(df):
            logging.info(f"FIX 17: dropped {n_before - len(df):,} lower-confidence "
                        f"duplicate PSM rows sharing a scan_id (kept best {conf_col})")
        df = df[["scan_id","peptide"]]
    else:
        n_before = len(df)
        df = df.drop_duplicates(subset="scan_id", keep="first")
        if n_before != len(df):
            logging.warning(f"No confidence/score column found — dropped "
                           f"{n_before - len(df):,} duplicate-scan PSM rows "
                           f"arbitrarily (kept first). Consider adding a "
                           f"confidence column for a principled choice.")
    df = df.drop_duplicates()
    logging.info(f"Loaded {len(df):,} PSMs (FDR<={fdr}) from {Path(path).name}")
    return df

def load_psm_comet(path: str, fdr: float) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", comment="#", low_memory=False)
    df.columns = df.columns.str.strip()
    if "e-value" in df.columns:
        df = df[pd.to_numeric(df["e-value"], errors="coerce") <= fdr]
    df = df.rename(columns={"plain_peptide":"peptide","scan":"scan_id"})
    if "peptide" not in df.columns and "modified_peptide" in df.columns:
        df = df.rename(columns={"modified_peptide":"peptide"})
    df["scan_id"] = df["scan_id"].astype(str).str.strip()
    df["peptide"] = df["peptide"].astype(str).str.strip().str.upper()
    return df[["scan_id","peptide"]].dropna().drop_duplicates()

def load_psm_generic(path: str, fdr: float,
                     scan_col="scan_id", peptide_col="peptide",
                     qval_col="q_value") -> pd.DataFrame:
    sep = "\t" if path.endswith((".tsv",".txt")) else ","
    df  = pd.read_csv(path, sep=sep, low_memory=False)
    df.columns = df.columns.str.strip()
    if qval_col in df.columns:
        df = df[pd.to_numeric(df[qval_col], errors="coerce") <= fdr]
    df = df.rename(columns={scan_col:"scan_id", peptide_col:"peptide"})
    df["scan_id"] = df["scan_id"].astype(str).str.strip()
    df["peptide"] = df["peptide"].astype(str).str.strip().str.upper()
    return df[["scan_id","peptide"]].dropna().drop_duplicates()

def load_psm(path: str, fmt: str, fdr: float) -> pd.DataFrame:
    if fmt == "proteome_discoverer": return load_psm_proteome_discoverer(path, fdr)
    if fmt == "comet":               return load_psm_comet(path, fdr)
    return load_psm_generic(path, fdr)


# ──────────────────────────────────────────────────────────────────────────
# AUGMENTATION: peak drop + intensity noise + m/z jitter
# ──────────────────────────────────────────────────────────────────────────

def passes_quality_filter(mz: np.ndarray, min_peaks: int, min_mz_range: float) -> bool:
    """falcon-style quality pre-filter (Bittremieux et al.) — MUST match the
    identically-named function in proteoclust_fixed.py exactly if you use
    both quality_filter_enabled=True here AND cfg.quality_filter_enabled=True
    there, so training and inference see the same spectrum population."""
    if len(mz) < min_peaks:
        return False
    if len(mz) > 0 and (mz.max() - mz.min()) < min_mz_range:
        return False
    return True


def remove_precursor_peaks(mz: np.ndarray, intensity: np.ndarray,
                           precursor_mz: float, charge: int,
                           tol_da: float = 1.5):
    """falcon-style precursor-peak removal. MUST be applied identically here
    and in proteoclust_fixed.py's UniversalSpectrumEncoder — see
    PretrainConfig.remove_precursor_peaks docstring. If you enable this for
    a NEW training run, the resulting checkpoint requires
    remove_precursor_peaks=True in proteoclust_fixed.py's Config too; it is
    NOT compatible with pretrained_encoder_v8.pt or any checkpoint trained
    before this flag existed."""
    PROTON_MASS = 1.007276
    if len(mz) == 0 or charge <= 0:
        return mz, intensity
    mask = np.ones(len(mz), dtype=bool)
    neutral_mass = precursor_mz * charge - PROTON_MASS * charge
    for z in range(1, charge + 1):
        mz_at_z = neutral_mass / z + PROTON_MASS
        mask &= np.abs(mz - mz_at_z) > tol_da
    return mz[mask], intensity[mask]


def spectrum_to_tokens(mz: np.ndarray, intensity: np.ndarray,
                       top_k: int, mz_min: float, mz_max: float,
                       augment: bool = False,
                       aug_drop: float = 0.15,
                       aug_noise: float = 0.08,
                       aug_mz_jitter: float = 0.01,
                       remove_precursor: bool = False,
                       precursor_mz: float = 0.0,
                       precursor_charge: int = 0,
                       precursor_tol_da: float = 1.5) -> Tuple[np.ndarray, np.ndarray]:
    if remove_precursor and len(mz) > 0:
        mz, intensity = remove_precursor_peaks(
            mz, intensity, precursor_mz, precursor_charge, precursor_tol_da)

    if augment and len(mz) > 0:
        if aug_drop > 0:
            keep = np.random.rand(len(mz)) > aug_drop
            if keep.sum() > 0:
                mz, intensity = mz[keep], intensity[keep]
        if aug_noise > 0 and len(intensity) > 0:
            noise = np.random.randn(len(intensity)).astype(np.float32) * aug_noise
            intensity = np.clip(intensity + noise * intensity.max(), 0, None)
        if aug_mz_jitter > 0 and len(mz) > 0:
            jitter = np.random.randn(len(mz)).astype(np.float32) * aug_mz_jitter
            mz = np.clip(mz + jitter * (mz_max - mz_min), mz_min, mz_max)

    tokens = np.zeros((top_k, 2), dtype=np.float32)
    if len(mz) == 0:
        return tokens, np.ones(top_k, dtype=bool)
    if len(mz) > top_k:
        idx = np.argpartition(intensity, -top_k)[-top_k:]
        mz, intensity = mz[idx], intensity[idx]
    order = np.argsort(mz)
    mz, intensity = mz[order], intensity[order]
    norm_mz  = np.clip((mz - mz_min) / (mz_max - mz_min), 0.0, 1.0)
    max_i    = intensity.max()
    norm_int = np.sqrt(intensity / max_i) if max_i > 0 else intensity
    n = len(mz)
    tokens[:n, 0] = norm_mz
    tokens[:n, 1] = norm_int
    pad_mask = np.ones(top_k, dtype=bool)
    pad_mask[:n] = False
    return tokens, pad_mask


# ──────────────────────────────────────────────────────────────────────────
# DATASET — O(1) scan matching via SCANS= index
# ──────────────────────────────────────────────────────────────────────────

class SpectrumContrastiveDataset(Dataset):
    def __init__(self, mgf_spectra: Dict[str, Tuple],
                 psm_df: pd.DataFrame, cfg: PretrainConfig,
                 augment: bool = True):
        self.cfg     = cfg
        self.augment = augment

        mgf_key_set = set(mgf_spectra.keys())
        logging.info(f"MGF scan index: {len(mgf_key_set):,} spectra")

        key_by_scan: Dict[str, str] = {}
        unmatched = 0
        for _, row in tqdm(psm_df.iterrows(), total=len(psm_df),
                           desc="Matching PSMs to MGF", unit=" PSM"):
            sid = str(row["scan_id"]).strip()
            if sid in mgf_key_set:
                key_by_scan[sid] = sid
            else:
                unmatched += 1

        logging.info(f"PSM matching: {len(key_by_scan):,} matched, "
                     f"{unmatched:,} unmatched")
        if unmatched > len(key_by_scan) and unmatched > 0:
            sample_sids  = [str(r["scan_id"]) for _,r in psm_df.head(3).iterrows()]
            sample_keys  = list(mgf_key_set)[:3]
            logging.warning(f"Most PSMs unmatched!\n"
                            f"  PSM scan_ids  : {sample_sids}\n"
                            f"  MGF scan keys : {sample_keys}\n"
                            f"  -> Check that MGF has SCANS= lines matching First Scan")

        # FIX 19: group by (peptide, charge) — NOT peptide alone. A peptide
        # observed at 2+ vs 3+ produces a genuinely different fragment-ion
        # spectrum (different charge-reduced species, different peaks
        # surviving the top-K intensity filter), so treating both charge
        # states as one SupCon class was forcing the encoder to pull apart
        # genuinely-different spectra together as "positive pairs". Measured
        # on a real 506K-spectrum COREAD run: charge mismatch was present in
        # 42% of intra-peptide pairs overall but 99.9% of the worst
        # (near-zero-similarity) pairs — i.e. it explains almost the entire
        # catastrophic tail of the similarity distribution. Charge comes
        # from the MGF spectrum itself (mgf_spectra[...][3]), not from the
        # PSM CSV, so this doesn't depend on a "Charge" column being present
        # or consistently named in the PSM export.
        peptide_to_scans: Dict[str, List[str]] = {}
        n_quality_dropped = 0
        for _, row in psm_df.iterrows():
            sid = str(row["scan_id"])
            if sid not in key_by_scan:
                continue
            spec_mz = mgf_spectra[key_by_scan[sid]][0]
            if cfg.quality_filter_enabled and not passes_quality_filter(
                    spec_mz, cfg.quality_min_peaks, cfg.quality_min_mz_range):
                n_quality_dropped += 1
                continue
            charge = mgf_spectra[key_by_scan[sid]][3]
            composite_label = f"{row['peptide']}|z{charge}"
            peptide_to_scans.setdefault(composite_label, []).append(sid)

        if cfg.quality_filter_enabled and n_quality_dropped:
            logging.info(f"Quality filter: {n_quality_dropped:,} spectra dropped "
                        f"(< {cfg.quality_min_peaks} peaks or "
                        f"< {cfg.quality_min_mz_range} Da range)")

        peptide_to_scans = {p: s for p, s in peptide_to_scans.items()
                            if len(s) >= cfg.min_spectra_per_peptide}

        if cfg.max_peptides:
            keys = list(peptide_to_scans.keys())
            random.shuffle(keys)
            peptide_to_scans = {k: peptide_to_scans[k] for k in keys[:cfg.max_peptides]}

        self.samples:     List[Tuple[str, int]] = []
        self.label_names: List[str]             = sorted(peptide_to_scans.keys())
        self.label_to_idx = {p: i for i, p in enumerate(self.label_names)}
        for pep, scans in peptide_to_scans.items():
            lbl = self.label_to_idx[pep]
            for sid in scans:
                self.samples.append((sid, lbl))

        self.mgf_spectra = mgf_spectra
        self.key_by_scan = key_by_scan
        logging.info(f"Dataset: {len(self.samples):,} spectra, "
                     f"{len(self.label_names):,} unique peptides")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        scan_id, label_idx = self.samples[idx]
        mz, intensity, pmz, charge = self.mgf_spectra[self.key_by_scan[scan_id]]
        tokens, pad_mask = spectrum_to_tokens(
            mz, intensity,
            top_k=self.cfg.top_k_peaks,
            mz_min=self.cfg.mz_min, mz_max=self.cfg.mz_max,
            augment=self.augment and self.cfg.augment,
            aug_drop=self.cfg.aug_peak_drop_prob,
            aug_noise=self.cfg.aug_intensity_noise,
            aug_mz_jitter=self.cfg.aug_mz_jitter,
            remove_precursor=self.cfg.remove_precursor_peaks,
            precursor_mz=pmz, precursor_charge=charge,
            precursor_tol_da=self.cfg.precursor_removal_tol_da,
        )
        return (torch.from_numpy(tokens),
                torch.from_numpy(pad_mask),
                torch.tensor(label_idx, dtype=torch.long))

    def class_weights(self) -> torch.Tensor:
        counts = np.zeros(len(self.label_names), dtype=np.float32)
        for _, lbl in self.samples:
            counts[lbl] += 1
        w = 1.0 / np.maximum(counts, 1)
        return torch.tensor([w[lbl] for _, lbl in self.samples], dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────
# FIX 16 — P x K BATCH SAMPLER: guarantees positive pairs every batch
# ──────────────────────────────────────────────────────────────────────────

class PKBatchSampler(torch.utils.data.Sampler):
    """Yields batches of P classes x K samples each.

    A per-sample WeightedRandomSampler draws each batch slot independently,
    so with thousands of unique peptides the expected number of draws for
    any one peptide in a single batch is ~batch_size / num_classes, usually
    far below 2 — meaning most classes present in a batch never get a
    second draw and SupCon silently discards them (`has_pos.any()` is
    False for that row). This sampler instead explicitly selects P peptide
    classes per batch and K spectra of each, so every included class is
    GUARANTEED at least one positive pair. Effective batch size is P*K.

    If a class has fewer than K spectra, its scans are sampled with
    replacement to fill K slots (mild augmentation-like repetition, fine
    since spectrum_to_tokens still applies independent augmentation to
    each draw when cfg.augment=True).
    """
    def __init__(self, dataset: "SpectrumContrastiveDataset", P: int, K: int,
                 seed: int = 0):
        self.P, self.K = P, K
        self.class_to_indices: Dict[int, List[int]] = {}
        for idx, (_, lbl) in enumerate(dataset.samples):
            self.class_to_indices.setdefault(lbl, []).append(idx)
        # every class here already has >= min_spectra_per_peptide (>=2) by
        # construction of SpectrumContrastiveDataset, so all are usable
        self.classes = list(self.class_to_indices.keys())
        if len(self.classes) < P:
            logging.warning(f"PKBatchSampler: only {len(self.classes)} classes "
                           f"available but P={P} requested; reducing P to "
                           f"{len(self.classes)}.")
            self.P = len(self.classes)
        self.batches_per_epoch = max(1, len(dataset) // (self.P * self.K))
        self._rng = random.Random(seed)

    def __iter__(self):
        for _ in range(self.batches_per_epoch):
            chosen_classes = self._rng.sample(self.classes, self.P)
            batch: List[int] = []
            for c in chosen_classes:
                idxs = self.class_to_indices[c]
                if len(idxs) >= self.K:
                    batch.extend(self._rng.sample(idxs, self.K))
                else:
                    batch.extend(self._rng.choices(idxs, k=self.K))
            self._rng.shuffle(batch)
            yield batch

    def __len__(self):
        return self.batches_per_epoch


# ──────────────────────────────────────────────────────────────────────────
# TRANSFORMER: projection head (SimCLR trick)
# ──────────────────────────────────────────────────────────────────────────

class LightTransformerEncoder(nn.Module):
    def __init__(self, cfg: PretrainConfig):
        super().__init__()
        d = cfg.embed_dim
        self.input_proj = nn.Linear(2, d)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d, nhead=cfg.n_heads, dim_feedforward=cfg.ff_dim,
            dropout=0.1, batch_first=True, norm_first=True,
        )
        self.encoder  = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.proj_norm = nn.Sequential(nn.Linear(d, d), nn.LayerNorm(d))
        self.proj_head = nn.Sequential(
            nn.Linear(d, d * 2), nn.ReLU(), nn.Linear(d * 2, 256),
        )

    def forward(self, x: torch.Tensor, pad_mask: torch.Tensor,
                return_proj: bool = False) -> torch.Tensor:
        x     = self.input_proj(x)
        x     = self.encoder(x, src_key_padding_mask=pad_mask)
        valid = (~pad_mask).unsqueeze(-1).float()
        x     = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        backbone = F.normalize(self.proj_norm(x), dim=-1)   # inference output
        if return_proj:
            return F.normalize(self.proj_head(backbone), dim=-1)  # training loss
        return backbone


# ──────────────────────────────────────────────────────────────────────────
# SUPCON LOSS + UNIFORMITY REGULARISATION  (FIX 12)
# ──────────────────────────────────────────────────────────────────────────

class SupConLoss(nn.Module):
    """SupCon (Khosla et al., NeurIPS 2020) + Uniformity (Wang & Isola, ICML 2020).

    FIX 12: The uniformity term now ONLY acts on pairs from different classes
    (true negatives). Previously it was computed over all off-diagonal pairs,
    including same-peptide (positive) pairs — which directly fights SupCon's
    alignment objective and was capping val_intra_sim almost immediately
    after the first epoch. lambda_uniform is also annealed in from 0 over
    `lambda_uniform_warmup_epochs`, via `set_uniform_weight()`, so alignment
    gets to establish class structure before the spreading pressure kicks in.

    After each forward() call, `self.last_supcon` and `self.last_uniform`
    hold the two components separately (both as floats) for logging.
    """
    def __init__(self, temperature: float = 0.10, lambda_uniform: float = 0.15):
        super().__init__()
        self.temperature      = temperature
        self.target_lambda    = lambda_uniform
        self.current_lambda   = lambda_uniform   # can be overridden by scheduler
        self.last_supcon      = 0.0
        self.last_uniform     = 0.0

    def set_uniform_weight(self, w: float):
        self.current_lambda = w

    def uniformity_loss(self, emb: torch.Tensor, exclude_mask: torch.Tensor) -> torch.Tensor:
        """exclude_mask: True where the pair (i,j) should NOT be used
        (self-pairs and same-class/positive pairs)."""
        sim     = torch.matmul(emb, emb.T)
        sq_dist = 2.0 - 2.0 * sim
        valid   = ~exclude_mask
        if valid.sum() == 0:
            return torch.tensor(0.0, device=emb.device)
        return torch.log(torch.exp(-2.0 * sq_dist[valid]).mean() + 1e-8)

    def forward(self, emb: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        N         = emb.shape[0]
        device    = emb.device
        sim       = torch.matmul(emb, emb.T) / self.temperature
        self_mask = torch.eye(N, dtype=torch.bool, device=device)
        sim       = sim.masked_fill(self_mask, -9e15)
        labels    = labels.view(-1, 1)
        pos_mask  = (labels == labels.T) & ~self_mask
        has_pos   = pos_mask.any(dim=1)

        if not has_pos.any():
            self.last_supcon, self.last_uniform = 0.0, 0.0
            return torch.tensor(0.0, device=device, requires_grad=True)

        log_denom     = torch.logsumexp(sim, dim=1, keepdim=True)
        log_prob      = sim - log_denom
        mean_log_prob = (pos_mask * log_prob).sum(dim=1) / pos_mask.sum(dim=1).clamp(min=1)
        supcon        = -mean_log_prob[has_pos].mean()

        # FIX 12: exclude both self-pairs and positive pairs from uniformity
        exclude_mask = self_mask | pos_mask
        uniform      = self.uniformity_loss(emb, exclude_mask)

        self.last_supcon  = supcon.item()
        self.last_uniform = uniform.item()
        return supcon + self.current_lambda * uniform


# ──────────────────────────────────────────────────────────────────────────
# LEARNING RATE SCHEDULE
# ──────────────────────────────────────────────────────────────────────────

def cosine_lr_schedule(optimizer, epoch, total_epochs, base_lr, warmup_epochs):
    if epoch < warmup_epochs:
        lr = base_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    for pg in optimizer.param_groups:
        pg["lr"] = lr
    return lr

def uniform_weight_schedule(epoch, target_lambda, warmup_epochs):
    """FIX 12: ramp lambda_uniform from 0 -> target over warmup_epochs so
    SupCon establishes class clustering before the spreading term engages."""
    if warmup_epochs <= 0:
        return target_lambda
    return target_lambda * min(1.0, (epoch + 1) / warmup_epochs)


# ──────────────────────────────────────────────────────────────────────────
# TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────

def train(cfg: PretrainConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-7s  %(message)s",
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler("pretrain.log")],
    )
    logger = logging.getLogger("pretrain")
    random.seed(cfg.seed); np.random.seed(cfg.seed); torch.manual_seed(cfg.seed)

    device = torch.device(cfg.device)
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    logger.info("Loading MGF ...")
    mgf_spectra = parse_mgf(cfg.mgf_path)
    logger.info("Loading PSM results ...")
    psm_df = load_psm(cfg.psm_path, cfg.psm_format, cfg.fdr_threshold)

    full_ds = SpectrumContrastiveDataset(mgf_spectra, psm_df, cfg, augment=True)
    if len(full_ds) == 0:
        logger.error("No matched PSM-MGF pairs found. Check SCANS= field in MGF.")
        return

    # FIX 19 fallout: full_ds.label_names are now composite "PEPTIDE|zN"
    # labels, but psm_df["peptide"] only ever holds the plain sequence — so
    # splitting/filtering directly against the composite labels matched
    # nothing (psm_train/psm_val came back empty -> ZeroDivisionError in
    # PKBatchSampler). Split at the BASE peptide level instead: this also
    # keeps all charge-state variants of a given peptide entirely in train
    # or entirely in val (no leakage), and correctly matches psm_df's
    # plain-sequence "peptide" column.
    base_peptides = sorted({lbl.rsplit("|z", 1)[0] for lbl in full_ds.label_names})
    random.shuffle(base_peptides)
    n_val          = max(1, int(len(base_peptides) * cfg.val_split))
    val_peptides   = set(base_peptides[:n_val])
    train_peptides = set(base_peptides[n_val:])

    psm_train = psm_df[psm_df["peptide"].isin(train_peptides)]
    psm_val   = psm_df[psm_df["peptide"].isin(val_peptides)]

    train_ds = SpectrumContrastiveDataset(mgf_spectra, psm_train, cfg, augment=True)
    val_ds   = SpectrumContrastiveDataset(mgf_spectra, psm_val,   cfg, augment=False)
    logger.info(f"Train: {len(train_ds):,} spectra / {len(train_peptides):,} base peptides "
                f"({len({l for l in train_ds.label_names}):,} peptide-charge classes)")
    logger.info(f"Val  : {len(val_ds):,}   spectra / {len(val_peptides):,} base peptides "
                f"({len({l for l in val_ds.label_names}):,} peptide-charge classes)")

    # FIX 16: P x K batch sampler replaces WeightedRandomSampler — every
    # batch now guarantees positive pairs for each of the P classes in it,
    # instead of relying on independent per-slot draws to coincide.
    pk_sampler = PKBatchSampler(train_ds, P=cfg.pk_classes_per_batch,
                                K=cfg.pk_samples_per_class, seed=cfg.seed)
    effective_batch_size = pk_sampler.P * pk_sampler.K
    logger.info(f"PK sampler: P={pk_sampler.P} classes x K={pk_sampler.K} "
                f"samples/class = {effective_batch_size} effective batch size, "
                f"{len(pk_sampler)} batches/epoch")
    loader     = DataLoader(train_ds, batch_sampler=pk_sampler,
                            num_workers=0, pin_memory=(device.type=="cuda"))
    val_loader = DataLoader(val_ds,   batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, pin_memory=(device.type=="cuda"))

    model     = LightTransformerEncoder(cfg).to(device)
    criterion = SupConLoss(temperature=cfg.temperature,
                           lambda_uniform=cfg.lambda_uniform)

    # FIX 18: don't apply weight decay to 1-D params (LayerNorm scale/bias,
    # all biases) — standard practice, decaying these toward 0 rather than
    # letting them be purely gradient-driven can slow/destabilize training.
    decay_params, no_decay_params = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay_params if p.ndim <= 1 else decay_params).append(p)
    optimizer = torch.optim.AdamW([
        {"params": decay_params,    "weight_decay": cfg.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ], lr=cfg.lr)
    n_params  = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,}")
    logger.info(f"Training: {cfg.epochs} epochs, batch={cfg.batch_size}, "
                f"lr={cfg.lr}, tau={cfg.temperature}, "
                f"lambda_u(target)={cfg.lambda_uniform} "
                f"(warmup {cfg.lambda_uniform_warmup_epochs} epochs)")

    # FIX 15: checkpoint on class separation (intra - inter), not raw intra
    best_metric      = -1e9
    best_epoch       = 0
    patience_counter = 0

    for epoch in range(cfg.epochs):
        lr = cosine_lr_schedule(optimizer, epoch, cfg.epochs, cfg.lr,
                                cfg.warmup_epochs)
        uw = uniform_weight_schedule(epoch, cfg.lambda_uniform,
                                     cfg.lambda_uniform_warmup_epochs)
        criterion.set_uniform_weight(uw)

        model.train()
        epoch_loss = epoch_supcon = epoch_uniform = 0.0; n_batches = 0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{cfg.epochs} lr={lr:.2e} "
                                  f"lam_u={uw:.3f}", unit=" batch")
        for tokens, pad_mask, labels in pbar:
            tokens   = tokens.to(device)
            pad_mask = pad_mask.to(device)
            labels   = labels.to(device)
            emb  = model(tokens, pad_mask, return_proj=True)
            loss = criterion(emb, labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss    += loss.item()
            epoch_supcon  += criterion.last_supcon
            epoch_uniform += criterion.last_uniform
            n_batches += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}",
                             supcon=f"{criterion.last_supcon:.4f}",
                             unif=f"{criterion.last_uniform:.4f}")
        avg_loss    = epoch_loss    / max(n_batches, 1)
        avg_supcon  = epoch_supcon  / max(n_batches, 1)
        avg_uniform = epoch_uniform / max(n_batches, 1)

        # Val metric: intra-class AND inter-class similarity on backbone
        model.eval()
        all_emb, all_lbl = [], []
        with torch.no_grad():
            for tokens, pad_mask, labels in val_loader:
                emb = model(tokens.to(device), pad_mask.to(device),
                            return_proj=False).cpu()
                all_emb.append(emb); all_lbl.append(labels)
        all_emb = torch.cat(all_emb); all_lbl = torch.cat(all_lbl)

        intra_sims = []
        for lbl in all_lbl.unique():
            mask = all_lbl == lbl
            if mask.sum() < 2: continue
            ce = all_emb[mask]
            sm = ce @ ce.T
            k  = ce.shape[0]
            upper = sm[torch.triu(torch.ones(k,k), diagonal=1).bool()]
            intra_sims.append(upper.mean().item())
        val_intra = float(np.mean(intra_sims)) if intra_sims else 0.0

        # FIX 14: inter-class similarity for context (full pairwise, off-diag,
        # different-label pairs)
        full_sim  = all_emb @ all_emb.T
        n         = full_sim.shape[0]
        diff_mask = (all_lbl.view(-1,1) != all_lbl.view(1,-1)) & \
                    ~torch.eye(n, dtype=torch.bool)
        val_inter = full_sim[diff_mask].mean().item() if diff_mask.any() else 0.0
        val_separation = val_intra - val_inter

        logger.info(f"Epoch {epoch+1:3d}/{cfg.epochs}  "
                    f"loss={avg_loss:.4f} (supcon={avg_supcon:.4f} "
                    f"uniform={avg_uniform:.4f})  "
                    f"val_intra={val_intra:.4f}  val_inter={val_inter:.4f}  "
                    f"val_separation={val_separation:.4f}  lr={lr:.2e}")

        # FIX 15: checkpoint/early-stop on separation, not raw intra-sim
        if val_separation > best_metric:
            best_metric      = val_separation
            best_epoch       = epoch + 1
            patience_counter = 0
            torch.save({
                "epoch":            epoch + 1,
                "train_loss":       avg_loss,
                "val_intra_sim":    val_intra,
                "val_inter_sim":    val_inter,
                "val_separation":   val_separation,
                "loss":             avg_loss,        # compatibility key
                "model_state_dict": model.state_dict(),
                "cfg":              cfg.__dict__,
            }, cfg.output_path)
            logger.info(f"  Saved best (val_separation={val_separation:.4f})"
                        f" -> {cfg.output_path}")
        else:
            patience_counter += 1
            logger.info(f"  No improvement. Patience {patience_counter}/"
                        f"{cfg.early_stopping_patience}")
            if patience_counter >= cfg.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break

    logger.info(f"Done. Best epoch: {best_epoch}, val_separation: {best_metric:.4f}")
    logger.info(f"Weights saved to: {cfg.output_path}")


# ──────────────────────────────────────────────────────────────────────────
# VERIFY CHECKPOINT
# ──────────────────────────────────────────────────────────────────────────

def verify_checkpoint(path: str) -> None:
    ckpt = torch.load(path, map_location="cpu")
    print(f"\nCheckpoint: {path}")
    print(f"  Epoch          : {ckpt['epoch']}")
    print(f"  val_intra_sim  : {ckpt.get('val_intra_sim', 'N/A')}")
    print(f"  val_inter_sim  : {ckpt.get('val_inter_sim', 'N/A')}")
    print(f"  val_separation : {ckpt.get('val_separation', 'N/A')}")
    print(f"  train_loss     : {ckpt.get('train_loss', 'N/A')}")
    print(f"  embed_dim      : {ckpt['cfg']['embed_dim']}")
    print(f"  n_heads        : {ckpt['cfg']['n_heads']}")
    print(f"  n_layers       : {ckpt['cfg']['n_layers']}")


# ──────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────

def build_parser():
    p = argparse.ArgumentParser(prog="pretrain_transformer",
        description="SupCon pre-training for ProteoCLUST (fixed uniformity/LR)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--mgf")
    p.add_argument("--psm")
    p.add_argument("--format", default="proteome_discoverer", dest="psm_format",
                   choices=["proteome_discoverer","comet","msfragger","generic"])
    p.add_argument("--out",    default="pretrained_encoder.pt")
    p.add_argument("--epochs", type=int,   default=30)
    p.add_argument("--batch",  type=int,   default=512,  dest="batch_size",
                   help="used for the validation loader only; train batch "
                        "size is --pk-classes * --pk-samples (FIX 16)")
    p.add_argument("--pk-classes", type=int, default=64, dest="pk_classes_per_batch",
                   help="P: unique peptides per training batch")
    p.add_argument("--pk-samples", type=int, default=8,  dest="pk_samples_per_class",
                   help="K: spectra per peptide per training batch")
    p.add_argument("--lr",     type=float, default=1e-4)
    p.add_argument("--temp",   type=float, default=0.10, dest="temperature")
    p.add_argument("--lambda-uniform", type=float, default=0.15, dest="lambda_uniform")
    p.add_argument("--lambda-uniform-warmup", type=int, default=5,
                   dest="lambda_uniform_warmup_epochs")
    p.add_argument("--weight-decay",   type=float, default=1e-2, dest="weight_decay")
    p.add_argument("--fdr",    type=float, default=0.01, dest="fdr_threshold")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--min-spectra", type=int, default=3,
                   dest="min_spectra_per_peptide")
    p.add_argument("--val-split",   type=float, default=0.15, dest="val_split")
    p.add_argument("--patience",    type=int,   default=8,
                   dest="early_stopping_patience")
    p.add_argument("--no-augment",  action="store_true")
    p.add_argument("--verify", metavar="CKPT_PATH")
    return p

def main():
    args = build_parser().parse_args()
    if args.verify:
        verify_checkpoint(args.verify); return
    if not args.mgf or not args.psm:
        build_parser().error("--mgf and --psm are required unless --verify is used")
    cfg = PretrainConfig(
        mgf_path=args.mgf, psm_path=args.psm, psm_format=args.psm_format,
        output_path=args.out, epochs=args.epochs, batch_size=args.batch_size,
        pk_classes_per_batch=args.pk_classes_per_batch,
        pk_samples_per_class=args.pk_samples_per_class,
        lr=args.lr, temperature=args.temperature, lambda_uniform=args.lambda_uniform,
        lambda_uniform_warmup_epochs=args.lambda_uniform_warmup_epochs,
        weight_decay=args.weight_decay, fdr_threshold=args.fdr_threshold,
        device=args.device, min_spectra_per_peptide=args.min_spectra_per_peptide,
        val_split=args.val_split, early_stopping_patience=args.early_stopping_patience,
        augment=not args.no_augment,
    )
    train(cfg)


# ──────────────────────────────────────────────────────────────────────────
# SPYDER / DIRECT EXECUTION
# ──────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        main()
    else:
        # ── Edit paths below ────────────────────────────────────────────
        cfg = PretrainConfig(
            mgf_path = r"D:\Shraddha\Original_MGFs\COREAD\20201022_FS_Choudhary_LMS2_FS03_MS2_16plex_2.mgf",
            psm_path = r"D:\Shraddha\sCOREAD\20201022_FS_Choudhary_LMS2_FS03_MS2_16plex(2)\20201022_FS_Choudhary_LMS2_FS03_MS2_16plex(2)_PSMs.csv",
            psm_format  = "proteome_discoverer",
            output_path = r"D:\Shraddha\falcon\files\pretrained_encoder_v9_falcon-style.pt",

            epochs                        = 50,
            batch_size                    = 512,     # val loader only now
            pk_classes_per_batch          = 64,       # FIX 16: P
            pk_samples_per_class          = 8,        # FIX 16: K (P*K=512)
            val_split                     = 0.15,
            temperature                   = 0.10,
            lambda_uniform                = 0.15,   # FIX 12: was 0.5
            lambda_uniform_warmup_epochs  = 5,       # FIX 12: new
            lr                            = 5e-5,    # FIX 13: was 2e-5
            warmup_epochs                 = 3,
            weight_decay                  = 1e-2,
            min_spectra_per_peptide       = 3,
            aug_peak_drop_prob            = 0.15,
            aug_intensity_noise           = 0.08,
            aug_mz_jitter                 = 0.01,
            early_stopping_patience       = 15,
            device = "cuda",
        )
        train(cfg)