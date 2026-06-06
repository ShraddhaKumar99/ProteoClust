"""
Synthetic smoke-test for ProteoCLUST — no real MGF file required.
Run from Spyder (F5) or terminal: python test_proteoclust.py
"""
import os
import tempfile
import numpy as np
from proteoclust import Config, load_spectra, UniversalSpectrumEncoder
from proteoclust import run_hgce, run_bar, run_pipeline, Spectrum


# ── 1. Build a tiny synthetic MGF ────────────────────────────────────────────

def _make_synthetic_mgf(path: str, n_clusters: int = 8,
                        members_per_cluster: int = 12,
                        noise_spectra: int = 10) -> None:
    rng = np.random.RandomState(42)
    with open(path, "w") as fh:
        scan = 0
        for cid in range(n_clusters):
            base_mz = rng.uniform(200, 1200, 50)
            base_int = rng.exponential(1000, 50)
            pmz = rng.uniform(400, 900)
            for _ in range(members_per_cluster):
                scan += 1
                mz = base_mz + rng.normal(0, 0.01, 50)
                inten = base_int * (1 + rng.normal(0, 0.05, 50))
                inten = np.clip(inten, 1, None)
                fh.write("BEGIN IONS\n")
                fh.write(f"TITLE=scan_{scan}\n")
                fh.write(f"PEPMASS={pmz:.4f}\n")
                fh.write("CHARGE=2+\n")
                for m, i in zip(np.sort(mz), inten):
                    fh.write(f"{m:.4f} {i:.2f}\n")
                fh.write("END IONS\n\n")
        # noise spectra — random peaks, varied precursor
        for _ in range(noise_spectra):
            scan += 1
            mz = rng.uniform(100, 1400, 20)
            inten = rng.exponential(500, 20)
            pmz = rng.uniform(300, 1100)
            fh.write("BEGIN IONS\n")
            fh.write(f"TITLE=scan_{scan}_noise\n")
            fh.write(f"PEPMASS={pmz:.4f}\n")
            fh.write("CHARGE=2+\n")
            for m, i in zip(np.sort(mz), inten):
                fh.write(f"{m:.4f} {i:.2f}\n")
            fh.write("END IONS\n\n")


# ── 2. Run pipeline on synthetic data ────────────────────────────────────────

def test_pipeline():
    with tempfile.TemporaryDirectory() as tmpdir:
        mgf_path = os.path.join(tmpdir, "test.mgf")
        out_dir  = os.path.join(tmpdir, "out")
        _make_synthetic_mgf(mgf_path, n_clusters=8, members_per_cluster=12)

        cfg = Config(
            mgf_path=mgf_path,
            output_dir=out_dir,
            embed_dim=64,
            n_heads=2,
            n_layers=1,
            ff_dim=128,
            batch_size=32,
            n_eigenvectors=10,
            ann_n_neighbours=20,
            dbscan_eps=0.20,
            dbscan_min_samples=2,
            edge_threshold=0.60,
            verbose=False,
        )

        clusters = run_pipeline(cfg)

        n_total = 8 * 12 + 10   # 106 spectra
        assert len(clusters) > 0, "Expected at least one cluster"
        print(f"PASS — {len(clusters)} clusters found from {n_total} spectra")

        # Check output files exist
        for fname in ("clusters.tsv", "consensus_spectra.mgf", "summary.txt"):
            fpath = os.path.join(out_dir, fname)
            assert os.path.exists(fpath), f"Missing output: {fname}"
        print("PASS — all output files written")

        # Check quality scores are bounded
        for cl in clusters:
            assert 0.0 <= cl.quality_score <= 1.0, f"Q out of range: {cl.quality_score}"
            assert cl.confidence in ("HIGH", "MEDIUM", "LOW")
        print("PASS — quality scores and confidence tiers valid")

        print("\n=== All tests passed ===")
        return clusters


if __name__ == "__main__":
    results = test_pipeline()
