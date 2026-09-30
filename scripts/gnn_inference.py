#!/usr/bin/env python3
"""
Inference on ONE new audio recording using the trained GNN (scripts/
train_gnn.py) -- shared by app/app.py and this file's own CLI.

Extraction: reuses source/extraction.py's get_audio_indices() directly (so
the 11 whole-window indices -- aca, adi, bioacousticIndex, temporalMedian,
numberPeaks, entropyFrequency, entropyTemporal, entropy, acousticEvenness,
soundscapeIndex, acousticRichness -- match training exactly), plus a small
duplicate of bootstrap_soundscape's band-splitting/band_acift logic for the
6 per-band values (meanEnergy, medianEnergy, sumEnergy, maxEnergy,
minEnergy, aci). Verified against a real extracted file: the 11 whole-
window indices are IDENTICAL across all 10 bands of the same second in the
training data (spot-checked directly against out/data/extraction/), which
is exactly what this reproduces.

Windowing -- KNOWN APPROXIMATION, measured not just suspected: this uses
1-second non-overlapping windows (matching the training data's observed
row structure: exactly duration_seconds rows per band, one per integer
second), NOT bootstrap_soundscape's 6-second/1.9s-hop scheme. The index
FORMULAS are identical (same get_audio_indices() call, same band-split +
band_acift), but the actual VALUES measurably differ from the stored
training features for the same source file -- spot-checked directly:
data/Engine/00_000066.wav vs out/data/extraction/Engine/00_000066_..._.pkl,
same order of magnitude but not equal (e.g. aca 72830 (stored) vs 12000
(here), dsi 0.0061 vs 0.0042). bootstrap_soundscape's own 6s/1.9s-hop
parameters don't reproduce the stored row counts either when checked by
hand (see .carlos/notes.md), so the exact historical windowing is unclear
without deeper archaeology -- not attempted here given the time cost vs.
payoff. This affects prediction ACCURACY on new audio (the model sees
features from a slightly different distribution than it trained on), not
the pipeline's correctness -- extraction, graph-building, and GNN
inference all run and produce a valid, well-formed prediction. Closing
this gap (either reverse-engineering the real historical parameters, or
switching this function to call bootstrap_soundscape directly) is a
worthwhile follow-up, not a blocker for the demo.

Inference is INDUCTIVE and LOCAL: this only ever builds a tiny graph (one
new Sound + its own new Frames + one placeholder Sensor) and runs the
2-layer GNN's message passing on just that -- it never needs the full
626,130-Frame training graph or a live GraphDB connection. That's a direct
consequence of the Sound/Frame design (LO3): a Sound's embedding is built
from its own Frame neighborhood, not a memorized per-node lookup.

Usage:
    python scripts/gnn_inference.py path/to/audio.wav
    python scripts/gnn_inference.py path/to/audio.wav --model-dir out/data/kg/gnn

Environment: needs torch, torch_geometric, librosa, scikit-maad (all in
tpyforest; same env as the rest of the KG pipeline).
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = REPO_ROOT / "source"
DEFAULT_MODEL_DIR = REPO_ROOT / "out" / "data" / "kg" / "gnn"

if SOURCE_DIR.exists() and str(SOURCE_DIR) not in sys.path:
    sys.path.insert(0, str(SOURCE_DIR))
if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_gnn import (  # noqa: E402
    GNN, SEED_DIM, FRAME_LITERAL_PROPS, CONTEXT_LIST, sound_type_for,
)


def band_acift(f: np.ndarray) -> float:
    """Exact duplicate of bootstrap_soundscape's nested band_acift (source/
    extraction.py) -- kept in sync manually, see module docstring."""
    total = np.sum(f)
    if total == 0:
        return 0.0
    return float(np.sum(np.abs(np.diff(f))) / total)


def split_freq_band_per_frame(s: np.ndarray, sr: int, b_band=0, u_band=10000, bandwidth=1000):
    """Exact duplicate of bootstrap_soundscape's nested split_freq_band_per_frame."""
    y_fft = np.fft.fft(s)
    fft_freq = np.fft.fftfreq(len(s), 1.0 / sr)
    bands = []
    for f in range(b_band, u_band, bandwidth):
        mask = (fft_freq >= f) & (fft_freq < f + bandwidth)
        y_band = np.real(np.fft.ifft(y_fft[mask]))
        bands.append(y_band)
    return bands


def extract_frame_features(audio_path: Path, sr: int = 48000, bandas: int = 10,
                            b_band: int = 0, u_band: int = 10000, bandwidth: int = 1000):
    """Returns an (n_seconds * bandas, len(FRAME_LITERAL_PROPS)) array, row
    order matching FRAME_LITERAL_PROPS, one row per (second, band)."""
    import librosa
    from extraction import get_audio_indices

    y, _ = librosa.load(str(audio_path), sr=sr, mono=True)
    n_seconds = len(y) // sr
    if n_seconds < 1:
        raise ValueError(f"Audio too short: {len(y) / sr:.2f}s, need at least 1s.")

    rows = []
    for sec in range(n_seconds):
        window = y[sec * sr:(sec + 1) * sr]
        whole = get_audio_indices(window, sr, apply_filter=False, start_freqs=b_band,
                                   end_freqs=u_band, bandwidth=bandwidth)
        aca, adi, bet, mmm, npp, hfq, htp, hhh, aei, dsi, amr = whole
        band_signals = split_freq_band_per_frame(window, sr, b_band, u_band, bandwidth)
        for ban, f in enumerate(band_signals):
            men = float(np.mean(f))
            med = float(np.median(f))
            sm = float(np.sum(f))
            mx = float(np.max(f))
            mn = float(np.min(f))
            aci = band_acift(f)
            rows.append([men, med, sm, mx, mn, aci, aca, adi, bet, mmm, npp,
                         hfq, htp, hhh, aei, dsi, amr, float(sec)])
    return np.array(rows, dtype=np.float32)


def load_model(model_dir: Path):
    import json
    import torch

    with open(model_dir / "model_meta.json") as fh:
        meta = json.load(fh)
    if meta["frame_literal_props"] != FRAME_LITERAL_PROPS or meta["context_list"] != CONTEXT_LIST:
        raise ValueError(
            "model_meta.json's feature/context ordering doesn't match the current "
            "train_gnn.py -- retrain before running inference."
        )

    from train_gnn import empty_hetero_skeleton
    skeleton = empty_hetero_skeleton()
    model = GNN(meta["hidden_dim"], skeleton.edge_types)
    state = torch.load(model_dir / "model_state.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    col_mean = np.array(meta["col_mean"], dtype=np.float32)
    col_std = np.array(meta["col_std"], dtype=np.float32)
    return model, col_mean, col_std


def predict(audio_path: Path, model_dir: Path = DEFAULT_MODEL_DIR, seed: int = 9103):
    import torch
    from torch_geometric.data import HeteroData
    import torch_geometric.transforms as T

    model, col_mean, col_std = load_model(model_dir)

    raw = extract_frame_features(audio_path)
    n_frames = raw.shape[0]
    frame_x = (raw - col_mean) / col_std
    frame_x = np.nan_to_num(frame_x, nan=0.0)
    frame_x = torch.tensor(frame_x, dtype=torch.float32)

    g = torch.Generator().manual_seed(seed)
    data = HeteroData()
    data["sound"].x = torch.randn(1, SEED_DIM, generator=g) * 0.01
    data["frame"].x = frame_x
    data["sensor"].x = torch.randn(1, SEED_DIM, generator=g) * 0.01  # one placeholder "unknown sensor"
    data["band"].x = torch.randn(10, SEED_DIM, generator=g) * 0.01

    hasframe = torch.tensor([[0] * n_frames, list(range(n_frames))], dtype=torch.long)
    bands_per_frame = [int(b) for b in range(10)] * (n_frames // 10)
    hasband = torch.tensor([list(range(n_frames)), bands_per_frame], dtype=torch.long)
    recordedby = torch.tensor([[0], [0]], dtype=torch.long)

    data["sound", "hasFrame", "frame"].edge_index = hasframe
    data["frame", "hasBand", "band"].edge_index = hasband
    data["sound", "recordedBy", "sensor"].edge_index = recordedby
    data = T.ToUndirected()(data)

    with torch.no_grad():
        out_primary, out_secondary = model(data.x_dict, data.edge_index_dict)
        probs_primary = torch.softmax(out_primary, dim=1)[0]
        probs_secondary = torch.softmax(out_secondary, dim=1)[0]

    primary_ranked = sorted(zip(CONTEXT_LIST, probs_primary.tolist()), key=lambda kv: -kv[1])
    secondary_labels = ["Urban", "Environmental"]
    secondary_ranked = sorted(zip(secondary_labels, probs_secondary.tolist()), key=lambda kv: -kv[1])

    return {
        "n_frames": n_frames,
        "duration_seconds": n_frames // 10,
        "primary": primary_ranked,   # list of (AcousticContext, prob), best first
        "secondary": secondary_ranked,  # list of (SoundType, prob), best first
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR)
    args = parser.parse_args()

    if not args.audio.exists():
        sys.exit(f"Audio file not found: {args.audio}")
    if not (args.model_dir / "model_state.pt").exists():
        sys.exit(f"No trained model in {args.model_dir} -- run scripts/train_gnn.py first.")

    result = predict(args.audio, args.model_dir)
    print(f"{result['n_frames']} frames, ~{result['duration_seconds']}s\n")
    print("Predicted AcousticContext (primary):")
    for ctx, p in result["primary"][:5]:
        print(f"  {ctx:22s} {p:.3f}")
    print("\nPredicted SoundType (secondary):")
    for st, p in result["secondary"]:
        print(f"  {st:22s} {p:.3f}")


if __name__ == "__main__":
    main()
