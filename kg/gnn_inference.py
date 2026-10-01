#!/usr/bin/env python3
"""
Inference on one new audio recording using the trained GNN (train_gnn.py).

Superseded by soundforest/kg_inference.py, which calls the real
bootstrap_soundscape() instead of this file's reimplemented 1-second
windowing. That reimplementation is a known approximation: formulas match
training exactly, but values measurably differ since the real training
windowing isn't reproduced here (see .carlos/notes.md). Kept for reference.

Usage:
    python scripts/gnn_inference.py path/to/audio.wav --model-dir out/data/kg/gnn
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
    total = np.sum(f)
    if total == 0:
        return 0.0
    return float(np.sum(np.abs(np.diff(f))) / total)


def split_freq_band_per_frame(s: np.ndarray, sr: int, b_band=0, u_band=10000, bandwidth=1000):
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
    data["sensor"].x = torch.randn(1, SEED_DIM, generator=g) * 0.01
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
        "primary": primary_ranked,
        "secondary": secondary_ranked,
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
