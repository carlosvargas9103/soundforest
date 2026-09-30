#!/usr/bin/env python3
"""
FOREST-KG Gradio app (commit 9): upload a new audio recording, extract its
acoustic-index features, and infer AcousticContext (primary, 13-way) and
SoundType (secondary, 2-way) using the trained GNN.

Why GNN, not the KGE (TransE/ComplEx) models from commit 5: KGE entity
embeddings are looked up by ID from a table fixed at training time -- a
brand-new Sound has no such embedding, so TransE/ComplEx can't score it
without retraining ("cold start", the standard KG-embedding limitation,
see LO8's completion/evolution material). The GNN doesn't have this
problem: it builds a Sound's representation from its own Frame
neighborhood's real content features via message passing, so it's
inductive by construction -- see scripts/gnn_inference.py's docstring.

Inference is entirely local: no GraphDB connection needed at request time
(only the trained model files, bundled with this app). That also means
this Space can run standalone on Hugging Face without needing to reach
back into any localhost service.

Run locally:
    python app/app.py
Deploy: see app/README.md for the Hugging Face Spaces steps (not done from
here -- pushing to a Space is an external, user-owned action).
"""
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
REPO_ROOT = APP_DIR.parent
for p in (REPO_ROOT / "scripts", REPO_ROOT / "source"):
    if p.exists() and str(p) not in sys.path:
        sys.path.insert(0, str(p))

import gradio as gr

from gnn_inference import predict

MODEL_DIR = REPO_ROOT / "out" / "data" / "kg" / "gnn"
if not (MODEL_DIR / "model_state.pt").exists():
    MODEL_DIR = APP_DIR / "model"  # bundled copy, for a standalone deploy


def infer(audio_path):
    if audio_path is None:
        return None, None, "Upload or record an audio clip first."
    try:
        result = predict(Path(audio_path), model_dir=MODEL_DIR)
    except Exception as e:
        return None, None, f"Could not run inference: {e}"

    primary = {ctx: prob for ctx, prob in result["primary"]}
    secondary = {st: prob for st, prob in result["secondary"]}
    note = (
        f"{result['n_frames']} frames (~{result['duration_seconds']}s analyzed). "
        f"Model: GNN trained on a 10% stratified sample -- see .carlos/notes.md "
        f"for accuracy figures and known limitations (feature-extraction is an "
        f"approximation of the original training pipeline's windowing)."
    )
    return primary, secondary, note


with gr.Blocks(title="FOREST-KG Region Inference") as demo:
    gr.Markdown(
        "# FOREST-KG: infer a recording's Acoustic Context\n"
        "Upload or record audio. The app extracts the same 17 acoustic indices "
        "used throughout FOREST-KG, builds a small local graph (this recording's "
        "own Frames), and runs the trained GNN (commit 6) to predict which of "
        "13 AcousticContext categories it belongs to, and whether it's Urban or "
        "Environmental."
    )
    with gr.Row():
        audio_in = gr.Audio(sources=["upload", "microphone"], type="filepath", label="Audio recording")
    run_btn = gr.Button("Predict", variant="primary")
    with gr.Row():
        primary_out = gr.Label(label="AcousticContext (primary, 13-way)", num_top_classes=5)
        secondary_out = gr.Label(label="SoundType (secondary)", num_top_classes=2)
    note_out = gr.Markdown()

    run_btn.click(fn=infer, inputs=audio_in, outputs=[primary_out, secondary_out, note_out])
    audio_in.change(fn=infer, inputs=audio_in, outputs=[primary_out, secondary_out, note_out])

if __name__ == "__main__":
    demo.launch()
