# tokenizer/app/main_app.py

import json
from typing import List, Dict, Any, Optional
import sys

import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

import torch
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

from tokenizer.core.mt_translate_core import (
    translate,
    translate_batch,
    translate_beam,
    translate_with_attention,
)
from tokenizer.core.config import (
    d_model,
    nhead,
    num_layers,
    bos_id,
    eos_id,
    pad_id,
)

# Paths / constants
st.set_page_config(
    page_title="EN→HI Transformer Lab",
    page_icon="🌙",
    layout="wide",
)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RUNS_DIR = os.path.join(DATA_DIR, "runs")
MODEL_DIR = os.path.join(DATA_DIR, "trained_models")

# model used by Model I/O tab and training script keeps this updated
CURRENT_MODEL = os.path.join(MODEL_DIR, "en_hi_latest.pt")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# C Layout
st.markdown(
    """
    <style>
    /* Layout tweaks */
    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #111319;
    }

    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #ff4b4b, #ff7979);
        color: white;
        border-radius: 999px;
        border: none;
        padding: 0.35rem 1.3rem;
        font-weight: 600;
        transition: transform 0.12s ease-out, box-shadow 0.12s ease-out,
                    filter 0.12s ease-out;
    }
    .stButton>button:hover {
        transform: translateY(-1px) scale(1.01);
        box-shadow: 0 0 14px rgba(255, 75, 75, 0.55);
        filter: brightness(1.05);
    }

    /* Small helper text */
    .small-label {
        font-size: 0.8rem;
        opacity: 0.7;
    }

    /* Cards (used implicitly via columns) */
    .metric-card {
        padding: 0.7rem 0.9rem;
        border-radius: 0.6rem;
        background: #141821;
        border: 1px solid #262b3a;
    }

    /* Headings */
    h1, h2, h3 {
        letter-spacing: 0.02em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Utility functions
def load_loss_curve(run_dir: str) -> Optional[List[float]]:
    """
    Robustly load a loss curve from run_dir/loss_curve.json.
    Supports:
      1) [0.54, 0.41, ...]
      2) [{"epoch": 0, "loss": 0.54}, ...]
      3) {"loss_curve": [...]}   (what your current script writes)
    """
    path = os.path.join(run_dir, "loss_curve.json")
    if not os.path.exists(path):
        return None

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Case 3: {"loss_curve": [...]}
    if isinstance(raw, dict) and "loss_curve" in raw:
        raw = raw["loss_curve"]

    # Case 2: list of dicts with "loss"
    if isinstance(raw, list) and raw and isinstance(raw[0], dict) and "loss" in raw[0]:
        return [float(x["loss"]) for x in raw]

    # Case 1: plain numeric list
    if isinstance(raw, list) and raw and isinstance(raw[0], (int, float)):
        return [float(x) for x in raw]

    return None


def load_metadata(run_dir: str) -> Optional[Dict[str, Any]]:
    """
    Load metadata from either metadata.json (new) or meta.json (older script).
    Adds a lightweight 'schema_version' if missing.
    """
    meta_path_new = os.path.join(run_dir, "metadata.json")
    meta_path_old = os.path.join(run_dir, "meta.json")

    path = None
    if os.path.exists(meta_path_new):
        path = meta_path_new
    elif os.path.exists(meta_path_old):
        path = meta_path_old

    if path is None:
        return None

    with open(path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    if "schema_version" not in meta:
        # v1 = original fields from training script, v2+ could add more later
        meta["schema_version"] = "v1"

    return meta


def list_runs() -> List[str]:
    if not os.path.exists(RUNS_DIR):
        return []
    return sorted(
        [d for d in os.listdir(RUNS_DIR) if os.path.isdir(os.path.join(RUNS_DIR, d))]
    )

# Section renderers

def render_translator():
    st.title("English → Hindi Translator")

    col_left, col_right = st.columns([2.2, 0.8])

    with col_left:
        mode = st.radio("Mode", ["Single", "Batch", "Beam search"], horizontal=True)

        # SINGLE
        if mode == "Single":
            text = st.text_area("Enter English text", height=140)
            max_len = st.slider("Max output length", 10, 80, 40)

            if st.button("Translate"):
                if not text.strip():
                    st.warning("Enter some text.")
                else:
                    with st.spinner("Translating…"):
                        out = translate(text.strip(), max_len=max_len)
                    st.subheader("Hindi translation")
                    st.write(out)

        # BATCH
        elif mode == "Batch":
            st.markdown(
                "<div class='small-label'>One sentence per line.</div>",
                unsafe_allow_html=True,
            )
            raw = st.text_area("Batch input", height=190)
            max_len = st.slider("Max output length", 10, 80, 40, key="batch_len")

            if st.button("Translate batch"):
                lines = [ln.strip() for ln in raw.split("\n") if ln.strip()]
                if not lines:
                    st.warning("No non-empty lines to translate.")
                else:
                    with st.spinner(f"Translating {len(lines)} sentences…"):
                        outs = translate_batch(lines, max_len=max_len)

                    df = pd.DataFrame({"EN": lines, "HI": outs})
                    st.subheader("Results")
                    st.dataframe(df, use_container_width=True)

        # BEAM
        else:
            text = st.text_area("Enter English text", height=140, key="beam_text")
            max_len = st.slider("Max output length", 10, 80, 40, key="beam_len")
            beam = st.slider("Beam size", 2, 8, 4)

            if st.button("Translate (beam search)"):
                if not text.strip():
                    st.warning("Enter some text.")
                else:
                    with st.spinner(f"Beam search (k={beam})…"):
                        out = translate_beam(
                            text.strip(),
                            max_len=max_len,
                            beam_size=beam,
                        )
                    st.subheader("Hindi translation")
                    st.write(out)

    with col_right:
        st.markdown("### Model config")
        st.json(
            {
                "d_model": d_model,
                "n_layers": num_layers,
                "nhead": nhead,
                "pad_id": pad_id,
                "bos_id": bos_id,
                "eos_id": eos_id,
                "device": DEVICE,
            }
        )


def render_attention_heatmap():
    st.title("Attention-style heatmap")

    col_input, col_opts = st.columns([2.2, 0.8])

    with col_input:
        text = st.text_area("English input", "I love Java.", height=130)

    with col_opts:
        max_len = st.slider("Max output length", 10, 80, 40)
        cmap_name = st.selectbox(
            "Colormap",
            ["magma", "inferno", "plasma", "viridis"],
            index=0,
        )
        normalize = st.checkbox("Normalize to [0, 1]", value=True)

    if st.button("Generate heatmap"):
        if not text.strip():
            st.warning("Enter some text.")
            return

        with st.spinner("Running model and extracting states…"):
            out = translate_with_attention(text.strip(), max_len=max_len)

        st.subheader("Translation")
        st.write(out["translation"])

        src_tokens: List[str] = out["src_tokens"]
        tgt_tokens: List[str] = out["tgt_tokens"]
        H = out["heatmap"]  # [T, S] CPU tensor

        H = H.numpy()
        if normalize:
            mn, mx = H.min(), H.max()
            if mx > mn:
                H = (H - mn) / (mx - mn)

        st.markdown("#### Encoder–decoder similarity")

        fig_width = min(12, 0.7 * max(1, len(src_tokens)))
        fig_height = min(9, 0.55 * max(1, len(tgt_tokens)))
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        im = ax.imshow(
            H,
            aspect="auto",
            cmap=cmap_name,
            interpolation="nearest",
        )

        ax.set_xticks(range(len(src_tokens)))
        ax.set_yticks(range(len(tgt_tokens)))
        ax.set_xticklabels(src_tokens, rotation=60, ha="right", fontsize=8)
        ax.set_yticklabels(tgt_tokens, fontsize=9)

        ax.set_xlabel("Source tokens")
        ax.set_ylabel("Target tokens")

        cbar = fig.colorbar(im, ax=ax, shrink=0.8)
        cbar.ax.tick_params(labelsize=8)

        st.pyplot(fig)


def render_training_dashboard():
    st.title("Training dashboard")

    runs = list_runs()
    if not runs:
        st.info("No runs logged yet in `tokenizer/data/runs`.")
        return

    col_sel, col_cmp = st.columns([1.2, 1.0])

    with col_sel:
        run = st.selectbox("Primary run", runs)

    with col_cmp:
        cmp_runs = st.multiselect(
            "Compare with",
            runs,
            default=[run] if run in runs else [],
        )

    run_dir = os.path.join(RUNS_DIR, run)

    # ----- single-run loss
    loss_list = load_loss_curve(run_dir)

    if loss_list is None:
        st.error(f"No valid `loss_curve.json` found in {run_dir}")
    else:
        epochs = list(range(len(loss_list)))

        st.subheader("Loss curve (selected run)")
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(epochs, loss_list, label=run)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.grid(True, alpha=0.35)
        ax.legend(fontsize=8)
        st.pyplot(fig)

    # ----- metadata for primary run
    st.subheader("Metadata")
    meta = load_metadata(run_dir)
    if meta is not None:
        st.json(meta)
    else:
        st.info("No metadata found for this run.")

    # ----- comparison across runs
    if cmp_runs:
        st.subheader("Run comparison")

        summary_rows = []
        fig2, ax2 = plt.subplots(figsize=(11, 4))

        best_final = None

        for r in cmp_runs:
            rd = os.path.join(RUNS_DIR, r)
            l = load_loss_curve(rd)
            if l is None:
                continue
            ep = list(range(len(l)))

            # shorten legend label for readability
            label = r
            if len(label) > 32:
                label = "…" + label[-31:]

            ax2.plot(ep, l, label=label)

            final_loss = float(l[-1])
            summary_rows.append(
                {
                    "run": r,
                    "epochs": len(l),
                    "final_loss": final_loss,
                }
            )
            if best_final is None or final_loss < best_final:
                best_final = final_loss

        if summary_rows:
            ax2.set_xlabel("Epoch")
            ax2.set_ylabel("Loss")
            ax2.grid(True, alpha=0.35)
            ax2.legend(fontsize=7, ncol=2)
            st.pyplot(fig2)

            # add Δ from best
            if best_final is not None:
                for row in summary_rows:
                    row["Δ_from_best"] = round(row["final_loss"] - best_final, 4)

            df = pd.DataFrame(summary_rows).sort_values("final_loss")
            st.dataframe(df, use_container_width=True)
        else:
            st.info("No comparable runs had a valid loss curve.")


def render_rest_api_tester():
    st.title("REST API tester (FastAPI)")

    st.markdown(
        "Backend expected at: "
        "`uvicorn tokenizer.app.api:app --reload --port 8000`"
    )

    # url = st.text_input("Endpoint URL", "http://127.0.0.1:8000/translate")
    url = st.text_input(
        "Endpoint URL",
        "https://how-transformer-llms-work.onrender.com/translate"
    )
    text = st.text_area("English text", "Hello world")
    max_len = st.slider("Max length", 10, 80, 40)

    if st.button("Call API"):
        try:
            with st.spinner("Calling API…"):
                resp = requests.post(
                    url,
                    json={"text": text, "max_len": max_len},
                    timeout=30,
                    verify=True   # or False if needed, but True is recommended
                )
            st.subheader("HTTP status")
            st.write(resp.status_code)
            st.subheader("Response JSON")
            st.json(resp.json())
        except Exception as e:
            st.error(f"Error calling API: {e}")


def render_model_io():
    st.title("Model upload / download")

    st.markdown(f"**Current model path (used for I/O tab):** `{CURRENT_MODEL}`")

    # download
    if os.path.exists(CURRENT_MODEL):
        with open(CURRENT_MODEL, "rb") as f:
            data = f.read()
        st.download_button(
            "Download current model (.pt)",
            data=data,
            file_name=os.path.basename(CURRENT_MODEL),
            mime="application/octet-stream",
        )
    else:
        st.warning("Current model file does not exist yet.")

    st.markdown("---")

    # upload
    uploaded = st.file_uploader("Upload new model checkpoint (.pt)", type=["pt"])
    if uploaded is not None:
        tmp_path = os.path.join(MODEL_DIR, "uploaded_temp.pt")
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        try:
            state = torch.load(tmp_path, map_location="cpu")
            if isinstance(state, dict):
                os.replace(tmp_path, CURRENT_MODEL)
                st.success(
                    "New checkpoint uploaded and saved as current model.\n\n"
                    "Note: `mt_translate_core` may still load its own default "
                    "checkpoint; keep those paths in sync if you want this file "
                    "to be used for inference."
                )
            else:
                st.error("Uploaded file is not a valid state_dict.")
        except Exception as e:
            st.error(f"Failed to load checkpoint: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

# Sidebar routing

st.sidebar.title("EN→HI Transformer Lab")
section = st.sidebar.radio(
    "Section",
    ["Translator", "Attention heatmap", "Training dashboard", "REST API tester", "Model I/O"],
)

if section == "Translator":
    render_translator()
elif section == "Attention heatmap":
    render_attention_heatmap()
elif section == "Training dashboard":
    render_training_dashboard()
elif section == "REST API tester":
    render_rest_api_tester()
elif section == "Model I/O":
    render_model_io()
