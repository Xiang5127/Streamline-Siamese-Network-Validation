# python -m streamlit run app.py --server.headless true --server.port 8502

import itertools
import os
from pathlib import Path

import altair as alt
import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf

THUMB = 200  # thumbnail width in pixels for image previews

st.set_page_config(page_title="Building Identity Verification", layout="wide")

# ── Design System (DESIGN.md) ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;475;500;600&display=swap');

/* ── CSS Custom Properties (Design Tokens) ── */
:root {
    --primary: #181d26;
    --primary-active: #0d1218;
    --ink: #181d26;
    --body: #333840;
    --muted: #41454d;
    --hairline: #dddddd;
    --border-strong: #9297a0;
    --canvas: #ffffff;
    --surface-soft: #f8fafc;
    --surface-strong: #e0e2e6;
    --surface-dark: #181d26;
    --signature-coral: #aa2d00;
    --signature-forest: #0a2e0e;
    --signature-cream: #f5e9d4;
    --signature-peach: #fcab79;
    --signature-mint: #a8d8c4;
    --on-primary: #ffffff;
    --on-dark: #ffffff;
    --link: #1b61c9;
    --link-active: #1a3866;
    --success: #006400;
    --success-border: #39bf45;
    --info: #254fad;
    --info-border: #458fff;
    --spacing-xs: 8px;
    --spacing-sm: 12px;
    --spacing-md: 16px;
    --spacing-lg: 24px;
    --spacing-xl: 32px;
    --spacing-xxl: 48px;
    --spacing-section: 96px;
    --rounded-xs: 2px;
    --rounded-sm: 6px;
    --rounded-md: 10px;
    --rounded-lg: 12px;
    --rounded-pill: 9999px;
}

/* ── Base ── */
.stApp {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
                 Oxygen, Ubuntu, Cantarell, "Fira Sans", "Droid Sans",
                 "Helvetica Neue", sans-serif;
    background-color: var(--canvas);
}

[data-testid="stHeader"] {
    background-color: var(--canvas);
}

/* ── Sidebar (light editorial) ── */
[data-testid="stSidebar"] {
    background-color: var(--surface-soft) !important;
    border-right: 1px solid var(--hairline) !important;
}
[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
    background-color: var(--surface-soft) !important;
}
[data-testid="stSidebar"] * {
    color: var(--ink) !important;
}
/* Slider track (entire bar) */
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: var(--primary) !important;
    border: 2px solid var(--canvas) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.18) !important;
}
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
    color: var(--ink) !important;
    font-weight: 500 !important;
}
/* Filled portion of the track */
[data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] div[role="progressbar"] > div {
    background-color: var(--primary) !important;
}
/* Unfilled track */
[data-testid="stSidebar"] .stSlider div[data-baseweb="slider"] div[role="progressbar"] {
    background-color: var(--hairline) !important;
}
[data-testid="stSidebar"] hr {
    border-color: var(--hairline) !important;
}
[data-testid="stSidebar"] .stRadio > label {
    font-size: 13px !important;
    font-weight: 500 !important;
    letter-spacing: 0.16px;
    text-transform: uppercase;
    color: var(--muted) !important;
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
    font-size: 15px !important;
    font-weight: 400 !important;
    text-transform: none;
    color: var(--body) !important;
    padding: 6px 0;
}
[data-testid="stSidebar"] .stRadio [role="radiogroup"] label[data-checked="true"] {
    font-weight: 500 !important;
    color: var(--ink) !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background-color: var(--canvas) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: var(--rounded-sm) !important;
    color: var(--ink) !important;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: var(--info-border) !important;
    box-shadow: 0 0 0 1px var(--info-border) !important;
}
[data-testid="stSidebar"] .stNumberInput input {
    background-color: var(--canvas) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: var(--rounded-sm) !important;
    color: var(--ink) !important;
}

/* ── Typography ── */
h1 {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    font-size: 40px !important;
    font-weight: 400 !important;
    line-height: 1.2 !important;
    letter-spacing: -0.01em !important;
    color: var(--ink) !important;
    padding-bottom: 0 !important;
}
h2 {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    font-size: 32px !important;
    font-weight: 400 !important;
    line-height: 1.2 !important;
    color: var(--ink) !important;
}
h3 {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    font-size: 24px !important;
    font-weight: 400 !important;
    line-height: 1.35 !important;
    letter-spacing: 0.12px !important;
    color: var(--ink) !important;
}
p, li, span {
    color: var(--body);
}

/* ── Hero band ── */
.hero-band {
    padding: 64px 0 48px 0;
    text-align: left;
}
.hero-band h1 {
    margin-bottom: 8px !important;
}
.hero-band .subtitle {
    font-size: 18px;
    font-weight: 400;
    color: var(--muted);
    line-height: 1.5;
    max-width: 600px;
}
.mode-badge {
    display: inline-block;
    background-color: var(--surface-soft);
    color: var(--muted);
    font-size: 13px;
    font-weight: 500;
    padding: 4px 14px;
    border-radius: var(--rounded-pill);
    border: 1px solid var(--hairline);
    margin-bottom: 16px;
    letter-spacing: 0.16px;
}

/* ── Buttons (primary = near-black) ── */
.stButton > button {
    background-color: var(--primary) !important;
    color: #ffffff !important;
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    font-size: 16px !important;
    font-weight: 500 !important;
    line-height: 1.4 !important;
    border: none !important;
    border-radius: var(--rounded-lg) !important;
    padding: 12px 24px !important;
    min-height: 48px;
    transition: background-color 0.15s ease;
}
.stButton > button p,
.stButton > button span,
.stButton > button div {
    color: #ffffff !important;
}
.stButton > button:active {
    background-color: var(--primary-active) !important;
}
.stButton > button:focus:not(:active) {
    outline: 2px solid var(--info-border);
    outline-offset: 2px;
    box-shadow: none !important;
}
.stButton > button[disabled] {
    background-color: var(--surface-strong) !important;
    color: var(--muted) !important;
    border: 1px solid var(--hairline) !important;
    opacity: 1;
}
.stButton > button[disabled] p,
.stButton > button[disabled] span,
.stButton > button[disabled] div {
    color: var(--muted) !important;
}

/* ── Sidebar buttons (secondary style) ── */
[data-testid="stSidebar"] .stButton > button {
    background-color: var(--canvas) !important;
    color: var(--ink) !important;
    border: 1px solid var(--hairline) !important;
}

/* ── Metrics ── */
[data-testid="stMetric"] {
    background-color: var(--surface-soft);
    border-radius: var(--rounded-md);
    padding: var(--spacing-lg);
    border: 1px solid var(--hairline);
}
[data-testid="stMetric"] label {
    font-size: 13px !important;
    font-weight: 500 !important;
    color: var(--muted) !important;
    letter-spacing: 0.16px;
}
[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    font-size: 28px !important;
    font-weight: 475 !important;
    color: var(--ink) !important;
}

/* ── Expanders (editorial cards) ── */
[data-testid="stExpander"] {
    border: 1px solid var(--hairline) !important;
    border-radius: var(--rounded-md) !important;
    overflow: hidden;
    margin-bottom: var(--spacing-sm);
}
[data-testid="stExpander"] details > summary {
    font-family: "Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif !important;
    font-weight: 500 !important;
    font-size: 15px !important;
    color: var(--ink) !important;
    padding: var(--spacing-md) var(--spacing-lg) !important;
}
[data-testid="stExpander"] details > summary:hover {
    background-color: var(--surface-soft);
}

/* ── File uploaders ── */
[data-testid="stFileUploader"] section {
    border: 2px dashed var(--hairline) !important;
    border-radius: var(--rounded-md) !important;
    padding: var(--spacing-lg) !important;
    background-color: var(--surface-soft);
    transition: border-color 0.15s ease;
}
[data-testid="stFileUploader"] section:hover {
    border-color: var(--border-strong) !important;
}
[data-testid="stFileUploader"] button {
    background-color: var(--canvas) !important;
    color: var(--ink) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: var(--rounded-sm) !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
    min-height: auto;
}

/* ── Text inputs ── */
.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background-color: var(--canvas) !important;
    color: var(--ink) !important;
    border: 1px solid var(--hairline) !important;
    border-radius: var(--rounded-sm) !important;
    padding: 12px 16px !important;
    height: 44px !important;
    font-size: 14px !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: var(--info-border) !important;
    box-shadow: 0 0 0 1px var(--info-border) !important;
}

/* ── Alerts (info / success / error / warning) ── */
[data-testid="stAlert"] {
    border-radius: var(--rounded-md) !important;
    font-size: 14px !important;
    padding: var(--spacing-md) var(--spacing-lg) !important;
}

/* ── Success result card ── */
.result-card-match {
    background: linear-gradient(135deg, #f0fdf4, #ecfdf5);
    border: 1px solid var(--success-border);
    border-radius: var(--rounded-lg);
    padding: var(--spacing-xl);
    margin: var(--spacing-lg) 0;
}
.result-card-match h2 {
    color: var(--success) !important;
    font-size: 24px !important;
    margin-bottom: 8px !important;
}
.result-card-match .detail {
    color: var(--body);
    font-size: 15px;
}

/* ── Reject result card ── */
.result-card-reject {
    background: linear-gradient(135deg, #fef2f2, #fff1f2);
    border: 1px solid var(--signature-coral);
    border-radius: var(--rounded-lg);
    padding: var(--spacing-xl);
    margin: var(--spacing-lg) 0;
}
.result-card-reject h2 {
    color: var(--signature-coral) !important;
    font-size: 24px !important;
    margin-bottom: 8px !important;
}
.result-card-reject .detail {
    color: var(--body);
    font-size: 15px;
}

/* ── Tables / DataFrames ── */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border-radius: var(--rounded-md) !important;
    overflow: hidden;
}

/* ── Progress bar ── */
.stProgress > div > div > div > div {
    background-color: var(--primary) !important;
    border-radius: var(--rounded-pill) !important;
}

/* ── Slider (global) ── */
.stSlider [data-baseweb="slider"] [role="slider"] {
    background-color: var(--primary) !important;
    border: 2px solid var(--canvas) !important;
    box-shadow: 0 1px 4px rgba(0,0,0,0.15) !important;
    width: 18px !important;
    height: 18px !important;
}
.stSlider div[data-baseweb="slider"] div[role="progressbar"] > div {
    background-color: var(--primary) !important;
}
.stSlider div[data-baseweb="slider"] div[role="progressbar"] {
    background-color: var(--hairline) !important;
}
.stSlider [data-testid="stThumbValue"] {
    color: var(--ink) !important;
    font-weight: 500 !important;
    font-size: 13px !important;
}

/* ── Dividers ── */
hr {
    border-color: var(--hairline) !important;
    margin: var(--spacing-xxl) 0 !important;
}

/* ── Images ── */
[data-testid="stImage"] img {
    border-radius: var(--rounded-md);
}

/* ── Main content area ── */
.main .block-container {
    max-width: 1280px;
    padding: var(--spacing-xl) var(--spacing-xxl) var(--spacing-section);
}

/* ── Altair charts ── */
.vega-embed {
    border-radius: var(--rounded-md);
    overflow: hidden;
}

/* ── Section headers ── */
.section-header {
    font-size: 14px;
    font-weight: 500;
    color: var(--muted);
    letter-spacing: 0.16px;
    text-transform: uppercase;
    margin-bottom: 4px;
}

/* ── Signature card surfaces ── */
.signature-card-coral {
    background-color: var(--signature-coral);
    color: var(--on-primary);
    border-radius: var(--rounded-lg);
    padding: var(--spacing-xxl);
}
.signature-card-forest {
    background-color: var(--signature-forest);
    color: var(--on-primary);
    border-radius: var(--rounded-lg);
    padding: var(--spacing-xxl);
}
.signature-card-dark {
    background-color: var(--surface-dark);
    color: var(--on-dark);
    border-radius: var(--rounded-lg);
    padding: var(--spacing-xxl);
}
.signature-card-dark h2, .signature-card-dark p,
.signature-card-coral h2, .signature-card-coral p,
.signature-card-forest h2, .signature-card-forest p {
    color: var(--on-dark) !important;
}
.cream-card {
    background-color: var(--signature-cream);
    border-radius: var(--rounded-md);
    padding: var(--spacing-lg);
}

/* ── Report summary card ── */
.report-summary {
    background-color: var(--surface-soft);
    border: 1px solid var(--hairline);
    border-radius: var(--rounded-lg);
    padding: var(--spacing-xl);
    margin: var(--spacing-lg) 0;
}

/* ── Upload zone label ── */
.upload-label {
    font-size: 16px;
    font-weight: 500;
    color: var(--ink);
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown('<p style="font-size:11px; font-weight:600; letter-spacing:1.5px; text-transform:uppercase; color:#9297a0; margin-bottom:4px;">BUILDING DNA</p>', unsafe_allow_html=True)
st.sidebar.title("Verification")
st.sidebar.markdown("---")
mode = st.sidebar.radio("Navigation", ["Single Comparison", "Batch Testing", "Auto Pairing", "Model Validation"])
st.sidebar.markdown("---")
threshold = st.sidebar.slider("Match Threshold", 0.1, 2.0, 0.8, step=0.01)

# ── Hero Band ────────────────────────────────────────────────────────────────
st.markdown(f'<div class="hero-band"><span class="mode-badge">{mode}</span>', unsafe_allow_html=True)
st.title("Building Identity Verification")
st.markdown('<p class="subtitle">Compare building photographs using a Siamese neural network to verify structural identity through deep feature embeddings.</p></div>', unsafe_allow_html=True)

# ── Load TFLite model (cached) ───────────────────────────────────────────────
@st.cache_resource
def load_interpreter():
    interpreter = tf.lite.Interpreter(model_path="building_dna_extractor.tflite")
    interpreter.allocate_tensors()
    return interpreter

interpreter = load_interpreter()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# ── Preprocessing ────────────────────────────────────────────────────────────
def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize((224, 224))
    arr = np.array(image, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)

# ── Inference ────────────────────────────────────────────────────────────────
def get_embedding(img_array: np.ndarray) -> np.ndarray:
    interpreter.set_tensor(input_details[0]["index"], img_array)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]["index"]).flatten()

# ── Helper: display result box ───────────────────────────────────────────────
def show_result(distance: float, threshold: float):
    st.metric(label="Euclidean Distance", value=f"{distance:.4f}")
    if distance < threshold:
        st.markdown(
            f'<div class="result-card-match">'
            f'<h2>MATCH &mdash; Same Building</h2>'
            f'<p class="detail">Distance <strong>{distance:.4f}</strong> is below threshold <strong>{threshold:.2f}</strong></p>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div class="result-card-reject">'
            f'<h2>REJECTED &mdash; Different Buildings</h2>'
            f'<p class="detail">Distance <strong>{distance:.4f}</strong> meets or exceeds threshold <strong>{threshold:.2f}</strong></p>'
            f'</div>',
            unsafe_allow_html=True,
        )

# ═════════════════════════════════════════════════════════════════════════════
# MODE 1 ─ Single Comparison
# ═════════════════════════════════════════════════════════════════════════════
if mode == "Single Comparison":
    st.markdown('<p class="section-header">Upload two images to compare</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<p class="upload-label">Reference Image</p>', unsafe_allow_html=True)
        file1 = st.file_uploader("Upload reference image", type=["jpg", "jpeg", "png"], key="ref")
        if file1:
            img1 = Image.open(file1)
            st.image(img1, width=THUMB)

    with col2:
        st.markdown('<p class="upload-label">Live Image</p>', unsafe_allow_html=True)
        file2 = st.file_uploader("Upload live image", type=["jpg", "jpeg", "png"], key="live")
        if file2:
            img2 = Image.open(file2)
            st.image(img2, width=THUMB)

    if file1 and file2:
        emb1 = get_embedding(preprocess(img1))
        emb2 = get_embedding(preprocess(img2))
        distance = float(np.linalg.norm(emb1 - emb2))
        st.markdown("---")
        show_result(distance, threshold)

# ═════════════════════════════════════════════════════════════════════════════
# MODE 2 ─ Batch Testing
# ═════════════════════════════════════════════════════════════════════════════
elif mode == "Batch Testing":
    st.markdown('<p class="section-header">Batch evaluation</p>', unsafe_allow_html=True)
    st.subheader("Batch Testing")
    num_pairs = st.sidebar.number_input("Number of test pairs", min_value=1, max_value=50, value=3, step=1)

    st.info(f"Upload **{num_pairs}** image pair(s) below, then click **Run Batch Test**.")

    ref_files = []
    live_files = []

    for i in range(num_pairs):
        st.markdown(f"### Pair {i + 1}")
        c1, c2 = st.columns(2)
        with c1:
            f_ref = st.file_uploader(f"Reference #{i + 1}", type=["jpg", "jpeg", "png"], key=f"batch_ref_{i}")
            if f_ref:
                st.image(Image.open(f_ref), width=THUMB)
            ref_files.append(f_ref)
        with c2:
            f_live = st.file_uploader(f"Live #{i + 1}", type=["jpg", "jpeg", "png"], key=f"batch_live_{i}")
            if f_live:
                st.image(Image.open(f_live), width=THUMB)
            live_files.append(f_live)

    all_uploaded = all(r is not None and l is not None for r, l in zip(ref_files, live_files))

    if st.button("Run Batch Test", disabled=not all_uploaded):
        results = []
        st.markdown("---")
        st.subheader("Individual Results")

        for i, (r, l) in enumerate(zip(ref_files, live_files)):
            img_r = Image.open(r)
            img_l = Image.open(l)
            emb_r = get_embedding(preprocess(img_r))
            emb_l = get_embedding(preprocess(img_l))
            dist = float(np.linalg.norm(emb_r - emb_l))
            matched = dist < threshold

            with st.expander(f"Pair {i + 1} — {'✅ MATCH' if matched else '❌ REJECTED'}  (Distance: {dist:.4f})", expanded=False):
                p1, p2, p3 = st.columns([1, 1, 2])
                with p1:
                    st.image(img_r, caption="Reference", width=THUMB)
                with p2:
                    st.image(img_l, caption="Live", width=THUMB)
                with p3:
                    show_result(dist, threshold)

            results.append({
                "Pair": i + 1,
                "Distance": round(dist, 4),
                "Threshold": threshold,
                "Verdict": "MATCH" if matched else "REJECTED",
            })

        # ── Summary Report ───────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="report-summary">', unsafe_allow_html=True)
        st.subheader("Batch Test Report")

        df = pd.DataFrame(results)
        total = len(df)
        matches = int((df["Verdict"] == "MATCH").sum())
        rejections = total - matches

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total Pairs", total)
        m2.metric("Matches", matches)
        m3.metric("Rejections", rejections)
        m4.metric("Match Rate", f"{matches / total * 100:.1f}%")

        st.dataframe(df, use_container_width=True, hide_index=True)
        st.markdown('</div>', unsafe_allow_html=True)

# ═════════════════════════════════════════════════════════════════════════════
# MODE 3 ─ Auto Pairing
# ═════════════════════════════════════════════════════════════════════════════
elif mode == "Auto Pairing":
    st.markdown('<p class="section-header">Combinatorial analysis</p>', unsafe_allow_html=True)
    st.subheader("Auto Pairing")
    st.info("Upload multiple images below. Every possible pair will be compared automatically.")

    uploaded = st.file_uploader(
        "Upload images", type=["jpg", "jpeg", "png"],
        accept_multiple_files=True, key="auto_pair",
    )

    if uploaded and len(uploaded) >= 2:
        # show thumbnails
        cols = st.columns(min(len(uploaded), 6))
        for idx, f in enumerate(uploaded):
            with cols[idx % len(cols)]:
                st.image(Image.open(f), caption=f.name, width=THUMB)

        pairs = list(itertools.combinations(range(len(uploaded)), 2))
        st.write(f"**{len(uploaded)}** images → **{len(pairs)}** pairs to compare.")

        if st.button("Run Auto Pairing"):
            # compute all embeddings once
            embeddings = []
            names = []
            progress = st.progress(0, text="Computing embeddings…")
            for idx, f in enumerate(uploaded):
                img = Image.open(f)
                embeddings.append(get_embedding(preprocess(img)))
                names.append(f.name)
                progress.progress((idx + 1) / len(uploaded), text=f"Embedding {idx + 1}/{len(uploaded)}")
            progress.empty()

            # compare all pairs
            results = []
            st.markdown("---")
            st.subheader("Individual Results")

            for i, j in pairs:
                dist = float(np.linalg.norm(embeddings[i] - embeddings[j]))
                matched = dist < threshold
                label = "✅ MATCH" if matched else "❌ REJECTED"

                with st.expander(f"{names[i]}  ↔  {names[j]} — {label}  (Distance: {dist:.4f})", expanded=False):
                    p1, p2, p3 = st.columns([1, 1, 2])
                    with p1:
                        st.image(Image.open(uploaded[i]), caption=names[i], width=THUMB)
                    with p2:
                        st.image(Image.open(uploaded[j]), caption=names[j], width=THUMB)
                    with p3:
                        show_result(dist, threshold)

                results.append({
                    "Image A": names[i],
                    "Image B": names[j],
                    "Distance": round(dist, 4),
                    "Threshold": threshold,
                    "Verdict": "MATCH" if matched else "REJECTED",
                })

            # ── Summary Report ───────────────────────────────────────
            st.markdown("---")
            st.markdown('<div class="report-summary">', unsafe_allow_html=True)
            st.subheader("Auto Pairing Report")

            df = pd.DataFrame(results)
            total = len(df)
            matches = int((df["Verdict"] == "MATCH").sum())
            rejections = total - matches

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Pairs", total)
            m2.metric("Matches", matches)
            m3.metric("Rejections", rejections)
            m4.metric("Match Rate", f"{matches / total * 100:.1f}%")

            st.dataframe(df, use_container_width=True, hide_index=True)
            st.markdown('</div>', unsafe_allow_html=True)

    elif uploaded and len(uploaded) < 2:
        st.warning("Please upload at least **2** images to generate pairs.")

# ═════════════════════════════════════════════════════════════════════════════
# MODE 4 ─ Model Validation
# ═════════════════════════════════════════════════════════════════════════════
else:
    st.markdown('<p class="section-header">Performance evaluation</p>', unsafe_allow_html=True)
    st.subheader("Model Validation")
    st.info("Point to a **validation folder** and its **pairs CSV** to evaluate model performance.")

    val_folder = st.sidebar.text_input("Validation folder path", value="validation_folder")
    val_csv = st.sidebar.text_input("Pairs CSV path", value="pairs_val.csv")

    folder_ok = os.path.isdir(val_folder)
    csv_ok = os.path.isfile(val_csv)

    if not folder_ok:
        st.warning(f"Folder not found: `{val_folder}`")
    if not csv_ok:
        st.warning(f"CSV not found: `{val_csv}`")

    can_run = folder_ok and csv_ok
    if st.button("Run Validation", disabled=not can_run):
        df_pairs = pd.read_csv(val_csv)
        total_pairs = len(df_pairs)

        distances = []
        labels = []
        skipped = 0
        progress = st.progress(0, text="Validating…")

        for idx, row in df_pairs.iterrows():
            path_a = os.path.join(val_folder, row["path_a"].replace("\\", "/"))
            path_b = os.path.join(val_folder, row["path_b"].replace("\\", "/"))
            gt = int(row["label"])  # 1 = same, 0 = different

            if not os.path.isfile(path_a) or not os.path.isfile(path_b):
                skipped += 1
                continue

            emb_a = get_embedding(preprocess(Image.open(path_a)))
            emb_b = get_embedding(preprocess(Image.open(path_b)))
            dist = float(np.linalg.norm(emb_a - emb_b))

            distances.append(dist)
            labels.append(gt)
            progress.progress((idx + 1) / total_pairs, text=f"Pair {idx + 1}/{total_pairs}")

        progress.empty()

        if skipped:
            st.warning(f"Skipped {skipped} pair(s) due to missing files.")

        distances = np.array(distances)
        labels = np.array(labels)
        preds = (distances < threshold).astype(int)  # 1 = match

        # ── Confusion matrix components ──────────────────────────────
        tp = int(((preds == 1) & (labels == 1)).sum())
        tn = int(((preds == 0) & (labels == 0)).sum())
        fp = int(((preds == 1) & (labels == 0)).sum())
        fn = int(((preds == 0) & (labels == 1)).sum())

        accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-9)
        far = fp / max(fp + tn, 1)   # False Accept Rate
        frr = fn / max(fn + tp, 1)   # False Reject Rate

        # ── EER approximation (sweep thresholds) ────────────────────
        sweep = np.linspace(0.0, 2.5, 500)
        fars, frrs = [], []
        for t in sweep:
            p = (distances < t).astype(int)
            _fp = ((p == 1) & (labels == 0)).sum()
            _fn = ((p == 0) & (labels == 1)).sum()
            _tn = ((p == 0) & (labels == 0)).sum()
            _tp = ((p == 1) & (labels == 1)).sum()
            fars.append(_fp / max(_fp + _tn, 1))
            frrs.append(_fn / max(_fn + _tp, 1))
        fars = np.array(fars)
        frrs = np.array(frrs)
        eer_idx = np.argmin(np.abs(fars - frrs))
        eer = float((fars[eer_idx] + frrs[eer_idx]) / 2)
        eer_threshold = float(sweep[eer_idx])

        # ── AUC (trapezoidal on ROC from sweep) ────────────────────
        tprs = 1.0 - frrs  # TPR = 1 - FRR
        sorted_idx = np.argsort(fars)
        auc = float(np.trapz(tprs[sorted_idx], fars[sorted_idx]))

        # ── Display metrics ─────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="report-summary">', unsafe_allow_html=True)
        st.subheader("Validation Report")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", f"{accuracy:.4f}")
        c2.metric("Precision", f"{precision:.4f}")
        c3.metric("Recall", f"{recall:.4f}")
        c4.metric("F1 Score", f"{f1:.4f}")

        c5, c6, c7, c8 = st.columns(4)
        c5.metric("FAR (False Accept)", f"{far:.4f}")
        c6.metric("FRR (False Reject)", f"{frr:.4f}")
        c7.metric("EER", f"{eer:.4f}")
        c8.metric("AUC-ROC", f"{auc:.4f}")

        st.caption(f"EER optimal threshold ≈ **{eer_threshold:.4f}** (current slider: {threshold:.2f})")
        st.markdown('</div>', unsafe_allow_html=True)

        # ── Confusion matrix ────────────────────────────────────────
        st.markdown("---")
        st.subheader("Confusion Matrix")
        cm_df = pd.DataFrame(
            [[tp, fn], [fp, tn]],
            index=["Actual: Same", "Actual: Different"],
            columns=["Pred: Same", "Pred: Different"],
        )
        st.table(cm_df)

        # ── Distance distribution ───────────────────────────────────
        st.markdown("---")
        st.subheader("Distance Distribution")
        hist_df = pd.DataFrame({
            "Distance": distances,
            "Ground Truth": ["Same Building" if l == 1 else "Different Building" for l in labels],
        })

        chart = (
            alt.Chart(hist_df)
            .mark_bar(opacity=0.6)
            .encode(
                alt.X("Distance:Q", bin=alt.Bin(maxbins=40)),
                alt.Y("count()"),
                alt.Color("Ground Truth:N"),
            )
            .properties(width=700, height=350)
        )
        rule = (
            alt.Chart(pd.DataFrame({"x": [threshold]}))
            .mark_rule(color="red", strokeDash=[5, 5], size=2)
            .encode(x="x:Q")
        )
        st.altair_chart(chart + rule, use_container_width=True)
        st.caption("Red dashed line = current threshold")

        # ── FAR vs FRR curve ────────────────────────────────────────
        st.markdown("---")
        st.subheader("FAR vs FRR Curve")
        rate_df = pd.DataFrame({"Threshold": sweep, "FAR": fars, "FRR": frrs})
        rate_melted = rate_df.melt("Threshold", var_name="Rate", value_name="Value")
        rate_chart = (
            alt.Chart(rate_melted)
            .mark_line()
            .encode(
                x="Threshold:Q",
                y="Value:Q",
                color="Rate:N",
            )
            .properties(width=700, height=300)
        )
        eer_rule = (
            alt.Chart(pd.DataFrame({"x": [eer_threshold]}))
            .mark_rule(color="green", strokeDash=[4, 4], size=2)
            .encode(x="x:Q")
        )
        st.altair_chart(rate_chart + eer_rule, use_container_width=True)
        st.caption(f"Green dashed line = EER threshold ({eer_threshold:.4f})")

        # ── Per-pair detail table ───────────────────────────────────
        st.markdown("---")
        st.subheader("Per-Pair Results")
        detail_rows = []
        for i in range(len(distances)):
            detail_rows.append({
                "#": i + 1,
                "Distance": round(distances[i], 4),
                "Ground Truth": "Same" if labels[i] == 1 else "Different",
                "Prediction": "Same" if preds[i] == 1 else "Different",
                "Correct": "✅" if preds[i] == labels[i] else "❌",
            })
        st.dataframe(pd.DataFrame(detail_rows), use_container_width=True, hide_index=True)
