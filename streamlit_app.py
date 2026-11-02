from __future__ import annotations

import io
from pathlib import Path

import streamlit as st
from PIL import Image, ImageOps, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "backend" / "models" / "model.keras"
CLASS_MAP_PATH = ROOT / "backend" / "models" / "class_indices.json"
MAX_BYTES = 5 * 1024 * 1024

st.set_page_config(
    page_title="Image Forensics | Tampering Detection",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;500;600;700;800&display=swap');
    :root { --ink:#e9edf5; --muted:#9ba7ba; --line:rgba(164,181,208,.17); }
    .stApp { background: radial-gradient(ellipse at 15% 0%, #172a43 0%, #0b1220 42%, #080d17 100%); color:var(--ink); }
    html, body, [class*="css"] { font-family:'Manrope',sans-serif; }
    .block-container { max-width:1120px; padding-top:2.2rem; padding-bottom:3rem; }
    .eyebrow { color:#75d6c3; font:500  .76rem 'DM Mono',monospace; letter-spacing:.18em; text-transform:uppercase; }
    .hero { font-size:clamp(2.25rem,5vw,4.3rem); line-height:1.04; font-weight:800; letter-spacing:-.055em; margin:.6rem 0 1rem; }
    .hero span { color:#79d9c5; }
    .sub { color:#aab6c8; max-width:680px; font-size:1.04rem; line-height:1.7; }
    .panel { background:rgba(19,31,49,.72); border:1px solid var(--line); border-radius:18px; padding:1.25rem 1.35rem; }
    .metric-label { color:#9ba7ba; font: .72rem 'DM Mono',monospace; letter-spacing:.12em; text-transform:uppercase; }
    .metric-value { color:#f3f6fb; font-size:1.55rem; font-weight:800; margin-top:.25rem; }
    .small-note { color:#9ba7ba; font-size:.83rem; line-height:1.55; }
    div[data-testid="stFileUploader"] { background:rgba(17,29,46,.75); border:1px dashed rgba(121,217,197,.42); border-radius:16px; padding:1rem; }
    .stButton>button { border-radius:10px; font-weight:700; min-height:3rem; }
    footer { visibility:hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="eyebrow">CASIA 2.0 · VGG19 · IMAGE FORENSICS</div>', unsafe_allow_html=True)
st.markdown('<div class="hero">Look closer.<br/><span>Verify the image.</span></div>', unsafe_allow_html=True)
st.markdown(
    '<p class="sub">An experimental deep-learning tool that screens an image for signs associated with manipulation. Upload a JPEG or PNG to run the VGG19 binary classifier.</p>',
    unsafe_allow_html=True,
)
st.write("")

left, right = st.columns([1.12, .88], gap="large")
with left:
    st.markdown("### 01 / Upload image")
    uploaded = st.file_uploader(
        "Choose an image to inspect",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
        help="JPEG or PNG · Maximum 5 MB",
        label_visibility="collapsed",
    )
    if uploaded:
        raw = uploaded.getvalue()
        if len(raw) > MAX_BYTES:
            st.error("This file is larger than 5 MB. Choose a smaller image.")
            uploaded = None
        else:
            try:
                with Image.open(io.BytesIO(raw)) as im:
                    preview = ImageOps.exif_transpose(im).convert("RGB")
                    preview.load()
                st.image(preview, caption=f"{uploaded.name} · {preview.width} × {preview.height}", use_container_width=True)
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("This file could not be decoded as a valid image.")
                uploaded = None
    else:
        st.markdown(
            '<div class="panel"><div class="metric-label">SUPPORTED INPUT</div><div class="metric-value">JPG / JPEG / PNG</div><p class="small-note">For best results, use a clear image. The model resizes the input to 224 × 224 pixels before inference.</p></div>',
            unsafe_allow_html=True,
        )

with right:
    st.markdown("### 02 / Analysis")
    st.markdown(
        '<div class="panel"><div class="metric-label">MODEL</div><div class="metric-value">VGG19 CNN</div><p class="small-note">Transfer-learning classifier trained for two dataset classes: authentic and tampered. The deployed class mapping is read from the exported training artifact.</p></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    run = st.button("Analyze image", type="primary", use_container_width=True, disabled=uploaded is None)

    if run and uploaded is not None:
        if not MODEL_PATH.is_file() or not CLASS_MAP_PATH.is_file():
            st.error("The model is not deployed yet. Add backend/models/model.keras and backend/models/class_indices.json to the deployment, then restart the app.")
            st.info("The trained model files are intentionally excluded from Git. Train and export the model using training/train_and_export.py before deploying.")
        else:
            with st.spinner("Running image analysis…"):
                try:
                    from backend.inference import predict_image
                    result = predict_image(uploaded.getvalue())
                except Exception as exc:
                    st.error("Inference could not be completed. Check that the model artifacts and dependencies match the training setup.")
                    st.caption(f"Diagnostic: {type(exc).__name__}: {exc}")
                else:
                    label = str(result["prediction"]).replace("_", " ").title()
                    st.markdown("---")
                    st.markdown("### 03 / Result")
                    if label.lower() == "tampered":
                        st.error(f"Possible manipulation detected · {label}")
                    else:
                        st.success(f"Classifier result · {label}")
                    c1, c2 = st.columns(2)
                    with c1:
                        st.metric("Model score", f'{result["confidence"] * 100:.1f}%')
                    with c2:
                        st.metric("Inference time", f'{result["inference_time_ms"]:.0f} ms')
                    st.progress(float(result["confidence"]))
                    st.caption(f'Threshold: {result["threshold"]:.2f} · Class mapping: index 1 = {result["class_1_name"]}')
                    st.warning(result["disclaimer"])

st.write("")
st.markdown("---")
a, b, c = st.columns(3)
with a:
    st.markdown("**01 · Preprocess**")
    st.caption("EXIF correction, RGB conversion and 224 × 224 resizing.")
with b:
    st.markdown("**02 · Extract features**")
    st.caption("VGG19 convolutional backbone with ImageNet initialization.")
with c:
    st.markdown("**03 · Classify**")
    st.caption("Binary sigmoid output mapped to the saved dataset classes.")

st.markdown(
    '<p class="small-note">Research / educational demonstration only. A model prediction is not proof of authenticity, does not identify the editing method, and does not localize altered pixels. Scores are not calibrated probabilities. Avoid uploading sensitive or private images.</p>',
    unsafe_allow_html=True,
)
