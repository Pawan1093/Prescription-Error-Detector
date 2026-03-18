import sys
sys.path.append('.')

import streamlit as st
from PIL import Image
import tempfile
import os

st.set_page_config(
    page_title="Prescription Error Detector",
    page_icon="🏥",
    layout="wide"
)

# ── Load pipeline once (cached) ───────────────────────────────
@st.cache_resource
def load_pipeline():
    from src.pipeline import PrescriptionPipeline
    return PrescriptionPipeline()

# ── Header ────────────────────────────────────────────────────
st.title("Prescription Error Detector")
st.markdown("AI-powered safety check for handwritten prescriptions · Built with BioBERT + EasyOCR")
st.divider()

# ── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.header("About")
    st.info(
        "This system reads handwritten prescription images and flags:\n\n"
        "- Dangerous drug interactions\n"
        "- Overdose / underdose risks\n"
        "- Missing critical information"
    )
    st.header("Model Info")
    st.success("OCR: EasyOCR")
    st.success("NER: BioBERT fine-tuned")
    st.success("DB: 20 drugs · 10 interaction rules")

# ── Main layout ───────────────────────────────────────────────
col1, col2 = st.columns([1, 1.2])

with col1:
    st.subheader("Upload Prescription")
    uploaded = st.file_uploader(
        "Drop a prescription image here",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded:
        image = Image.open(uploaded)
        st.image(image, caption="Uploaded prescription", use_container_width=True)
        analyse_btn = st.button("Analyse Prescription", type="primary", use_container_width=True)
    else:
        st.info("Upload a prescription image to begin analysis.")
        analyse_btn = False

with col2:
    st.subheader("Analysis Results")

    if uploaded and analyse_btn:
        with st.spinner("Running AI analysis..."):
            # save uploaded file temporarily
            suffix = os.path.splitext(uploaded.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.getvalue())
                tmp_path = tmp.name

            try:
                pipeline = load_pipeline()
                result = pipeline.run(tmp_path)
            finally:
                os.unlink(tmp_path)

        # ── OCR output ────────────────────────────────────────
        with st.expander("Extracted text (OCR)", expanded=True):
            st.code(result["ocr_text"])
            st.caption(f"OCR confidence: {result['ocr_confidence']}")

        # ── Entities ──────────────────────────────────────────
        with st.expander("Detected entities (NER)", expanded=True):
            ecol1, ecol2 = st.columns(2)
            with ecol1:
                st.metric("Drugs found", len(result["entities"]["DRUG"]))
                st.write(result["entities"]["DRUG"] or "None detected")
            with ecol2:
                st.metric("Dosages found", len(result["entities"]["DOSAGE"]))
                st.write(result["entities"]["DOSAGE"] or "None detected")

        # ── Risk summary ──────────────────────────────────────
        st.subheader("Risk Assessment")
        risk = result["summary"]["risk_level"]
        rcol1, rcol2, rcol3 = st.columns(3)

        with rcol1:
            st.metric("Critical", result["summary"]["critical"],
                      delta="High risk" if result["summary"]["critical"] > 0 else None,
                      delta_color="inverse")
        with rcol2:
            st.metric("Warnings", result["summary"]["warnings"])
        with rcol3:
            st.metric("Risk Level", risk)

        if risk == "HIGH":
            st.error(f"HIGH RISK — {result['summary']['critical']} critical issue(s) found. Do not dispense without review.")
        elif risk == "MEDIUM":
            st.warning(f"MEDIUM RISK — {result['summary']['warnings']} warning(s) found. Pharmacist review recommended.")
        else:
            st.success("LOW RISK — No safety issues detected. Prescription appears safe.")

        # ── Alerts ────────────────────────────────────────────
        if result["alerts"]:
            st.subheader("Detailed Alerts")
            for alert in result["alerts"]:
                if alert["severity"] == "CRITICAL":
                    st.error(f"**{alert['type']}** — {alert['message']} (confidence: {alert.get('confidence', '-')})")
                else:
                    st.warning(f"**{alert['type']}** — {alert['message']} (confidence: {alert.get('confidence', '-')})")

    elif not uploaded:
        st.markdown("""
        **How it works:**
        1. Upload a prescription image (JPG or PNG)
        2. Click Analyse Prescription
        3. AI reads the handwriting via OCR
        4. BioBERT extracts drug names and dosages
        5. Error detector checks for safety issues
        6. You get a full safety report in seconds
        """)