<div align="center">

# Prescription Error Detector
### AI-powered safety system for handwritten medical prescriptions

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-red.svg)
![BioBERT](https://img.shields.io/badge/BioBERT-Fine--tuned-green.svg)
![EasyOCR](https://img.shields.io/badge/EasyOCR-1.7-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.55-ff4b4b.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

</div>

---

## The Problem

Every year, **1.3 million people** are harmed by preventable medication errors globally. In India, most prescriptions are still handwritten — making manual verification slow, error-prone, and dangerous. Pharmacists spend 2+ minutes cross-referencing each prescription against drug databases. This system reduces that to **under 3 seconds**.

---

## Demo

> Upload a prescription image → AI reads the handwriting → Extracts drugs and dosages → Flags dangerous errors instantly

### Full App View
![Demo](assets/demo.png)

### OCR Text Extraction
![OCR](assets/ocr.png)

### Entity Detection (NER)
![NER](assets/ner.png)

### Safety Alerts
![Alerts](assets/alerts.png)

---

## What It Detects

| Error Type | Example | Severity |
|------------|---------|----------|
| Drug interaction | Warfarin + Aspirin = bleeding risk | CRITICAL |
| Overdose | Metformin 3000mg (max: 2000mg) | CRITICAL |
| Underdose | Amoxicillin 100mg (min: 250mg) | WARNING |
| Missing info | No dosage found on prescription | WARNING |
| Unknown drug | Unrecognized medication name | WARNING |

---

## System Architecture
```
Prescription Image
        ↓
   EasyOCR Engine          ← Handwriting recognition + image preprocessing
        ↓
  4-Layer NER Pipeline
  ├── d4data Biomedical NER    ← 8 medical datasets, thousands of drug names
  ├── Fine-tuned BioBERT       ← Trained on prescription data (F1: 0.87)
  ├── Positional Extraction    ← Word after Tab/Syp/Cap = drug name
  └── Smart Regex              ← Dosages, frequencies, durations
        ↓
  Error Detection Engine
  ├── Dosage range checker     ← 20 drugs with safe min/max limits
  ├── Drug interaction checker ← 10 dangerous drug pairs
  ├── Missing info checker     ← Flags incomplete prescriptions
  └── Fuzzy drug matching      ← Handles OCR typos and brand names
        ↓
  Risk Report + Confidence Scores
```

---

## Results

| Metric | Score |
|--------|-------|
| NER F1 Score | **0.867** |
| NER Precision | 0.812 |
| NER Recall | 0.929 |
| OCR Avg Confidence | 0.42 (real handwriting) |
| Drugs in database | 20 generic + 50+ brand names |
| Interaction rules | 10 critical drug pairs |
| Dataset size | 129 real prescription images |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Handwriting OCR | EasyOCR 1.7 + OpenCV preprocessing |
| Medical NER | BioBERT (dmis-lab) + d4data/biomedical-ner-all |
| Error Detection | Custom rule engine + fuzzy matching |
| Frontend | Streamlit + Gradio |
| ML Framework | PyTorch + HuggingFace Transformers |
| Data | Kaggle illegible prescriptions dataset (129 images) |

---

## Project Structure
```
prescription-error-detector/
├── data/
│   ├── raw/              ← 129 real prescription images (Kaggle)
│   └── processed/        ← OCR results + NER training data
├── models/
│   └── ner/              ← Fine-tuned BioBERT model
├── src/
│   ├── ocr/
│   │   ├── extractor.py       ← EasyOCR pipeline
│   │   └── batch_processor.py ← Batch OCR on all images
│   ├── ner/
│   │   ├── prepare_data.py    ← BIO tagging + dataset builder
│   │   ├── train_ner.py       ← BioBERT fine-tuning
│   │   └── predict.py         ← 4-layer entity extraction
│   └── detection/
│       └── error_detector.py  ← Error detection engine
├── app/
│   └── main.py           ← Streamlit UI
├── app_gradio.py         ← Gradio UI
├── src/pipeline.py       ← End-to-end pipeline
└── notebooks/            ← Experiments + data generation
```

---

## Run Locally
```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/prescription-error-detector
cd prescription-error-detector

# Create virtual environment
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # Mac/Linux

# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app/main.py

# OR run Gradio app
python app_gradio.py
```

---

## Key Challenges Solved

**OCR on real handwriting** — Handwritten prescriptions have average OCR confidence of 0.42. Solved using CLAHE image enhancement, denoising, and contrast normalization before OCR.

**Unknown brand names** — Generic NER models don't know Indian brand names like Calpol, Delcon, Levolin. Solved using positional extraction (word after Tab/Syp is always a drug) + biomedical NER + keyword fallback.

**Subword tokenization in BERT** — BioBERT splits "Warfarin" into "War" + "##farin". Solved by aligning labels to word IDs, not token IDs, during training and inference.

**Duplicate entity detection** — Multiple extractors produce overlapping results. Solved using case-insensitive deduplication and substring filtering across all layers.

---

## Limitations and Future Work

- OCR accuracy drops below 0.4 on very messy handwriting — Google Vision API would improve this to 95%+
- NER trained on 20 samples — 10,000+ labeled prescriptions would significantly improve generalization
- Drug database covers 20 generic drugs — OpenFDA API integration would add 100,000+ drugs
- No multilingual support — Hindi/Tamil/Malayalam prescriptions need language-specific OCR

---

## What I Learned

This project taught me how real ML systems fail in production — OCR errors cascade into NER failures which cascade into wrong safety alerts. Building defenses at each layer (preprocessing, multi-model extraction, fuzzy matching, confidence thresholds) is what separates a demo from a production system.

---

## License

MIT License — free to use, modify, and distribute.

---

<div align="center">
Built by Pawan Tatyaso Pawar · Fresher ML Engineer
</div>