

import sys
sys.path.append('.')

import gradio as gr
import tempfile
import os
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_SERVER_PORT"] = str(os.environ.get("PORT", "7860"))
import numpy as np
from PIL import Image

print("Loading pipeline...")
from src.pipeline import PrescriptionPipeline
pipeline = PrescriptionPipeline()
print("Pipeline ready!")


def analyse_prescription(image):
    if image is None:
        return "No image uploaded", "No entities detected", "Upload an image to begin"

    try:
        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        result = pipeline.run(tmp_path)

        # OCR output
        ocr_output = f"OCR Text:\n{result['ocr_text']}\n\nConfidence: {result['ocr_confidence']}"

        # entities output
        entities = result["entities"]
        entities_output = (
            f"Drugs found    : {entities['DRUG']}\n"
            f"Dosages found  : {entities['DOSAGE']}\n"
            f"Frequency      : {entities['FREQUENCY']}\n"
            f"Duration       : {entities['DURATION']}"
        )

        # safety report
        summary = result["summary"]
        risk = summary["risk_level"]

        if risk == "HIGH":
            risk_label = "HIGH RISK"
        elif risk == "MEDIUM":
            risk_label = "MEDIUM RISK"
        else:
            risk_label = "LOW RISK - Safe"

        alerts_text = f"Risk Level: {risk_label}\n"
        alerts_text += f"Critical: {summary['critical']} | Warnings: {summary['warnings']}\n\n"

        if result["alerts"]:
            alerts_text += "Detailed Alerts:\n"
            for alert in result["alerts"]:
                alerts_text += f"[{alert['severity']}] {alert['message']} (conf: {alert.get('confidence', '-')})\n"
        else:
            alerts_text += "No safety issues detected."

        os.unlink(tmp_path)
        return ocr_output, entities_output, alerts_text

    except Exception as e:
        return f"Error: {str(e)}", "", "Analysis failed — please try another image"


# ── Gradio UI ─────────────────────────────────────────────────
with gr.Blocks(title="Prescription Error Detector") as demo:

    gr.Markdown("# Prescription Error Detector")
    gr.Markdown("AI-powered safety check for handwritten prescriptions · BioBERT + EasyOCR")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(
                label="Upload Prescription Image",
                type="pil"
            )
            analyse_btn = gr.Button(
                "Analyse Prescription",
                variant="primary"
            )

        with gr.Column():
            ocr_output    = gr.Textbox(label="Extracted Text (OCR)", lines=6)
            entity_output = gr.Textbox(label="Detected Entities (NER)", lines=6)
            alert_output  = gr.Textbox(label="Safety Report", lines=10)

    analyse_btn.click(
        fn=analyse_prescription,
        inputs=image_input,
        outputs=[ocr_output, entity_output, alert_output]
    )

    gr.Markdown("### How it works")
    gr.Markdown(
        "1. Upload a prescription image (JPG or PNG)\n"
        "2. EasyOCR reads the handwriting\n"
        "3. BioBERT extracts drug names and dosages\n"
        "4. Error detector checks for dangerous interactions and overdoses\n"
        "5. You get a full color-coded safety report in seconds"
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        show_error=True
    )