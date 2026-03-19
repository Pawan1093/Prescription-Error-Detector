import sys
sys.path.append('.')

import gradio as gr
from src.pipeline import PrescriptionPipeline
import json

# load pipeline once
print("Loading pipeline...")
pipeline = PrescriptionPipeline()
print("Pipeline ready!")

def analyse_prescription(image):
    """Main function called by Gradio."""
    if image is None:
        return "No image uploaded", "{}", "Upload an image to begin"

    try:
        # save temp image
        import tempfile
        import numpy as np
        from PIL import Image

        if isinstance(image, np.ndarray):
            img = Image.fromarray(image)
        else:
            img = image

        with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
            img.save(tmp.name)
            tmp_path = tmp.name

        result = pipeline.run(tmp_path)

        # format OCR output
        ocr_text = result["ocr_text"]
        ocr_conf = result["ocr_confidence"]
        ocr_output = f"OCR Text:\n{ocr_text}\n\nConfidence: {ocr_conf}"

        # format entities
        entities = result["entities"]
        entities_output = (
            f"Drugs found    : {entities['DRUG']}\n"
            f"Dosages found  : {entities['DOSAGE']}\n"
            f"Frequency      : {entities['FREQUENCY']}\n"
            f"Duration       : {entities['DURATION']}"
        )

        # format alerts
        summary = result["summary"]
        risk = summary["risk_level"]

        if risk == "HIGH":
            risk_label = "HIGH RISK"
        elif risk == "MEDIUM":
            risk_label = "MEDIUM RISK"
        else:
            risk_label = "LOW RISK — Safe"

        alerts_text = f"Risk Level: {risk_label}\n"
        alerts_text += f"Critical: {summary['critical']} | Warnings: {summary['warnings']}\n\n"

        if result["alerts"]:
            alerts_text += "Detailed Alerts:\n"
            for alert in result["alerts"]:
                severity = alert["severity"]
                message = alert["message"]
                conf = alert.get("confidence", "-")
                alerts_text += f"[{severity}] {message} (conf: {conf})\n"
        else:
            alerts_text += "No safety issues detected."

        import os
        os.unlink(tmp_path)

        return ocr_output, entities_output, alerts_text

    except Exception as e:
        return f"Error: {str(e)}", "", "Analysis failed"


# build Gradio UI
with gr.Blocks(title="Prescription Error Detector") as demo:
    gr.Markdown("# Prescription Error Detector")
    gr.Markdown("AI-powered safety check for handwritten prescriptions · BioBERT + EasyOCR")

    with gr.Row():
        with gr.Column():
            image_input = gr.Image(label="Upload Prescription Image", type="pil")
            analyse_btn = gr.Button("Analyse Prescription", variant="primary")

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
        "1. Upload a prescription image\n"
        "2. EasyOCR reads the handwriting\n"
        "3. BioBERT extracts drug names and dosages\n"
        "4. Error detector checks for dangerous interactions and overdoses\n"
        "5. You get a full safety report"
    )

    gr.Examples(
        examples=[["data/raw/108.jpg"], ["data/raw/75.jpg"]],
        inputs=image_input
    )

if __name__ == "__main__":
    demo.launch()