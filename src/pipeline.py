import sys
sys.path.append('.')

from src.ocr.extractor import PrescriptionOCR
from src.ner.predict import load_model, extract_entities
from src.detection.error_detector import PrescriptionAnalyzer

class PrescriptionPipeline:
    def __init__(self):
        print("Loading OCR engine...")
        self.ocr = PrescriptionOCR()
        print("Loading NER models...")
        self.tokenizer, self.ner_model, self.bio_ner = load_model()
        print("Loading error detector...")
        self.analyzer = PrescriptionAnalyzer()
        print("\nPipeline ready. All systems operational.\n")

    def run(self, image_path):
        print(f"Processing: {image_path}")
        print("-" * 50)

        # step 1: OCR
        ocr_result = self.ocr.extract_text(image_path)
        text = ocr_result["full_text"]
        print(f"OCR text       : {text[:80]}...")
        print(f"OCR confidence : {ocr_result['avg_confidence']}")

        # step 2: 3-layer NER
        entities = extract_entities(
            text, self.tokenizer, self.ner_model, self.bio_ner
        )

        print(f"Drugs found    : {entities['DRUG']}")
        print(f"Dosages found  : {entities['DOSAGE']}")
        print(f"Frequency found: {entities['FREQUENCY']}")

        # step 3: error detection
        result = self.analyzer.analyse(
            entities,
            ner_confidence=ocr_result["avg_confidence"]
        )

        print(f"\n--- SAFETY REPORT ---")
        print(f"Risk level  : {result['summary']['risk_level']}")
        print(f"Critical    : {result['summary']['critical']}")
        print(f"Warnings    : {result['summary']['warnings']}")

        if result["alerts"]:
            for alert in result["alerts"]:
                print(f"  [{alert['severity']}] {alert['message']}")
        else:
            print("No safety issues detected.")

        print("-" * 50)

        return {
            "image": image_path,
            "ocr_text": text,
            "ocr_confidence": ocr_result["avg_confidence"],
            "entities": entities,
            "alerts": result["alerts"],
            "summary": result["summary"]
        }


if __name__ == "__main__":
    pipeline = PrescriptionPipeline()
    result = pipeline.run("data/raw/75.jpg")