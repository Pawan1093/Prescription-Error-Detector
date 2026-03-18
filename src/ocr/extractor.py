import easyocr
import cv2
import numpy as np

class PrescriptionOCR:
    def __init__(self):
        print("Loading OCR model... (first time takes 1-2 mins to download)")
        self.reader = easyocr.Reader(['en'], gpu=False)
        print("OCR model ready!")

    def preprocess_image(self, image_path: str) -> np.ndarray:
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        denoised = cv2.fastNlMeansDenoising(gray, h=10)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        return enhanced

    def extract_text(self, image_path: str) -> dict:
        processed = self.preprocess_image(image_path)
        results = self.reader.readtext(processed)

        extracted = []
        for (bbox, text, confidence) in results:
            extracted.append({
                "text": text,
                "confidence": round(confidence, 3),
            })

        full_text = " ".join([r["text"] for r in extracted])
        avg_confidence = round(
            sum(r["confidence"] for r in extracted) / len(extracted), 3
        ) if extracted else 0

        return {
            "full_text": full_text,
            "words": extracted,
            "word_count": len(extracted),
            "avg_confidence": avg_confidence
        }


if __name__ == "__main__":
    ocr = PrescriptionOCR()
    result = ocr.extract_text("data/raw/1.jpg")
    print("\n--- Extracted Text ---")
    print(result["full_text"])
    print(f"\nWords found: {result['word_count']}")
    print(f"Avg confidence: {result['avg_confidence']}")