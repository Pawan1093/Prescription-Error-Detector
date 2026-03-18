from src.ocr.extractor import PrescriptionOCR

ocr = PrescriptionOCR()

for img in ['10.jpg', '27.jpg', '52.jpg', '75.jpg']:
    result = ocr.extract_text(f'data/raw/{img}')
    print(f'\n--- {img} ---')
    print(result['full_text'])
    print(f'Confidence: {result["avg_confidence"]}')
    print(f'Words: {result["word_count"]}')