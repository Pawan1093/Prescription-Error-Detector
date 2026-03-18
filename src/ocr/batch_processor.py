import sys
sys.path.append('.')

import os
import json
from src.ocr.extractor import PrescriptionOCR

def process_all_prescriptions(raw_folder="data/raw", output_folder="data/processed"):
    os.makedirs(output_folder, exist_ok=True)
    
    ocr = PrescriptionOCR()
    
    images = [f for f in os.listdir(raw_folder) if f.endswith('.jpg') or f.endswith('.png')]
    images.sort(key=lambda x: int(x.split('.')[0]))
    
    print(f"\nFound {len(images)} images to process\n")
    
    results = []
    failed = []

    for i, filename in enumerate(images):
        image_path = os.path.join(raw_folder, filename)
        print(f"Processing {i+1}/{len(images)}: {filename}", end=" ... ")

        try:
            result = ocr.extract_text(image_path)

            record = {
                "id": filename.split('.')[0],
                "filename": filename,
                "full_text": result["full_text"],
                "word_count": result["word_count"],
                "avg_confidence": result["avg_confidence"],
                "status": "success"
            }
            results.append(record)
            print(f"OK (confidence: {result['avg_confidence']}, words: {result['word_count']})")

        except Exception as e:
            failed.append(filename)
            print(f"FAILED — {str(e)}")

    # save all results to one JSON file
    output_path = os.path.join(output_folder, "ocr_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # save summary
    summary = {
        "total_images": len(images),
        "successful": len(results),
        "failed": len(failed),
        "failed_files": failed,
        "avg_confidence_overall": round(
            sum(r["avg_confidence"] for r in results) / len(results), 3
        ) if results else 0
    }

    summary_path = os.path.join(output_folder, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n--- Batch Processing Complete ---")
    print(f"Total images     : {summary['total_images']}")
    print(f"Successfully done: {summary['successful']}")
    print(f"Failed           : {summary['failed']}")
    print(f"Avg confidence   : {summary['avg_confidence_overall']}")
    print(f"\nResults saved to : {output_path}")
    print(f"Summary saved to : {summary_path}")

if __name__ == "__main__":
    process_all_prescriptions()