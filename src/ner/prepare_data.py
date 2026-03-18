import json
import random
import os

random.seed(42)
os.makedirs("data/processed", exist_ok=True)

# Manually labelled prescription sentences
# Format: (sentence, [(start, end, label)])
TRAINING_SAMPLES = [
    ("Tab Metformin 500mg twice daily after meals", [
        (4, 13, "DRUG"), (14, 19, "DOSAGE"), (20, 25, "FREQUENCY"), (26, 31, "FREQUENCY")]),
    ("Warfarin 5mg at night for 30 days", [
        (0, 8, "DRUG"), (9, 12, "DOSAGE"), (16, 21, "FREQUENCY"), (22, 24, "DURATION"), (25, 29, "DURATION")]),
    ("Tab Aspirin 325mg once daily", [
        (4, 11, "DRUG"), (12, 17, "DOSAGE"), (18, 22, "FREQUENCY"), (23, 28, "FREQUENCY")]),
    ("Amoxicillin 250mg thrice daily for 7 days", [
        (0, 11, "DRUG"), (12, 17, "DOSAGE"), (18, 24, "FREQUENCY"), (25, 30, "FREQUENCY"), (31, 32, "DURATION"), (33, 37, "DURATION")]),
    ("Amlodipine 10mg once daily in the morning", [
        (0, 10, "DRUG"), (11, 15, "DOSAGE"), (16, 20, "FREQUENCY"), (21, 26, "FREQUENCY")]),
    ("Atorvastatin 40mg once at bedtime", [
        (0, 12, "DRUG"), (13, 17, "DOSAGE"), (18, 22, "FREQUENCY"), (26, 33, "FREQUENCY")]),
    ("Paracetamol 500mg three times a day for 5 days", [
        (0, 11, "DRUG"), (12, 17, "DOSAGE"), (18, 29, "FREQUENCY"), (30, 31, "DURATION"), (32, 36, "DURATION")]),
    ("Azithromycin 500mg once daily for 3 days", [
        (0, 12, "DRUG"), (13, 18, "DOSAGE"), (19, 23, "FREQUENCY"), (24, 29, "FREQUENCY"), (30, 31, "DURATION"), (32, 36, "DURATION")]),
    ("Metoprolol 25mg twice a day", [
        (0, 9, "DRUG"), (10, 14, "DOSAGE"), (15, 20, "FREQUENCY"), (23, 26, "FREQUENCY")]),
    ("Omeprazole 20mg before breakfast daily", [
        (0, 10, "DRUG"), (11, 15, "DOSAGE"), (23, 32, "FREQUENCY"), (33, 38, "FREQUENCY")]),
    ("Ciprofloxacin 500mg twice daily for 10 days", [
        (0, 13, "DRUG"), (14, 19, "DOSAGE"), (20, 25, "FREQUENCY"), (26, 31, "FREQUENCY"), (32, 34, "DURATION"), (35, 39, "DURATION")]),
    ("Losartan 50mg once daily in morning", [
        (0, 8, "DRUG"), (9, 13, "DOSAGE"), (14, 18, "FREQUENCY"), (19, 24, "FREQUENCY")]),
    ("Insulin glargine 10 units at bedtime", [
        (0, 15, "DRUG"), (16, 18, "DOSAGE"), (19, 24, "DOSAGE"), (28, 35, "FREQUENCY")]),
    ("Prednisolone 10mg daily for 5 days tapering", [
        (0, 12, "DRUG"), (13, 17, "DOSAGE"), (18, 23, "FREQUENCY"), (24, 25, "DURATION"), (26, 30, "DURATION")]),
    ("Tab Diazepam 5mg at night only", [
        (4, 12, "DRUG"), (13, 16, "DOSAGE"), (20, 25, "FREQUENCY")]),
    ("Clopidogrel 75mg once daily", [
        (0, 11, "DRUG"), (12, 16, "DOSAGE"), (17, 21, "FREQUENCY"), (22, 27, "FREQUENCY")]),
    ("Pantoprazole 40mg before meals twice daily", [
        (0, 12, "DRUG"), (13, 17, "DOSAGE"), (25, 30, "FREQUENCY"), (31, 36, "FREQUENCY")]),
    ("Hydroxychloroquine 200mg twice daily for 6 months", [
        (0, 18, "DRUG"), (19, 24, "DOSAGE"), (25, 30, "FREQUENCY"), (31, 36, "FREQUENCY"), (37, 38, "DURATION"), (39, 45, "DURATION")]),
    ("Ramipril 5mg once daily in the evening", [
        (0, 8, "DRUG"), (9, 12, "DOSAGE"), (13, 17, "FREQUENCY"), (18, 23, "FREQUENCY")]),
    ("Furosemide 40mg once in the morning", [
        (0, 10, "DRUG"), (11, 15, "DOSAGE"), (16, 20, "FREQUENCY")]),
]

def sentence_to_bio_tags(sentence, annotations):
    """Convert sentence + annotations to BIO token labels."""
    words = sentence.split()
    labels = []
    char_idx = 0
    word_spans = []

    for word in words:
        start = sentence.index(word, char_idx)
        end = start + len(word)
        word_spans.append((start, end))
        char_idx = end

    for word, (w_start, w_end) in zip(words, word_spans):
        label = "O"
        for ann_start, ann_end, ann_label in annotations:
            if w_start >= ann_start and w_end <= ann_end:
                if w_start == ann_start:
                    label = f"B-{ann_label}"
                else:
                    label = f"I-{ann_label}"
                break
        labels.append(label)

    return words, labels

def build_dataset():
    all_data = []
    for sentence, annotations in TRAINING_SAMPLES:
        words, labels = sentence_to_bio_tags(sentence, annotations)
        all_data.append({
            "id": len(all_data),
            "tokens": words,
            "ner_tags": labels,
            "sentence": sentence
        })

    # shuffle and split 80/20
    random.shuffle(all_data)
    split = int(len(all_data) * 0.8)
    train_data = all_data[:split]
    test_data = all_data[split:]

    with open("data/processed/ner_train.json", "w") as f:
        json.dump(train_data, f, indent=2)

    with open("data/processed/ner_test.json", "w") as f:
        json.dump(test_data, f, indent=2)

    print(f"Training samples : {len(train_data)}")
    print(f"Test samples     : {len(test_data)}")
    print(f"\nSample record:")
    sample = train_data[0]
    for token, tag in zip(sample["tokens"], sample["ner_tags"]):
        print(f"  {token:20s} {tag}")
    print("\nData saved to data/processed/")

if __name__ == "__main__":
    build_dataset()