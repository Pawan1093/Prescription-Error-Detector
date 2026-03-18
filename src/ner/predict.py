import sys
sys.path.append('.')

from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch

MODEL_PATH = "models/ner"

def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(MODEL_PATH)
    model.eval()
    return tokenizer, model

def predict(text, tokenizer, model):
    words = text.split()
    inputs = tokenizer(
        words,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)

    predictions = torch.argmax(outputs.logits, dim=2)[0]
    word_ids = inputs.word_ids()
    id2label = model.config.id2label

    results = []
    seen = set()
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id in seen:
            continue
        seen.add(word_id)
        label = id2label[predictions[idx].item()]
        results.append((words[word_id], label))
    return results

def extract_entities(text, tokenizer, model):
    predictions = predict(text, tokenizer, model)
    entities = {"DRUG": [], "DOSAGE": [], "FREQUENCY": [], "DURATION": []}
    for word, label in predictions:
        if label.startswith("B-"):
            entity_type = label[2:]
            if entity_type in entities:
                entities[entity_type].append(word)
        elif label.startswith("I-"):
            entity_type = label[2:]
            if entity_type in entities and len(entities[entity_type]) > 0:
                entities[entity_type][-1] += " " + word
    return entities

if __name__ == "__main__":
    print("Loading model...")
    tokenizer, model = load_model()
    print("Model loaded!\n")

    test_sentences = [
        "Tab Metformin 500mg twice daily after meals",
        "Warfarin 5mg at night for 30 days",
        "Amoxicillin 250mg thrice daily for 7 days",
        "Amlodipine 10mg once daily in morning",
    ]

    for sentence in test_sentences:
        print(f"Input : {sentence}")
        entities = extract_entities(sentence, tokenizer, model)
        print(f"Drug      : {entities['DRUG']}")
        print(f"Dosage    : {entities['DOSAGE']}")
        print(f"Frequency : {entities['FREQUENCY']}")
        print(f"Duration  : {entities['DURATION']}")
        print()