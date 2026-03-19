import sys
sys.path.append('.')

import json
import numpy as np
import os
from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
        DataCollatorForTokenClassification
    )
from datasets import Dataset
import evaluate

# ── label setup ──────────────────────────────────────────────
LABELS = ["O", "B-DRUG", "I-DRUG", "B-DOSAGE", "I-DOSAGE",
          "B-FREQUENCY", "I-FREQUENCY", "B-DURATION", "I-DURATION"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}

MODEL_NAME = "dmis-lab/biobert-base-cased-v1.2"

# ── load data ─────────────────────────────────────────────────
def load_data(path):
    with open(path) as f:
        raw = json.load(f)
    return Dataset.from_list([
        {"tokens": d["tokens"],
         "ner_tags": [LABEL2ID.get(t, 0) for t in d["ner_tags"]]}
        for d in raw
    ])

# ── tokenize + align labels ───────────────────────────────────
def tokenize_and_align(examples, tokenizer):
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
    )
    all_labels = []
    for i, label_ids in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        aligned = []
        prev_word = None
        for word_id in word_ids:
            if word_id is None:
                aligned.append(-100)
            elif word_id != prev_word:
                aligned.append(label_ids[word_id])
            else:
                aligned.append(-100)
            prev_word = word_id
        all_labels.append(aligned)
    tokenized["labels"] = all_labels
    return tokenized

# ── metrics ───────────────────────────────────────────────────
def compute_metrics(p, metric):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    true_labels = [
        [ID2LABEL[l] for l in label if l != -100]
        for label in labels
    ]
    true_preds = [
        [ID2LABEL[p] for p, l in zip(pred, label) if l != -100]
        for pred, label in zip(predictions, labels)
    ]
    results = metric.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": round(results["overall_precision"], 4),
        "recall":    round(results["overall_recall"], 4),
        "f1":        round(results["overall_f1"], 4),
        "accuracy":  round(results["overall_accuracy"], 4),
    }

# ── main ──────────────────────────────────────────────────────
def train():
    print("Loading BioBERT tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    print("Loading datasets...")
    train_ds = load_data("data/processed/ner_train.json")
    test_ds  = load_data("data/processed/ner_test.json")

    print("Tokenizing...")
    fn = lambda x: tokenize_and_align(x, tokenizer)
    train_ds = train_ds.map(fn, batched=True)
    test_ds  = test_ds.map(fn, batched=True)

    print("Loading BioBERT model...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
        ignore_mismatched_sizes=True
    )

    args = TrainingArguments(
        output_dir="models/ner",
        num_train_epochs=10,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=5,
        report_to="none"
    )

    metric = evaluate.load("seqeval")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=lambda p: compute_metrics(p, metric),
    )

    print("\nStarting training...")
    print("This will take 5-10 minutes on CPU. Please wait...\n")
    trainer.train()

    print("\nSaving model...")
    os.makedirs("models/ner", exist_ok=True)
    trainer.save_model("models/ner")
    tokenizer.save_pretrained("models/ner")

    print("\nEvaluating on test set...")
    results = trainer.evaluate()
    print(f"\n--- Results ---")
    print(f"F1 Score  : {results.get('eval_f1', 'N/A')}")
    print(f"Precision : {results.get('eval_precision', 'N/A')}")
    print(f"Recall    : {results.get('eval_recall', 'N/A')}")
    print(f"Accuracy  : {results.get('eval_accuracy', 'N/A')}")
    print("\nModel saved to models/ner/")

if __name__ == "__main__":
    train()