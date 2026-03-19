import sys
sys.path.append('.')

import re
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline

LOCAL_MODEL_PATH = "models/ner"
BIOMEDICAL_MODEL = "d4data/biomedical-ner-all"

DRUG_PREFIXES = {
    "syp", "syrup", "tab", "tablet", "cap", "capsule",
    "inj", "injection", "drops", "oint", "cream", "gel",
    "rx", "sig", "dt", "no", "sr", "xr", "er", "cr"
}

FREQUENCY_PATTERNS = [
    r'\bonce\s+daily\b', r'\btwice\s+daily\b', r'\bthrice\s+daily\b',
    r'\bonce\s+a\s+day\b', r'\btwice\s+a\s+day\b',
    r'\bthree\s+times\s+a\s+day\b', r'\bfour\s+times\s+a\s+day\b',
    r'\bafter\s+meals?\b', r'\bbefore\s+meals?\b', r'\bwith\s+meals?\b',
    r'\bq\.?4\.?h\b', r'\bq\.?6\.?h\b', r'\bq\.?8\.?h\b', r'\bq\.?12\.?h\b',
    r'\btds\b', r'\bqid\b', r'\bbd\b', r'\bod\b',
    r'\bmorning\b', r'\bnight\b', r'\bbedtime\b', r'\bsos\b',
    r'\b1[-\s]0[-\s]1\b', r'\b1[-\s]1[-\s]1\b',
    r'\b0[-\s]0[-\s]1\b', r'\b1[-\s]0[-\s]0\b',
    r'\bx\s*\d+\s*d\b',
]

DOSAGE_PATTERN = re.compile(
    r'\b(\d+\.?\d*)\s*(mg|mcg|ml|g|units?|tabs?|caps?|tsp|tbsp|iu|drops?)\b',
    re.IGNORECASE
)

DURATION_PATTERN = re.compile(
    r'\b(\d+)\s*(day|days|week|weeks|month|months)\b',
    re.IGNORECASE
)


def clean_drug_name(name):
    name = name.strip().strip("-").strip()
    name = re.sub(r'\s*\d+.*$', '', name).strip()
    if len(name) <= 2:
        return None
    if name.lower() in DRUG_PREFIXES:
        return None
    if re.match(r'^\d+\.?\d*$', name):
        return None
    return name


def load_model():
    print("Loading local BioBERT model...")
    tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
    model = AutoModelForTokenClassification.from_pretrained(LOCAL_MODEL_PATH)
    model.eval()

    print("Loading d4data biomedical NER model...")
    try:
        bio_ner = pipeline(
            "ner",
            model=BIOMEDICAL_MODEL,
            aggregation_strategy="first",
            device=-1
        )
        print("Biomedical NER model ready!")
    except Exception as e:
        print(f"Could not load biomedical model: {e}")
        bio_ner = None

    return tokenizer, model, bio_ner


def extract_entities_biomedical(text, bio_ner):
    if bio_ner is None:
        return {"DRUG": [], "DOSAGE": [], "FREQUENCY": [], "DURATION": []}

    entities = {"DRUG": [], "DOSAGE": [], "FREQUENCY": [], "DURATION": []}

    try:
        results = bio_ner(text[:512])
        for entity in results:
            word = re.sub(r'\s*##\s*', '', entity["word"]).strip()
            word = re.sub(r'\s+', '', word)
            label = entity["entity_group"].upper()
            score = entity["score"]

            if score < 0.6 or len(word) < 3:
                continue

            if any(x in label for x in ["CHEM", "DRUG", "MEDICATION", "MEDICINE"]):
                cleaned = clean_drug_name(word)
                if cleaned and cleaned.lower() not in [
                    d.lower() for d in entities["DRUG"]
                ]:
                    entities["DRUG"].append(cleaned)

    except Exception as e:
        print(f"Biomedical NER error: {e}")

    return entities


def extract_entities_biobert(text, tokenizer, model):
    words = text.split()
    if not words:
        return {"DRUG": [], "DOSAGE": [], "FREQUENCY": [], "DURATION": []}

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

    entities = {"DRUG": [], "DOSAGE": [], "FREQUENCY": [], "DURATION": []}
    for word, label in results:
        if label.startswith("B-"):
            entity_type = label[2:]
            if entity_type in entities:
                cleaned = clean_drug_name(word) if entity_type == "DRUG" else word
                if cleaned:
                    entities[entity_type].append(cleaned)
        elif label.startswith("I-"):
            entity_type = label[2:]
            if entity_type in entities and entities[entity_type]:
                entities[entity_type][-1] += " " + word

    return entities


def extract_entities_regex(text):
    entities = {"DRUG": [], "DOSAGE": [], "FREQUENCY": [], "DURATION": []}

    # dosages
    for match in DOSAGE_PATTERN.finditer(text):
        dosage_str = match.group(0).strip()
        if dosage_str not in entities["DOSAGE"]:
            entities["DOSAGE"].append(dosage_str)

    # frequencies — collect all matches with positions
    all_matches = []
    for pattern in FREQUENCY_PATTERNS:
        for m in re.finditer(pattern, text, re.IGNORECASE):
            all_matches.append((m.start(), m.end(), m.group(0).strip()))

    # sort by length descending so longer matches take priority
    all_matches.sort(key=lambda x: -(x[1] - x[0]))

    # keep only non-overlapping matches
    found_spans = []
    kept_matches = []
    for start, end, match_text in all_matches:
        overlapping = any(s <= start < e or s < end <= e for s, e in found_spans)
        if not overlapping:
            found_spans.append((start, end))
            kept_matches.append((start, match_text))

    # sort by position in original text
    kept_matches.sort(key=lambda x: x[0])
    entities["FREQUENCY"] = [m for _, m in kept_matches]

    # durations
    for m in DURATION_PATTERN.finditer(text):
        dur = m.group(0).strip()
        if dur not in entities["DURATION"]:
            entities["DURATION"].append(dur)

    return entities

def deduplicate_frequencies(freq_list):
    """Remove entries that are substrings of longer entries."""
    result = []
    for item in freq_list:
        # keep item only if no other item contains it
        is_substring = any(
            item.lower() != other.lower() and item.lower() in other.lower()
            for other in freq_list
        )
        if not is_substring and item not in result:
            result.append(item)
    return result


def merge_entities(*entity_dicts):
    merged = {"DRUG": [], "DOSAGE": [], "FREQUENCY": [], "DURATION": []}

    for entities in entity_dicts:
        for key in merged:
            for item in entities.get(key, []):
                item_clean = item.strip()
                if not item_clean:
                    continue
                if item_clean.lower() not in [x.lower() for x in merged[key]]:
                    merged[key].append(item_clean)
    merged["FREQUENCY"] = deduplicate_frequencies(merged["FREQUENCY"])
    return merged

def extract_drugs_by_position(text):
    """
    Extract drug names positionally — word immediately after
    Syp/Tab/Cap/Inj is almost always a drug name.
    """
    entities = {"DRUG": [], "DOSAGE": [], "FREQUENCY": [], "DURATION": []}
    pattern = re.compile(
        r'\b(?:syp\.?|syrup|tab\.?|tablet|cap\.?|capsule|inj\.?)\s+([A-Z][a-zA-Z\-]+)',
        re.IGNORECASE
    )
    for match in pattern.finditer(text):
        drug = match.group(1).strip()
        cleaned = clean_drug_name(drug)
        if cleaned and cleaned.lower() not in [
            d.lower() for d in entities["DRUG"]
        ]:
            entities["DRUG"].append(cleaned)
    return entities


def extract_entities(text, tokenizer, model, bio_ner=None):
    print("\n--- Entity Extraction ---")

    bio_entities  = extract_entities_biomedical(text, bio_ner)
    bert_entities = extract_entities_biobert(text, tokenizer, model)
    regex_entities = extract_entities_regex(text)
    pos_entities  = extract_drugs_by_position(text)

    print(f"Biomedical NER : drugs={bio_entities['DRUG']}")
    print(f"BioBERT NER    : drugs={bert_entities['DRUG']}")
    print(f"Positional     : drugs={pos_entities['DRUG']}")
    print(f"Regex          : dosages={regex_entities['DOSAGE']}, freq={regex_entities['FREQUENCY']}")

    final = merge_entities(bio_entities, bert_entities, pos_entities, regex_entities)
    print(f"Final merged   : {final}")
    return final


if __name__ == "__main__":
    print("Loading models...")
    tokenizer, model, bio_ner = load_model()
    print("\nAll models ready!\n")

    test_cases = [
        "Syp Calpol 250/5 4ml Q6H x 3d",
        "Syp Delcon 3ml TDS x 5d",
        "Syp Levolin 3ml TDS x 5d",
        "Syp Meftal-P 100/5 3ml SOS",
        "Tab Metformin 500mg twice daily after meals",
        "Warfarin 5mg at night Aspirin 75mg once daily",
    ]

    for text in test_cases:
        print(f"\nInput : {text}")
        entities = extract_entities(text, tokenizer, model, bio_ner)
        print(f"Drugs     : {entities['DRUG']}")
        print(f"Dosages   : {entities['DOSAGE']}")
        print(f"Frequency : {entities['FREQUENCY']}")
        print(f"Duration  : {entities['DURATION']}")