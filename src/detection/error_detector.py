import sys
sys.path.append('.')

import json
import re
import os

# ── Drug database ─────────────────────────────────────────────
DRUG_DOSAGE_LIMITS = {
    "metformin":          {"min": 500,  "max": 2000, "unit": "mg"},
    "warfarin":           {"min": 1,    "max": 10,   "unit": "mg"},
    "aspirin":            {"min": 75,   "max": 325,  "unit": "mg"},
    "amoxicillin":        {"min": 250,  "max": 500,  "unit": "mg"},
    "amlodipine":         {"min": 2.5,  "max": 10,   "unit": "mg"},
    "atorvastatin":       {"min": 10,   "max": 80,   "unit": "mg"},
    "paracetamol":        {"min": 500,  "max": 1000, "unit": "mg"},
    "azithromycin":       {"min": 250,  "max": 500,  "unit": "mg"},
    "metoprolol":         {"min": 25,   "max": 200,  "unit": "mg"},
    "omeprazole":         {"min": 10,   "max": 40,   "unit": "mg"},
    "ciprofloxacin":      {"min": 250,  "max": 750,  "unit": "mg"},
    "losartan":           {"min": 25,   "max": 100,  "unit": "mg"},
    "prednisolone":       {"min": 5,    "max": 60,   "unit": "mg"},
    "diazepam":           {"min": 2,    "max": 10,   "unit": "mg"},
    "clopidogrel":        {"min": 75,   "max": 150,  "unit": "mg"},
    "pantoprazole":       {"min": 20,   "max": 40,   "unit": "mg"},
    "ramipril":           {"min": 1.25, "max": 10,   "unit": "mg"},
    "furosemide":         {"min": 20,   "max": 80,   "unit": "mg"},
    "hydroxychloroquine": {"min": 200,  "max": 400,  "unit": "mg"},
    "insulin glargine":   {"min": 10,   "max": 50,   "unit": "units"},
}

DRUG_INTERACTIONS = [
    {"drugs": ["warfarin", "aspirin"],      "severity": "CRITICAL", "message": "Warfarin + Aspirin significantly increases bleeding risk"},
    {"drugs": ["warfarin", "ibuprofen"],    "severity": "CRITICAL", "message": "Warfarin + Ibuprofen increases bleeding risk dangerously"},
    {"drugs": ["warfarin", "ciprofloxacin"],"severity": "CRITICAL", "message": "Ciprofloxacin + Warfarin greatly increases bleeding risk"},
    {"drugs": ["losartan", "ramipril"],     "severity": "CRITICAL", "message": "Two ACE inhibitors together cause dangerous blood pressure drop"},
    {"drugs": ["aspirin", "ibuprofen"],     "severity": "WARNING",  "message": "Aspirin + Ibuprofen reduces effectiveness of both drugs"},
    {"drugs": ["amlodipine", "simvastatin"],"severity": "WARNING",  "message": "Amlodipine + Simvastatin increases risk of muscle damage"},
    {"drugs": ["diazepam", "metoprolol"],   "severity": "WARNING",  "message": "Diazepam + Metoprolol may cause excessive sedation"},
    {"drugs": ["furosemide", "metformin"],  "severity": "WARNING",  "message": "Furosemide + Metformin may increase risk of lactic acidosis"},
    {"drugs": ["prednisolone", "aspirin"],  "severity": "WARNING",  "message": "Prednisolone + Aspirin increases risk of GI bleeding"},
    {"drugs": ["metformin", "alcohol"],     "severity": "WARNING",  "message": "Metformin + Alcohol increases risk of lactic acidosis"},
]

# ── Fuzzy matching without external library ───────────────────
def simple_similarity(a, b):
    """Character overlap similarity — no external library needed."""
    a, b = a.lower(), b.lower()
    if a == b:
        return 100
    if a in b or b in a:
        return 90
    # count common characters
    common = sum(min(a.count(c), b.count(c)) for c in set(a))
    return int(100 * (2 * common) / (len(a) + len(b))) if (len(a) + len(b)) > 0 else 0

def normalize_drug_name(drug_name):
    """Clean and match drug name against database."""
    # remove common prefixes like Tab, Cap, Inj
    name = re.sub(
        r'^(tab\.?|tablet|cap\.?|capsule|inj\.?|injection|syrup|drops)\s*',
        '', str(drug_name), flags=re.IGNORECASE
    ).strip().lower()

    # remove trailing dosage if stuck to name e.g. "Metformin500"
    name = re.sub(r'\d+.*$', '', name).strip()

    # exact match
    if name in DRUG_DOSAGE_LIMITS:
        return name

    # fuzzy match against database
    best_match, best_score = None, 0
    for db_drug in DRUG_DOSAGE_LIMITS:
        score = simple_similarity(name, db_drug)
        if score > best_score:
            best_score = score
            best_match = db_drug

    if best_score >= 80:
        return best_match

    return name  # fallback to cleaned name

def parse_dosage(dosage_str):
    """Parse dosage — handles 500mg, 500 mg, 1g, 10 units, ranges."""
    text = str(dosage_str).lower()

    # convert grams to mg
    g_match = re.search(r'(\d+\.?\d*)\s*g\b', text)
    if g_match:
        return float(g_match.group(1)) * 1000

    # extract all numbers, return the highest (handles ranges like 500-1000mg)
    numbers = re.findall(r'(\d+\.?\d*)', text)
    if numbers:
        return max(float(n) for n in numbers)

    return None

# ── Core detection functions ──────────────────────────────────
def check_missing_info(entities):
    alerts = []
    if not entities.get("DRUG"):
        alerts.append({
            "type": "MISSING_DRUG", "severity": "CRITICAL",
            "message": "No drug name could be identified in this prescription",
            "confidence": 0.99
        })
    if not entities.get("DOSAGE"):
        alerts.append({
            "type": "MISSING_DOSAGE", "severity": "WARNING",
            "message": "No dosage information found — pharmacist must verify",
            "confidence": 0.99
        })
    if not entities.get("FREQUENCY"):
        alerts.append({
            "type": "MISSING_FREQUENCY", "severity": "WARNING",
            "message": "No frequency information found — pharmacist must verify",
            "confidence": 0.99
        })
    return alerts

def check_dosage(drug_raw, dosage_list, ner_confidence=0.9):
    alerts = []
    drug_key = normalize_drug_name(drug_raw)

    if drug_key not in DRUG_DOSAGE_LIMITS:
        return alerts

    limits = DRUG_DOSAGE_LIMITS[drug_key]

    for dosage_str in dosage_list:
        value = parse_dosage(dosage_str)
        if value is None:
            continue
        if value > limits["max"]:
            alerts.append({
                "type": "OVERDOSE", "severity": "CRITICAL",
                "drug": drug_key,
                "message": (f"{drug_raw} {dosage_str} exceeds safe maximum "
                            f"({limits['max']}{limits['unit']}). Risk of toxicity."),
                "confidence": round(ner_confidence * 0.95, 3)
            })
        elif value < limits["min"]:
            alerts.append({
                "type": "UNDERDOSE", "severity": "WARNING",
                "drug": drug_key,
                "message": (f"{drug_raw} {dosage_str} is below minimum effective "
                            f"dose ({limits['min']}{limits['unit']})."),
                "confidence": round(ner_confidence * 0.85, 3)
            })
    return alerts

def check_interactions(drug_list):
    alerts = []
    normalized = [normalize_drug_name(d) for d in drug_list]

    for interaction in DRUG_INTERACTIONS:
        matched = []
        for inter_drug in interaction["drugs"]:
            for norm_drug in normalized:
                if simple_similarity(norm_drug, inter_drug) >= 80:
                    matched.append(inter_drug)
                    break
        if len(matched) == len(interaction["drugs"]):
            alerts.append({
                "type": "INTERACTION",
                "severity": interaction["severity"],
                "drugs": interaction["drugs"],
                "message": interaction["message"],
                "confidence": 0.96
            })
    return alerts

# ── Main analyzer class ───────────────────────────────────────
class PrescriptionAnalyzer:
    def __init__(self):
        print("Prescription Analyzer ready.")
        print(f"Drug database   : {len(DRUG_DOSAGE_LIMITS)} drugs")
        print(f"Interaction rules: {len(DRUG_INTERACTIONS)} pairs\n")

    def analyse(self, entities, ner_confidence=0.9):
        """Full pipeline — takes NER output dict, returns alerts + summary."""
        alerts = []

        # step 1: missing info
        alerts += check_missing_info(entities)

        # step 2: dosage check per drug
        drugs   = entities.get("DRUG", [])
        dosages = entities.get("DOSAGE", [])

        for i, drug in enumerate(drugs):
            # only use the corresponding dosage, not all dosages
            if i < len(dosages):
                drug_dosages = [dosages[i]]
            else:
                drug_dosages = []
            alerts += check_dosage(drug, drug_dosages, ner_confidence)

        # step 3: drug interactions
        alerts += check_interactions(drugs)

        # step 4: build summary
        critical = [a for a in alerts if a["severity"] == "CRITICAL"]
        warnings = [a for a in alerts if a["severity"] == "WARNING"]

        return {
            "alerts": alerts,
            "drugs_normalized": [normalize_drug_name(d) for d in drugs],
            "summary": {
                "total_alerts": len(alerts),
                "critical":     len(critical),
                "warnings":     len(warnings),
                "safe_count":   max(len(drugs) - len(critical), 0),
                "risk_level":   "HIGH" if critical else "MEDIUM" if warnings else "LOW"
            }
        }


if __name__ == "__main__":
    analyzer = PrescriptionAnalyzer()

    tests = [
        {
            "label": "Dangerous combination: Warfarin + Aspirin",
            "entities": {
                "DRUG":      ["Warfarin", "Aspirin"],
                "DOSAGE":    ["5mg", "325mg"],
                "FREQUENCY": ["at night", "once daily"],
                "DURATION":  []
            }
        },
        {
            "label": "Overdose: Metformin 3000mg",
            "entities": {
                "DRUG":      ["Tab Metformin"],
                "DOSAGE":    ["3000mg"],
                "FREQUENCY": ["twice daily"],
                "DURATION":  []
            }
        },
        {
            "label": "Safe prescription: Amoxicillin 250mg",
            "entities": {
                "DRUG":      ["Amoxicillin"],
                "DOSAGE":    ["250mg"],
                "FREQUENCY": ["thrice daily"],
                "DURATION":  ["7 days"]
            }
        },
        {
            "label": "Fuzzy match test: Tab Metformin + Warfrin (typo)",
            "entities": {
                "DRUG":      ["Tab Metformin", "Warfrin"],
                "DOSAGE":    ["500mg", "5mg"],
                "FREQUENCY": ["twice daily"],
                "DURATION":  []
            }
        },
    ]

    for test in tests:
        print("=" * 60)
        print(f"TEST: {test['label']}")
        print("=" * 60)
        result = analyzer.analyse(test["entities"])
        print(f"Normalized drugs: {result['drugs_normalized']}")
        if not result["alerts"]:
            print("No issues found — prescription is safe.")
        for alert in result["alerts"]:
            print(f"[{alert['severity']}] {alert['message']} (conf: {alert.get('confidence', '-')})")
        print(f"Risk level: {result['summary']['risk_level']}\n")