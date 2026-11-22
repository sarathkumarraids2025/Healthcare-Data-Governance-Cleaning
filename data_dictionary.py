"""Data dictionary generation and privacy classification.

Classification is inferred from (1) column names and (2) value patterns (e-mail / phone / SSN-like
text, all-unique text columns). Descriptions are inferred from column names because EdVyro does
not supply an official data dictionary; unknown columns are explicitly marked as undocumented.
"""
from __future__ import annotations

import re
from typing import Optional

import pandas as pd

import config


def normalize_column_name(name) -> str:
    """'PatientID' -> 'patient_id', 'Treatment Duration (Days)' -> 'treatment_duration_days'."""
    s = str(name).strip()
    s = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", s)          # fooBar -> foo_Bar
    s = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", "_", s)        # PatientID -> Patient_ID
    s = re.sub(r"[^0-9a-zA-Z]+", "_", s).strip("_").lower()
    return s or "unnamed"


COLUMN_DESCRIPTIONS = {
    "patient_id": "Unique patient record identifier (pseudonymous code, not a name).",
    "age": "Patient age in years.",
    "gender": "Patient gender category.",
    "diagnosis": "Recorded primary diagnosis / health condition.",
    "primary_medication": "Primary medication prescribed.",
    "dosage_mg": "Prescribed dose in milligrams (medication-specific).",
    "treatment_duration_days": "Length of treatment in days.",
    "readmitted": "Whether the patient was readmitted (Yes/No).",
    "blood_pressure": "Blood pressure reading recorded as 'systolic/diastolic' text.",
    "cholesterol_level": "Cholesterol measurement (assumed mg/dL; unit not documented).",
}

# (privacy_category, classification, risk, regex on normalised column name) - first match wins.
_NAME_RULES = [
    ("Direct identifier", "Potentially Identifiable", "High",
     r"(^|_)(name|first_name|last_name|ssn|phone|mobile|email|address|passport|license|aadhaar|"
     r"national_id|ip_address|account|insurance_id)($|_)"),
    ("Direct identifier (pseudonymous ID)", "Potentially Identifiable", "High",
     r"^(patient_?id|patientid|mrn|record_id|id)$"),
    ("Clinical information", "Sensitive", "High", r"(diagnosis|condition|disease|allergy|procedure)"),
    ("Clinical information", "Sensitive", "Medium",
     r"(medication|drug|dosage|dose|blood|pressure|cholesterol|glucose|bmi|heart|pulse|temperature|"
     r"treatment|readmit|outcome)|(^|_)lab($|_)"),
    ("Quasi-identifier", "Potentially Identifiable", "Medium",
     r"(^|_)(age|dob|birth|gender|sex|zip|postal|pin_code|city|district|state|race|ethnic|occupation)($|_)"),
    ("Quasi-identifier", "Potentially Identifiable", "Medium", r"(date|admission|discharge)"),
    ("Administrative information", "Internal", "Low",
     r"(facility|hospital|ward|department|billing|claim|physician|doctor)"),
    ("Non-sensitive metadata", "Public", "Low", r"(dataset_version|data_source|report_date)"),
]

_VALUE_PATTERNS = {
    "e-mail address": r"[\w.+-]+@[\w-]+\.[\w.-]+",
    "phone number": r"^\+?\d[\d\s().-]{8,}\d$",
    "SSN-like number": r"^\d{3}-\d{2}-\d{4}$",
}

_ACTIONS = {
    "Direct identifier": "Remove or tokenise before sharing; restrict access; never display in public reports.",
    "Direct identifier (pseudonymous ID)": "Keep only as a pseudonymous key; verify it is not derived from real identifiers; "
                                           "mask in displays; store the re-identification key separately.",
    "Quasi-identifier": "Generalise (e.g. age bands) or suppress rare combinations; assess k-anonymity before release.",
    "Clinical information": "Treat as sensitive health data; role-based access; do not alter values - flag for review only.",
    "Administrative information": "Internal use; apply standard access controls.",
    "Non-sensitive metadata": "Low risk; can be shared with documentation.",
    "Possible identifier (all values unique)": "Unclear purpose and all values unique; confirm with the data owner whether it can identify a person.",
}


def detect_pattern_identifiers(series: pd.Series) -> list:
    """Return names of direct-identifier value patterns found in a text column."""
    if series.dtype != object and not pd.api.types.is_string_dtype(series):
        return []
    vals = series.dropna().astype(str).str.strip()
    found = []
    for label, pattern in _VALUE_PATTERNS.items():
        if len(vals) and vals.str.contains(pattern, regex=True).mean() >= 0.5:
            found.append(label)
    return found


def classify_column(norm_name: str, series: Optional[pd.Series] = None) -> dict:
    """Return privacy_category, classification, risk and recommended action for one column."""
    if series is not None:
        patterns = detect_pattern_identifiers(series)
        if patterns:
            return dict(privacy_category="Direct identifier", classification="Potentially Identifiable",
                        risk="High", basis=f"value pattern: {', '.join(patterns)}",
                        action=_ACTIONS["Direct identifier"])
    for category, classification, risk, pattern in _NAME_RULES:
        if re.search(pattern, norm_name):
            return dict(privacy_category=category, classification=classification, risk=risk,
                        basis="column name", action=_ACTIONS.get(category, _ACTIONS["Clinical information"]))
    if series is not None and len(series.dropna()) >= 5 and series.dtype != "float64" \
            and not pd.api.types.is_numeric_dtype(series) and series.nunique() == len(series.dropna()):
        cat = "Possible identifier (all values unique)"
        return dict(privacy_category=cat, classification="Potentially Identifiable", risk="Medium",
                    basis="all values unique", action=_ACTIONS[cat])
    return dict(privacy_category="Non-sensitive metadata", classification="Internal", risk="Low",
                basis="no sensitive indicators found", action=_ACTIONS["Non-sensitive metadata"])


def describe_validation_rule(norm_name: str, series: pd.Series) -> str:
    """Human-readable validation rule applied to a column."""
    if norm_name in config.ID_COLUMN_ALIASES:
        return "Required; trimmed; duplicates flagged (not auto-removed)."
    if norm_name in config.BLOOD_PRESSURE_ALIASES:
        sb, db = config.CLINICAL_RULES["systolic_bp"], config.CLINICAL_RULES["diastolic_bp"]
        return (f"Format 'sys/dia'; systolic {sb['min']}-{sb['max']}, diastolic {db['min']}-{db['max']} mmHg; "
                "systolic must exceed diastolic (screening only).")
    hit = config.rule_for_column(norm_name)
    if hit:
        _, r = hit
        op = ">" if r.get("min_exclusive") else ">="
        return f"Numeric; {op} {r['min']} and <= {r['max']} {r['unit']} (screening range, not diagnostic)."
    if norm_name in config.CATEGORY_DOMAINS:
        return "Allowed values: " + ", ".join(config.CATEGORY_DOMAINS[norm_name]) + " (case-insensitive)."
    if pd.api.types.is_numeric_dtype(series):
        return "Numeric; negatives prohibited; IQR outlier screen."
    return "Text; trimmed; consistent capitalisation and spelling checked."


def build_data_dictionary(raw_df: pd.DataFrame, cleaned_df: Optional[pd.DataFrame] = None,
                          exceptions: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Build the data dictionary from the actual dataset (nothing hard-coded about its contents)."""
    rows = []
    n = len(raw_df)
    for pos, col in enumerate(raw_df.columns):
        s = raw_df[col]
        norm = normalize_column_name(col)
        info = classify_column(norm, s)
        miss = int(s.isna().sum())
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            miss = int((s.isna() | s.fillna("").astype(str).str.strip().eq("")).sum())
        desc = COLUMN_DESCRIPTIONS.get(norm, "Undocumented field - no official description supplied.")
        row = {
            "Column Name": col,
            "Cleaned Column Name": norm,
            "Data Type": str(s.dtype),
            "Cleaned Data Type": (str(cleaned_df[cleaned_df.columns[pos]].dtype)
                                  if cleaned_df is not None and pos < len(cleaned_df.columns) else ""),
            "Description": desc,
            "Data Classification": info["classification"],
            "Privacy Category": info["privacy_category"],
            "Classification Basis": info["basis"],
            "Missing Count": miss,
            "Missing Percentage": round(100 * miss / n, 2) if n else 0.0,
            "Unique Values": int(s.nunique(dropna=True)),
            "Validation Rule": describe_validation_rule(norm, s),
            "Risk Level": info["risk"],
            "Recommended Action": info["action"],
        }
        if exceptions is not None and len(exceptions):
            row["Exceptions Logged"] = int((exceptions["column_name"] == col).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def privacy_assessment(raw_df: pd.DataFrame) -> dict:
    """Evidence-based privacy assessment. Never claims full de-identification."""
    direct, pseudo, quasi, sensitive = [], [], [], []
    for col in raw_df.columns:
        info = classify_column(normalize_column_name(col), raw_df[col])
        cat = info["privacy_category"]
        if cat == "Direct identifier":
            direct.append(col)
        elif cat.startswith("Direct identifier (pseudonymous"):
            pseudo.append(col)
        elif cat.startswith("Quasi") or cat.startswith("Possible identifier"):
            quasi.append(col)
        elif info["classification"] == "Sensitive":
            sensitive.append(col)

    # k-anonymity over quasi-identifiers (age/gender etc.), excluding pseudonymous IDs
    qcols = [c for c in quasi if any(k in normalize_column_name(c) for k in config.QUASI_IDENTIFIER_KEYWORDS)]
    k_info = {"columns": qcols, "min_k": None, "unique_rows": None, "unique_pct": None}
    if qcols and len(raw_df):
        sizes = raw_df.groupby(qcols, dropna=False)[qcols[0]].transform("size")
        k_info.update(min_k=int(sizes.min()), unique_rows=int((sizes == 1).sum()),
                      unique_pct=round(100 * float((sizes == 1).mean()), 1))

    if direct:
        status = "Not de-identified - direct identifiers detected"
        why = f"Columns with direct-identifier characteristics were detected: {', '.join(direct)}."
    elif pseudo or quasi:
        status = "Partially de-identified - NOT verified"
        why = ("No names, contact details or government IDs were detected by name or value pattern, but the "
               "dataset still holds a patient-level identifier and/or quasi-identifiers. Without documentation "
               "of the de-identification method, full de-identification cannot be confirmed.")
    else:
        status = "Not verified"
        why = "No identifiers were detected, but absence of detection is not proof of de-identification."

    risks = []
    if pseudo:
        risks.append(f"Patient-level identifier column(s) present ({', '.join(pseudo)}); linkage to source systems could re-identify patients.")
    if k_info["min_k"] is not None and k_info["min_k"] < config.K_ANONYMITY_TARGET:
        risks.append(f"Smallest quasi-identifier group has k={k_info['min_k']} (target >= {config.K_ANONYMITY_TARGET}); "
                     f"{k_info['unique_rows']} of {len(raw_df)} records ({k_info['unique_pct']}%) are unique on {', '.join(qcols)}.")
    if sensitive:
        risks.append(f"Sensitive health information present in: {', '.join(sensitive)}.")
    if len(raw_df) < 100:
        risks.append("Very small dataset: rare combinations of attributes make individuals easier to single out.")
    risks.append("No data-use agreement, consent record or de-identification documentation accompanies the dataset.")

    safeguards = [
        "Use the dataset only for its stated educational purpose; do not attempt re-identification.",
        "Mask patient identifiers in dashboards and reports (enabled by default in this app).",
        "Generalise quasi-identifiers (e.g. 10-year age bands) before any external sharing.",
        "Apply role-based access control and encrypt files at rest and in transit.",
        "Keep any re-identification key separate from the data; log access to patient-level files.",
        "Do not commit patient-level outputs to public repositories (outputs/ is git-ignored).",
        "Document the data source, purpose, retention period and deletion plan.",
    ]
    return dict(status=status, explanation=why, direct_identifiers=direct, pseudonymous_identifiers=pseudo,
                quasi_identifiers=quasi, sensitive_columns=sensitive, k_anonymity=k_info,
                risks=risks, safeguards=safeguards)
