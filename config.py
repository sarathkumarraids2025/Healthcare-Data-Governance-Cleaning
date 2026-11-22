"""Central configuration for the Healthcare Data Governance & Quality Analytics project.

All validation thresholds live here. IMPORTANT: clinical thresholds are *data-quality
screening ranges* (plausibility checks for data-entry errors / unit problems). They are NOT
diagnostic criteria and must never be used to label a patient as healthy or unwell.
"""
from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------- paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_URL = "https://edvyro.in/datasets/healthcare_patients_sample.csv"
LOCAL_DATASET_PATH = BASE_DIR / "dataset" / "healthcare_patients_sample.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
REPORT_DIR = BASE_DIR / "reports"

# --------------------------------------------------------------------------- loading
URL_TIMEOUT_SECONDS = 20
ENCODING_CANDIDATES = ("utf-8-sig", "utf-8", "cp1252", "latin-1")

# --------------------------------------------------------------------------- schema
ID_COLUMN_ALIASES = ("patient_id", "patientid", "mrn", "record_id", "id")

# Columns the official EdVyro sample is expected to contain (normalised names).
# Extra columns are reported as Informational; missing ones as Medium.
EXPECTED_COLUMNS = [
    "patient_id", "age", "gender", "diagnosis", "primary_medication", "dosage_mg",
    "treatment_duration_days", "readmitted", "blood_pressure", "cholesterol_level",
]

# Expected logical type per column: "numeric" or "text".
EXPECTED_TYPES = {
    "patient_id": "text", "age": "numeric", "gender": "text", "diagnosis": "text",
    "primary_medication": "text", "dosage_mg": "numeric",
    "treatment_duration_days": "numeric", "readmitted": "text",
    "blood_pressure": "text", "cholesterol_level": "numeric",
}

BLOOD_PRESSURE_ALIASES = ("blood_pressure", "bp", "bp_reading")
BLOOD_PRESSURE_PATTERN = r"^\s*(\d{2,3})\s*/\s*(\d{2,3})\s*$"

# --------------------------------------------------------------------------- range rules
# Each rule: aliases (normalised column names), min, max, unit, severity and explanation.
# If no alias matches a dataset column the rule is skipped automatically.
# "min_exclusive": True means the value must be strictly greater than min.
CLINICAL_RULES = {
    "age": dict(label="Age", aliases=("age", "patient_age", "age_years"), min=0, max=120,
                unit="years", severity="High",
                explanation="Age outside a plausible human lifespan usually indicates a data-entry error."),
    "systolic_bp": dict(label="Systolic blood pressure",
                        aliases=("systolic", "systolic_bp", "systolic_blood_pressure", "bp_systolic"),
                        min=70, max=250, unit="mmHg", severity="High",
                        explanation="Screening range for implausible readings (entry/unit errors), not a hypertension criterion."),
    "diastolic_bp": dict(label="Diastolic blood pressure",
                         aliases=("diastolic", "diastolic_bp", "diastolic_blood_pressure", "bp_diastolic"),
                         min=40, max=150, unit="mmHg", severity="High",
                         explanation="Screening range for implausible readings, not a diagnostic criterion."),
    "heart_rate": dict(label="Heart rate", aliases=("heart_rate", "pulse", "pulse_rate", "hr"),
                       min=30, max=220, unit="bpm", severity="High",
                       explanation="Screening range for implausible readings only."),
    "bmi": dict(label="BMI", aliases=("bmi", "body_mass_index"), min=10, max=70, unit="kg/m2",
                severity="High", explanation="Screening range for implausible BMI values only."),
    "glucose": dict(label="Glucose",
                    aliases=("glucose", "blood_glucose", "glucose_level", "fasting_glucose"),
                    min=20, max=600, unit="mg/dL", severity="High",
                    explanation="Screening range for implausible readings only (check units: mg/dL vs mmol/L)."),
    "temperature": dict(label="Body temperature",
                        aliases=("temperature", "body_temperature", "temp", "temperature_c"),
                        min=30, max=45, unit="deg C", severity="High",
                        explanation="Screening range in Celsius; Fahrenheit values would be flagged."),
    "cholesterol": dict(label="Cholesterol",
                        aliases=("cholesterol", "cholesterol_level", "total_cholesterol"),
                        min=50, max=500, unit="mg/dL", severity="High",
                        explanation="Screening range for implausible readings, not a risk classification."),
    "dosage": dict(label="Medication dosage", aliases=("dosage_mg", "dosage", "dose_mg", "dose"),
                   min=0, max=10000, unit="mg", severity="High", min_exclusive=True,
                   explanation="Dose must be positive and below a generous ceiling. Appropriate dose is "
                               "medication-specific and is NOT assessed here."),
    "treatment_duration": dict(label="Treatment duration",
                               aliases=("treatment_duration_days", "treatment_days", "duration_days"),
                               min=1, max=3650, unit="days", severity="Medium",
                               explanation="Duration must be at least 1 day and at most ~10 years."),
}

# Statistical outlier screen (IQR). Dosage is excluded because it is pooled across different
# medications whose doses are not comparable; outliers there would be meaningless.
OUTLIER_IQR_MULTIPLIER = 1.5
OUTLIER_MIN_ROWS = 8
OUTLIER_EXCLUDED_COLUMNS = {"dosage_mg", "dosage", "dose", "dose_mg"}

# --------------------------------------------------------------------------- categorical
CATEGORICAL_MAX_UNIQUE = 20
SPELLING_SIMILARITY_THRESHOLD = 0.90
CATEGORY_DOMAINS = {
    "gender": ["Male", "Female", "Other", "Unknown"],
    "sex": ["Male", "Female", "Other", "Unknown"],
    "readmitted": ["Yes", "No"],
}

# --------------------------------------------------------------------------- privacy
QUASI_IDENTIFIER_KEYWORDS = ("age", "dob", "birth", "gender", "sex", "zip", "postal", "pin_code",
                             "city", "district", "state", "race", "ethnic", "occupation")
K_ANONYMITY_TARGET = 5

# --------------------------------------------------------------------------- scoring
SEVERITY_ORDER = ["Critical", "High", "Medium", "Low", "Informational"]
QUALITY_WEIGHTS = {"Completeness": 0.25, "Validity": 0.30, "Uniqueness": 0.15,
                   "Consistency": 0.15, "Structural Integrity": 0.15}

SCORE_FORMULA_TEXT = (
    "Overall = 0.25 x Completeness + 0.30 x Validity + 0.15 x Uniqueness + 0.15 x Consistency "
    "+ 0.15 x Structural Integrity (each dimension is 0-100).\n"
    "Completeness = 100 x (1 - missing cells / total cells).\n"
    "Validity = 100 x (1 - distinct invalid cells / total cells). Invalid = wrong data type, "
    "negative where prohibited, outside screening range, or category outside a known domain "
    "(Informational findings are excluded).\n"
    "Uniqueness = 100 x (1 - rows that repeat an earlier patient ID or an earlier complete record / total rows).\n"
    "Consistency = 100 x (1 - distinct text cells with whitespace/capitalisation/spelling issues / total text cells).\n"
    "Structural Integrity = 100 x (structural checks passed / structural checks applicable)."
)

# --------------------------------------------------------------------------- anomaly detection
MIN_ANOMALY_ROWS = 8
ANOMALY_CONTAMINATION = 0.1
ANOMALY_RANDOM_STATE = 42

DISCLAIMER = ("This application is designed for educational data-quality analysis. It does not provide "
              "medical diagnosis or clinical decision-making. Clinical values are flagged for "
              "data-quality review only.")


def rule_for_column(normalized_name: str):
    """Return (rule_key, rule_dict) whose aliases contain the normalised column name, else None."""
    for key, rule in CLINICAL_RULES.items():
        if normalized_name in rule["aliases"]:
            return key, rule
    return None
