"""Unit tests for validation_rules.py.

NOTE: the official sample is perfectly clean, so these tests build tiny, clearly artificial
frames purely to prove that each rule detects the defect it claims to detect. These frames are
test fixtures only; they are never used by the app, notebook or reports.
"""
import numpy as np
import pandas as pd

import config
import validation_rules as vr


def _validate(df):
    """Validate with the expected-schema check disabled (fixtures only contain a few columns)."""
    saved = config.EXPECTED_COLUMNS
    config.EXPECTED_COLUMNS = []
    try:
        return vr.validate(df)
    finally:
        config.EXPECTED_COLUMNS = saved


def _base(n=10):
    return pd.DataFrame({
        "PatientID": [f"P-{i}" for i in range(n)],
        "Age": [30 + i for i in range(n)],
        "Gender": ["Male", "Female"] * (n // 2),
        "BloodPressure": ["120/80"] * n,
        "CholesterolLevel": [180 + i for i in range(n)],
    })


def _cats(out):
    return set(out.exceptions["issue_category"])


def test_clean_frame_has_no_issues_above_informational():
    out = _validate(_base())
    scores = vr.compute_quality_scores(out)
    assert scores["issues_detected"] == 0
    assert scores["overall"] == 100.0


def test_empty_dataset_is_critical_structural_issue():
    out = _validate(pd.DataFrame())
    assert out.exceptions.iloc[0]["severity"] == "Critical"
    assert vr.compute_quality_scores(out)["overall"] == 0.0


def test_missing_values_detected_including_blank_strings():
    df = _base()
    df.loc[2, "Age"] = np.nan
    df.loc[3, "Gender"] = "   "
    out = _validate(df)
    miss = out.exceptions[out.exceptions["issue_category"] == "Missing value"]
    assert set(zip(miss["row_number"], miss["column_name"])) == {(3, "Age"), (4, "Gender")}
    assert vr.compute_quality_scores(out)["missing_values"] == 2


def test_duplicate_patient_id_flagged_not_assumed_error():
    df = _base()
    df.loc[5, "PatientID"] = df.loc[1, "PatientID"]
    out = _validate(df)
    dup = out.exceptions[out.exceptions["issue_category"] == "Duplicate patient ID"]
    assert sorted(dup["row_number"]) == [2, 6]
    assert "repeat encounters" in dup.iloc[0]["recommended_action"]
    assert vr.compute_quality_scores(out)["dimensions"]["Uniqueness"] == 90.0  # one excess row of 10


def test_duplicate_complete_record_detected():
    df = _base()
    df.loc[7] = df.loc[2]
    out = _validate(df)
    assert "Duplicate record" in _cats(out)
    assert {"Duplicate patient ID"} <= _cats(out)


def test_invalid_numeric_conversion_and_negative_values():
    df = _base()
    df["Age"] = df["Age"].astype(object)
    df.loc[1, "Age"] = "abc"
    df.loc[2, "CholesterolLevel"] = -5
    out = _validate(df)
    e = out.exceptions
    assert ((e["issue_category"] == "Invalid data type") & (e["column_name"] == "Age")).any()
    assert ((e["validation_rule"] == "Negative values are prohibited") & (e["column_name"] == "CholesterolLevel")).any()


def test_clinical_range_flags_value_but_preserves_it():
    df = _base()
    df.loc[4, "Age"] = 200
    before = df.copy()
    out = _validate(df)
    flagged = out.exceptions[(out.exceptions["column_name"] == "Age") & (out.exceptions["issue_category"] == "Out-of-range value")]
    assert len(flagged) == 1 and flagged.iloc[0]["original_value"] == "200"
    assert "not diagnostic" in flagged.iloc[0]["validation_rule"]
    pd.testing.assert_frame_equal(df, before)  # validation never edits data


def test_blood_pressure_format_range_and_swap():
    df = _base()
    df.loc[0, "BloodPressure"] = "abc"
    df.loc[1, "BloodPressure"] = "300/80"
    df.loc[2, "BloodPressure"] = "80/120"
    out = _validate(df)
    rules = set(out.exceptions["validation_rule"])
    assert any("systolic/diastolic" in r for r in rules)
    assert any("Systolic blood pressure" in r for r in rules)
    assert "Systolic should be greater than diastolic" in rules


def test_missing_clinical_columns_are_skipped_not_failed():
    out = _validate(_base())
    status = dict(zip(out.rules["rule"], out.rules["status"]))
    assert status["Heart rate range"] == "Skipped"
    assert status["Glucose range"] == "Skipped"


def test_category_whitespace_case_domain_and_spelling():
    df = _base()
    df.loc[0, "Gender"] = " male"
    df.loc[1, "Gender"] = "Unknwn"
    df["Diagnosis"] = ["Hypertension"] * 5 + ["Hypertention"] * 1 + ["Asthma"] * 4
    out = _validate(df)
    cats = _cats(out)
    assert "Inconsistent formatting" in cats      # whitespace / capitalisation
    assert "Invalid category" in cats             # 'Unknwn' outside gender domain
    assert "Inconsistent spelling" in cats        # Hypertention
    sp = out.exceptions[out.exceptions["issue_category"] == "Inconsistent spelling"]
    assert sp.iloc[0]["original_value"] == "Hypertention"


def test_digit_only_differences_are_not_spelling_errors():
    df = _base()
    df["Diagnosis"] = ["Type 1 Diabetes"] * 5 + ["Type 2 Diabetes"] * 5
    assert "Inconsistent spelling" not in _cats(_validate(df))


def test_exception_log_schema_and_severity_values():
    df = _base(); df.loc[0, "Age"] = -1
    e = _validate(df).exceptions
    assert list(e.columns) == vr.EXCEPTION_COLUMNS
    assert set(e["severity"]) <= set(config.SEVERITY_ORDER)
    assert e["exception_id"].is_unique


def test_quality_score_is_weighted_and_bounded():
    df = _base(); df.loc[0, "Age"] = np.nan
    s = vr.compute_quality_scores(_validate(df))
    expected = sum(s["dimensions"][k] * w for k, w in config.QUALITY_WEIGHTS.items())
    assert abs(s["overall"] - round(expected, 2)) < 0.01
    assert 0 <= s["overall"] < 100
    assert s["dimensions"]["Completeness"] == 98.0   # 1 missing of 50 cells
