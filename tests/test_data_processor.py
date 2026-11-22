"""Tests for data_processor.py: loading, cleaning pipeline, dictionary, export."""
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

import config
import data_processor as dp
from data_dictionary import build_data_dictionary, normalize_column_name, privacy_assessment


def _dirty():
    return pd.DataFrame({
        "PatientID": ["A1", "A2", "A3", "A3", "A5", "A6", "A7", "A8", "A9", "A10"],
        "Age": [30, 41, 52, 52, 200, 34, 45, 56, 67, 78],
        "Gender": ["male", "Male ", "Female", "Female", "Male", "Female", "male", "Female", "Male", "Female"],
        "BloodPressure": ["120/80"] * 10,
        "CholesterolLevel": [180, 190, 200, 200, 210, 220, np.nan, 240, 250, 260],
    })


def test_normalize_column_name():
    assert normalize_column_name("PatientID") == "patient_id"
    assert normalize_column_name("TreatmentDurationDays") == "treatment_duration_days"
    assert normalize_column_name(" Treatment Duration (Days) ") == "treatment_duration_days"


def test_official_dataset_file_loads():
    df, info = dp.load_from_path(config.LOCAL_DATASET_PATH)
    assert df.shape[0] > 0 and info["source_type"] == "Local file"
    assert info["file_size_bytes"] == config.LOCAL_DATASET_PATH.stat().st_size


def test_empty_and_malformed_inputs_raise_explicit_errors():
    for bad in (b"", b"   \n  "):
        try:
            dp.load_from_upload(bad, "x.csv")
        except dp.DataLoadError as e:
            assert "empty" in str(e).lower()
        else:
            raise AssertionError("empty file should raise DataLoadError")
    try:
        dp.load_from_path("does_not_exist.csv")
    except dp.DataLoadError:
        pass
    else:
        raise AssertionError("missing file should raise")


def test_upload_encoding_fallback():
    df, info = dp.load_from_upload("Name,City\nJos\u00e9,Z\u00fcrich\n".encode("cp1252"), "latin.csv")
    assert df.iloc[0, 0] == "Jos\u00e9" and info["source_type"] == "User upload"


def test_pipeline_never_modifies_original_and_preserves_rows():
    raw = _dirty()
    snapshot = raw.copy(deep=True)
    res = dp.run_pipeline(raw)
    pd.testing.assert_frame_equal(raw, snapshot)           # original untouched
    assert len(res.cleaned) == len(raw)                    # no rows deleted
    assert len(res.raw) == len(raw)


def test_cleaning_standardises_text_but_not_clinical_values():
    res = dp.run_pipeline(_dirty())
    assert set(res.cleaned["gender"]) == {"Male", "Female"}
    assert res.cleaned.loc[4, "age"] == 200                # invalid clinical value kept, only flagged
    assert pd.isna(res.cleaned.loc[6, "cholesterol_level"])  # missing NOT imputed
    step10 = next(s for s in res.steps if s["step"].startswith("10."))
    assert step10["status"] == "Verified" and step10["affected"] == 0


def test_derived_blood_pressure_columns_keep_original_text():
    res = dp.run_pipeline(_dirty())
    assert (res.cleaned["blood_pressure"] == "120/80").all()
    assert (res.cleaned["systolic_bp"] == 120).all() and (res.cleaned["diastolic_bp"] == 80).all()


def test_exception_log_records_modifications_and_review_flags():
    res = dp.run_pipeline(_dirty())
    e = res.exceptions
    assert {"Duplicate patient ID", "Out-of-range value", "Missing value", "Inconsistent formatting"} <= set(e["issue_category"])
    fmt = e[e["issue_category"] == "Inconsistent formatting"]
    assert fmt["record_modified"].all()                    # whitespace/case fixes are marked as modified
    oor = e[e["issue_category"] == "Out-of-range value"]
    assert not oor["record_modified"].any()                # clinical flags are never modified
    assert res.cleaned["dq_needs_review"].sum() == res.scores_before["records_requiring_review"]


def test_after_cleaning_score_not_worse_and_before_score_is_honest():
    res = dp.run_pipeline(_dirty())
    assert res.scores_before["overall"] < 100              # defects exist, score must reflect them
    assert res.scores_after["dimensions"]["Consistency"] >= res.scores_before["dimensions"]["Consistency"]
    assert res.scores_after["dimensions"]["Validity"] == res.scores_before["dimensions"]["Validity"]


def test_data_dictionary_is_computed_from_data():
    raw = _dirty()
    d = build_data_dictionary(raw)
    assert list(d["Column Name"]) == list(raw.columns)
    row = d[d["Column Name"] == "CholesterolLevel"].iloc[0]
    assert row["Missing Count"] == 1 and row["Missing Percentage"] == 10.0
    assert d[d["Column Name"] == "PatientID"].iloc[0]["Data Classification"] == "Potentially Identifiable"
    assert set(d["Risk Level"]) <= {"Low", "Medium", "High"}


def test_privacy_assessment_never_claims_full_deidentification():
    p = privacy_assessment(_dirty())
    assert "NOT verified" in p["status"] or "Not" in p["status"]
    assert p["k_anonymity"]["min_k"] is not None
    emails = pd.DataFrame({"email": ["a@b.com", "c@d.org", "e@f.net", "g@h.io", "i@j.com"]})
    assert "Not de-identified" in privacy_assessment(emails)["status"]


def test_export_writes_real_files_matching_results():
    res = dp.run_pipeline(_dirty())
    with tempfile.TemporaryDirectory() as tmp:
        paths = dp.export_outputs(res, Path(tmp) / "outputs", Path(tmp) / "reports")
        assert all(p.exists() and p.stat().st_size > 0 for p in paths.values())
        log = pd.read_csv(paths["exception_log.csv"])
        assert len(log) == len(res.exceptions)
        cleaned = pd.read_csv(paths["cleaned_healthcare_data.csv"])
        assert cleaned.shape == res.cleaned.shape
        assert "Overall score" in paths["data_quality_report.md"].read_text(encoding="utf-8")


def test_empty_dataframe_pipeline_does_not_crash():
    res = dp.run_pipeline(pd.DataFrame())
    assert res.scores_before["overall"] == 0.0 and res.steps[0]["status"] == "Failed"


def test_anomaly_detection_graceful_with_insufficient_data_and_excludes_ids():
    few = dp.run_pipeline(_dirty().head(4), enable_anomaly=True)
    assert few.anomalies is None and few.anomaly_info["status"].startswith("Skipped")
    res = dp.run_pipeline(_dirty(), enable_anomaly=True)
    assert "patient_id" not in res.anomaly_info["features"]
    assert res.anomalies is not None and "anomaly_score" in res.anomalies
