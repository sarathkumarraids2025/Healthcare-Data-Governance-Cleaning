"""Data loading, reproducible cleaning pipeline, quality reporting and export.

Run from the command line to execute the whole pipeline and write the output files:
    python data_processor.py                # uses dataset/healthcare_patients_sample.csv
    python data_processor.py --url          # downloads the official EdVyro URL
    python data_processor.py --csv my.csv   # any other CSV
"""
from __future__ import annotations

import argparse
import io
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import config
from data_dictionary import build_data_dictionary, normalize_column_name, privacy_assessment
from validation_rules import (ValidationEngine, compute_quality_scores, is_text_series, missing_mask)


# =============================================================================== loading
class DataLoadError(Exception):
    """Raised for any dataset loading problem. Never swallowed silently."""


def parse_csv_bytes(content: bytes, label: str = "CSV"):
    """Decode bytes with a safe encoding fallback chain and parse. Returns (df, encoding)."""
    if content is None or len(content.strip()) == 0:
        raise DataLoadError(f"{label} is empty (0 bytes of content).")
    last_err = None
    for enc in config.ENCODING_CANDIDATES:
        try:
            text = content.decode(enc)
        except UnicodeDecodeError as e:
            last_err = e
            continue
        try:
            df = pd.read_csv(io.StringIO(text))
        except pd.errors.EmptyDataError:
            raise DataLoadError(f"{label} contains no parsable data.")
        except pd.errors.ParserError as e:
            raise DataLoadError(f"{label} could not be parsed as CSV: {e}")
        if df.shape[1] == 0:
            raise DataLoadError(f"{label} has no columns.")
        return df, enc
    raise DataLoadError(f"{label} could not be decoded with {config.ENCODING_CANDIDATES}: {last_err}")


def _info(df, source_type, source_label, size, encoding, note=""):
    return dict(source_type=source_type, source_label=source_label, file_size_bytes=int(size),
                encoding=encoding, rows=int(df.shape[0]), columns=int(df.shape[1]),
                loaded_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"), note=note)


def load_from_url(url: str = config.DATASET_URL, timeout: int = config.URL_TIMEOUT_SECONDS):
    """Download the CSV from a URL. Returns (df, info). Raises DataLoadError on failure."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Healthcare-Data-Governance/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            content = resp.read()
    except Exception as e:  # network, DNS, HTTP, TLS ... all reported explicitly
        raise DataLoadError(f"Could not download dataset from {url}: {type(e).__name__}: {e}")
    df, enc = parse_csv_bytes(content, label=f"Downloaded file from {url}")
    return df, _info(df, "Official URL", url, len(content), enc)


def load_from_path(path=config.LOCAL_DATASET_PATH):
    """Load a local CSV file. Returns (df, info)."""
    p = Path(path)
    if not p.exists():
        raise DataLoadError(f"File not found: {p}")
    content = p.read_bytes()
    df, enc = parse_csv_bytes(content, label=f"File {p.name}")
    return df, _info(df, "Local file", str(p), len(content), enc)


def load_from_upload(content: bytes, filename: str = "uploaded.csv"):
    """Load a CSV uploaded by the user (bytes). Returns (df, info)."""
    df, enc = parse_csv_bytes(content, label=f"Uploaded file {filename}")
    return df, _info(df, "User upload", filename, len(content), enc)


def load_with_fallback(url: str = config.DATASET_URL, local_path=config.LOCAL_DATASET_PATH):
    """Try the official URL; on failure fall back to the local copy and SAY SO in info['note']."""
    try:
        return load_from_url(url)
    except DataLoadError as url_err:
        df, info = load_from_path(local_path)
        info["note"] = f"Official URL unavailable ({url_err}). Loaded the local copy of the dataset instead."
        return df, info


# =============================================================================== cleaning
def _unique_names(names):
    seen, out = {}, []
    for n in names:
        seen[n] = seen.get(n, 0) + 1
        out.append(n if seen[n] == 1 else f"{n}_{seen[n]}")
    return out


def clean_dataset(raw: pd.DataFrame):
    """Apply safe, documented cleaning to a protected copy of `raw`.

    Returns (cleaned_core_df, steps, changes_df). Nothing is deleted, imputed or clinically altered.
    """
    original_cols = list(raw.columns)
    df = raw.copy(deep=True).reset_index(drop=True)
    steps, changes = [], []

    def log_change(row, col, old, new, action):
        changes.append(dict(row_number=int(row) + 1, column_name=col, original_value=old,
                            new_value=new, cleaning_action=action))

    n_rows = len(df)
    steps.append(dict(step="1. Load original dataset & create protected copy", status="Completed",
                      affected=n_rows, explanation="The original DataFrame is never modified; every later step "
                      "works on a deep copy.", before=f"{n_rows} rows x {len(original_cols)} columns",
                      after="Protected working copy created"))

    # 2 normalise column names ------------------------------------------------
    new_names = _unique_names([normalize_column_name(c) for c in original_cols])
    renamed = sum(1 for a, b in zip(original_cols, new_names) if a != b)
    df.columns = new_names
    name_map = dict(zip(new_names, original_cols))
    steps.append(dict(step="2. Normalise column names", status="Completed" if renamed else "No changes needed",
                      affected=renamed, explanation="Names converted to lower snake_case so rules can match them "
                      "reliably. Original names are kept in the data dictionary and exception log.",
                      before=", ".join(map(str, original_cols)), after=", ".join(new_names)))

    # 3 trim whitespace / standardise blanks ----------------------------------
    trimmed = blanks = 0
    for c in df.columns:
        if not is_text_series(df[c]):
            continue
        s = df[c].astype("object")
        for idx in range(len(s)):
            v = s.iloc[idx]
            if isinstance(v, str):
                t = v.strip()
                if t == "":
                    log_change(idx, name_map[c], repr(v), "(missing)", "Whitespace-only text standardised to missing")
                    s.iloc[idx] = np.nan
                    blanks += 1
                elif t != v:
                    log_change(idx, name_map[c], repr(v), t, "Trimmed leading/trailing whitespace")
                    s.iloc[idx] = t
                    trimmed += 1
        df[c] = s
    steps.append(dict(step="3. Trim whitespace in text values", status="Completed" if trimmed + blanks else "No changes needed",
                      affected=trimmed + blanks, explanation="Leading/trailing spaces removed. Whitespace-only cells "
                      "become missing (no value is invented).", before=f"{trimmed} cells with extra whitespace, {blanks} blank cells",
                      after="0 cells with extra whitespace"))

    # 4 standardise categoricals (case-only differences / known domains) ------
    std = 0
    id_norm = next((a for a in config.ID_COLUMN_ALIASES if a in df.columns), None)
    for c in df.columns:
        if c == id_norm or c in config.BLOOD_PRESSURE_ALIASES or not is_text_series(df[c]):
            continue
        s = df[c]
        vals = s.dropna()
        if vals.nunique() > config.CATEGORICAL_MAX_UNIQUE or vals.empty:
            continue
        mapping = {}
        groups = {}
        for v in vals.value_counts().index:
            groups.setdefault(str(v).casefold(), []).append(v)
        dom = config.CATEGORY_DOMAINS.get(c)
        dom_map = {d.casefold(): d for d in dom} if dom else {}
        for key, variants in groups.items():
            target = dom_map.get(key, variants[0])
            for v in variants:
                if v != target:
                    mapping[v] = target
        if mapping:
            s = s.astype("object")
            for idx in range(len(s)):
                v = s.iloc[idx]
                if isinstance(v, str) and v in mapping:
                    log_change(idx, name_map[c], v, mapping[v], "Standardised capitalisation of category label")
                    s.iloc[idx] = mapping[v]
                    std += 1
            df[c] = s
    steps.append(dict(step="4. Standardise categorical labels", status="Completed" if std else "No changes needed",
                      affected=std, explanation="Only case-only variants (e.g. 'male' vs 'Male') or known-domain labels are "
                      "unified. Spelling differences are flagged, never auto-corrected.",
                      before=f"{std} cells with inconsistent capitalisation", after="Consistent labels"))

    # 5 convert data types ----------------------------------------------------
    converted, kept = [], []
    for c in df.columns:
        if c == id_norm or c in config.BLOOD_PRESSURE_ALIASES:
            continue
        if is_text_series(df[c]) and config.EXPECTED_TYPES.get(c) == "numeric":
            conv = pd.to_numeric(df[c], errors="coerce")
            bad = int((df[c].notna() & conv.isna()).sum())
            if bad == 0:
                converted.append(f"{c}: {df[c].dtype} -> {conv.dtype}")
                df[c] = conv
            else:
                kept.append(f"{c}: {bad} non-numeric value(s) -> kept as text and flagged")
    steps.append(dict(step="5. Convert data types where valid",
                      status="Completed" if converted else ("Flagged" if kept else "No changes needed"),
                      affected=len(converted), explanation="A column is converted only if EVERY non-missing value is a valid number; "
                      "otherwise it is left untouched and the bad cells are flagged.",
                      before="; ".join(kept) or "All expected-numeric columns already numeric",
                      after="; ".join(converted) or "No conversion required"))
    return df, steps, pd.DataFrame(changes, columns=["row_number", "column_name", "original_value",
                                                     "new_value", "cleaning_action"])


def add_derived_columns(df: pd.DataFrame):
    """Add systolic/diastolic numeric columns parsed from the BP text. Original BP text is preserved."""
    out = df.copy()
    bp = next((a for a in config.BLOOD_PRESSURE_ALIASES if a in out.columns), None)
    if bp is None:
        return out, dict(step="6. Derive systolic / diastolic from blood pressure", status="Skipped", affected=0,
                         explanation="No blood-pressure column in this dataset.", before="-", after="-")
    parts = out[bp].fillna("").astype(str).str.extract(config.BLOOD_PRESSURE_PATTERN)
    s_name = "systolic_bp" if "systolic_bp" not in out.columns else "derived_systolic_bp"
    d_name = "diastolic_bp" if "diastolic_bp" not in out.columns else "derived_diastolic_bp"
    out[s_name] = pd.to_numeric(parts[0], errors="coerce").astype("Int64")
    out[d_name] = pd.to_numeric(parts[1], errors="coerce").astype("Int64")
    parsed = int(out[s_name].notna().sum())
    unparsed = int((out[bp].notna() & out[s_name].isna()).sum())
    step = dict(step="6. Derive systolic / diastolic from blood pressure",
                status="Completed" if unparsed == 0 else "Flagged", affected=parsed,
                explanation="Derived NEW columns are added to make numeric validation and charting possible. The original "
                            "text column is kept unchanged.",
                before=f"1 text column ('{bp}')", after=f"{s_name}, {d_name} added; {parsed} parsed, {unparsed} unparseable (left empty)")
    return out, step


# =============================================================================== pipeline
@dataclass
class PipelineResult:
    raw: pd.DataFrame
    cleaned: pd.DataFrame
    steps: list
    changes: pd.DataFrame
    exceptions: pd.DataFrame
    rules: pd.DataFrame
    dictionary: pd.DataFrame
    scores_before: dict
    scores_after: dict
    stats: dict
    privacy: dict
    source_info: dict
    run_info: dict = field(default_factory=dict)
    anomalies: Optional[pd.DataFrame] = None
    anomaly_info: dict = field(default_factory=dict)


def _clinical_columns(cols):
    keys = {a for r in config.CLINICAL_RULES.values() for a in r["aliases"]} | set(config.BLOOD_PRESSURE_ALIASES)
    return [c for c in cols if normalize_column_name(c) in keys]


def run_pipeline(raw_df: pd.DataFrame, source_info: Optional[dict] = None, enable_anomaly: bool = False,
                 contamination: float = config.ANOMALY_CONTAMINATION) -> PipelineResult:
    """Execute the full, reproducible pipeline. `raw_df` is never modified."""
    started = datetime.now()
    raw = raw_df.copy(deep=True).reset_index(drop=True)
    source_info = source_info or dict(source_type="In-memory DataFrame", source_label="n/a", file_size_bytes=0,
                                      encoding="n/a", rows=len(raw), columns=raw.shape[1], loaded_at="", note="")

    if raw.empty or raw.shape[1] == 0:
        val = ValidationEngine(raw).run()
        scores = compute_quality_scores(val)
        empty = pd.DataFrame()
        return PipelineResult(raw, empty, [dict(step="Pipeline halted", status="Failed", affected=0,
                              explanation="Dataset is empty.", before="-", after="-")], pd.DataFrame(), val.exceptions,
                              val.rules, pd.DataFrame(), scores, scores, val.stats, privacy_assessment(raw),
                              source_info, dict(started=started.isoformat(timespec="seconds"), seconds=0.0))

    core, steps, changes = clean_dataset(raw)
    before_val = ValidationEngine(raw).run()
    after_val = ValidationEngine(core).run()
    scores_before, scores_after = compute_quality_scores(before_val), compute_quality_scores(after_val)

    exceptions = before_val.exceptions.copy()
    mod_keys = {(int(r.row_number), r.column_name): r.cleaning_action for r in changes.itertuples()}
    for i, r in exceptions.iterrows():
        key = (r["row_number"], r["column_name"])
        if pd.notna(r["row_number"]) and (int(r["row_number"]), r["column_name"]) in mod_keys and \
                r["issue_category"] in ("Inconsistent formatting", "Missing value"):
            exceptions.at[i, "cleaning_action"] = mod_keys[(int(r["row_number"]), r["column_name"])]
            exceptions.at[i, "record_modified"] = True
    exceptions["record_modified"] = exceptions["record_modified"].astype(bool)

    cleaned, bp_step = add_derived_columns(core)
    steps.append(bp_step)

    # 7-9: identification steps (detect, never delete)
    stats = before_val.stats
    s = scores_before
    steps.append(dict(step="7. Identify missing values", status="Flagged" if s["missing_values"] else "Verified",
                      affected=s["missing_values"], explanation="Null and blank cells are counted and logged. They are NOT "
                      "imputed - no means, medians or invented values.", before=f"{s['missing_values']} missing cells",
                      after=f"{s['missing_values']} missing cells (unchanged by design)"))
    dup_total = s["duplicate_id_rows"] + s["duplicate_record_rows"]
    steps.append(dict(step="8. Identify duplicates", status="Flagged" if dup_total else "Verified", affected=dup_total,
                      explanation="Duplicate patient IDs and duplicate complete records are flagged. No record is deleted "
                      "because a repeat ID may be a legitimate repeat encounter.",
                      before=f"{s['duplicate_id_rows']} rows share a patient ID; {s['duplicate_record_rows']} rows are exact duplicates",
                      after="Same rows retained and flagged"))
    steps.append(dict(step="9. Flag invalid values", status="Flagged" if s["invalid_values"] else "Verified",
                      affected=s["invalid_values"], explanation="Type, range and category violations are flagged in the exception log "
                      "using configurable data-quality thresholds (not diagnostic criteria).",
                      before=f"{s['invalid_values']} invalid cells", after="Original values retained; see exception log"))

    # 10 verify clinical values preserved
    diffs = 0
    clin = _clinical_columns(raw.columns)
    for c in clin:
        a = raw[c]
        b = core[normalize_column_name(c)]
        if is_text_series(a) and is_text_series(b):
            same = a.fillna("").astype(str).str.strip().eq(b.fillna("").astype(str).str.strip())
        else:
            same = (pd.to_numeric(a, errors="coerce").fillna(-1e18) == pd.to_numeric(b, errors="coerce").fillna(-1e18))
        diffs += int((~same).sum())
    steps.append(dict(step="10. Verify clinical values preserved", status="Verified" if diffs == 0 else "VIOLATION",
                      affected=diffs, explanation="Every clinical column in the cleaned data is compared cell-by-cell with the original.",
                      before=f"{len(clin)} clinical columns checked: {', '.join(map(str, clin)) or 'none'}",
                      after=f"{diffs} clinical cells differ from the original"))

    # add review flags to the cleaned dataset
    cleaned["dq_issue_count"] = 0
    cleaned["dq_flags"] = ""
    non_info = exceptions[exceptions["row_number"].notna() & (exceptions["severity"] != "Informational")]
    for rn, grp in non_info.groupby("row_number"):
        cleaned.loc[int(rn) - 1, "dq_issue_count"] = len(grp)
        cleaned.loc[int(rn) - 1, "dq_flags"] = "; ".join(sorted(grp["issue_category"].unique()))
    cleaned["dq_needs_review"] = cleaned["dq_issue_count"] > 0
    steps.append(dict(step="11. Add review flags", status="Completed", affected=int(cleaned["dq_needs_review"].sum()),
                      explanation="Columns dq_issue_count, dq_flags and dq_needs_review are appended so analysts can filter "
                      "records needing review. No rows are removed.", before=f"{cleaned.shape[1] - 3} columns",
                      after=f"{cleaned.shape[1]} columns; {int(cleaned['dq_needs_review'].sum())} records flagged for review"))
    steps.append(dict(step="12. Generate exception log, data dictionary and quality report", status="Completed",
                      affected=len(exceptions), explanation="Produced dynamically from the validation results.",
                      before="-", after=f"{len(exceptions)} exception rows; {len(raw.columns)} dictionary rows"))

    dictionary = build_data_dictionary(raw, core, exceptions)
    result = PipelineResult(raw=raw, cleaned=cleaned, steps=steps, changes=changes, exceptions=exceptions,
                            rules=before_val.rules, dictionary=dictionary, scores_before=scores_before,
                            scores_after=scores_after, stats=stats, privacy=privacy_assessment(raw),
                            source_info=source_info,
                            run_info=dict(started=started.isoformat(timespec="seconds"),
                                          seconds=round((datetime.now() - started).total_seconds(), 3)))
    if enable_anomaly:
        result.anomalies, result.anomaly_info = detect_anomalies(cleaned, contamination)
    return result


# =============================================================================== anomaly detection (optional)
def detect_anomalies(df: pd.DataFrame, contamination: float = config.ANOMALY_CONTAMINATION):
    """Unsupervised Isolation Forest on numeric, non-identifier columns.

    Returns (scores_df or None, info). Anomalies are statistical only - not medical findings - and
    no value is modified. No accuracy is reported because there are no ground-truth labels.
    """
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    exclude = {"dq_issue_count", "dq_needs_review"}
    feats = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) and not pd.api.types.is_bool_dtype(df[c])
             and c not in exclude and normalize_column_name(c) not in config.ID_COLUMN_ALIASES]
    X = df[feats].apply(pd.to_numeric, errors="coerce").astype(float) if feats else pd.DataFrame()
    dropped = [c for c in feats if X[c].isna().all() or X[c].nunique(dropna=True) < 2]
    feats = [c for c in feats if c not in dropped]
    X = X[feats]
    rows_ok = X.dropna().index if feats else []
    info = dict(features=feats, dropped_constant_or_empty=dropped, contamination=contamination,
                n_rows_used=len(rows_ok), model="IsolationForest(n_estimators=200)",
                limitations=["Anomalies are statistical outliers, not necessarily medically abnormal or erroneous.",
                             "No ground-truth labels exist, so no accuracy/precision/recall can be reported.",
                             "With very few rows the scores are unstable; treat them as a review aid only."])
    if len(feats) < 2 or len(rows_ok) < config.MIN_ANOMALY_ROWS:
        info["status"] = (f"Skipped: needs >= 2 usable numeric features and >= {config.MIN_ANOMALY_ROWS} complete rows "
                          f"(found {len(feats)} features, {len(rows_ok)} rows).")
        return None, info
    Xs = StandardScaler().fit_transform(X.loc[rows_ok])
    model = IsolationForest(n_estimators=200, contamination=contamination, random_state=config.ANOMALY_RANDOM_STATE)
    model.fit(Xs)
    out = X.loc[rows_ok].copy()
    out.insert(0, "row_number", np.asarray(rows_ok) + 1)
    out["anomaly_score"] = model.decision_function(Xs).round(4)      # lower = more anomalous
    out["is_anomaly"] = model.predict(Xs) == -1
    info["status"] = "Completed"
    info["n_flagged"] = int(out["is_anomaly"].sum())
    return out, info


# =============================================================================== reports / export
def to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def build_quality_report_df(r: PipelineResult) -> pd.DataFrame:
    rows = []
    add = lambda sec, m, v, note="": rows.append(dict(section=sec, metric=m, value=v, notes=note))
    si = r.source_info
    add("Dataset", "Source type", si.get("source_type", ""), si.get("source_label", ""))
    add("Dataset", "Rows", r.raw.shape[0]); add("Dataset", "Columns", r.raw.shape[1])
    add("Dataset", "File size (bytes)", si.get("file_size_bytes", ""))
    sb, sa = r.scores_before, r.scores_after
    add("Score", "Overall quality score (original data)", sb["overall"], "0-100; formula in report")
    add("Score", "Overall quality score (after cleaning)", sa["overall"], "Re-validated cleaned data")
    for k, v in sb["dimensions"].items():
        add("Dimension (original)", k, v); add("Dimension (cleaned)", k, sa["dimensions"][k])
    for k in ("total_exceptions", "issues_detected", "records_requiring_review", "validated_records",
              "missing_values", "duplicate_id_rows", "duplicate_record_rows", "invalid_values"):
        add("Counts", k, sb[k])
    for sev, n in sb["severity_counts"].items():
        add("Severity", sev, n)
    for st_name, n in r.rules["status"].value_counts().items():
        add("Rule status", st_name, int(n))
    add("Privacy", "De-identification status", r.privacy["status"])
    km = r.privacy["k_anonymity"]
    add("Privacy", "Minimum k-anonymity (quasi-identifiers)", km["min_k"] if km["min_k"] is not None else "n/a",
        ", ".join(km["columns"]))
    add("Cleaning", "Cells modified by cleaning", len(r.changes))
    return pd.DataFrame(rows)


def _md_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "_None_\n"
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "---|" * len(df.columns) + "\n"
    body = "".join("| " + " | ".join(str(v).replace("|", "/").replace("\n", " ") for v in row) + " |\n"
                   for row in df.itertuples(index=False))
    return head + body


def build_report_markdown(r: PipelineResult) -> str:
    si, sb, sa = r.source_info, r.scores_before, r.scores_after
    dims = pd.DataFrame({"Dimension": list(sb["dimensions"]), "Original data": list(sb["dimensions"].values()),
                         "After cleaning": [sa["dimensions"][k] for k in sb["dimensions"]]})
    sev = pd.DataFrame({"Severity": list(sb["severity_counts"]), "Exceptions": list(sb["severity_counts"].values())})
    rules = r.rules[["rule_id", "rule", "status", "records_affected", "severity"]]
    steps = pd.DataFrame(r.steps)[["step", "status", "affected"]]
    p = r.privacy
    exc = r.exceptions[["exception_id", "record_identifier", "column_name", "issue_category", "severity", "original_value"]]
    md = [f"# Data Quality Report\n\n_Generated {datetime.now():%Y-%m-%d %H:%M:%S} by Healthcare Data Governance & Quality Analytics._\n",
          f"> {config.DISCLAIMER}\n",
          "## 1. Data source\n",
          f"- Source type: **{si.get('source_type')}**\n- Location: `{si.get('source_label')}`\n"
          f"- File size: {si.get('file_size_bytes')} bytes; encoding: {si.get('encoding')}\n"
          f"- Rows x columns: **{r.raw.shape[0]} x {r.raw.shape[1]}**\n"
          + (f"- Note: {si['note']}\n" if si.get("note") else ""),
          "## 2. Quality score\n",
          f"**Overall score (original data): {sb['overall']} / 100** - after cleaning: {sa['overall']} / 100\n\n"
          + _md_table(dims) + "\n```text\n" + config.SCORE_FORMULA_TEXT + "\n```\n",
          "## 3. Summary counts\n",
          f"- Exceptions logged: {sb['total_exceptions']} (of which {sb['issues_detected']} are above Informational)\n"
          f"- Records requiring review: {sb['records_requiring_review']} / {sb['n_rows']}\n"
          f"- Validated records: {sb['validated_records']}\n- Missing cells: {sb['missing_values']}\n"
          f"- Rows sharing a patient ID: {sb['duplicate_id_rows']}; duplicate-record exceptions: {sb['duplicate_record_rows']}\n"
          f"- Invalid cells: {sb['invalid_values']}\n\n" + _md_table(sev),
          "## 4. Validation rules\n", _md_table(rules),
          "## 5. Exceptions\n", _md_table(exc),
          "## 6. Cleaning pipeline\n", _md_table(steps),
          f"Cells modified by cleaning: **{len(r.changes)}**. Clinical values preserved: "
          f"**{next((s['after'] for s in r.steps if s['step'].startswith('10.')), 'n/a')}**.\n",
          "## 7. Privacy assessment\n",
          f"- Status: **{p['status']}**\n- Reasoning: {p['explanation']}\n"
          f"- Pseudonymous identifier columns: {', '.join(p['pseudonymous_identifiers']) or 'none'}\n"
          f"- Quasi-identifiers: {', '.join(p['quasi_identifiers']) or 'none'}\n"
          f"- Sensitive columns: {', '.join(p['sensitive_columns']) or 'none'}\n"
          f"- k-anonymity on {', '.join(p['k_anonymity']['columns']) or 'n/a'}: min k = {p['k_anonymity']['min_k']}\n\n"
          "Remaining risks:\n\n" + "".join(f"- {x}\n" for x in p["risks"]) +
          "\nRecommended safeguards:\n\n" + "".join(f"- {x}\n" for x in p["safeguards"]),
          "## 8. Limitations\n",
          "- Column descriptions are inferred from column names (no official data dictionary supplied).\n"
          "- Clinical thresholds are plausibility screens, not diagnostic criteria.\n"
          "- Small samples make IQR outlier flags and k-anonymity estimates unstable.\n"
          "- Units (e.g. mg/dL) are assumed, not documented.\n"]
    if r.anomaly_info:
        md.append("## 9. Anomaly detection (optional)\n\n" + f"Status: {r.anomaly_info.get('status')}\n\n"
                  + "".join(f"- {x}\n" for x in r.anomaly_info["limitations"]))
    return "\n".join(md)


def export_outputs(r: PipelineResult, output_dir=config.OUTPUT_DIR, report_dir=config.REPORT_DIR) -> dict:
    """Write all deliverables to disk and return {name: path}."""
    out, rep = Path(output_dir), Path(report_dir)
    out.mkdir(parents=True, exist_ok=True); rep.mkdir(parents=True, exist_ok=True)
    paths = {"cleaned_healthcare_data.csv": out / "cleaned_healthcare_data.csv",
             "data_dictionary.csv": out / "data_dictionary.csv",
             "exception_log.csv": out / "exception_log.csv",
             "data_quality_report.csv": out / "data_quality_report.csv",
             "data_quality_report.md": rep / "data_quality_report.md"}
    r.cleaned.to_csv(paths["cleaned_healthcare_data.csv"], index=False)
    r.dictionary.to_csv(paths["data_dictionary.csv"], index=False)
    r.exceptions.to_csv(paths["exception_log.csv"], index=False)
    build_quality_report_df(r).to_csv(paths["data_quality_report.csv"], index=False)
    paths["data_quality_report.md"].write_text(build_report_markdown(r), encoding="utf-8")
    return paths


def main():
    ap = argparse.ArgumentParser(description="Run the healthcare data governance pipeline.")
    ap.add_argument("--url", action="store_true", help="download the official EdVyro dataset")
    ap.add_argument("--csv", help="path to a CSV file")
    ap.add_argument("--anomaly", action="store_true", help="also run optional Isolation Forest")
    a = ap.parse_args()
    try:
        df, info = load_from_url() if a.url else load_from_path(a.csv or config.LOCAL_DATASET_PATH)
    except DataLoadError as e:
        raise SystemExit(f"ERROR: {e}")
    res = run_pipeline(df, info, enable_anomaly=a.anomaly)
    paths = export_outputs(res)
    print(f"Source: {info['source_type']} ({info['source_label']}), {info['rows']} rows x {info['columns']} cols")
    print(f"Overall quality score: {res.scores_before['overall']} (after cleaning {res.scores_after['overall']})")
    print(f"Exceptions: {res.scores_before['total_exceptions']}; cells modified: {len(res.changes)}")
    for name, p in paths.items():
        print("wrote", p)


if __name__ == "__main__":
    main()
