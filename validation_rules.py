"""Reusable data validation engine.

Validation only *detects and documents*. It never edits values. Every finding becomes a row in the
exception log and every check becomes a row in the rule-results table.
"""
from __future__ import annotations

import difflib
import re
from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

import config
from data_dictionary import classify_column, detect_pattern_identifiers, normalize_column_name

EXCEPTION_COLUMNS = ["exception_id", "row_number", "record_identifier", "column_name", "original_value",
                     "issue_category", "validation_rule", "severity", "recommended_action",
                     "cleaning_action", "record_modified", "dimension"]
RULE_COLUMNS = ["rule_id", "rule", "category", "description", "status", "records_affected",
                "severity", "dimension"]


@dataclass
class ValidationOutput:
    exceptions: pd.DataFrame
    rules: pd.DataFrame
    stats: dict


def missing_mask(df: pd.DataFrame) -> pd.DataFrame:
    """True where a cell is null OR a blank / whitespace-only string."""
    mask = df.isna()
    for c in df.columns:
        s = df[c]
        if s.dtype == object or pd.api.types.is_string_dtype(s):
            mask[c] = mask[c] | s.fillna("").astype(str).str.strip().eq("")
    return mask


def is_text_series(s: pd.Series) -> bool:
    return s.dtype == object or pd.api.types.is_string_dtype(s)


class ValidationEngine:
    """Run all validation families on a DataFrame and return exceptions, rule results and stats."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)
        self.n_rows, self.n_cols = self.df.shape
        self.norm = {c: normalize_column_name(c) for c in self.df.columns}
        self.by_norm = {}
        for c, n in self.norm.items():
            self.by_norm.setdefault(n, c)
        self.id_col = next((self.by_norm[a] for a in config.ID_COLUMN_ALIASES if a in self.by_norm), None)
        self.bp_col = next((self.by_norm[a] for a in config.BLOOD_PRESSURE_ALIASES if a in self.by_norm), None)
        self.mask = missing_mask(self.df) if self.n_rows and self.n_cols else pd.DataFrame()
        self._exc, self._rules = [], []
        self.excess_dup_rows = set()
        self.numeric_cols, self.text_cols, self.categorical_cols = [], [], []

    # ------------------------------------------------------------------ helpers
    def _rec(self, idx):
        if idx is None:
            return "(dataset-level)"
        if self.id_col is not None and not self.mask[self.id_col].iloc[idx]:
            return str(self.df[self.id_col].iloc[idx]).strip()
        return f"Row {idx + 1}"

    @staticmethod
    def _fmt(v):
        return "(missing)" if v is None or (not isinstance(v, str) and pd.isna(v)) else str(v)

    def _add(self, idx, col, category, rule, severity, action, dimension, value=None, use_cell=True):
        if use_cell and idx is not None and col is not None and value is None:
            value = self.df[col].iloc[idx]
        self._exc.append(dict(
            row_number=None if idx is None else idx + 1, record_identifier=self._rec(idx),
            column_name=col if col is not None else "(all columns)", original_value=self._fmt(value),
            issue_category=category, validation_rule=rule, severity=severity,
            recommended_action=action, cleaning_action="None - flagged only; original value preserved",
            record_modified=False, dimension=dimension))

    def _rule(self, name, category, description, affected, severity, dimension, applicable=True):
        if not applicable:
            status = "Skipped"
        elif affected == 0:
            status = "Passed"
        else:
            status = "Failed" if severity in ("Critical", "High", "Medium") else "Warning"
        self._rules.append(dict(rule_id=f"R{len(self._rules) + 1:02d}", rule=name, category=category,
                                description=description, status=status,
                                records_affected=int(affected) if applicable else 0,
                                severity=severity, dimension=dimension))

    # ------------------------------------------------------------------ run
    def run(self) -> ValidationOutput:
        empty = self._structural()
        if not empty:
            self._missing()
            self._duplicates()
            self._numeric()
            self._clinical()
            self._categorical()
            self._privacy()
        return self._finish()

    # ------------------------------------------------------------------ structural
    def _structural(self) -> bool:
        S = "Structural Integrity"
        empty = self.n_rows == 0 or self.n_cols == 0
        if empty:
            self._add(None, None, "Structural issue", "Dataset must contain rows and columns", "Critical",
                      "Provide a non-empty CSV file.", S)
        self._rule("Dataset not empty", "Structural issue", "Dataset has at least one row and one column.",
                   1 if empty else 0, "Critical", S)
        if empty:
            return True

        raw_cols = list(self.df.columns)
        unnamed = [c for c in raw_cols if str(c).strip() == "" or str(c).startswith("Unnamed:")]
        for c in unnamed:
            self._add(None, c, "Structural issue", "Column names must not be missing", "High",
                      "Provide a header name for this column.", S, value=c, use_cell=False)
        self._rule("No missing column names", "Structural issue", "Every column has a header.",
                   len(unnamed), "High", S)

        counts = Counter(self.norm.values())
        dup_names = [c for c in raw_cols if counts[self.norm[c]] > 1 or
                     (re.search(r"\.\d+$", str(c)) and re.sub(r"\.\d+$", "", str(c)) in map(str, raw_cols))]
        for c in dup_names:
            self._add(None, c, "Structural issue", "Column names must be unique", "High",
                      "Rename or merge duplicate columns.", S, value=c, use_cell=False)
        self._rule("No duplicate column names", "Structural issue", "Column names are unique after normalisation.",
                   len(dup_names), "High", S)

        empties = [c for c in raw_cols if self.mask[c].all()]
        for c in empties:
            self._add(None, c, "Structural issue", "Columns must not be completely empty", "High",
                      "Remove the column or obtain its data.", S, value="(entire column empty)", use_cell=False)
        self._rule("No completely empty columns", "Structural issue", "No column is entirely missing.",
                   len(empties), "High", S)

        # expected type check
        bad_type = []
        for c in raw_cols:
            exp = config.EXPECTED_TYPES.get(self.norm[c])
            if exp == "numeric" and not pd.api.types.is_numeric_dtype(self.df[c]) and not self.mask[c].all():
                bad_type.append(c)
                self._add(None, c, "Invalid data type", "Column expected to be numeric",
                          "High", "Inspect non-numeric entries listed separately; fix at source.", S,
                          value=str(self.df[c].dtype), use_cell=False)
        self._rule("Column types match expectation", "Invalid data type",
                   "Columns documented as numeric are numeric.", len(bad_type), "High", S,
                   applicable=any(config.EXPECTED_TYPES.get(n) == "numeric" for n in self.norm.values()))

        if config.EXPECTED_COLUMNS:
            unexpected = [c for c in raw_cols if self.norm[c] not in config.EXPECTED_COLUMNS]
            for c in unexpected:
                self._add(None, c, "Structural issue", "Unexpected column (not in expected schema)",
                          "Informational", "Confirm the column is intended and document it.", S,
                          value=c, use_cell=False)
            self._rule("No unexpected columns", "Structural issue", "All columns belong to the expected schema.",
                       len(unexpected), "Informational", S)
            missing_exp = [e for e in config.EXPECTED_COLUMNS if e not in self.by_norm]
            for e in missing_exp:
                self._add(None, e, "Structural issue", "Expected column is missing", "Medium",
                          "Obtain the missing column or update the schema configuration.", S,
                          value="(column absent)", use_cell=False)
            self._rule("All expected columns present", "Structural issue", "Every expected column exists.",
                       len(missing_exp), "Medium", S)
        return False

    # ------------------------------------------------------------------ missing
    def _missing(self):
        C = "Completeness"
        rows_with_missing = set()
        for c in self.df.columns:
            sev = "Critical" if c == self.id_col else ("High" if config.rule_for_column(self.norm[c]) else "Medium")
            for idx in np.flatnonzero(self.mask[c].to_numpy()):
                if self.mask[c].all():
                    continue  # whole-column emptiness is reported as structural
                rows_with_missing.add(int(idx))
                self._add(int(idx), c, "Missing value", "Value must be present", sev,
                          "Obtain the value from the source system; do NOT impute clinical values.", C)
        partial = [c for c in self.df.columns if self.mask[c].any() and not self.mask[c].all()]
        self._rule("No missing values", "Missing value", "No null or blank cells.",
                   len(rows_with_missing), "Medium", C)
        self._rule("No partially missing columns", "Missing value",
                   "Columns with some (not all) values missing.", len(partial), "Low", C)

    # ------------------------------------------------------------------ duplicates
    def _duplicates(self):
        U = "Uniqueness"
        id_dup_rows = 0
        if self.id_col is not None:
            key = self.df[self.id_col].fillna("").astype(str).str.strip().str.casefold()
            valid = ~self.mask[self.id_col]
            dup_all = key.duplicated(keep=False) & valid
            dup_extra = key.duplicated(keep="first") & valid
            self.excess_dup_rows |= set(np.flatnonzero(dup_extra.to_numpy()).tolist())
            id_dup_rows = int(dup_all.sum())
            for idx in np.flatnonzero(dup_all.to_numpy()):
                self._add(int(idx), self.id_col, "Duplicate patient ID", "Patient ID should be unique", "High",
                          "Review: repeat encounters can legitimately share an ID; otherwise a data-entry "
                          "error. Not auto-removed.", U)
        self._rule("Patient ID unique", "Duplicate patient ID", "No patient ID appears more than once.",
                   id_dup_rows, "High", U, applicable=self.id_col is not None)

        full = self.df.astype(str).duplicated(keep=False)
        full_extra = self.df.astype(str).duplicated(keep="first")
        self.excess_dup_rows |= set(np.flatnonzero(full_extra.to_numpy()).tolist())
        for idx in np.flatnonzero(full.to_numpy()):
            self._add(int(idx), None, "Duplicate record", "Complete records should not repeat", "Medium",
                      "Confirm whether this is a true duplicate before any removal.", U,
                      value="(entire row)", use_cell=False)
        self._rule("No complete duplicate records", "Duplicate record", "No two rows are identical in every column.",
                   int(full.sum()), "Medium", U)

        applicable = self.id_col is not None and self.n_cols > 2
        n_alt = 0
        if applicable:
            rest = self.df.drop(columns=[self.id_col]).astype(str)
            alt = rest.duplicated(keep=False) & ~full
            n_alt = int(alt.sum())
            for idx in np.flatnonzero(alt.to_numpy()):
                self._add(int(idx), self.id_col, "Duplicate record",
                          "Identical non-ID attributes under different patient IDs", "Low",
                          "Check whether the same patient was registered under multiple IDs.", U)
        self._rule("No identical records under different IDs", "Duplicate record",
                   "Rows identical except for the patient ID.", n_alt, "Low", U, applicable=applicable)

    # ------------------------------------------------------------------ numeric
    def _to_numeric(self, c) -> pd.Series:
        return pd.to_numeric(self.df[c], errors="coerce")

    def _numeric(self):
        V = "Validity"
        for c in self.df.columns:
            if c in (self.id_col, self.bp_col) or self.mask[c].all():
                continue
            if pd.api.types.is_numeric_dtype(self.df[c]) or config.EXPECTED_TYPES.get(self.norm[c]) == "numeric":
                self.numeric_cols.append(c)
            elif is_text_series(self.df[c]):
                conv = self._to_numeric(c)
                ok = (~self.mask[c]) & conv.notna()
                if (~self.mask[c]).sum() and ok.sum() / (~self.mask[c]).sum() >= 0.8:
                    self.numeric_cols.append(c)

        bad_idx, neg_idx, outliers = set(), set(), set()
        for c in self.numeric_cols:
            conv = self._to_numeric(c)
            invalid = (~self.mask[c]) & conv.isna()
            for idx in np.flatnonzero(invalid.to_numpy()):
                bad_idx.add((int(idx), c))
                self._add(int(idx), c, "Invalid data type", "Value must be convertible to a number", "High",
                          "Correct at source. The original text is preserved.", V)
            for idx in np.flatnonzero((conv < 0).to_numpy()):
                neg_idx.add((int(idx), c))
                self._add(int(idx), c, "Out-of-range value", "Negative values are prohibited", "High",
                          "Verify the value at source; do not auto-correct.", V)
            vals = conv.dropna()
            if len(vals) >= config.OUTLIER_MIN_ROWS and self.norm[c] not in config.OUTLIER_EXCLUDED_COLUMNS:
                q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
                iqr = q3 - q1
                lo, hi = q1 - config.OUTLIER_IQR_MULTIPLIER * iqr, q3 + config.OUTLIER_IQR_MULTIPLIER * iqr
                for idx in np.flatnonzero(((conv < lo) | (conv > hi)).to_numpy()):
                    outliers.add((int(idx), c))
                    self._add(int(idx), c, "Statistical outlier",
                              f"Outside IQR fence [{lo:.2f}, {hi:.2f}] ({config.OUTLIER_IQR_MULTIPLIER}xIQR)",
                              "Informational", "Statistical outlier only - not necessarily an error. Review in context.", V)
        has = bool(self.numeric_cols)
        self._rule("Numeric conversion valid", "Invalid data type", "Numeric columns contain only numbers.",
                   len(bad_idx), "High", V, applicable=has)
        self._rule("No negative values where prohibited", "Out-of-range value", "Numeric values are not negative.",
                   len(neg_idx), "High", V, applicable=has)
        self._rule("IQR outlier screen", "Statistical outlier",
                   f"Values beyond {config.OUTLIER_IQR_MULTIPLIER}xIQR (needs >= {config.OUTLIER_MIN_ROWS} rows; "
                   "dosage excluded: pooled across medications).", len(outliers), "Informational", V,
                   applicable=has and self.n_rows >= config.OUTLIER_MIN_ROWS)

    # ------------------------------------------------------------------ clinical
    def _range_check(self, col, conv, rule, label_prefix=""):
        V = "Validity"
        lo, hi, excl = rule["min"], rule["max"], rule.get("min_exclusive", False)
        low = (conv <= lo) if excl else (conv < lo)
        flagged = low | (conv > hi)
        op = ">" if excl else ">="
        name = (f"{label_prefix}{rule['label']} must be {op} {lo} and <= {hi} {rule['unit']} "
                "(data-quality screen, not diagnostic)")
        n = 0
        for idx in np.flatnonzero(flagged.fillna(False).to_numpy()):
            n += 1
            self._add(int(idx), col, "Out-of-range value", name, rule["severity"],
                      "Verify against source / units. Value preserved; no automatic correction.", V)
        return n

    def _clinical(self):
        V = "Validity"
        bp_handled = False
        for key, rule in config.CLINICAL_RULES.items():
            col = next((self.by_norm[a] for a in rule["aliases"] if a in self.by_norm), None)
            if col is not None:
                conv = self._to_numeric(col)
                n = self._range_check(col, conv, rule)
                self._rule(f"{rule['label']} range", "Out-of-range value", rule["explanation"], n,
                           rule["severity"], V)
                continue
            if key in ("systolic_bp", "diastolic_bp") and self.bp_col is not None:
                if not bp_handled:
                    self._blood_pressure()
                    bp_handled = True
                continue
            self._rule(f"{rule['label']} range", "Out-of-range value", rule["explanation"] + " (column not in dataset)",
                       0, rule["severity"], V, applicable=False)

    def _blood_pressure(self):
        V = "Validity"
        c = self.bp_col
        sb, db = config.CLINICAL_RULES["systolic_bp"], config.CLINICAL_RULES["diastolic_bp"]
        parts = self.df[c].fillna("").astype(str).str.extract(config.BLOOD_PRESSURE_PATTERN)
        sys_, dia_ = pd.to_numeric(parts[0], errors="coerce"), pd.to_numeric(parts[1], errors="coerce")
        unparsed = (~self.mask[c]) & sys_.isna()
        for idx in np.flatnonzero(unparsed.to_numpy()):
            self._add(int(idx), c, "Invalid data type", "Blood pressure must look like 'systolic/diastolic'",
                      "High", "Correct the format at source. Original text preserved.", V)
        self._rule("Blood pressure format", "Invalid data type", "Values match 'systolic/diastolic'.",
                   int(unparsed.sum()), "High", V)
        n_s = self._range_check(c, sys_, sb, label_prefix="[BP] ")
        n_d = self._range_check(c, dia_, db, label_prefix="[BP] ")
        self._rule("Systolic BP range", "Out-of-range value", sb["explanation"], n_s, sb["severity"], V)
        self._rule("Diastolic BP range", "Out-of-range value", db["explanation"], n_d, db["severity"], V)
        inverted = (sys_ <= dia_).fillna(False)
        for idx in np.flatnonzero(inverted.to_numpy()):
            self._add(int(idx), c, "Out-of-range value", "Systolic should be greater than diastolic", "High",
                      "Check whether the two numbers were swapped at entry. Value preserved.", V)
        self._rule("Systolic greater than diastolic", "Out-of-range value", "Logical consistency of BP reading.",
                   int(inverted.sum()), "High", V)

    # ------------------------------------------------------------------ categorical
    def _categorical(self):
        K, V = "Consistency", "Validity"
        self.text_cols = [c for c in self.df.columns if is_text_series(self.df[c]) and not self.mask[c].all()
                          and c not in self.numeric_cols]
        self.categorical_cols = [c for c in self.text_cols if c not in (self.id_col, self.bp_col)
                                 and self.df[c][~self.mask[c]].nunique() <= config.CATEGORICAL_MAX_UNIQUE]
        ws_cells, case_cells, spell_cells, dom_cells = set(), set(), set(), set()

        for c in self.text_cols:
            s = self.df[c].fillna("").astype(str)
            for idx in np.flatnonzero(((s != s.str.strip()) & ~self.mask[c]).to_numpy()):
                ws_cells.add((int(idx), c))
                self._add(int(idx), c, "Inconsistent formatting", "No leading/trailing whitespace", "Low",
                          "Trim whitespace (applied automatically in cleaning).", K, value=repr(self.df[c].iloc[idx]))

        for c in self.categorical_cols:
            s = self.df[c].where(~self.mask[c]).astype("object").map(lambda v: v.strip() if isinstance(v, str) else v)
            freq = s.value_counts()
            groups = {}
            for v in freq.index:
                groups.setdefault(v.casefold(), []).append(v)
            for _, variants in groups.items():
                if len(variants) > 1:
                    canon = variants[0]  # most frequent (value_counts order)
                    for idx in np.flatnonzero(s.isin(variants[1:]).to_numpy()):
                        case_cells.add((int(idx), c))
                        self._add(int(idx), c, "Inconsistent formatting",
                                  f"Inconsistent capitalisation (majority form: '{canon}')", "Low",
                                  "Standardise to the majority form (applied automatically).", K)
            uniq = [g[0] for g in groups.values()]
            if 1 < len(uniq) <= 200:
                for a, b in [(uniq[i], uniq[j]) for i in range(len(uniq)) for j in range(i + 1, len(uniq))]:
                    if re.sub(r"\d", "", a.casefold()) == re.sub(r"\d", "", b.casefold()):
                        continue  # differ only by digits (e.g. 'Type 1' vs 'Type 2') - legitimately different
                    if difflib.SequenceMatcher(None, a.casefold(), b.casefold()).ratio() >= config.SPELLING_SIMILARITY_THRESHOLD:
                        rare = a if freq[a] < freq[b] else b
                        other = b if rare == a else a
                        for idx in np.flatnonzero((s.str.casefold() == rare.casefold()).fillna(False).to_numpy()):
                            spell_cells.add((int(idx), c))
                            self._add(int(idx), c, "Inconsistent spelling",
                                      f"Possible misspelling of '{other}' (similarity >= {config.SPELLING_SIMILARITY_THRESHOLD})",
                                      "Low", "Review and correct at source; not auto-changed.", K)
            dom = config.CATEGORY_DOMAINS.get(self.norm[c])
            if dom:
                allowed = {d.casefold() for d in dom}
                canon_map = {d.casefold(): d for d in dom}
                for idx in np.flatnonzero((s.notna() & s.str.casefold().isin(allowed) & (s != s.str.casefold().map(canon_map))).to_numpy()):
                    if (int(idx), c) in case_cells:
                        continue
                    case_cells.add((int(idx), c))
                    self._add(int(idx), c, "Inconsistent formatting",
                              f"Category should use the standard form '{canon_map[str(s.iloc[idx]).casefold()]}'", "Low",
                              "Standardise capitalisation (applied automatically).", K)
                for idx in np.flatnonzero((~s.str.casefold().isin(allowed) & s.notna()).to_numpy()):
                    dom_cells.add((int(idx), c))
                    self._add(int(idx), c, "Invalid category",
                               f"Value must be one of: {', '.join(dom)}", "Medium",
                               "Correct at source; unknown categories are not guessed.", V)
        self._rule("No leading/trailing whitespace", "Inconsistent formatting", "Text cells are trimmed.",
                   len(ws_cells), "Low", K, applicable=bool(self.text_cols))
        self._rule("Consistent capitalisation", "Inconsistent formatting",
                   "Category labels use one capitalisation.", len(case_cells), "Low", K,
                   applicable=bool(self.categorical_cols))
        self._rule("Consistent spelling", "Inconsistent spelling",
                   "No near-identical category labels (digit-only differences ignored).", len(spell_cells), "Low", K,
                   applicable=bool(self.categorical_cols))
        has_dom = any(self.norm[c] in config.CATEGORY_DOMAINS for c in self.categorical_cols)
        self._rule("Values within known category domains", "Invalid category",
                   "Gender / Readmitted values belong to the configured domain.", len(dom_cells), "Medium", V,
                   applicable=has_dom)

    # ------------------------------------------------------------------ privacy
    def _privacy(self):
        P = "Privacy"
        flagged = 0
        for c in self.df.columns:
            if self.mask[c].all():
                continue
            info = classify_column(self.norm[c], self.df[c])
            if info["privacy_category"] == "Direct identifier":
                flagged += 1
                self._add(None, c, "Privacy risk", "Direct identifier detected", "Critical",
                          info["action"], P, value=f"({info['basis']})", use_cell=False)
            elif info["privacy_category"].startswith("Direct identifier (pseudonymous"):
                flagged += 1
                self._add(None, c, "Privacy risk", "Patient-level identifier present", "Informational",
                          info["action"], P, value="(column-level finding)", use_cell=False)
            elif info["privacy_category"] == "Quasi-identifier":
                self._add(None, c, "Privacy risk", "Quasi-identifier can contribute to re-identification",
                          "Informational", info["action"], P, value="(column-level finding)", use_cell=False)
        self._rule("No direct identifiers", "Privacy risk", "No names / contact details / government IDs detected.",
                   sum(1 for e in self._exc if e["issue_category"] == "Privacy risk" and e["severity"] == "Critical"),
                   "Critical", P)

    # ------------------------------------------------------------------ finish
    def _finish(self) -> ValidationOutput:
        exc = pd.DataFrame(self._exc, columns=[c for c in EXCEPTION_COLUMNS if c != "exception_id"])
        if len(exc):
            exc["_sev"] = exc["severity"].map({s: i for i, s in enumerate(config.SEVERITY_ORDER)})
            exc["_row"] = exc["row_number"].fillna(0)
            exc = exc.sort_values(["_sev", "_row", "column_name"], kind="stable").drop(columns=["_sev", "_row"])
        exc = exc.reset_index(drop=True)
        exc.insert(0, "exception_id", [f"EX-{i + 1:04d}" for i in range(len(exc))])
        rules = pd.DataFrame(self._rules, columns=RULE_COLUMNS)
        stats = dict(missing_cells=int(self.mask.to_numpy().sum()) if len(self.mask) else 0,
                     total_cells=int(self.n_rows * self.n_cols), n_rows=self.n_rows, n_cols=self.n_cols,
                     text_cells=int(self.n_rows * len(self.text_cols)),
                     excess_dup_rows=len(self.excess_dup_rows), id_col=self.id_col, bp_col=self.bp_col,
                     numeric_cols=list(self.numeric_cols), categorical_cols=list(self.categorical_cols),
                     text_cols=list(self.text_cols))
        return ValidationOutput(exc, rules, stats)


def validate(df: pd.DataFrame) -> ValidationOutput:
    """Convenience wrapper: run the full engine on a DataFrame."""
    return ValidationEngine(df).run()


def _pct(bad: float, total: float) -> float:
    return 100.0 if total <= 0 else round(100 * max(0.0, 1 - bad / total), 2)


def compute_quality_scores(out: ValidationOutput) -> dict:
    """Transparent 0-100 scores. Formula documented in config.SCORE_FORMULA_TEXT."""
    st, exc, rules = out.stats, out.exceptions, out.rules
    n, cells = st["n_rows"], st["total_cells"]
    if n == 0 or st["n_cols"] == 0:
        dims = {k: 0.0 for k in config.QUALITY_WEIGHTS}
        return dict(dimensions=dims, overall=0.0, total_exceptions=len(exc), issues_detected=len(exc),
                    records_requiring_review=0, validated_records=0, missing_values=0, duplicate_id_rows=0,
                    duplicate_record_rows=0, invalid_values=0, severity_counts={}, **{"n_rows": n, "n_cols": st["n_cols"]})
    row_level = exc[exc["row_number"].notna()]
    non_info = exc[exc["severity"] != "Informational"]
    invalid = row_level[(row_level["dimension"] == "Validity") & (row_level["severity"] != "Informational")]
    invalid_cells = invalid.drop_duplicates(["row_number", "column_name"]).shape[0]
    cons = row_level[row_level["dimension"] == "Consistency"].drop_duplicates(["row_number", "column_name"]).shape[0]
    struct = rules[(rules["dimension"] == "Structural Integrity") & (rules["status"] != "Skipped")]
    struct_score = 100.0 if len(struct) == 0 else round(float(100 * (struct["status"] == "Passed").mean()), 2)
    dims = {"Completeness": _pct(st["missing_cells"], cells),
            "Validity": _pct(invalid_cells, cells),
            "Uniqueness": _pct(st["excess_dup_rows"], n),
            "Consistency": _pct(cons, st["text_cells"]),
            "Structural Integrity": struct_score}
    overall = round(float(sum(dims[k] * w for k, w in config.QUALITY_WEIGHTS.items())), 2)
    review_rows = row_level[row_level["severity"] != "Informational"]["row_number"].nunique()
    id_dups = int((exc["issue_category"] == "Duplicate patient ID").sum())
    rec_dups = int((exc["issue_category"] == "Duplicate record").sum())
    return dict(dimensions=dims, overall=overall, total_exceptions=len(exc), issues_detected=len(non_info),
                records_requiring_review=int(review_rows), validated_records=int(n - review_rows),
                missing_values=int(st["missing_cells"]), duplicate_id_rows=id_dups,
                duplicate_record_rows=rec_dups,
                invalid_values=int(invalid_cells),
                severity_counts={s: int((exc["severity"] == s).sum()) for s in config.SEVERITY_ORDER},
                n_rows=n, n_cols=st["n_cols"])

