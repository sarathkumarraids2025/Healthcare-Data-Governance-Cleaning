"""Healthcare Data Governance & Quality Analytics - Streamlit dashboard.

Run: streamlit run app.py
Every number on every page is computed from the loaded dataset by data_processor.run_pipeline().
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import config
import data_processor as dp
from data_processor import DataLoadError
from validation_rules import is_text_series, missing_mask

st.set_page_config(
    page_title="Healthcare Data Governance",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

BLUE, TEAL, MINT, GREEN, NAVY = "#1D6FDC", "#0E9AA7", "#5FD3BC", "#2FB36D", "#12355B"
AMBER, RED, GREY = "#F2A93B", "#E0525A", "#8AA0B4"

SEQ = [BLUE, TEAL, MINT, GREEN, "#7FB5F5", "#9BE3D2", "#B5E3A8", NAVY]

SEV_COLORS = {
    "Critical": "#B3261E",
    "High": RED,
    "Medium": AMBER,
    "Low": "#7FB5F5",
    "Informational": GREY
}

STATUS_COLORS = {
    "Passed": GREEN,
    "Failed": RED,
    "Warning": AMBER,
    "Skipped": GREY
}

PAGES = [
    "Dashboard Overview",
    "Dataset Explorer",
    "Data Quality Analysis",
    "Data Dictionary",
    "Validation Results",
    "Exception Log",
    "Cleaning Pipeline",
    "Privacy & Governance",
    "Download Center",
    "About Project"
]

CSS = """
<style>
.block-container {padding-top: 1.6rem; max-width: 1350px;}
section[data-testid="stSidebar"] {background: #FFFFFF; border-right: 1px solid #DCE9F0;}
h1, h2, h3, h4 {color: #12355B; letter-spacing: -0.01em;}

.hero {
    background: linear-gradient(120deg, #1D6FDC 0%, #0E9AA7 60%, #2FB36D 100%);
    color: #fff;
    padding: 22px 28px;
    border-radius: 16px;
    margin-bottom: 18px;
}

.hero h1 {
    color: #fff;
    margin: 0;
    font-size: 1.9rem;
}

.hero p {
    margin: 4px 0 0 0;
    opacity: .95;
}

.kpi {
    background: #fff;
    border: 1px solid #DCE9F0;
    border-left: 6px solid var(--c);
    border-radius: 12px;
    padding: 14px 16px;
    box-shadow: 0 1px 3px rgba(18,53,91,.07);
    margin-bottom: 10px;
}

.kpi .l {
    font-size: .74rem;
    color: #5B7083;
    text-transform: uppercase;
    letter-spacing: .05em;
    font-weight: 600;
}

.kpi .v {
    font-size: 1.85rem;
    font-weight: 700;
    color: #12355B;
    line-height: 1.25;
}

.kpi .s {
    font-size: .74rem;
    color: #7C93A6;
}

.note {
    background: #EAF6F8;
    border: 1px solid #BFE3E8;
    border-radius: 10px;
    padding: 10px 14px;
    color: #12355B;
    margin-bottom: 12px;
}

.warn {
    background: #FFF6E5;
    border: 1px solid #F5D9A0;
    border-radius: 10px;
    padding: 10px 14px;
    color: #5C3B00;
    margin-bottom: 12px;
}

.disc {
    font-size: .78rem;
    color: #5B7083;
    border-top: 1px solid #DCE9F0;
    padding-top: 10px;
    margin-top: 28px;
}
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

_CHART_KW = (
    {"width": "stretch"}
    if "width" in inspect.signature(st.plotly_chart).parameters
    else {"use_container_width": True}
)


# ----------------------------------------------------------------------
# UI HELPERS
# ----------------------------------------------------------------------

def chart(fig, key):
    fig.update_layout(
        template="plotly_white",
        margin=dict(l=10, r=10, t=50, b=10),
        font=dict(family="Segoe UI, Inter, sans-serif", color=NAVY),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=SEQ
    )

    st.plotly_chart(fig, key=key, **_CHART_KW)


def hero(title, subtitle):
    st.markdown(
        f'<div class="hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True
    )


def kpi(col, label, value, sub="", color=TEAL):
    col.markdown(
        f'<div class="kpi" style="--c:{color}">'
        f'<div class="l">{label}</div>'
        f'<div class="v">{value}</div>'
        f'<div class="s">{sub}&nbsp;</div></div>',
        unsafe_allow_html=True
    )


def note(text, warn=False):
    st.markdown(
        f'<div class="{"warn" if warn else "note"}">{text}</div>',
        unsafe_allow_html=True
    )


def mask_series(s: pd.Series) -> pd.Series:
    return s.map(
        lambda v: v if pd.isna(v) else (str(v)[:3] + "***")
    )


def id_columns(df: pd.DataFrame):
    return [
        c for c in df.columns
        if str(c).lower().replace("_", "")
        in ("patientid", "mrn", "recordid", "id")
    ]


def display_df(df: pd.DataFrame, mask_on: bool) -> pd.DataFrame:
    out = df.copy()

    if mask_on:
        for c in id_columns(out):
            out[c] = mask_series(out[c])

    return out


def mask_log(df: pd.DataFrame, id_col, mask_on: bool) -> pd.DataFrame:
    out = df.copy()

    if mask_on and len(out):
        out["record_identifier"] = out["record_identifier"].map(
            lambda v: v if str(v).startswith(("(", "Row"))
            else str(v)[:3] + "***"
        )

        if id_col is not None:
            m = out["column_name"] == id_col

            out.loc[m, "original_value"] = out.loc[m, "original_value"].map(
                lambda v: str(v)[:3] + "***"
                if v != "(column-level finding)"
                else v
            )

    return out


# FIXED: height=None is not accepted by newer Streamlit versions.
def table(df, height=None):
    st.dataframe(
        df,
        hide_index=True,
        height=height if height is not None else "content"
    )


def dl(label, df, name, key):
    st.download_button(
        label,
        data=dp.to_csv_bytes(df),
        file_name=name,
        mime="text/csv",
        key=key
    )


def needs_data(res):
    if res.cleaned.empty:
        st.warning(
            "The dataset has no usable rows/columns. "
            "See the Validation Results and Exception Log pages."
        )
        return True

    return False


# ----------------------------------------------------------------------
# DASHBOARD OVERVIEW
# ----------------------------------------------------------------------

def page_overview(res):
    hero(
        "Healthcare Data Governance & Quality Analytics",
        "Privacy-Aware Patient Data Validation, Cleaning & Quality Monitoring"
    )

    si, sb = res.source_info, res.scores_before

    note(
        f"<b>Data source:</b> {si['source_type']} &nbsp;|&nbsp; "
        f"<code>{si['source_label']}</code> &nbsp;|&nbsp; "
        f"{si['file_size_bytes']:,} bytes &nbsp;|&nbsp; "
        f"encoding {si['encoding']} &nbsp;|&nbsp; loaded {si['loaded_at']}"
    )

    if si.get("note"):
        note(si["note"], warn=True)

    if needs_data(res):
        return

    c = st.columns(4)

    kpi(c[0], "Total Records", f"{sb['n_rows']:,}", "rows in dataset", BLUE)
    kpi(c[1], "Total Columns", sb["n_cols"], "original columns", BLUE)
    kpi(c[2], "Missing Values", sb["missing_values"], "null or blank cells", TEAL)
    kpi(c[3], "Duplicate Patient IDs", sb["duplicate_id_rows"], "rows sharing an ID", TEAL)

    c = st.columns(4)

    kpi(c[0], "Duplicate Records", sb["duplicate_record_rows"], "duplicate-record flags", GREEN)
    kpi(c[1], "Invalid Values", sb["invalid_values"], "distinct cells failing rules", GREEN)
    kpi(c[2], "Data Quality Score", f"{sb['overall']}", "out of 100", MINT)
    kpi(
        c[3],
        "Records Requiring Review",
        sb["records_requiring_review"],
        f"{sb['validated_records']} validated",
        MINT
    )

    a, b = st.columns(2)

    with a:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=sb["overall"],
                title={"text": "Overall data quality score"},
                gauge=dict(
                    axis=dict(range=[0, 100]),
                    bar=dict(color=TEAL),
                    steps=[
                        dict(range=[0, 60], color="#FBE3E4"),
                        dict(range=[60, 85], color="#FFF1D6"),
                        dict(range=[85, 100], color="#DDF5E8")
                    ]
                )
            )
        )

        chart(fig, "ov_gauge")

    with b:
        d = pd.DataFrame({
            "Dimension": list(sb["dimensions"]),
            "Score": list(sb["dimensions"].values())
        })

        fig = px.bar(
            d,
            x="Score",
            y="Dimension",
            orientation="h",
            range_x=[0, 100],
            text="Score",
            title="Data quality dimensions",
            color_discrete_sequence=[BLUE]
        )

        fig.update_layout(yaxis=dict(autorange="reversed"))

        chart(fig, "ov_dims")

    a, b = st.columns(2)

    with a:
        m = missing_mask(res.raw).sum().reset_index()
        m.columns = ["Column", "Missing"]

        fig = px.bar(
            m,
            x="Column",
            y="Missing",
            title="Missing values by column",
            color_discrete_sequence=[TEAL]
        )

        chart(fig, "ov_missing")

        if m["Missing"].sum() == 0:
            st.caption("No missing values were detected in any column.")

    with b:
        v = pd.DataFrame({
            "Status": [
                "Validated (no issues above Informational)",
                "Requires review"
            ],
            "Records": [
                sb["validated_records"],
                sb["records_requiring_review"]
            ]
        })

        fig = px.pie(
            v,
            names="Status",
            values="Records",
            hole=.55,
            title="Valid vs invalid records",
            color="Status",
            color_discrete_map={
                v["Status"][0]: GREEN,
                v["Status"][1]: RED
            }
        )

        chart(fig, "ov_valid")

    a, b = st.columns(2)

    exc = res.exceptions

    with a:
        if len(exc):
            sev = (
                exc["severity"]
                .value_counts()
                .reindex(config.SEVERITY_ORDER)
                .dropna()
                .reset_index()
            )

            sev.columns = ["Severity", "Exceptions"]

            fig = px.pie(
                sev,
                names="Severity",
                values="Exceptions",
                hole=.55,
                title="Issue severity distribution",
                color="Severity",
                color_discrete_map=SEV_COLORS
            )

            chart(fig, "ov_sev")

        else:
            st.info("No exceptions were logged.")

    with b:
        if len(exc):
            cat = exc["issue_category"].value_counts().reset_index()
            cat.columns = ["Issue category", "Exceptions"]

            fig = px.bar(
                cat,
                x="Exceptions",
                y="Issue category",
                orientation="h",
                title="Exceptions by category",
                color_discrete_sequence=[BLUE]
            )

            fig.update_layout(yaxis=dict(autorange="reversed"))

            chart(fig, "ov_cat")

    a, b = st.columns(2)

    with a:
        cl = res.dictionary["Data Classification"].value_counts().reset_index()
        cl.columns = ["Classification", "Columns"]

        chart(
            px.bar(
                cl,
                x="Classification",
                y="Columns",
                title="Data classification distribution",
                color_discrete_sequence=[TEAL]
            ),
            "ov_class"
        )

    with b:
        dt = res.dictionary["Data Type"].value_counts().reset_index()
        dt.columns = ["Data type", "Columns"]

        chart(
            px.pie(
                dt,
                names="Data type",
                values="Columns",
                hole=.5,
                title="Data type distribution"
            ),
            "ov_dtype"
        )

    if sb["issues_detected"] == 0:
        note(
            "No issues above <b>Informational</b> severity were found in this dataset. "
            "Informational items (privacy notes, statistical observations) may still be present."
        )
        def explorer_table(df, key, mask_on):
    disp = display_df(df, mask_on)

    cols = st.multiselect(
        "Columns to display",
        list(disp.columns),
        default=list(disp.columns),
        key=f"{key}_cols"
    )

    c1, c2 = st.columns(2)

    q = c1.text_input(
        "Search (matches any displayed text)",
        key=f"{key}_q"
    )

    cands = [
        c for c in disp.columns
        if is_text_series(disp[c])
        and disp[c].nunique() <= config.CATEGORICAL_MAX_UNIQUE
    ]

    fcol = c2.selectbox(
        "Filter by column",
        ["(none)"] + cands,
        key=f"{key}_fcol"
    )

    view = disp[cols] if cols else disp.iloc[:, :0]

    if fcol != "(none)":
        vals = st.multiselect(
            f"Keep values of {fcol}",
            sorted(disp[fcol].dropna().unique()),
            key=f"{key}_fval"
        )

        if vals:
            view = view[disp[fcol].isin(vals)]

    if q:
        hay = view.astype(str).apply(
            lambda r: " | ".join(r),
            axis=1
        )

        view = view[hay.str.contains(q, case=False, regex=False)]

    p1, p2 = st.columns(2)

    size = p1.selectbox(
        "Rows per page",
        [10, 25, 50, 100],
        key=f"{key}_size"
    )

    pages = max(1, int(np.ceil(len(view) / size)))

    page = p2.number_input(
        f"Page (1-{pages})",
        min_value=1,
        max_value=pages,
        value=1,
        step=1,
        key=f"{key}_page"
    )

    lo = (int(page) - 1) * size

    table(view.iloc[lo:lo + size])

    st.caption(
        f"Showing rows {lo + 1 if len(view) else 0}-"
        f"{min(lo + size, len(view))} of {len(view)} "
        f"(from {len(df)} total)."
        f"{' Patient identifiers are masked.' if mask_on else ''}"
    )

    dl(
        "Download this view (CSV)",
        view,
        f"{key}_view.csv",
        f"{key}_dl"
    )


def page_explorer(res, mask_on):
    hero(
        "Dataset Explorer",
        "Browse, search and filter the original and cleaned data"
    )

    if needs_data(res):
        return

    t1, t2 = st.tabs([
        "Original data (unchanged)",
        "Cleaned data"
    ])

    with t1:
        st.caption(
            f"{res.raw.shape[0]} rows x {res.raw.shape[1]} columns "
            "- exactly as loaded."
        )

        explorer_table(res.raw, "orig", mask_on)

    with t2:
        st.caption(
            f"{res.cleaned.shape[0]} rows x {res.cleaned.shape[1]} columns "
            "- includes derived and review-flag columns."
        )

        explorer_table(res.cleaned, "clean", mask_on)


def column_scores(res):
    n = len(res.raw)
    e = res.exceptions

    rows = []

    for c in res.raw.columns:
        flagged = e[
            (e["column_name"] == c)
            & e["row_number"].notna()
            & (e["severity"] != "Informational")
        ]["row_number"].nunique()

        rows.append({
            "Column": c,
            "FlaggedRecords": int(flagged),
            "QualityScore": round(100 * (1 - flagged / n), 2) if n else 0.0
        })

    return pd.DataFrame(rows)


def page_quality(res, mask_on):
    hero(
        "Data Quality Analysis",
        "Missing values, duplicates, distributions, outliers and consistency"
    )

    if needs_data(res):
        return

    raw, e, stats = res.raw, res.exceptions, res.stats

    tabs = st.tabs([
        "Missing values",
        "Duplicates",
        "Numeric distributions",
        "Outliers",
        "Data types",
        "Category consistency",
        "Column quality scores",
        "Anomaly detection"
    ])

    # Missing values
    with tabs[0]:
        m = missing_mask(raw)

        pct = (m.mean() * 100).round(2).reset_index()
        pct.columns = ["Column", "Missing %"]

        sample = m.head(500)

        fig = go.Figure(
            go.Heatmap(
                z=sample.T.astype(int).values,
                x=list(range(1, len(sample) + 1)),
                y=list(sample.columns),
                colorscale=[[0, "#DDF5E8"], [1, RED]],
                showscale=False
            )
        )

        fig.update_layout(
            title=f"Missing-value heatmap (red = missing; first {len(sample)} rows)",
            xaxis_title="Row number"
        )

        chart(fig, "q_heat")

        chart(
            px.bar(
                pct,
                x="Column",
                y="Missing %",
                range_y=[0, 100],
                title="Missing percentage by column",
                color_discrete_sequence=[TEAL]
            ),
            "q_misspct"
        )

        table(pct)

    # Duplicates
    with tabs[1]:
        d = e[e["issue_category"].isin([
            "Duplicate patient ID",
            "Duplicate record"
        ])]

        c = st.columns(3)

        kpi(
            c[0],
            "Rows sharing a patient ID",
            int((e["issue_category"] == "Duplicate patient ID").sum()),
            "",
            BLUE
        )

        kpi(
            c[1],
            "Exact duplicate-record flags",
            int(
                (
                    (e["issue_category"] == "Duplicate record")
                    & (e["validation_rule"].str.startswith("Complete"))
                ).sum()
            ),
            "",
            TEAL
        )

        kpi(
            c[2],
            "Same attributes, different ID",
            int(
                (
                    (e["issue_category"] == "Duplicate record")
                    & (e["validation_rule"].str.startswith("Identical"))
                ).sum()
            ),
            "",
            GREEN
        )

        if d.empty:
            st.success(
                "No duplicate patient IDs or duplicate records were found."
            )

        else:
            st.warning(
                "Duplicates are flagged, not deleted: a repeated ID "
                "can be a legitimate repeat encounter."
            )

            table(
                mask_log(d, stats["id_col"], mask_on)[[
                    "exception_id",
                    "row_number",
                    "record_identifier",
                    "issue_category",
                    "validation_rule",
                    "severity"
                ]]
            )

    num = [
        c for c in stats["numeric_cols"]
        if c in raw.columns
    ]

    # Numeric distributions
    with tabs[2]:
        if not num:
            st.info("No numeric columns available.")

        else:
            col = st.selectbox(
                "Numeric column",
                num,
                key="q_numcol"
            )

            s = pd.to_numeric(raw[col], errors="coerce")

            chart(
                px.histogram(
                    pd.DataFrame({col: s.dropna()}),
                    x=col,
                    nbins=20,
                    title=f"Distribution of {col}",
                    color_discrete_sequence=[BLUE],
                    marginal="box"
                ),
                "q_hist"
            )

            table(
                s.describe()
                .round(3)
                .reset_index()
                .rename(columns={
                    "index": "statistic",
                    col: "value"
                })
            )

    # Outliers
    with tabs[3]:
        rows = []

        for c in num:
            s = pd.to_numeric(raw[c], errors="coerce").dropna()

            excluded = (
                c.lower().replace("_", "")
                in {
                    x.replace("_", "")
                    for x in config.OUTLIER_EXCLUDED_COLUMNS
                }
            )

            if len(s) >= config.OUTLIER_MIN_ROWS and not excluded:
                q1, q3 = s.quantile(.25), s.quantile(.75)

                lo = q1 - config.OUTLIER_IQR_MULTIPLIER * (q3 - q1)
                hi = q3 + config.OUTLIER_IQR_MULTIPLIER * (q3 - q1)

                rows.append({
                    "Column": c,
                    "Lower_fence": round(lo, 2),
                    "Upper_fence": round(hi, 2),
                    "Outliers": int(((s < lo) | (s > hi)).sum()),
                    "Note": ""
                })

            else:
                rows.append({
                    "Column": c,
                    "Lower_fence": None,
                    "Upper_fence": None,
                    "Outliers": None,
                    "Note": (
                        "Excluded (pooled across medications)"
                        if excluded
                        else f"Needs >= {config.OUTLIER_MIN_ROWS} rows"
                    )
                })

        if rows:
            table(pd.DataFrame(rows))

            col = st.selectbox(
                "Box plot for",
                num,
                key="q_boxcol"
            )

            chart(
                px.box(
                    pd.DataFrame({
                        col: pd.to_numeric(raw[col], errors="coerce").dropna()
                    }),
                    y=col,
                    points="all",
                    title=f"Outlier view: {col}",
                    color_discrete_sequence=[TEAL]
                ),
                "q_box"
            )

            st.caption(
                "IQR outliers are statistical only and never modify data. "
                "They are logged as Informational."
            )

        else:
            st.info("No numeric columns available.")

    # Data types
    with tabs[4]:
        table(
            res.dictionary[[
                "Column Name",
                "Data Type",
                "Cleaned Column Name",
                "Cleaned Data Type",
                "Missing Count",
                "Unique Values"
            ]]
        )

    # Category consistency
    with tabs[5]:
        cats = [
            c for c in stats["categorical_cols"]
            if c in raw.columns
        ]

        cons = e[e["issue_category"].isin([
            "Inconsistent formatting",
            "Inconsistent spelling",
            "Invalid category"
        ])]

        if cons.empty:
            st.success(
                "No whitespace, capitalisation, spelling "
                "or invalid-category issues were found."
            )

        else:
            table(
                mask_log(cons, stats["id_col"], mask_on)[[
                    "exception_id",
                    "row_number",
                    "column_name",
                    "original_value",
                    "issue_category",
                    "validation_rule"
                ]]
            )

        if cats:
            col = st.selectbox(
                "Category column",
                cats,
                key="q_catcol"
            )

            vc = raw[col].astype(str).value_counts().reset_index()
            vc.columns = [col, "Count"]

            chart(
                px.bar(
                    vc,
                    x=col,
                    y="Count",
                    title=f"Value counts: {col}",
                    color_discrete_sequence=[MINT]
                ),
                "q_cat"
            )

    # Column quality scores
    with tabs[6]:
        cs = column_scores(res)

        chart(
            px.bar(
                cs,
                x="QualityScore",
                y="Column",
                orientation="h",
                range_x=[0, 100],
                title="Column-level quality score (% of records without issues above Informational)",
                color_discrete_sequence=[GREEN]
            ),
            "q_colscore"
        )

        table(cs)

    # Anomaly detection
    with tabs[7]:
        if res.anomalies is None:
            st.info(
                res.anomaly_info.get(
                    "status",
                    "Anomaly detection is off. Enable it in the sidebar (optional, unsupervised)."
                )
            )

        else:
            ai, an = res.anomaly_info, res.anomalies

            note(
                "Isolation Forest scores are <b>statistical</b>, not medical. "
                "No accuracy is reported because there are no ground-truth labels. "
                "Values are never changed based on these scores."
            )

            st.caption(
                f"Features used: {', '.join(ai['features'])} | "
                f"contamination {ai['contamination']} | "
                f"rows used {ai['n_rows_used']} | "
                f"flagged {ai['n_flagged']}"
            )

            f = ai["features"]

            x = st.selectbox(
                "X axis",
                f,
                index=0,
                key="an_x"
            )

            y = st.selectbox(
                "Y axis",
                f,
                index=min(1, len(f) - 1),
                key="an_y"
            )

            chart(
                px.scatter(
                    an,
                    x=x,
                    y=y,
                    color="is_anomaly",
                    hover_data=["row_number", "anomaly_score"],
                    color_discrete_map={
                        True: RED,
                        False: TEAL
                    },
                    title="Isolation Forest anomaly view"
                ),
                "an_scatter"
            )

            table(an.sort_values("anomaly_score"))

            st.caption(
                "Limitations: " + " ".join(ai["limitations"])
            )