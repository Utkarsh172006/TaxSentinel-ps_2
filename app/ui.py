from __future__ import annotations

import html
import json
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.intake import read_uploaded_table
from app.service import PipelineResult, load_demo_tables, run_reconciliation
from detect.anomaly import benford_summary
from detect.issue import Issue
from explain.llm import explain
from ingest.clean import standardize_table, validation_report
from ingest.loader import assert_inference_safe
from ingest.schema_map import TABLE_SCHEMAS, detect_table_type, suggest_mapping
from liability.liability import calculate_liability, estimate_interest_and_penalty
from matching.model import train_pair_model
from reports.excel import build_excel_report
from reports.pdf import build_pdf_report
from review.store import (
    add_comment,
    initialize_store,
    list_audit_log,
    list_cases,
    list_comments,
    list_feedback,
    save_feedback,
    sync_issues,
    transition_case,
)
from vendor_risk.score import score_vendors

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REQUIRED_APP_TABLES = ("invoices", "bank_transactions", "ledger", "supplier_filings", "tax_rates")
SEVERITY_ORDER = {"Critical": 0, "High": 1, "Medium": 2, "Low": 3}
NAVIGATION = (
    "Command Center",
    "Data intake",
    "Issues inbox",
    "Anomaly explorer",
    "Vendor risk",
    "Tax liability",
    "Chat with your books",
    "Audit trail",
    "Reports",
    "Model & metrics",
    "Demo mode",
)


def _css(light_mode: bool) -> None:
    background = "#F4F6FB" if light_mode else "#0B1220"
    foreground = "#15213A" if light_mode else "#E7ECF5"
    card = "#FFFFFF" if light_mode else "#111C2F"
    border = "#D9E0EC" if light_mode else "#26344C"
    st.markdown(
        f"""
        <style>
        :root {{ color-scheme: {'light' if light_mode else 'dark'}; }}
        html, body, [class*="css"] {{ font-family: Inter, ui-sans-serif, system-ui, -apple-system, "Segoe UI", sans-serif; }}
        .stApp {{ background: {background}; color: {foreground}; }}
        [data-testid="stSidebar"] {{ background: {'#E9EEF8' if light_mode else '#0E1728'}; border-right: 1px solid {border}; }}
        [data-testid="stMetric"] {{ background: {card}; border: 1px solid {border}; padding: 16px 18px; border-radius: 16px; box-shadow: 0 8px 22px rgba(0,0,0,.12); }}
        [data-testid="stMetricValue"] {{ font-weight: 700; }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 18px; }}
        .ts-hero {{ padding: 24px 26px; border-radius: 20px; border: 1px solid {border}; background: linear-gradient(125deg, {card}, {'#E5E9FF' if light_mode else '#182441'}); box-shadow: 0 12px 30px rgba(0,0,0,.15); }}
        .ts-step {{ display:inline-block; padding:8px 13px; margin:5px 4px 0 0; border:1px solid {border}; border-radius:999px; font-size:13px; }}
        .ts-chip {{ display:inline-block; padding:4px 9px; border-radius:999px; font-size:12px; font-weight:600; background:{'rgba(75,90,210,.12)' if light_mode else 'rgba(111,126,255,.16)'}; }}
        .ts-muted {{ opacity:.72; }}
        button[kind="primary"] {{ border-radius: 11px; background: #4F46E5; border: 0; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _current_result() -> PipelineResult | None:
    value = st.session_state.get("pipeline_result")
    return value if isinstance(value, PipelineResult) else None


def _run_tables(tables: dict[str, pd.DataFrame], origin: str) -> None:
    progress = st.progress(0)
    status = st.empty()

    def stage(message: str, percent: int) -> None:
        status.info(message)
        progress.progress(percent)

    try:
        assert_inference_safe({f"{key}.csv": frame for key, frame in tables.items()})
        result = run_reconciliation(tables, on_stage=stage)
        sync_issues(result.issues)
        st.session_state["tables"] = tables
        st.session_state["pipeline_result"] = result
        st.session_state["dataset_origin"] = origin
        status.success("Reconciliation complete. Review the cases in the inbox.")
    except (ValueError, FileNotFoundError, KeyError) as exc:
        status.error(str(exc))
        raise


def _active_cases(result: PipelineResult) -> list[dict[str, Any]]:
    sync_issues(result.issues)
    allowed = {item.issue_id for item in result.issues}
    return [case for case in list_cases() if case["issue_id"] in allowed]


def _format_money(value: Any) -> str:
    try:
        return f"Rs {float(value):,.2f}"
    except (TypeError, ValueError):
        return "Rs 0.00"


def _issue_table(cases: list[dict[str, Any]]) -> pd.DataFrame:
    columns = ["issue_id", "issue_type", "severity", "vendor", "rupee_impact", "confidence", "status"]
    return pd.DataFrame(cases, columns=columns)


def _matched_invoice_ids(result: PipelineResult, minimum: float = 0.60) -> set[str]:
    return {
        invoice_id
        for match in result.reconciliation.matches
        if match.confidence >= minimum
        for invoice_id in match.invoice_ids
    }


def _command_center(result: PipelineResult, cases: list[dict[str, Any]]) -> None:
    invoice_count = len(result.tables["invoices"])
    matched_ids = _matched_invoice_ids(result)
    auto_ids = _matched_invoice_ids(result, minimum=0.90)
    match_rate = 100 * len(matched_ids) / max(invoice_count, 1)
    auto_rate = 100 * len(auto_ids) / max(invoice_count, 1)
    open_cases = [case for case in cases if case["status"] != "Resolved"]
    total_impact = sum(float(case.get("rupee_impact", 0) or 0) for case in open_cases)
    liability = result.liability["totals"]
    kpis = st.columns(6)
    kpis[0].metric("Match rate", f"{match_rate:.1f}%")
    kpis[1].metric("Auto-matched", f"{auto_rate:.1f}%")
    kpis[2].metric("Open issues", f"{len(open_cases):,}")
    kpis[3].metric("Open issue impact", _format_money(total_impact))
    kpis[4].metric("ITC at risk", _format_money(liability["itc_at_risk"]))
    kpis[5].metric("Net liability", _format_money(liability["net_liability_after"]))

    left, right = st.columns([1.25, 1])
    with left:
        st.subheader("Invoice → payment → books → filing")
        match_ids = matched_ids
        ledger_ids = set(result.reconciliation.ledger_links)
        filing_ids = set(result.reconciliation.filing_links)
        matched_count = len(match_ids)
        labels = ["Invoices", "Bank matched", "No confident payment", "Ledger linked", "Ledger gap", "Filing linked", "Filing gap"]
        sources = [0, 0, 1, 1, 1, 1]
        targets = [1, 2, 3, 4, 5, 6]
        values = [
            matched_count,
            max(invoice_count - matched_count, 0),
            len(ledger_ids),
            max(matched_count - len(ledger_ids), 0),
            len(filing_ids),
            max(matched_count - len(filing_ids), 0),
        ]
        fig = go.Figure(go.Sankey(
            node={"label": labels, "pad": 18, "thickness": 17,
                  "color": ["#4F46E5", "#14B8A6", "#F59E0B", "#14B8A6", "#EF4444", "#14B8A6", "#EF4444"]},
            link={"source": sources, "target": targets, "value": values,
                  "color": ["rgba(79,70,229,.35)", "rgba(245,158,11,.3)", "rgba(20,184,166,.3)",
                            "rgba(239,68,68,.3)", "rgba(20,184,166,.25)", "rgba(239,68,68,.25)"]},
        ))
        fig.update_layout(height=330, margin=dict(l=8, r=8, t=8, b=8), font_color="#DDE6F5")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.subheader("Issue types")
        by_type = Counter(case["issue_type"] for case in open_cases)
        chart = pd.DataFrame({"Issue type": list(by_type), "Cases": list(by_type.values())})
        if chart.empty:
            st.info("No open issues in this dataset.")
        else:
            fig = px.bar(chart.sort_values("Cases", ascending=True), x="Cases", y="Issue type", orientation="h",
                         color="Cases", color_continuous_scale=["#14B8A6", "#4F46E5"])
            fig.update_layout(height=330, showlegend=False, coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

    st.subheader("Monthly tax position")
    period_frame = pd.DataFrame(result.liability["periods"])
    if period_frame.empty:
        st.info("No dated invoices are available to trend.")
    else:
        trend = period_frame.melt(
            id_vars="period",
            value_vars=["output_tax", "eligible_itc", "net_liability_after"],
            var_name="Metric",
            value_name="Rupees",
        )
        fig = px.line(trend, x="period", y="Rupees", color="Metric", markers=True,
                      color_discrete_sequence=["#4F46E5", "#14B8A6", "#F59E0B"])
        fig.update_layout(height=310)
        st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Risk, liability, match rates and issue impact are computed from the loaded business records; "
        "the answer key is not used by the detector."
    )


def _read_case_records(case: dict[str, Any], tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    matches = []
    for table_name, frame in tables.items():
        id_columns = [column for column in ("record_id", "txn_id", "entry_id", "filing_id") if column in frame.columns]
        if not id_columns:
            continue
        for column in id_columns:
            selected = frame[frame[column].astype(str).isin(set(map(str, case.get("record_ids", []))))]
            for row in selected.to_dict("records"):
                matches.append({"Source": table_name, **row})
    return pd.DataFrame(matches)


def _case_detail(case: dict[str, Any], tables: dict[str, pd.DataFrame]) -> None:
    safe_id = html.escape(str(case["issue_id"]))
    st.markdown(f"<span class='ts-chip'>{html.escape(case['status'])}</span> &nbsp; <b>{safe_id}</b>", unsafe_allow_html=True)
    st.subheader(f"{case['issue_type'].replace('_', ' ').title()} · {case['vendor']}")
    left, right = st.columns([1.3, 1])
    with left:
        st.markdown("#### Source records")
        records = _read_case_records(case, tables)
        if records.empty:
            st.info("No source row was directly addressable for this case.")
        else:
            st.dataframe(records, use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### Evidence")
        st.write("Expected:", case.get("expected"))
        st.write("Observed:", case.get("actual"))
        st.metric("Deterministic rupee impact", _format_money(case.get("rupee_impact", 0)))
        gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=float(case.get("confidence", 0)) * 100,
            number={"suffix": "%"},
            gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#14B8A6"},
                   "steps": [{"range": [0, 60], "color": "#3A2632"}, {"range": [60, 90], "color": "#342E24"},
                             {"range": [90, 100], "color": "#173B39"}]},
        ))
        gauge.update_layout(height=200, margin=dict(l=15, r=15, t=10, b=0))
        st.plotly_chart(gauge, use_container_width=True)
        st.json(case.get("evidence", {}), expanded=False)

    explanation = explain(case)
    st.markdown("#### Evidence-grounded explanation")
    st.info(explanation["text"])
    st.caption(f"Explanation provider: {explanation['provider']}; figures are copied from the structured issue.")
    st.markdown("#### Suggested action")
    st.write(case.get("suggested_fix", "Review the source records."))
    if float(case.get("rupee_impact", 0) or 0) > 0:
        st.write(
            f"Adjustment proposal: prepare a reviewer-approved journal or credit note for "
            f"{_format_money(case['rupee_impact'])}; account mapping is not automated."
        )

    status = str(case["status"])
    action_cols = st.columns(5)
    if status == "Open" and action_cols[0].button("Investigate", key=f"investigate-{safe_id}"):
        transition_case(case["issue_id"], "Investigating", "Investigation started")
        st.rerun()
    if status == "Investigating" and action_cols[1].button("Mark explained", key=f"explain-{safe_id}"):
        transition_case(case["issue_id"], "Explained", "Offline explanation reviewed")
        st.rerun()
    if status == "Explained" and action_cols[2].button("Approve fix", key=f"approve-{safe_id}"):
        transition_case(case["issue_id"], "Resolved", "Fix approved by reviewer")
        st.rerun()
    if status == "Explained" and action_cols[3].button("Resolve", key=f"resolve-{safe_id}"):
        transition_case(case["issue_id"], "Resolved", "Case marked resolved")
        st.rerun()
    if action_cols[4].button("Escalate", key=f"escalate-{safe_id}"):
        transition_case(case["issue_id"], status, "Escalated to tax reviewer")
        st.toast("Case escalation recorded in the audit trail.")
    feedback_cols = st.columns(2)
    if feedback_cols[0].button("Accept finding", key=f"accept-{safe_id}"):
        save_feedback(case["issue_id"], "accept", "Reviewer confirmed this issue.")
        st.toast("Review feedback stored.")
    if feedback_cols[1].button("Reject finding", key=f"reject-{safe_id}"):
        save_feedback(case["issue_id"], "reject", "Reviewer rejected this issue.")
        st.toast("Review feedback stored.")

    with st.expander("Add case comment"):
        with st.form(f"comment-form-{safe_id}"):
            comment = st.text_area("Comment")
            author = st.text_input("Reviewer", value="Reviewer")
            submitted = st.form_submit_button("Add comment")
            if submitted:
                if not comment.strip():
                    st.error("Enter a comment before submitting.")
                else:
                    add_comment(case["issue_id"], comment, author)
                    st.rerun()
    comments = list_comments(case["issue_id"])
    if comments:
        st.markdown("#### Comment thread")
        for item in comments:
            st.caption(f"{item['created_at']} · {item['author']}")
            st.write(item["body"])
    timeline = [item for item in list_audit_log() if item["issue_id"] == case["issue_id"]]
    if timeline:
        st.markdown("#### Case timeline")
        st.dataframe(pd.DataFrame(timeline), hide_index=True, use_container_width=True)


def _issues_inbox(result: PipelineResult, cases: list[dict[str, Any]]) -> None:
    st.title("Issues inbox")
    if not cases:
        st.success("No issues were produced for the loaded records.")
        return
    frame = _issue_table(cases)
    views = st.session_state.setdefault("issue_saved_views", {})
    view_cols = st.columns([2, 1, 1])
    view_name = view_cols[0].text_input("Saved view name", key="saved-view-name")
    if views:
        selected_view = view_cols[1].selectbox("Saved views", ["(none)", *views.keys()], key="saved-view-select")
        if view_cols[2].button("Load view") and selected_view != "(none)":
            saved = views[selected_view]
            for key, value in saved.items():
                st.session_state[f"issue-filter-{key}"] = value
            st.rerun()
    else:
        view_cols[1].caption("No saved views")
    query = st.text_input("Search issues, vendors or record IDs", key="issue-filter-query")
    severity = st.multiselect(
        "Severity", options=["Critical", "High", "Medium", "Low"],
        default=["Critical", "High", "Medium"], key="issue-filter-severity",
    )
    issue_types = sorted(frame.issue_type.unique().tolist())
    selected_types = st.multiselect("Issue type", issue_types, key="issue-filter-types")
    status_filter = st.multiselect(
        "Status", list(("Open", "Investigating", "Explained", "Resolved")),
        default=["Open", "Investigating", "Explained"], key="issue-filter-status",
    )
    if st.button("Save current filters") and view_name.strip():
        views[view_name.strip()] = {
            "query": query,
            "severity": severity,
            "types": selected_types,
            "status": status_filter,
        }
        st.toast(f"Saved view: {view_name.strip()}")
    if query:
        q = query.casefold()
        mask = frame.astype(str).apply(lambda row: row.str.casefold().str.contains(q, regex=False).any(), axis=1)
        frame = frame[mask]
    frame = frame[frame.severity.isin(severity)]
    if selected_types:
        frame = frame[frame.issue_type.isin(selected_types)]
    if status_filter:
        frame = frame[frame.status.isin(status_filter)]
    frame["_severity_order"] = frame.severity.map(SEVERITY_ORDER).fillna(9)
    frame = frame.sort_values(["_severity_order", "rupee_impact"], ascending=[True, False]).drop(columns="_severity_order")
    st.caption(f"{len(frame):,} cases match the current filters.")
    page_size = st.selectbox("Rows per page", [25, 50, 100], index=0)
    page = st.number_input("Page", min_value=1, max_value=max(1, (len(frame) + page_size - 1) // page_size), value=1)
    st.dataframe(frame.iloc[(page - 1) * page_size:page * page_size], use_container_width=True, hide_index=True)
    case_ids = frame.issue_id.tolist()
    if not case_ids:
        st.info("No cases match these filters.")
        return
    case_by_id = {case["issue_id"]: case for case in cases}
    selected_bulk = st.multiselect("Select cases for bulk action", case_ids, key="issue-bulk-selection")
    open_selected = [
        issue_id for issue_id in selected_bulk if case_by_id[issue_id]["status"] == "Open"
    ]
    explained_selected = [
        issue_id for issue_id in selected_bulk if case_by_id[issue_id]["status"] == "Explained"
    ]
    bulk_cols = st.columns(2)
    if bulk_cols[0].button("Start investigation for selected open cases", disabled=not open_selected):
        for issue_id in open_selected:
            transition_case(issue_id, "Investigating", "Bulk investigation started")
        st.rerun()
    if bulk_cols[1].button("Resolve selected explained cases", disabled=not explained_selected):
        for issue_id in explained_selected:
            transition_case(issue_id, "Resolved", "Bulk resolution approved")
        st.rerun()
    selected_id = st.selectbox("Open case", case_ids)
    selected_case = case_by_id[selected_id]
    _case_detail(selected_case, result.tables)


def _data_intake() -> None:
    st.title("Data intake")
    st.write("Drag CSV or Excel files below. Headers are detected automatically; confirm each mapping before use.")
    pdf_ocr = st.checkbox("Enable OCR for scanned PDFs (requires Tesseract)", value=False)
    uploads = st.file_uploader(
        "Upload invoices, bank, ledger, filing and rate tables",
        type=["csv", "xlsx", "xls", "pdf"],
        accept_multiple_files=True,
        key="taxsentinel-upload",
    )
    if "uploaded_tables" not in st.session_state:
        st.session_state["uploaded_tables"] = {}
    if not uploads:
        st.caption("PDF text extraction is opt-in. Image-only PDFs require the local Tesseract executable.")
    for upload in uploads or []:
        with st.expander(f"{upload.name} · {upload.size:,} bytes", expanded=True):
            try:
                frame, extracted_text = read_uploaded_table(upload.name, upload.getvalue(), enable_ocr=pdf_ocr)
            except (ValueError, RuntimeError, ImportError) as exc:
                st.error(str(exc))
                continue
            if frame is None:
                st.text_area("Extracted PDF text (review before use)", extracted_text or "", height=180)
                st.warning("PDF output is text only. Convert it to a structured CSV/Excel table and upload for mapping.")
                continue
            try:
                suggested, scores = detect_table_type(frame)
            except ValueError as exc:
                st.error(str(exc))
                continue
            table_types = list(TABLE_SCHEMAS)
            selected_type = st.selectbox(
                "Detected table type",
                table_types,
                index=table_types.index(suggested),
                key=f"type-{upload.name}",
            )
            st.caption(f"Header match scores: {scores}")
            suggestion = suggest_mapping(frame, selected_type)
            selected_mapping: dict[str, str | None] = {}
            with st.form(f"mapping-{upload.name}"):
                mapping_cols = st.columns(3)
                for index, canonical in enumerate(TABLE_SCHEMAS[selected_type]):
                    possible = ["(not mapped)", *list(frame.columns)]
                    initial = suggestion.get(canonical)
                    default_index = possible.index(initial) if initial in possible else 0
                    choice = mapping_cols[index % 3].selectbox(
                        f"{canonical} ←",
                        possible,
                        index=default_index,
                        key=f"col-{upload.name}-{canonical}",
                    )
                    selected_mapping[canonical] = None if choice == "(not mapped)" else choice
                confirm = st.form_submit_button("Apply mapping and validate")
            if confirm:
                try:
                    standardized, changes = standardize_table(frame, selected_type, selected_mapping)
                    report = validation_report(standardized, selected_type)
                    st.session_state["uploaded_tables"][selected_type] = standardized
                    st.session_state["upload_change_log"] = (
                        st.session_state.get("upload_change_log", []) + changes
                    )
                    st.dataframe(standardized.head(10), use_container_width=True, hide_index=True)
                    if report["ready"]:
                        st.success(f"Validation passed for {report['rows']:,} rows.")
                    else:
                        st.error("; ".join(report["issues"]) or "Validation failed.")
                    if report["issues"]:
                        for message in report["issues"]:
                            st.warning(message)
                    st.caption(f"{len(changes):,} field transformations logged; *_raw columns retain source values.")
                except (ValueError, TypeError) as exc:
                    st.error(f"Mapping failed: {exc}")

    mapped = st.session_state["uploaded_tables"]
    if mapped:
        st.markdown("#### Mapped tables")
        for name, frame in mapped.items():
            st.write(f"**{name}** · {len(frame):,} rows")
        missing = sorted(set(REQUIRED_APP_TABLES) - mapped.keys())
        if missing:
            st.info(f"Upload and map: {', '.join(missing)}")
        elif st.button("Reconcile uploaded tables", type="primary"):
            try:
                _run_tables(mapped, "uploaded")
                st.success("Uploaded records are ready for review.")
            except (ValueError, FileNotFoundError, KeyError):
                pass
    changes = st.session_state.get("upload_change_log", [])
    if changes:
        with st.expander("Transformation log"):
            st.dataframe(pd.DataFrame(changes), use_container_width=True, hide_index=True)


def _anomaly_explorer(result: PipelineResult) -> None:
    st.title("Anomaly explorer")
    cases = [
        case for case in _active_cases(result)
        if case["issue_type"].startswith("anomaly") or case["issue_type"] == "new_vendor_huge_invoice"
    ]
    if not cases:
        st.info("No anomaly rules or models flagged the current records.")
        return
    reason_rows = []
    for case in cases:
        evidence = case.get("evidence", {})
        reason_rows.append({
            "issue_id": case["issue_id"],
            "record_id": case["record_ids"][0] if case.get("record_ids") else "",
            "vendor": case["vendor"],
            "amount": evidence.get("amount", 0),
            "robust_z": evidence.get("robust_z", 0),
            "reasons": " · ".join(evidence.get("reasons", [])),
        })
    st.dataframe(pd.DataFrame(reason_rows), use_container_width=True, hide_index=True)
    fig = px.scatter(
        pd.DataFrame(reason_rows), x="amount", y="robust_z", color="vendor",
        hover_data=["record_id", "reasons"], title="Vendor-normalized amount signals",
    )
    st.plotly_chart(fig, use_container_width=True)
    benford = benford_summary(result.tables["invoices"])
    st.subheader("Benford first-digit review (n ≥ 300)")
    if benford:
        st.dataframe(pd.DataFrame(benford), use_container_width=True, hide_index=True)
        selected_vendor = st.selectbox("Vendor digit distribution", [row["vendor"] for row in benford])
        summary = next(row for row in benford if row["vendor"] == selected_vendor)
        digit_frame = pd.DataFrame({
            "Digit": list(range(1, 10)),
            "Observed": summary["observed_first_digits"],
            "Expected": [round((1 / d), 3) for d in range(1, 10)],
        })
        st.bar_chart(digit_frame.set_index("Digit")[["Observed"]])
    else:
        st.info("No supplier has 300 or more invoices in the loaded records; Benford is not applied.")


def _vendor_risk(result: PipelineResult) -> None:
    st.title("Vendor risk")
    st.caption("Formula: min(100, 100 × severity-weighted issue points ÷ (3 × invoice count)); "
               "Critical=4, High=3, Medium=2, Low=1.")
    vendors = result.vendors
    st.dataframe(vendors, use_container_width=True, hide_index=True)
    if vendors.empty:
        st.info("No vendors are available.")
        return
    selected = st.selectbox("Vendor drill-down", vendors.vendor.tolist())
    selected_rows = result.tables["invoices"][
        result.tables["invoices"].vendor_customer_name.astype(str).eq(selected)
    ]
    case_rows = [
        case for case in _active_cases(result)
        if any(str(record_id) in set(selected_rows.record_id.astype(str)) for record_id in case["record_ids"])
    ]
    st.metric("Computed score", f"{float(vendors.loc[vendors.vendor == selected, 'risk_score'].iloc[0]):.1f}/100")
    st.write(f"{len(selected_rows):,} invoice records · {_format_money(selected_rows.total.sum())} invoice total")
    if case_rows:
        st.dataframe(_issue_table(case_rows), use_container_width=True, hide_index=True)
    else:
        st.success("No detected cases are linked to this vendor.")


def _tax_liability(result: PipelineResult) -> None:
    st.title("Tax liability and ITC")
    totals = result.liability["totals"]
    metrics = st.columns(4)
    metrics[0].metric("Output tax", _format_money(totals["output_tax"]))
    metrics[1].metric("Eligible ITC", _format_money(totals["eligible_itc"]))
    metrics[2].metric("ITC at risk", _format_money(totals["itc_at_risk"]))
    metrics[3].metric("Net after reconciliation", _format_money(totals["net_liability_after"]))
    period_frame = pd.DataFrame(result.liability["periods"])
    if not period_frame.empty:
        st.subheader("Before / after reconciliation")
        waterfall = go.Figure(go.Waterfall(
            name="Liability",
            orientation="v",
            measure=["absolute", "relative", "total"],
            x=["Output tax less booked ITC", "ITC at risk removed", "Net liability after"],
            y=[totals["net_liability_before"], totals["itc_at_risk"], totals["net_liability_after"]],
            connector={"line": {"color": "#64748B"}},
            increasing={"marker": {"color": "#EF4444"}},
            decreasing={"marker": {"color": "#14B8A6"}},
            totals={"marker": {"color": "#4F46E5"}},
        ))
        waterfall.update_layout(height=360, showlegend=False)
        st.plotly_chart(waterfall, use_container_width=True)
        st.dataframe(period_frame, use_container_width=True, hide_index=True)
    st.subheader("ITC at risk by supplier invoice")
    risk_rows = result.liability["itc_at_risk_by_invoice"]
    if risk_rows:
        st.dataframe(pd.DataFrame(risk_rows), use_container_width=True, hide_index=True)
    else:
        st.success("No purchase ITC at risk was calculated.")

    st.subheader("What-if simulator")
    hsn_values = sorted(result.tables["invoices"].HSN.astype(str).unique().tolist())
    hsn = st.selectbox("HSN rate to simulate", hsn_values)
    current = result.tables["invoices"].loc[result.tables["invoices"].HSN.astype(str) == hsn, "tax_rate_pct"]
    current_rate = float(current.median()) if not current.empty else 18.0
    simulated_rate = st.slider("Simulated rate (%)", 0.0, 40.0, float(min(40, max(0, current_rate))), 0.5)
    fixable = [case for case in _active_cases(result) if case["status"] != "Resolved"]
    selected_fixed = st.multiselect(
        "Simulate reviewed issues as fixed",
        options=[case["issue_id"] for case in fixable],
        format_func=lambda issue_id: next(
            f"{case['issue_type']} · {case['vendor']}" for case in fixable if case["issue_id"] == issue_id
        ),
    )
    simulation = calculate_liability(
        result.tables["invoices"],
        result.tables["supplier_filings"],
        result.issues,
        resolved_issue_ids=set(selected_fixed),
        rate_overrides={hsn: simulated_rate},
    )
    sim_cols = st.columns(3)
    sim_cols[0].metric("Simulated net liability", _format_money(simulation["totals"]["net_liability_after"]))
    sim_cols[1].metric("Simulated ITC at risk", _format_money(simulation["totals"]["itc_at_risk"]))
    difference = simulation["totals"]["net_liability_after"] - totals["net_liability_after"]
    sim_cols[2].metric("Change from current", _format_money(difference))

    st.subheader("Illustrative interest / penalty estimator")
    amount = st.number_input("Tax amount (Rs)", min_value=0.0, value=0.0, step=1000.0)
    from_date = st.date_input("From date", value=date.today())
    to_date = st.date_input("To date", value=date.today())
    if to_date >= from_date:
        estimate = estimate_interest_and_penalty(amount, from_date, to_date)
        st.write(f"Estimated interest: {_format_money(estimate['interest'])}")
        st.write(f"Configured penalty: {_format_money(estimate['penalty'])}")
        st.caption(str(estimate["notice"]))


def _safe_chat(result: PipelineResult) -> None:
    st.title("Chat with your books")
    st.caption("Offline, read-only command interpreter. Only fixed report queries are supported; no generated SQL is executed.")
    query = st.text_input("Ask about your reconciliation")
    if not query:
        st.info("Try: “show open issues”, “ITC at risk”, “net liability”, or “top vendor risk”.")
        return
    normalized = query.casefold()
    cases = _active_cases(result)
    if "liability" in normalized or "net tax" in normalized:
        st.write("Computed net liability after reconciliation:", _format_money(result.liability["totals"]["net_liability_after"]))
        st.dataframe(pd.DataFrame(result.liability["periods"]), use_container_width=True, hide_index=True)
    elif "itc" in normalized or "credit at risk" in normalized:
        st.write("Computed ITC at risk:", _format_money(result.liability["totals"]["itc_at_risk"]))
        st.dataframe(pd.DataFrame(result.liability["itc_at_risk_by_invoice"]), use_container_width=True, hide_index=True)
    elif "vendor" in normalized and ("risk" in normalized or "top" in normalized):
        st.dataframe(result.vendors.head(10), use_container_width=True, hide_index=True)
    elif "issue" in normalized or "case" in normalized:
        open_rows = [case for case in cases if case["status"] != "Resolved"]
        st.write(f"Open review cases in the current issue set: {len(open_rows):,}")
        st.dataframe(_issue_table(open_rows), use_container_width=True, hide_index=True)
    else:
        st.warning("That query is not in the read-only whitelist. Ask about issues, ITC risk, liability, or vendor risk.")


def _audit_trail() -> None:
    st.title("Audit trail")
    st.caption("Append-only through the application API; the local SQLite file is not cryptographically tamper-proof.")
    entries = list_audit_log()
    if entries:
        st.dataframe(pd.DataFrame(entries), use_container_width=True, hide_index=True)
    else:
        st.info("Actions, comments, decisions and issue detections will appear here.")
    feedback = list_feedback()
    st.subheader("Stored reviewer feedback")
    if feedback:
        st.dataframe(pd.DataFrame(feedback), use_container_width=True, hide_index=True)
    else:
        st.caption("No review feedback has been recorded.")


def _reports(result: PipelineResult, cases: list[dict[str, Any]]) -> None:
    st.title("Reports")
    st.write("Exports contain the currently loaded records, computed issues, liability and audit events.")
    workbook = build_excel_report(cases, result.liability, result.vendors, list_audit_log())
    pdf = build_pdf_report(cases, result.liability)
    left, right = st.columns(2)
    with left:
        st.download_button(
            "Download multi-sheet Excel",
            workbook,
            file_name="taxsentinel-reconciliation.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
    with right:
        st.download_button(
            "Download CA review PDF",
            pdf,
            file_name="taxsentinel-review.pdf",
            mime="application/pdf",
        )
    st.caption("PDF contains the first 35 issues for a concise review; Excel contains the complete case table.")


def _metrics_page() -> None:
    st.title("Model & metrics")
    path = PROJECT_ROOT / "eval" / "results.json"
    if not path.exists():
        st.info("Evaluation results are not available yet. Run `python tasks.py eval` after training.")
    else:
        results = json.loads(path.read_text(encoding="utf-8"))
        datasets = results.get("datasets", {})
        if datasets:
            split = st.selectbox("Evaluation split", list(datasets), index=0)
            selected = datasets[split]
            detection = selected["issue_detection"]
            overall = detection["overall"]
            kpis = st.columns(4)
            kpis[0].metric("Issue precision", f"{overall['precision']:.1%}")
            kpis[1].metric("Issue recall", f"{overall['recall']:.1%}")
            kpis[2].metric("Issue F1", f"{overall['f1']:.1%}")
            kpis[3].metric(
                "Targets",
                "Met" if detection["targets"]["both_met"] else "Missed",
                help="Target: precision ≥85% and recall ≥90%.",
            )
            st.subheader("Per injected error type")
            per_type = pd.DataFrame.from_dict(detection["per_error_type"], orient="index")
            if not per_type.empty:
                per_type.index.name = "error_type"
                st.dataframe(per_type.reset_index(), use_container_width=True, hide_index=True)
            st.caption(
                "For injected subtypes that share one detector family, precision is the shared-family "
                "precision and recall is measured per injected label. See precision_scope in the table."
            )
            st.subheader("Matcher versus naive exact baseline")
            matcher_table = pd.DataFrame([
                {"matcher": "Calibrated model", **selected["matching"]["model"]},
                {"matcher": "Exact invoice number + amount only",
                 **selected["matching"]["naive_exact_invoice_number_and_amount"]},
            ])
            st.dataframe(matcher_table, use_container_width=True, hide_index=True)
            cols = st.columns(3)
            cols[0].metric("Many-to-many exact group accuracy", f"{selected['many_to_many']['exact_group_accuracy']:.1%}")
            cols[1].metric(
                f"Anomaly precision@{selected['anomaly_precision_at_k']['k']}",
                f"{selected['anomaly_precision_at_k']['precision_at_k']:.1%}",
            )
            cols[2].metric(
                "Net liability error",
                _format_money(selected["liability_error"]["net_liability"]["absolute_error_rupees"]),
            )
            st.dataframe(pd.DataFrame([
                {"measure": name, **detail}
                for name, detail in selected["liability_error"].items()
            ]), use_container_width=True, hide_index=True)
            if split == "test_b":
                st.warning(
                    "Test B conventions differ. `wrong_tax_split` and `invalid_gstin` remain "
                    "separate error types; their reported counts include convention-related mismatches."
                )
            validation_curve = datasets.get("validation", {}).get("matching", {}).get(
                "validation_threshold_curve", []
            )
            if validation_curve:
                st.subheader("Validation-only match threshold simulator")
                threshold = st.slider(
                    "Minimum match confidence",
                    min_value=0.60,
                    max_value=0.99,
                    value=0.90,
                    step=0.01,
                )
                closest = min(validation_curve, key=lambda point: abs(point["threshold"] - threshold))
                live = st.columns(2)
                live[0].metric("Validation precision", f"{closest['precision']:.1%}")
                live[1].metric("Validation recall", f"{closest['recall']:.1%}")
                st.caption(
                    f"At threshold {closest['threshold']:.2f}; this curve is validation-only. "
                    "Test A and Test B are never used to select thresholds."
                )
            with st.expander("Evaluation protocol"):
                st.json(results["protocol"], expanded=False)
                st.json(detection["weakest_error_types"], expanded=False)
    st.subheader("Training and calibration")
    card_path = PROJECT_ROOT / "models" / "model_card.json"
    if card_path.exists():
        st.json(json.loads(card_path.read_text(encoding="utf-8")), expanded=False)
    else:
        st.info("No trained model card is present. Run `python tasks.py train`.")
    feedback = list_feedback()
    st.caption(f"Stored reviewer decisions: {len(feedback):,}.")
    st.warning(
        "Feedback is persisted, but automated feedback-driven retraining is not enabled. "
        "Test A/Test B reviewer decisions are held out and must never enter training."
    )
    if st.button("Retrain on approved training seeds", type="primary"):
        try:
            card = train_pair_model()
            st.success(f"Training completed with {card['training_rows']:,} labeled candidate pairs.")
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))


def _demo_mode() -> None:
    st.title("Guided demo · 5 minutes")
    steps = [
        ("1 · Load", "Load the demo dataset from the landing screen or upload source tables."),
        ("2 · Reconcile", "Show exact, fuzzy and split/bulk matches in the Command Center."),
        ("3 · Investigate", "Open a tax, filing or payment issue and inspect source records and evidence."),
        ("4 · Explain and resolve", "Review the evidence-bound explanation, leave a comment and progress the case status."),
        ("5 · Report", "Compare before/after liability and download the Excel/PDF review package."),
    ]
    for title, detail in steps:
        st.markdown(f"**{title}** — {detail}")
    st.info("All rupee amounts, flags and risk values come from deterministic code; the demo uses offline templates.")


def render() -> None:
    initialize_store()
    with st.sidebar:
        st.markdown("## TaxSentinel")
        st.caption("Intelligent tax reconciliation")
        light_mode = st.toggle("Light mode", value=False)
        if st.session_state.pop("navigate_to_intake", False):
            st.session_state["workspace_page"] = "Data intake"
        default_page = st.session_state.get("workspace_page", NAVIGATION[0])
        page = st.radio("Workspace", NAVIGATION, index=NAVIGATION.index(default_page), key="workspace_page")
        st.divider()
        st.write("Workflow")
        st.markdown("`Upload` → `Map` → `Reconcile` → `Review` → `Report`")
        origin = st.session_state.get("dataset_origin")
        if origin:
            st.caption(f"Loaded source: {origin}")
    _css(light_mode)

    if page == "Data intake":
        _data_intake()
        return

    if page == "Command Center" and _current_result() is None:
        st.markdown(
            "<div class='ts-hero'><h1>TaxSentinel</h1><p>Intelligent GST reconciliation, with every flag explainable and every case reviewable.</p>"
            "<span class='ts-step'>1 Upload</span><span class='ts-step'>2 Map</span>"
            "<span class='ts-step'>3 Reconcile</span><span class='ts-step'>4 Review</span><span class='ts-step'>5 Report</span></div>",
            unsafe_allow_html=True,
        )
        st.write("")
        left, right = st.columns(2)
        with left:
            if st.button("Load demo dataset", type="primary", use_container_width=True):
                try:
                    demo_tables = load_demo_tables("test_a")
                    _run_tables(demo_tables, "Test A (evaluation-only demo)")
                    st.rerun()
                except (ValueError, FileNotFoundError, KeyError):
                    pass
        with right:
            if st.button("Upload your files", use_container_width=True):
                st.session_state["navigate_to_intake"] = True
                st.rerun()
        st.caption("Demo uses a held-out synthetic set for demonstration only. It is never used for model training.")
        return

    if page == "Model & metrics":
        _metrics_page()
        return
    if page == "Demo mode":
        _demo_mode()
        return
    if page == "Audit trail":
        _audit_trail()
        return

    result = _current_result()
    if result is None:
        st.title(page)
        st.info("Load the demo dataset from Command Center or map your tables in Data intake.")
        return
    cases = _active_cases(result)
    if page == "Command Center":
        _command_center(result, cases)
    elif page == "Issues inbox":
        _issues_inbox(result, cases)
    elif page == "Anomaly explorer":
        _anomaly_explorer(result)
    elif page == "Vendor risk":
        _vendor_risk(result)
    elif page == "Tax liability":
        _tax_liability(result)
    elif page == "Chat with your books":
        _safe_chat(result)
    elif page == "Reports":
        _reports(result, cases)
