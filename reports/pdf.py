from __future__ import annotations

from io import BytesIO
from typing import Any

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def build_pdf_report(
    cases: list[dict[str, Any]],
    liability: dict[str, Any],
) -> bytes:
    """Create a concise case/liability PDF that can be shared with an advisor."""
    buffer = BytesIO()
    document = SimpleDocTemplate(buffer, pagesize=A4, title="TaxSentinel Reconciliation Report")
    styles = getSampleStyleSheet()
    total = liability.get("totals", {})
    story = [
        Paragraph("TaxSentinel reconciliation summary", styles["Title"]),
        Paragraph("Not tax advice. Verify rates and eligibility against official GST Council/CBIC notifications.",
                  styles["BodyText"]),
        Spacer(1, 12),
        Paragraph("Liability summary", styles["Heading2"]),
        Table([
            ["Metric", "Amount (Rs)"],
            ["Output tax", f"{total.get('output_tax', 0):,.2f}"],
            ["Booked purchase ITC", f"{total.get('booked_purchase_itc', 0):,.2f}"],
            ["Eligible ITC", f"{total.get('eligible_itc', 0):,.2f}"],
            ["ITC at risk", f"{total.get('itc_at_risk', 0):,.2f}"],
            ["Net liability after reconciliation", f"{total.get('net_liability_after', 0):,.2f}"],
        ]),
        Spacer(1, 12),
        Paragraph(f"Open review cases: {sum(case.get('status') != 'Resolved' for case in cases)}", styles["Heading2"]),
    ]
    rows = [["Issue", "Severity", "Vendor", "Impact (Rs)", "Status"]]
    rows.extend([
        [
            str(case.get("issue_type", "")),
            str(case.get("severity", "")),
            str(case.get("vendor", ""))[:32],
            f"{float(case.get('rupee_impact', 0)):,.2f}",
            str(case.get("status", "")),
        ]
        for case in cases[:35]
    ])
    table = Table(rows, repeatRows=1)
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1E2A4A")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCD3E0")),
        ("FONTSIZE", (0, 0), (-1, -1), 8),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    story.append(table)
    document.build(story)
    return buffer.getvalue()
