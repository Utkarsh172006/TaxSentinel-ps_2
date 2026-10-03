from __future__ import annotations

from io import BytesIO
from typing import Any

import pandas as pd


def build_excel_report(
    cases: list[dict[str, Any]],
    liability: dict[str, Any],
    vendors: pd.DataFrame,
    audit: list[dict[str, Any]],
) -> bytes:
    """Create an in-memory multi-sheet review workbook."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        pd.DataFrame(cases).to_excel(writer, sheet_name="Issues", index=False)
        pd.DataFrame(liability.get("periods", [])).to_excel(writer, sheet_name="Liability", index=False)
        pd.DataFrame(liability.get("itc_at_risk_by_invoice", [])).to_excel(
            writer, sheet_name="ITC at risk", index=False
        )
        vendors.to_excel(writer, sheet_name="Vendor risk", index=False)
        pd.DataFrame(audit).to_excel(writer, sheet_name="Audit trail", index=False)
    return buffer.getvalue()
