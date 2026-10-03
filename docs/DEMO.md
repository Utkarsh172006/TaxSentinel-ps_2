# TaxSentinel · 5-minute walkthrough

## Before the demo

From the `taxsentinel` folder, use Python 3.11 and install the pinned dependencies:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python tasks.py data
python tasks.py train
python tasks.py app
```

Open the local Streamlit URL printed in the terminal. `data` needs the provided `files.zip` and `reconciliation_dataset.zip`. `train` uses only seeds 11–13. Loading Test A in the app is for demonstration only; neither held-out test is used for fitting or threshold selection.

## Walkthrough

1. **0:00–0:40 · Command Center.** Select **Load demo dataset**. Point out the actual match rate, open-case count, issue impact, ITC at risk and net liability. The progress messages show candidate matching, detection, liability and review preparation. No answer-key counts or hard-coded risk scores are displayed.
2. **0:40–1:20 · Matching flow.** Read the invoice → payment → books → filing Sankey. Explain that the matching view includes exact, fuzzy and bounded split/bulk paths; the demo data remains inference-safe.
3. **1:20–2:30 · Investigate.** Open **Issues inbox**, select a high-severity case and inspect the source records, evidence, confidence, impact and offline explanation. Advance Open → Investigating → Explained, add a comment, and record an accept/reject decision. These actions are stored in SQLite and visible in **Audit trail**.
4. **2:30–3:20 · Tax and vendor review.** Open **Tax liability** for monthly values, ITC at risk and the liability waterfall. Change the HSN what-if rate or simulate selected issues as fixed; the totals recompute from the business records. Open **Vendor risk** and show the displayed score formula.
5. **3:20–4:10 · Anomalies and metrics.** Open **Anomaly explorer** for reason-coded findings. In **Model & metrics**, compare the calibrated matcher with invoice-number-plus-amount-only baseline and use the validation-only confidence slider. Mention that current discrepancy detector targets are missed; use the saved per-error-type results rather than claiming success.
6. **4:10–5:00 · Report.** Download the multi-sheet Excel and concise CA review PDF. Finish by pointing to the audit log and the not-tax-advice / official-rate verification notice.

## Honest feature notes

- Explanations are offline templates. No LLM adapter is configured, and numeric outputs are never authored by a language model.
- Upload mapping and CSV/Excel validation are supported. PDF extraction is opt-in text extraction; image-only files require local Tesseract and do not become structured accounting rows automatically.
- Review decisions are persisted, but feedback-driven model retraining is not enabled. The training action uses only approved training seeds.
- The app does not create or post accounting entries. A case may show the measured adjustment amount, but account mapping and any journal/credit note require reviewer preparation.
