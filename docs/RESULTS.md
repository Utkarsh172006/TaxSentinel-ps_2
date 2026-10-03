# TaxSentinel evaluation results

**Result: the discrepancy-detector targets were not met.** Measurements below are from `python tasks.py eval`; full per-error-type counts and the validation threshold curve are in [`../eval/results.json`](../eval/results.json). Train seeds 11–13 were used only for fitting, validation is seed 14, and Test A/Test B were excluded from training and threshold selection. Each detector run received inference-safe raw tables; scoring labels and clean liability references were opened only after inference.

## Overall comparison

| Split | Issue detector precision | Issue detector recall | Issue detector F1 | Calibrated matcher P/R/F1 | Naive exact matcher P/R/F1 | Split/bulk exact-group accuracy | Anomaly precision@k |
|---|---:|---:|---:|---:|---:|---:|---:|
| Validation (seed 14) | 29.6% | 89.8% | 44.5% | 98.6 / 99.1 / 98.8% | 100.0 / 90.9 / 95.2% | 64.1% (k=39) | 66.7% (k=60) |
| Test A | 30.0% | 91.1% | 45.1% | 98.6 / 99.1 / 98.9% | 100.0 / 91.1 / 95.3% | 64.1% (k=39) | 66.7% (k=60) |
| Test B | 7.6% | 88.3% | 14.0% | 98.8 / 99.1 / 98.9% | 100.0 / 94.5 / 97.2% | 64.1% (k=39) | 51.7% (k=60) |

Issue metrics are event-level micro scores over mapped detector families. Where multiple injected labels share a generic detector family (for example, the several payment-amount labels), per-label precision uses that shared-family precision and recall is measured against that label’s rows; `precision_scope` in the JSON makes this explicit. This avoids counting one generic alert repeatedly as several false-positive events. The model/naive rows instead score bank invoice-payment links against the link answer key.

## Test B convention-sensitive results

| Error type | TP | FP | FN | Precision | Recall |
|---|---:|---:|---:|---:|---:|
| `wrong_tax_split` | 7 | 1,583 | 8 | 0.44% | 46.7% |
| `invalid_gstin` | 14 | 2,843 | 1 | 0.49% | 93.3% |

These categories are intentionally not merged. Their large false-positive counts reflect applying the specified Maharashtra and checksum conventions to a held-out set with different conventions.

## Tax liability error against clean references

| Split | Net liability absolute error | Net liability error | ITC-at-risk absolute error | ITC-at-risk error |
|---|---:|---:|---:|---:|
| Validation | Rs 1,391,414.80 | 33.5% | Rs 2,269,348.83 | Undefined (clean reference is Rs 0) |
| Test A | Rs 1,057,267.03 | 27.9% | Rs 2,103,704.47 | Undefined (clean reference is Rs 0) |
| Test B | Rs 932,053.76 | 29.2% | Rs 1,206,148.88 | 18.1% |

The reference is recalculated from each split’s isolated clean invoice and supplier-filing tables. A percentage is not meaningful when the clean ITC-at-risk reference is zero.

## Weak categories and causes

- **Vendor-name variants:** no true rows were detected in these runs. The current comparison primarily targets invoice-to-filing names and normalizes equivalent company suffix/case variants.
- **Anomalies:** broad vendor robust-z and Isolation Forest signals create many unrelated alerts; rule signals also miss anomalies when vendor history is sparse or distribution is skewed. Test A anomaly precision@60 is 66.7%, not a claim of high overall anomaly precision.
- **Test B `wrong_tax_split` and `invalid_gstin`:** these are reported as separate error types as required. The detector applies Maharashtra-state split rules and the strict 15-character/checksum GSTIN rule, while Test B uses different conventions; this results in many convention-related flags and low precision. Do not interpret Test B’s scores as equivalent to Test A.
- **Liability / ITC:** the large gap versus clean references shows that issue-to-eligibility logic still overstates risk and needs further validation.
- **Many-to-many:** exact invoice/payment group reconstruction is 64.1%; this remains a concrete matching weakness despite high aggregate link precision/recall.

The requested detector thresholds are precision ≥85% and recall ≥90%. The validation run missed both (29.6% precision and 89.8% recall); Test A met recall but not precision; Test B met neither. Do not present the system as production-ready or as meeting the accuracy goal. Rates, eligibility, issue disposition and all proposed corrections require qualified review. This is not tax advice.
