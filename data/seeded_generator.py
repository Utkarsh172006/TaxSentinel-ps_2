#!/usr/bin/env python3
"""
GST reconciliation synthetic dataset generator (same file layout as v2).

Usage:
    python generate_dataset.py --seed 2026 --out out_dir
    python generate_dataset.py --seed 7 --out heldout_dir --zip heldout.zip

Everything is synthetic. Not tax advice. Verify GST rates against official
CBIC / GST Council notifications before any real-world use.

Deliberate differences from the v2 files you uploaded (all documented in the
generated README):
  * All vendor GSTINs carry a VALID checksum, so "invalid_gstin" is a real,
    detectable error (in v2 only 2 of 35 vendor GSTINs passed the checksum).
  * The reporting entity is registered in Maharashtra (state code 27), so
    intra- vs inter-state supply (CGST+SGST vs IGST) is derivable from
    party_state. In v2 IGST usage was random, which made wrong_tax_split
    undetectable.
"""
import argparse
import io
import zipfile
from datetime import date, timedelta
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
import pandas as pd

START, END = date(2025, 4, 1), date(2026, 9, 30)
OPEN_AFTER = date(2026, 8, 25)          # invoices after this date may be unpaid
HOME_STATE = "Maharashtra"              # state of the reporting entity (code 27)
N_INVOICES = 3000

STATE_CODE = {"Maharashtra": "27", "Karnataka": "29", "West Bengal": "19", "Haryana": "06",
              "Uttar Pradesh": "09", "Gujarat": "24", "Delhi": "07", "Telangana": "36",
              "Rajasthan": "08", "Madhya Pradesh": "23"}

BIG_COMPANIES = [  # (name, state) -- same 20 anchor names as v2; GSTINs are synthetic
    ("Tata Motors Limited", "Maharashtra"), ("Reliance Industries Limited", "Maharashtra"),
    ("Infosys Limited", "Karnataka"), ("Wipro Limited", "Karnataka"),
    ("Hindustan Unilever Limited", "Maharashtra"), ("Larsen & Toubro Limited", "Maharashtra"),
    ("ITC Limited", "West Bengal"), ("Mahindra & Mahindra Limited", "Maharashtra"),
    ("Asian Paints Limited", "Maharashtra"), ("Maruti Suzuki India Limited", "Haryana"),
    ("HCL Technologies Limited", "Uttar Pradesh"), ("Adani Enterprises Limited", "Gujarat"),
    ("Godrej Consumer Products Limited", "Maharashtra"), ("Bajaj Auto Limited", "Maharashtra"),
    ("UltraTech Cement Limited", "Maharashtra"), ("Tech Mahindra Limited", "Maharashtra"),
    ("Dabur India Limited", "Uttar Pradesh"), ("Pidilite Industries Limited", "Maharashtra"),
    ("Havells India Limited", "Delhi"), ("Dr Reddy's Laboratories Limited", "Telangana"),
]
SURNAMES = ["Sharma", "Verma", "Iyer", "Nair", "Reddy", "Patel", "Mehta", "Shah", "Gupta", "Singh",
            "Kapoor", "Malhotra", "Bose", "Banerjee", "Chatterjee", "Das", "Joshi", "Kulkarni",
            "Deshmukh", "Pillai", "Menon", "Rao", "Naidu", "Chopra", "Bhatia", "Saxena", "Mishra",
            "Tiwari", "Agarwal", "Jain", "Khanna", "Sethi", "Ahuja", "Bhalla", "Dutta", "Ghosh",
            "Mukherjee", "Sinha", "Thakur", "Yadav", "Pandey", "Trivedi", "Desai", "Parekh"]
INDUSTRY = ["Industries", "Enterprises", "Traders", "Logistics", "Textiles", "Chemicals",
            "Engineering", "Foods", "Pharma", "Infotech", "Polymers", "Metals"]
SUFFIX = ["Limited", "Pvt Ltd", "Private Limited"]

# Date-effective rate master (HSN, rate, valid_from, valid_to, description). Same as v2.
RATES_CSV = """HSN,standard_tax_rate_pct,valid_from,valid_to,description
9983,18,2017-07-01,,"Other professional, technical and business services"
9985,18,2017-07-01,,Support services
8471,18,2017-07-01,,Computers and data-processing equipment
8504,18,2017-07-01,,Electrical transformers and static converters
8708,28,2017-07-01,2025-09-21,Motor vehicle parts
8708,18,2025-09-22,,Motor vehicle parts
8517,18,2017-07-01,,Telecom equipment
3926,18,2017-07-01,,Other plastic articles
7308,18,2017-07-01,,Structures and parts of structures
2523,28,2017-07-01,2025-09-21,Cement
2523,18,2025-09-22,,Cement
3004,12,2017-07-01,2025-09-21,Medicaments
3004,5,2025-09-22,,Medicaments
3401,18,2017-07-01,2025-09-21,Soap and cleansing preparations
3401,5,2025-09-22,,Soap and cleansing preparations
4819,18,2017-07-01,,Paper packaging
1006,5,2017-07-01,,Rice
901,5,2017-07-01,,Coffee
2710,18,2017-07-01,,Petroleum oils and preparations
"""

CH = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"


# --------------------------------------------------------------------------- helpers
def r2(x) -> float:
    return float(Decimal(repr(float(x))).quantize(Decimal("0.01"), ROUND_HALF_UP))


def gstin_checksum(g14: str) -> str:
    tot = 0
    for i, c in enumerate(g14):
        v = CH.index(c) * (1 if i % 2 == 0 else 2)
        tot += v // 36 + v % 36
    return CH[(36 - tot % 36) % 36]


def gstin_valid(g: str) -> bool:
    return isinstance(g, str) and len(g) == 15 and g[13] == "Z" and \
        all(c in CH for c in g) and g[14] == gstin_checksum(g[:14])


def money(x: float) -> str:
    return f"{x:,.2f}"


def d2s(d: date) -> str:
    return d.isoformat()


def clip_date(d: date) -> date:
    return min(max(d, START), END)


def tax_split(taxable: float, rate: int, inter: bool):
    if inter:
        return 0.0, 0.0, r2(taxable * rate / 100)
    half = r2(taxable * rate / 200)
    return half, half, 0.0


def load_rates():
    master = pd.read_csv(io.StringIO(RATES_CSV), dtype={"HSN": str})
    table = {}
    for r in master.itertuples():
        vf = date.fromisoformat(r.valid_from)
        vt = date.fromisoformat(r.valid_to) if isinstance(r.valid_to, str) else date(2100, 1, 1)
        table.setdefault(r.HSN, []).append((vf, vt, int(r.standard_tax_rate_pct)))
    return master, table


def rate_on(table, hsn, d):
    for vf, vt, rate in table[hsn]:
        if vf <= d <= vt:
            return rate
    raise ValueError(f"no rate for {hsn} on {d}")


# --------------------------------------------------------------------------- generator
def generate(seed: int):
    rng = np.random.default_rng(seed)
    rate_master, rate_table = load_rates()
    hsn_list = sorted(rate_table)

    # ---- vendors -----------------------------------------------------------
    used_names = {n for n, _ in BIG_COMPANIES}

    def new_company():
        while True:
            style = rng.integers(0, 4)
            a, b, c = rng.choice(SURNAMES, 3, replace=False)
            suf = rng.choice(SUFFIX)
            if style == 0:
                name = f"{a}, {b} and {c} {suf}"
            elif style == 1:
                name = f"{a} & Sons {suf}"
            elif style == 2:
                name = f"{a} {rng.choice(INDUSTRY)} {suf}"
            else:
                name = f"{a}-{b} {suf}"
            if name not in used_names:
                used_names.add(name)
                return name

    def make_gstin(state):
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        pan = "".join(rng.choice(list(letters), 3)) + "C" + rng.choice(list(letters)) + \
              "".join(str(rng.integers(0, 10)) for _ in range(4)) + rng.choice(list(letters))
        g14 = STATE_CODE[state] + pan + str(rng.integers(1, 10)) + "Z"
        return g14 + gstin_checksum(g14)

    states = list(STATE_CODE)
    state_w = np.array([5, 2, 2, 1.5, 2, 1.5, 1, 1.5, 1.5, 1.5])
    vend = [(n, s, False) for n, s in BIG_COMPANIES]
    for _ in range(11):
        vend.append((new_company(), rng.choice(states, p=state_w / state_w.sum()), False))
    for _ in range(4):
        vend.append((new_company(), rng.choice(states, p=state_w / state_w.sum()), True))
    vendors = pd.DataFrame([{"party_id": f"V{i + 1:03d}", "party_name": n,
                             "GSTIN": make_gstin(s), "state": s, "is_new_vendor": nv}
                            for i, (n, s, nv) in enumerate(vend)])
    assert vendors.GSTIN.map(gstin_valid).all()
    vinfo = vendors.set_index("party_id")

    # ---- base invoices -----------------------------------------------------
    w = np.clip(1 + 0.25 * rng.standard_normal(len(vendors)), 0.6, 1.4)
    w[vendors.is_new_vendor.values] = 0.95
    counts = rng.multinomial(N_INVOICES, w / w.sum())
    sales_p = rng.uniform(0.24, 0.40, len(vendors))
    scale = rng.uniform(0.80, 1.25, len(vendors))
    span = (END - START).days
    rows = []
    for vi, (pid, cnt) in enumerate(zip(vendors.party_id, counts)):
        for _ in range(cnt):
            d = START + timedelta(days=int(rng.integers(0, span + 1)))
            hsn = str(rng.choice(hsn_list))
            rate = rate_on(rate_table, hsn, d)
            taxable = r2(np.clip(rng.lognormal(np.log(15400 * scale[vi]), 0.95), 700, 340000))
            inter = vinfo.loc[pid, "state"] != HOME_STATE
            c, s, i = tax_split(taxable, rate, inter)
            rows.append(dict(
                invoice_no=None, document_type="sales" if rng.random() < sales_p[vi] else "purchase",
                date=d, vendor_customer_name=vinfo.loc[pid, "party_name"], GSTIN=vinfo.loc[pid, "GSTIN"],
                party_id=pid, party_state=vinfo.loc[pid, "state"], HSN=hsn, taxable_value=taxable,
                tax_rate_pct=rate, CGST=c, SGST=s, IGST=i, total=r2(taxable + c + s + i),
                is_new_vendor=bool(vinfo.loc[pid, "is_new_vendor"])))
    order = rng.permutation(len(rows))
    rows = [rows[k] for k in order]
    for n, r in enumerate(rows, 1):
        r["record_id"] = f"INVREC{n:05d}"
        r["invoice_no"] = f"INV-{n:05d}"
    cols = ["record_id", "invoice_no", "document_type", "date", "vendor_customer_name", "GSTIN",
            "party_id", "party_state", "HSN", "taxable_value", "tax_rate_pct", "CGST", "SGST", "IGST",
            "total", "is_new_vendor"]
    base = pd.DataFrame(rows)[cols]
    B = {r["record_id"]: r for r in rows}               # base invoices by id
    ids = list(B)
    purchases = [i for i in ids if B[i]["document_type"] == "purchase"]

    # ---- bank (clean): open items, splits, bulks, TDS, single payments ------
    def delay():
        return int(rng.integers(0, 36)) if rng.random() < 0.9 else int(rng.integers(36, 50))

    recent = [i for i in ids if B[i]["date"] > OPEN_AFTER]
    rw = np.array([1 + (B[i]["date"] - OPEN_AFTER).days for i in recent], dtype=float)
    open_ids = set(rng.choice(recent, 40, replace=False, p=rw / rw.sum()))

    taken = set(open_ids)
    elig_pool = [i for i in ids if B[i]["date"] <= date(2026, 7, 31) and i not in taken]
    split_ids = list(rng.choice(elig_pool, 25, replace=False))
    taken |= set(split_ids)

    bulk_groups = []
    tries = 0
    while len(bulk_groups) < 14 and tries < 5000:
        tries += 1
        anchor = str(rng.choice([i for i in elig_pool if i not in taken]))
        a = B[anchor]
        cand = [i for i in elig_pool if i not in taken and i != anchor
                and B[i]["party_id"] == a["party_id"] and B[i]["document_type"] == a["document_type"]
                and abs((B[i]["date"] - a["date"]).days) <= 45]
        if not cand:
            continue
        k = min(len(cand), int(rng.integers(1, 4)))
        grp = [anchor] + [str(x) for x in rng.choice(cand, k, replace=False)]
        bulk_groups.append(sorted(grp, key=lambda x: B[x]["date"]))
        taken |= set(grp)
    assert len(bulk_groups) == 14

    tds_pool = [i for i in purchases if i not in taken]
    tds_ids = set(rng.choice(tds_pool, 45, replace=False))

    bank = []  # dicts without txn_id yet
    sign = lambda i: 1.0 if B[i]["document_type"] == "purchase" else -1.0
    verb = lambda i: "Payment to" if B[i]["document_type"] == "purchase" else "Receipt from"

    split_set = set(split_ids)
    bulk_set = {i for g in bulk_groups for i in g}
    for i in ids:
        if i in open_ids or i in split_set or i in bulk_set:
            continue
        b = B[i]
        tds = r2(b["total"] * 0.02) if i in tds_ids else 0.0
        bank.append(dict(date=clip_date(b["date"] + timedelta(days=delay())),
                         amount=sign(i) * r2(b["total"] - tds),
                         narration=f"{verb(i)} {b['vendor_customer_name']} {b['invoice_no']}",
                         reference=b["invoice_no"], linked_invoice_record_id=i, tds_withheld=tds))
    for i in split_ids:
        b = B[i]
        n = 2 if rng.random() < 0.36 else 3
        cuts = np.sort(rng.uniform(0.2, 0.85, n - 1))
        shares = np.diff(np.concatenate([[0], cuts, [1]]))
        parts = [r2(b["total"] * s) for s in shares[:-1]]
        parts.append(r2(b["total"] - sum(parts)))
        d = b["date"] + timedelta(days=delay())
        for k, p in enumerate(parts, 1):
            bank.append(dict(date=clip_date(d), amount=sign(i) * p,
                             narration=f"{verb(i)} {b['vendor_customer_name']} {b['invoice_no']} (part {k}/{n})",
                             reference=b["invoice_no"], linked_invoice_record_id=i, tds_withheld=0.0))
            d = d + timedelta(days=int(rng.integers(3, 13)))
    for g_no, grp in enumerate(bulk_groups, 1):
        first = B[grp[0]]
        d = max(B[x]["date"] for x in grp) + timedelta(days=delay())
        tot = r2(sum(B[x]["total"] for x in grp))
        word = "Bulk payment to" if first["document_type"] == "purchase" else "Bulk receipt from"
        bank.append(dict(date=clip_date(d), amount=sign(grp[0]) * tot,
                         narration=f"{word} {first['vendor_customer_name']} " + " ".join(B[x]["invoice_no"] for x in grp),
                         reference=f"BULK{g_no:04d}", linked_invoice_record_id="|".join(grp), tds_withheld=0.0))
    bank.sort(key=lambda r: r["date"])
    for n, r in enumerate(bank, 1):
        r["txn_id"] = f"TXN{n:06d}"
    bank_cols = ["txn_id", "date", "amount", "narration", "reference", "linked_invoice_record_id", "tds_withheld"]
    bank_clean = pd.DataFrame(bank)[bank_cols]
    next_txn = len(bank) + 1

    # ---- ledger and supplier filings (clean) ---------------------------------
    ledger = []
    for n, i in enumerate(ids, 1):
        b = B[i]
        pur = b["document_type"] == "purchase"
        ledger.append(dict(entry_id=f"LED{n:06d}", date=b["date"],
                           account="Accounts Payable" if pur else "Accounts Receivable",
                           debit=0.0 if pur else b["total"], credit=b["total"] if pur else 0.0,
                           reference=b["invoice_no"], linked_invoice_record_id=i))
    ledger_clean = pd.DataFrame(ledger)

    filings = []
    for n, i in enumerate(purchases, 1):
        b = B[i]
        filings.append(dict(filing_id=f"FIL{n:06d}", supplier_GSTIN=b["GSTIN"], supplier_name=b["vendor_customer_name"],
                            invoice_no=b["invoice_no"], invoice_date=b["date"], taxable_value=b["taxable_value"],
                            tax_amount=r2(b["CGST"] + b["SGST"] + b["IGST"]), appears_in_supplier_filing=True,
                            filing_period=b["date"].strftime("%Y-%m"), linked_invoice_record_id=i))
    fil_clean = pd.DataFrame(filings)

    # ---- inject errors -------------------------------------------------------
    D = {i: dict(B[i]) for i in ids}                    # dirty invoices
    DB = {r["txn_id"]: dict(r) for r in bank}           # dirty bank rows by txn id
    DL = {r["entry_id"]: dict(r) for r in ledger}       # dirty ledger rows
    DF = {r["filing_id"]: dict(r) for r in filings}     # dirty filing rows
    errs, used = [], set()

    vw = {}
    for k, pid in enumerate(vendors.party_id):
        vw[pid] = 3.5 if vendors.is_new_vendor[k] else (1.8 if 20 <= k < 31 else 0.6)

    def pick(pool, n):
        pool = [x for x in pool if x not in used]
        p = np.array([vw[B[x]["party_id"]] for x in pool])
        chosen = [pool[k] for k in rng.choice(len(pool), n, replace=False, p=p / p.sum())]
        used.update(chosen)
        return chosen

    def log(rid, etype, table, row_id, details):
        errs.append(dict(record_id=rid, error_type=etype, table_name=table, row_id=row_id, details=details))

    protected = open_ids | split_set | bulk_set
    inv_pool = [i for i in ids if i not in protected]
    single_pay = {}
    for t, r in DB.items():
        l = r["linked_invoice_record_id"]
        if "|" not in l:
            single_pay.setdefault(l, []).append(t)
    single_pool = [i for i in ids if i not in protected and len(single_pay.get(i, [])) == 1]
    single_pur = [i for i in single_pool if i in set(purchases)]
    inter = lambda i: B[i]["party_state"] != HOME_STATE

    # bank errors first (most constrained)
    for i in pick(single_pool, 15):
        t = single_pay[i][0]
        del DB[t]
        log(i, "missing_payment", "bank_transactions", t, "Bank transaction removed for invoice")
    for i in pick(single_pur, 18):
        t = single_pay[i][0]
        pct = round(float(rng.uniform(3, 10)), 1)
        short = r2(B[i]["total"] * pct / 100)
        DB[t]["amount"] = r2(DB[t]["amount"] - short)
        log(i, "short_payment_tds", "bank_transactions", t,
            f"Payment short by {pct:.1f}% (₹{money(short)}); may be TDS or underpayment")
    for i in pick(single_pool, 12):
        t = single_pay[i][0]
        delta = float(rng.uniform(100, 1500)) * (1 if rng.random() < 0.5 else -1)
        a = DB[t]["amount"]
        DB[t]["amount"] = r2(np.sign(a) * max(abs(a) + delta, 50))
        log(i, "bank_amount_mismatch", "bank_transactions", t, f"Bank amount adjusted by about ₹{money(abs(delta))}")
    for i in pick([x for x in single_pool if B[x]["date"] >= date(2025, 4, 16)], 12):
        t = single_pay[i][0]
        nd = B[i]["date"] - timedelta(days=int(rng.integers(1, 16)))
        DB[t]["date"] = nd
        log(i, "payment_before_invoice", "bank_transactions", t,
            f"Payment dated {d2s(nd)}, before invoice date {d2s(B[i]['date'])}")
    extra_bank = []
    for i in pick(single_pool, 12):
        t = single_pay[i][0]
        new = dict(DB[t])
        new["txn_id"] = f"TXN{next_txn:06d}"
        next_txn += 1
        new["date"] = clip_date(DB[t]["date"] + timedelta(days=int(rng.integers(1, 11))))
        extra_bank.append(new)
        log(i, "duplicate_payment", "bank_transactions", new["txn_id"], f"Payment posted twice (original {t})")
    fake_nos = set()

    def fake_inv_no():
        while True:
            n = int(rng.integers(90000, 100000))
            if n not in fake_nos:
                fake_nos.add(n)
                return f"INV-{n}"

    for _ in range(12):
        v = vendors.iloc[int(rng.integers(0, 31))]
        no = fake_inv_no()
        row = dict(txn_id=f"TXN{next_txn:06d}", date=START + timedelta(days=int(rng.integers(0, span + 1))),
                   amount=r2(np.clip(rng.lognormal(10.0, 0.7), 3000, 150000)),
                   narration=f"Payment to {v.party_name} {no}", reference=no, linked_invoice_record_id=None, tds_withheld=0.0)
        extra_bank.append(row)
        next_txn += 1
        log(None, "payment_without_invoice", "bank_transactions", row["txn_id"],
            "Bank payment with no matching invoice or ledger entry")

    # invoice-level errors
    def reprice(r, taxable=None, rate=None):
        taxable = r["taxable_value"] if taxable is None else taxable
        rate = r["tax_rate_pct"] if rate is None else rate
        c, s, ig = tax_split(taxable, rate, r["party_state"] != HOME_STATE)
        r.update(taxable_value=taxable, tax_rate_pct=rate, CGST=c, SGST=s, IGST=ig, total=r2(taxable + c + s + ig))

    for i in pick(inv_pool, 25):
        r = D[i]
        comp = "IGST" if inter(i) else "CGST"
        delta = max(r2(r[comp] * rng.uniform(0.05, 0.20)), 2.0)
        r[comp] = r2(r[comp] + delta)
        r["total"] = r2(r["total"] + delta)
        log(i, "miscalculated_tax", "invoices", i, f"Tax component increased by ₹{money(delta)}")
    for i in pick(inv_pool, 18):
        r = D[i]
        old = r["tax_rate_pct"]
        new = int(rng.choice([x for x in (5, 12, 18, 28, 40) if x != old]))
        reprice(r, rate=new)
        log(i, "wrong_tax_rate", "invoices", i, f"Tax rate changed from {old}% to {new}% (HSN {r['HSN']})")
    intra_pool = [x for x in inv_pool if not inter(x)]
    inter_pool = [x for x in inv_pool if inter(x)]
    for i in pick(intra_pool, 10):
        r = D[i]
        r["IGST"] = r2(r["CGST"] + r["SGST"])
        r["CGST"] = r["SGST"] = 0.0
        log(i, "wrong_tax_split", "invoices", i, "CGST+SGST replaced by IGST on intra-state supply")
    for i in pick(inter_pool, 5):
        r = D[i]
        half = r2(r["IGST"] / 2)
        r["CGST"] = r["SGST"] = half
        r["IGST"] = 0.0
        log(i, "wrong_tax_split", "invoices", i, "IGST replaced by CGST+SGST on inter-state supply")
    for i in pick(inv_pool, 12):
        delta = round(float(rng.uniform(1.2, 4.9)), 2) * (1 if rng.random() < 0.5 else -1)
        D[i]["total"] = r2(D[i]["total"] + delta)
        log(i, "rounding_diff", "invoices", i, f"Invoice total off by ₹{abs(delta):.2f} (rounding-type difference)")
    for i in pick(inv_pool, 12):
        delta = r2(rng.uniform(60, 300))
        D[i]["total"] = r2(D[i]["total"] + delta)
        log(i, "amount_mismatch_small", "invoices", i, f"Total increased by ₹{money(delta)}")
    for i in pick(inv_pool, 15):
        delta = r2(rng.uniform(1600, 5800))
        D[i]["total"] = r2(D[i]["total"] + delta)
        log(i, "amount_mismatch_large", "invoices", i, f"Total increased by ₹{money(delta)}")
    for i in pick(inv_pool, 15):
        old = D[i]["date"]
        while True:
            nd = old + timedelta(days=int(rng.integers(1, 31)) * (1 if rng.random() < 0.5 else -1))
            if START <= nd <= END:
                break
        D[i]["date"] = nd
        log(i, "date_shift", "invoices", i, f"Invoice date shifted from {d2s(old)} to {d2s(nd)}")
    for i in pick([x for x in inv_pool if not B[x]["is_new_vendor"]], 15):
        reprice(D[i], taxable=r2(D[i]["taxable_value"] * 8))
        log(i, "anomaly_unusual_spike", "invoices", i, "Taxable value multiplied by 8x versus normal for this vendor")
    med = base.groupby("party_id").taxable_value.median()
    for i in pick([x for x in inv_pool if B[x]["is_new_vendor"]], 15):
        reprice(D[i], taxable=r2(med[B[i]["party_id"]] * rng.uniform(25, 35)))
        log(i, "new_vendor_huge_invoice", "invoices", i, "Large invoice (~30x typical) from a newly onboarded vendor")

    def force_total(i, target):
        r = D[i]
        reprice(r, taxable=r2(target / (1 + r["tax_rate_pct"] / 100)))
        r["total"] = float(target)

    for i in pick(inv_pool, 15):
        target = int(rng.choice([50000, 100000, 200000, 250000, 500000]))
        force_total(i, target)
        log(i, "anomaly_round_number", "invoices", i, f"Round-number total set to ₹{money(target)}")
    for i in pick(inv_pool, 15):
        target = int(rng.choice([49999, 99999, 199999, 499999]))
        force_total(i, target)
        log(i, "anomaly_approval_threshold", "invoices", i,
            f"Total set just below a common approval threshold: ₹{money(target)}")
    for i in pick(inv_pool, 15):
        g = D[i]["GSTIN"]
        kind = int(rng.integers(0, 3))
        if kind == 0:
            bad, why = g[:13] + "X" + g[14], "invalid 14th character"
        elif kind == 1:
            bad, why = g[:14], "truncated to 14 characters"
        else:
            alt = CH[(CH.index(g[14]) + int(rng.integers(1, 35))) % 36]
            bad, why = g[:14] + alt, "wrong checksum character"
        D[i]["GSTIN"] = bad
        log(i, "invalid_gstin", "invoices", i, f"GSTIN changed from {g} to {bad} ({why})")

    def name_variant(name):
        opts = []
        if name.endswith(" Private Limited"):
            opts.append(name.replace(" Private Limited", " Pvt Ltd"))
        if name.endswith(" Limited") and not name.endswith("Private Limited"):
            opts.append(name[:-len(" Limited")] + " Ltd")
        opts += [name.lower(), name.upper()]
        if "&" in name:
            opts.append(name.replace("&", "and"))
        if "," in name:
            opts.append(name.replace(",", ""))
        return str(rng.choice(opts))

    for i in pick(inv_pool, 18):
        old = D[i]["vendor_customer_name"]
        D[i]["vendor_customer_name"] = name_variant(old)
        log(i, "vendor_name_variant", "invoices", i, f"Name changed from '{old}' to '{D[i]['vendor_customer_name']}'")

    def id_variant(no):
        digits = no.split("-")[1]
        cands = []
        if "0" in digits:
            k = digits.index("0")
            cands.append(f"INV-{digits[:k]}O{digits[k + 1:]}")
        if "1" in digits:
            k = digits.index("1")
            cands.append(f"INV-{digits[:k]}I{digits[k + 1:]}")
        for k in range(len(digits) - 1):
            if digits[k] != digits[k + 1]:
                cands.append(f"INV-{digits[:k]}{digits[k + 1]}{digits[k]}{digits[k + 2:]}")
                break
        cands.append(f"INV{digits}")
        return str(rng.choice(cands))

    for i in pick(inv_pool, 16):
        old = D[i]["invoice_no"]
        D[i]["invoice_no"] = id_variant(old)
        log(i, "invoice_id_variant", "invoices", i, f"Invoice number changed from '{old}' to '{D[i]['invoice_no']}'")

    dups, dup_n = [], N_INVOICES
    for kind, n in (("exact", 15), ("near", 15)):
        for i in pick(inv_pool, n):
            dup_n += 1
            rid = f"DUP{dup_n:05d}"
            row = dict(B[i])
            row["record_id"] = rid
            if kind == "exact":
                log(rid, "duplicate_exact", "invoices", rid, f"Exact duplicate of {i}")
            elif rng.random() < 0.5:
                row["invoice_no"] = B[i]["invoice_no"] + "A"
                log(rid, "duplicate_near", "invoices", rid, f"Near duplicate of {i} (invoice number suffixed with A)")
            else:
                row["date"] = clip_date(B[i]["date"] + timedelta(days=int(rng.choice([-2, -1, 1, 2]))))
                log(rid, "duplicate_near", "invoices", rid, f"Near duplicate of {i} (date shifted by 1-2 days)")
            dups.append(row)

    # ledger errors
    led_of = {r["linked_invoice_record_id"]: k for k, r in DL.items()}
    for i in pick(ids, 20):
        k = led_of[i]
        del DL[k]
        log(i, "missing_ledger_entry", "ledger", k, "Ledger entry removed for invoice")
    for i in pick([x for x in ids if B[x]["date"] <= date(2026, 9, 5) and x in led_of and led_of[x] in DL], 15):
        k = led_of[i]
        nd = B[i]["date"] + timedelta(days=int(rng.integers(3, 21)))
        DL[k]["date"] = nd
        log(i, "ledger_date_gap", "ledger", k, f"Ledger posted {d2s(nd)} vs invoice {d2s(B[i]['date'])}")
    extra_led = []
    for n in range(12):
        row = dict(entry_id=f"LED{N_INVOICES + n + 1:06d}",
                   date=START + timedelta(days=int(rng.integers(0, span + 1))), account="Accounts Payable",
                   debit=0.0, credit=r2(rng.uniform(30000, 95000)), reference=fake_inv_no(),
                   linked_invoice_record_id=None)
        extra_led.append(row)
        log(None, "ledger_without_source", "ledger", row["entry_id"], "Ledger entry with no source invoice")

    # supplier-filing errors
    fil_of = {r["linked_invoice_record_id"]: k for k, r in DF.items()}
    for i in pick(purchases, 45):
        k = fil_of[i]
        DF[k]["appears_in_supplier_filing"] = False
        log(i, "supplier_not_filed", "supplier_filings", k, "Supplier invoice not in filed statement; ITC at risk")
    for i in pick(purchases, 15):
        k = fil_of[i]
        pct = round(float(rng.uniform(3, 14)) * (1 if rng.random() < 0.6 else -1), 1)
        DF[k]["tax_amount"] = r2(DF[k]["tax_amount"] * (1 + pct / 100))
        log(i, "gstr2b_tax_mismatch", "supplier_filings", k, f"Supplier-reported tax differs from books by {pct:+.1f}%")
    for i in pick([x for x in purchases if B[x]["date"] <= date(2026, 8, 31)], 12):
        k = fil_of[i]
        old = DF[k]["filing_period"]
        y, m = int(old[:4]), int(old[5:])
        new = f"{y + (m == 12)}-{(m % 12) + 1:02d}"
        DF[k]["filing_period"] = new
        log(i, "wrong_tax_period", "supplier_filings", k, f"Filed in period {new} instead of {old}")
    extra_fil = []
    for n in range(12):
        v = vendors.iloc[int(rng.integers(0, 31))]
        d = START + timedelta(days=int(rng.integers(0, span + 1)))
        taxable = r2(np.clip(rng.lognormal(9.6, 0.8), 1500, 120000))
        rate = int(rng.choice([5, 18, 18, 18]))
        row = dict(filing_id=f"FIL{len(filings) + n + 1:06d}", supplier_GSTIN=v.GSTIN, supplier_name=v.party_name,
                   invoice_no=fake_inv_no(), invoice_date=d, taxable_value=taxable, tax_amount=r2(taxable * rate / 100),
                   appears_in_supplier_filing=True, filing_period=d.strftime("%Y-%m"), linked_invoice_record_id=None)
        extra_fil.append(row)
        log(None, "in_gstr2b_not_in_books", "supplier_filings", row["filing_id"],
            "Supplier filed invoice that is missing from the books")

    # ---- assemble tables -------------------------------------------------------
    inv_dirty = pd.DataFrame(list(D.values()) + dups)[cols]
    bank_dirty = pd.DataFrame(list(DB.values()) + extra_bank)[bank_cols]
    led_cols = list(ledger_clean.columns)
    led_dirty = pd.DataFrame(list(DL.values()) + extra_led)[led_cols]
    fil_dirty = pd.DataFrame(list(DF.values()) + extra_fil)[list(fil_clean.columns)]
    errors = pd.DataFrame(errs)[["record_id", "error_type", "table_name", "row_id", "details"]]

    ch = []
    for g_no, grp in enumerate(bulk_groups, 1):
        ch.append(dict(record_id="|".join(grp), challenge_type="bulk_payment", table_name="bank_transactions",
                       row_id=f"BULK{g_no:04d}",
                       details=f"{len(grp)} invoices settled by one payment; should reconcile as many-to-one"))
    n_inst = bank_clean[bank_clean.linked_invoice_record_id.isin(split_ids)].groupby("linked_invoice_record_id").size()
    for i in split_ids:
        ch.append(dict(record_id=i, challenge_type="split_payment", table_name="bank_transactions",
                       row_id=B[i]["invoice_no"],
                       details=f"Invoice settled in {n_inst[i]} instalments; should reconcile as one-to-many"))
    for i in sorted(open_ids):
        ch.append(dict(record_id=i, challenge_type="open_item_not_due", table_name="bank_transactions", row_id=None,
                       details=f"Recent invoice (dated after {d2s(OPEN_AFTER)}) not yet paid; legitimately open, not an error"))
    challenges = pd.DataFrame(ch)

    return dict(vendors=vendors, rate_master=rate_master, inv_clean=base, inv=inv_dirty,
                bank_clean=bank_clean, bank=bank_dirty, ledger_clean=ledger_clean, ledger=led_dirty,
                fil_clean=fil_clean, fil=fil_dirty, errors=errors, challenges=challenges)


# --------------------------------------------------------------------------- output
def readme(t, seed):
    e = t["errors"]
    by_table = {tb: ", ".join(f"{k} ({v})" for k, v in g.error_type.value_counts().items())
                for tb, g in e.groupby("table_name")}
    n_aff = len(set(e.record_id.dropna()))
    return f"""GST RECONCILIATION SYNTHETIC DATASET (v2 layout, seed {seed})
Period: {d2s(START)} to {d2s(END)} | {N_INVOICES:,} base invoices | {len(t['vendors'])} parties | all data is synthetic
Not tax advice. GSTINs are GSTIN-shaped placeholders, not verified identities.

FILES
  *_clean.csv            known-good baseline (zero discrepancies)
  invoices.csv, ledger.csv, bank_transactions.csv, supplier_filings.csv
                         corrupted versions = input to your engine
  tax_rates.csv          date-effective rate master (HSN, rate, valid_from, valid_to)
  vendor_master.csv      {len(t['vendors'])} parties (V032-V035 are new vendors)
  injected_errors.csv    ANSWER KEY: record_id, error_type, table_name, row_id, details
  match_challenges.csv   LEGITIMATE tricky matches (NOT errors): split_payment, bulk_payment, open_item_not_due
  Keep both key files hidden from the detector. Use them only for scoring.

HOW THIS DIFFERS FROM THE PREVIOUS v2 FILES (same columns, same files, same error types and counts)
  - New random seed: different invoices, amounts, dates, vendors and error positions.
  - Reporting entity is registered in Maharashtra (state code 27). Intra-state supply
    (party_state = Maharashtra) uses CGST+SGST; every other state uses IGST. This makes
    wrong_tax_split detectable from the data alone.
  - Every vendor GSTIN in the clean data has a valid checksum (15 chars, 14th char 'Z'), so
    a regex + checksum validator produces no false positives; invalid_gstin rows are real failures.
  - Rates follow the 22 Sep 2025 GST rationalisation (28->18: HSN 8708, 2523; 12->5: HSN 3004;
    18->5: HSN 3401). Invoices before that date use old rates. Verify against official notifications.

CONVENTIONS
  - bank amount: purchases positive (payment out), sales negative (receipt in); abs(amount) = invoice total - tds_withheld
  - tds_withheld = 2% of invoice total on 45 purchases (legit, not an error)
  - ledger: purchases credit Accounts Payable, sales debit Accounts Receivable
  - IGST for inter-state, CGST+SGST (equal halves) for intra-state; tolerance for tax checks: Rs 1
  - supplier_filings (GSTR-2B style) exist for purchases only; appears_in_supplier_filing=False means ITC at risk
  - Bulk payments: one bank row, linked ids pipe-separated (INVREC1|INVREC2), reference BULKnnnn
  - Split payments: one invoice -> 2-3 bank rows, narration ends "(part k/n)"
  - Invoices dated after {d2s(OPEN_AFTER)} may be unpaid (open items). They are listed in match_challenges.csv, so do not flag them as missing payments
  - DUP##### record_ids are injected duplicate invoices
  - Dirty rows keep the clean ledger/bank/filing values, so invoice-side mutations show up as cross-table mismatches

INJECTED ERRORS ({len(e)} rows, ~{n_aff / N_INVOICES * 100:.0f}% of invoices affected)
""" + "".join(f"  {tb}: {txt}\n" for tb, txt in by_table.items()) + f"""
TABLE COLUMNS
  invoices: {', '.join(t['inv'].columns)}
  bank_transactions: {', '.join(t['bank'].columns)}
  ledger: {', '.join(t['ledger'].columns)}
  supplier_filings: {', '.join(t['fil'].columns)}
  linked_invoice_record_id is a hint for building ground truth; your engine should match WITHOUT it. Orphan rows have it blank.
"""


def write_all(t, out: Path, seed: int, zip_path=None):
    folder = out / "gst_reconciliation_dataset"
    folder.mkdir(parents=True, exist_ok=True)

    def w(df, name):
        df = df.copy()
        for c in df.columns:
            if c in ("date", "invoice_date"):
                df[c] = df[c].map(d2s)
        df.to_csv(folder / name, index=False, float_format="%.2f")

    w(t["inv_clean"], "invoices_clean.csv"); w(t["inv"], "invoices.csv")
    w(t["bank_clean"], "bank_transactions_clean.csv"); w(t["bank"], "bank_transactions.csv")
    w(t["ledger_clean"], "ledger_clean.csv"); w(t["ledger"], "ledger.csv")
    w(t["fil_clean"], "supplier_filings_clean.csv"); w(t["fil"], "supplier_filings.csv")
    w(t["vendors"], "vendor_master.csv"); w(t["errors"], "injected_errors.csv")
    w(t["challenges"], "match_challenges.csv")
    (folder / "tax_rates.csv").write_text(RATES_CSV, encoding="utf-8")
    (folder / "README.txt").write_text(readme(t, seed), encoding="utf-8")
    if zip_path:
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for f in sorted(folder.iterdir()):
                z.write(f, f"gst_reconciliation_dataset/{f.name}")
    return folder


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--out", default="generated")
    ap.add_argument("--zip", default=None, help="optional zip file path")
    a = ap.parse_args()
    tables = generate(a.seed)
    folder = write_all(tables, Path(a.out), a.seed, a.zip)
    print(f"written to {folder}")
    print(tables["errors"].error_type.value_counts().to_string())
    print("total injected:", len(tables["errors"]))
