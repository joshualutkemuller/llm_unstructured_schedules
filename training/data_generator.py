"""
Synthetic collateral schedule generator.

Produces (document_text, ground_truth_json) pairs for training the fine-tuned
model.  Two strategies:

1. TEMPLATE-BASED: Fill randomised values into legal-language sentence
   templates that mimic real CSA / GMRA wording.  Fast, fully controlled,
   but limited linguistic variety.

2. LLM-AUGMENTED: Use a capable model (Claude Sonnet) to paraphrase and
   re-express template documents, producing varied surface forms for the same
   ground truth.  Slower but crucial for robustness.

Usage:
    python -m training.data_generator --count 500 --type IM --output data/synthetic/
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import List, Tuple

from schemas.base import ScheduleType

# ── Value pools ───────────────────────────────────────────────────────────────

_COUNTERPARTIES = [
    ("Goldman Sachs International", "W22LROWP2IHZNBB6K528"),
    ("JPMorgan Chase Bank N.A.", "8I5DZWZKVSZI1NUHU748"),
    ("Deutsche Bank AG", "7LTWFZYICNSX8D621K86"),
    ("BNP Paribas S.A.", "R0MUWSFPU8MPRO8K5P83"),
    ("Barclays Bank PLC", "G5GSEF7VJP5I7OUK5573"),
    ("Citibank N.A.", "E57ODZWZ7FF32TWEFA76"),
    ("HSBC Bank PLC", "MP6I5ZYZBEU3UXPYFY54"),
    ("Société Générale S.A.", "O2RNE8IBXP4R0TD8PX32"),
]

_CUSTODIANS = [
    "BNY Mellon",
    "Euroclear Bank SA/NV",
    "Clearstream Banking S.A.",
    "State Street Bank and Trust Company",
    "JP Morgan Chase Bank N.A. (Custody)",
]

_COLLATERAL_ROWS = [
    ("US Treasury Securities", "AAA", 30, "USD", 0.5),
    ("German Bunds", "AAA", 30, "EUR", 1.0),
    ("UK Gilts", "AA+", 30, "GBP", 1.0),
    ("Agency MBS", "AA", 10, "USD", 4.0),
    ("Investment Grade Corporate Bonds", "A-", 5, "USD", 8.0),
    ("Cash (USD)", None, None, "USD", 0.0),
    ("Cash (EUR)", None, None, "EUR", 0.0),
]

_IM_RATES = ["SOFR flat", "SOFR - 5bps", "Fed Funds flat", "Fed Funds - 10bps"]
_SIMM_VERSIONS = ["SIMM v2.5", "SIMM v2.6", "SIMM v2.7"]


# ── Template-based generators ─────────────────────────────────────────────────


def _pick_counterparty():
    return random.choice(_COUNTERPARTIES)


def _pick_collateral(n: int = 3) -> List[dict]:
    rows = random.sample(_COLLATERAL_ROWS, min(n, len(_COLLATERAL_ROWS)))
    return [
        {
            "asset_class": r[0],
            "min_rating": r[1],
            "max_maturity_years": r[2],
            "currency": r[3],
            "haircut_pct": r[4],
        }
        for r in rows
    ]


def generate_im_sample() -> Tuple[str, dict]:
    """Return (document_text, ground_truth_dict) for one IM CSA."""
    cp_name, cp_lei = _pick_counterparty()
    threshold_a = random.choice([0, 0, 0, 500_000, 1_000_000])
    threshold_b = random.choice([0, 0, 0, 500_000, 1_000_000])
    mta_a = random.choice([100_000, 250_000, 500_000, 1_000_000])
    mta_b = random.choice([100_000, 250_000, 500_000, 1_000_000])
    ia_a = random.choice([0, 0, 1_000_000, 5_000_000])
    ia_b = 0
    custodian = random.choice(_CUSTODIANS)
    collateral = _pick_collateral(3)
    simm = random.choice(_SIMM_VERSIONS)
    rate_usd = random.choice(_IM_RATES)
    settlement = random.choice([1, 2])

    text = f"""\
CREDIT SUPPORT ANNEX
to the Schedule to the 2002 ISDA Master Agreement
(Security Interest – New York Law)

Dated as of January 15, 2024

Party A: Acme Capital Markets LLC
Party B: {cp_name} (LEI: {cp_lei})

Paragraph 13. Elections and Variables.

(b) Threshold. The Threshold for Party A shall be USD {threshold_a:,}.
    The Threshold for Party B shall be USD {threshold_b:,}.

(c) Minimum Transfer Amount. The Minimum Transfer Amount for Party A
    shall be USD {mta_a:,}. The Minimum Transfer Amount for Party B
    shall be USD {mta_b:,}.

(d) Independent Amount. The Independent Amount for Party A shall be
    USD {ia_a:,}. The Independent Amount for Party B shall be zero.

(e) Eligible Collateral.
    The following types of collateral shall be Eligible Credit Support:

    Asset Class                             | Haircut
    ----------------------------------------|--------
""" + "\n".join(
        f"    {row['asset_class']:<40}| {row['haircut_pct']}%"
        for row in collateral
    ) + f"""

(f) Custodian. {custodian} shall act as Custodian for both parties.
    Rehypothecation is not permitted.

(g) IM Calculation Method. The parties shall use {simm} for calculation
    of Initial Margin requirements.

(h) Settlement. Transfers shall be made within T+{settlement} Business Days.
    Notification time is 10:00 AM New York time.

(i) Interest Rate. Interest on USD cash collateral shall accrue at {rate_usd}.

Governing law: New York.
"""

    ground_truth = {
        "schedule_type": "IM",
        "counterparty_name": cp_name,
        "counterparty_lei": cp_lei,
        "governing_law": "NEW_YORK",
        "agreement_type": "ISDA_2016_IM",
        "threshold_party_a": threshold_a,
        "threshold_party_b": threshold_b,
        "minimum_transfer_amount_party_a": mta_a,
        "minimum_transfer_amount_party_b": mta_b,
        "independent_amount_party_a": ia_a,
        "independent_amount_party_b": ia_b,
        "eligible_collateral": collateral,
        "custodian_name": custodian,
        "rehypothecation_permitted": False,
        "im_calculation_method": "SIMM",
        "simm_version": simm,
        "settlement_day": settlement,
        "interest_rate_cash_usd": rate_usd,
    }

    return text, ground_truth


def generate_vm_sample() -> Tuple[str, dict]:
    """Return (document_text, ground_truth_dict) for one VM CSA."""
    cp_name, cp_lei = _pick_counterparty()
    mta_a = random.choice([500_000, 1_000_000])
    mta_b = random.choice([500_000, 1_000_000])
    currencies = random.sample(["USD", "EUR", "GBP"], k=random.randint(2, 3))
    rate_usd = random.choice(["SOFR flat", "SOFR - 5bps", "Fed Funds flat"])
    rate_eur = random.choice(["€STR flat", "€STR - 5bps"])
    valuation_agent = random.choice(["Party A", "Party B", "Calculation Agent"])

    text = f"""\
2016 ISDA Credit Support Annex for Variation Margin
(Title Transfer – English Law)

Dated: March 1, 2024
Party A: Zenith Asset Management Ltd
Party B: {cp_name} (LEI: {cp_lei})

Paragraph 13. Elections.

(a) Base Currency: USD

(b) Threshold: Party A: USD 0. Party B: USD 0.
    (Regulatory VM – thresholds must be zero.)

(c) Minimum Transfer Amount: Party A: USD {mta_a:,}. Party B: USD {mta_b:,}.

(d) Eligible Currencies: {', '.join(currencies)}.

(e) Interest on Cash Collateral:
    USD: {rate_usd}
    EUR: {rate_eur}

(f) Valuation Agent: {valuation_agent}
    Valuation Time: 4:00 PM London time on each Valuation Date.

(g) Settlement Day: T+1 Business Day.

(h) Close-out Netting: Applies.

Governing law: English law.
"""

    ground_truth = {
        "schedule_type": "VM",
        "counterparty_name": cp_name,
        "counterparty_lei": cp_lei,
        "governing_law": "ENGLISH",
        "agreement_type": "ISDA_2016_VM",
        "threshold_party_a": 0,
        "threshold_party_b": 0,
        "minimum_transfer_amount_party_a": mta_a,
        "minimum_transfer_amount_party_b": mta_b,
        "eligible_currencies": currencies,
        "interest_rate_usd": rate_usd,
        "interest_rate_eur": rate_eur,
        "valuation_agent": valuation_agent,
        "regular_settlement_day": 1,
        "close_out_netting_applies": True,
        "base_currency": "USD",
    }

    return text, ground_truth


def generate_repo_sample() -> Tuple[str, dict]:
    """Return (document_text, ground_truth_dict) for one GMRA REPO schedule."""
    cp_name, cp_lei = _pick_counterparty()
    margin_ratio = random.choice([1.02, 1.025, 1.05])
    maintenance = margin_ratio - 0.005
    tri_party = random.choice(_CUSTODIANS[:3])
    repricing = random.choice(["daily", "weekly", "each Business Day"])
    substitution_days = random.choice([2, 3, 5])
    gmra_version = random.choice(["GMRA_2000", "GMRA_2011"])

    securities = random.sample(
        [
            {"asset_class": "UK Gilts", "min_rating": "AA+", "max_maturity_years": 30, "currency": "GBP", "haircut_pct": 1.0},
            {"asset_class": "German Bunds", "min_rating": "AAA", "max_maturity_years": 30, "currency": "EUR", "haircut_pct": 0.5},
            {"asset_class": "US Treasuries", "min_rating": "AAA", "max_maturity_years": 30, "currency": "USD", "haircut_pct": 0.5},
        ],
        k=2,
    )

    text = f"""\
ANNEX TO THE GLOBAL MASTER REPURCHASE AGREEMENT

Dated: February 10, 2024
Seller: {cp_name} (LEI: {cp_lei})
Buyer: Apex Fixed Income Fund

Agreement Version: {gmra_version.replace('_', ' ')}

1. Margin Ratio: {margin_ratio} (i.e., {(margin_ratio-1)*100:.1f}% initial margin)
   Margin Maintenance Level: {maintenance}

2. Eligible Purchased Securities:
""" + "\n".join(
        f"   - {s['asset_class']} ({s['currency']}): haircut {s['haircut_pct']}%"
        for s in securities
    ) + f"""

3. Repricing: {repricing.capitalize()}.
   Repricing notice: 2 business hours.

4. Substitution: Permitted with {substitution_days} Business Days' notice.

5. Tri-party Agent: {tri_party}

6. Net Margin: Applies across all outstanding Transactions.

7. Income: Manufactured payments to be made on the Business Day
   on which the relevant income is paid.

8. Default Interest: SONIA + 200 basis points.

9. Set-off rights apply on close-out.

Governing law: English law.
"""

    ground_truth = {
        "schedule_type": "REPO",
        "counterparty_name": cp_name,
        "counterparty_lei": cp_lei,
        "governing_law": "ENGLISH",
        "agreement_type": gmra_version,
        "initial_margin_ratio": margin_ratio,
        "margin_maintenance_threshold": maintenance,
        "net_margin_applies": True,
        "eligible_securities": securities,
        "repricing_date": repricing,
        "repricing_notice_hours": 2,
        "substitution_permitted": True,
        "substitution_notice_days": substitution_days,
        "tri_party_agent": tri_party,
        "default_interest_rate": "SONIA + 200bps",
        "set_off_rights": True,
    }

    return text, ground_truth


# ── CLI ───────────────────────────────────────────────────────────────────────

_GENERATORS = {
    "IM": generate_im_sample,
    "VM": generate_vm_sample,
    "REPO": generate_repo_sample,
}


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic schedule data")
    parser.add_argument("--count", type=int, default=300, help="Samples per type")
    parser.add_argument(
        "--type", choices=["IM", "VM", "REPO", "ALL"], default="ALL"
    )
    parser.add_argument("--output", default="data/synthetic")
    args = parser.parse_args()

    types = ["IM", "VM", "REPO"] if args.type == "ALL" else [args.type]
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    for stype in types:
        samples = []
        generator = _GENERATORS[stype]
        for _ in range(args.count):
            text, gt = generator()
            samples.append({"input": text, "ground_truth": gt})

        out_path = out_dir / f"{stype.lower()}_synthetic.jsonl"
        with out_path.open("w") as f:
            for s in samples:
                f.write(json.dumps(s) + "\n")
        print(f"Wrote {len(samples)} {stype} samples → {out_path}")


if __name__ == "__main__":
    main()
