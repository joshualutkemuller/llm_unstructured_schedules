# ExtractionAgent — Instructions

## Purpose
The ExtractionAgent runs LLM-based field extraction on pre-classified,
chunked collateral schedule text. It produces a structured dict of field
values with per-field confidence scores.

It does **not** validate the schema or write to any downstream system —
that is the ValidationAgent's responsibility.

---

## Skills

| Skill | Description |
|-------|-------------|
| `chunk_text` | Pre-chunk long text if not already chunked by IngestAgent. |
| `extract_fields` | Call the LLM extraction prompt for a given schedule type. Returns all schema fields with `value`, `confidence`, and `source_text` per field. |
| `get_low_confidence` | Filter extracted fields by confidence threshold (default 0.7) to surface candidates for human review. |

---

## Field Coverage by Schedule Type

### IM (Initial Margin CSA)
Threshold (A/B), MTA (A/B), Independent Amount (A/B), Eligible Collateral table,
Custodian, Rehypothecation, Interest rates (USD/EUR/GBP), Valuation Agent,
SIMM version, IM Calculation Method, Dispute Resolution period,
Settlement Day, Notification Time, Governing Law, Counterparty (name + LEI).

### VM (Variation Margin CSA)
Threshold (A/B=0), MTA (A/B), Eligible Currencies, Interest rates (USD/EUR/GBP/JPY),
Interest netting, Valuation Agent, Valuation Time, Close-out netting,
Delivery Amount Floor, Settlement Day, Netting Set ID, Covered Transactions,
Governing Law, Counterparty (name + LEI).

### REPO (GMRA)
Initial Margin Ratio, Margin Maintenance Threshold, Net Margin flag,
Margin Call Method, Eligible Securities table, Concentration Limits,
Repricing Date, Repricing Notice Hours, Settlement Lag, Substitution flag,
Substitution Notice Days, Income Payment Method, Default Interest Rate,
Set-off Rights, Tri-party Agent, Pricing Rate Basis, Governing Law,
Counterparty (name + LEI).

---

## Decision Logic

```
receive text + schedule_type
    │
    ├─ schedule_type unknown? → ask user to clarify or run IngestAgent first
    │
    ▼
word_count > 2500?
    ├─ YES → chunk_text(text)
    └─ NO  → use as single chunk
    │
    ▼
for each chunk:
    extract_fields(chunk, schedule_type)
    │
    ├─ JSON error? → retry once with smaller chunk
    └─ success → accumulate fields (highest confidence per field wins)
    │
    ▼
get_low_confidence(merged_fields, threshold=0.7)
    │
    ▼
report summary
```

---

## Confidence Scoring

| Range | Meaning | Action |
|-------|---------|--------|
| 0.9 – 1.0 | High — verbatim match in source | Auto-accept |
| 0.7 – 0.9 | Medium — inferred with context | Accept, spot-check |
| 0.5 – 0.7 | Low — uncertain | Flag for human review |
| 0.0 – 0.5 | Very low / absent | Human must supply |

---

## Example Tasks

```bash
# Extract from a pre-classified IM document
python agent/extraction_agent/run.py \
    --file tests/fixtures/sample_im.txt \
    --type IM

# Extract from raw text piped in
echo "CREDIT SUPPORT ANNEX..." | \
    python agent/extraction_agent/run.py --stdin --type VM

# Lower the review threshold to 0.85 for high-stakes counterparties
python agent/extraction_agent/run.py \
    --file path/to/csa.pdf \
    --type IM \
    --threshold 0.85
```
