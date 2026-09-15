# Lab 1 Report

## Part A - Naive Extractor Failures

`python labs/lab1/v0_naive.py --n 40`

| Failure mode | Count in 40 | Example ticket id | T1 mapping |
|---|---:|---|---|
| Not valid JSON at all | 0 | - | Output parsing |
| JSON wrapped in a markdown fence | 40 | T0054 | Output parsing |
| Extra prose before or after the JSON | 0 | - | Output parsing |
| Valid JSON, missing a required field | 0 | - | Schema failure |
| Category outside the allowed set | 40 | T0054 | Schema failure |
| Urgency as a string instead of an int | 40 | T0054 | Schema failure |
| Policy number invented | 0 | - | Grounding/fabrication |
| Unhandled exception | 0 | - | Reliability/ops |

The two rows that do not map cleanly to the model-output taxonomy are markdown
fencing and unhandled exception. Fencing is an integration contract problem:
the content may be recoverable, but the caller cannot parse it. Unhandled
exception is an engineering reliability failure outside the model answer. The
arc was `0/40` parsed, `40/40` parsed after wrapper stripping, still `0/40`
clean.

That second number is the reason Part B exists. A one-line tolerant parser can
recover the JSON wrapper, but it does not make the record usable: every
recovered object still violated the contract because `urgency` was a string and
`category` was outside the allowed set. Part B adds the missing production
guarantee: schema validation, constrained fields, and repair/fallback paths.

A human reviewer would notice wrong business fields such as category, urgency,
missing required fields, or invented policy numbers because those affect
routing. They would usually not notice markdown fences, extra prose, or parser
exceptions directly, because those fail inside the integration layer before a
usable record reaches the review queue.

## Variant Comparison

| Variant | Split | Schema validity | Field accuracy | Record accuracy | Cost USD | p95 latency ms |
|---|---|---:|---:|---:|---:|---:|
| v0 | dev n=40 | 0.000 | 0.000 | 0.000 | 0.0017 | 1016 |
| B | dev | 1.000 | 0.8976 | 0.4500 | 0.0054 | 1039 |
| C | dev | 1.000 | 0.9167 | 0.5167 | 0.0222 | 1136 |
| C | test | 1.000 | 0.8969 | 0.4583 | 0.0458 | 1163 |

The test run was executed once and saved to `reports/lab1_test.json`. It missed
the 0.90 field-accuracy target by 0.0031, mainly because the Gemini free-tier
quota produced 39 `needs_human_review` fallbacks during the run. Those degraded
to valid records, not crashes, so schema validity and error rate held.

## Deterministic Fields

Part C moved `policy_number`, `contains_pii`, `product`, `language`, and
`escalate` into code. Policy numbers are read only from the live customer
message before quoted `>` history and signature-style tails; this avoids stale
identifiers from old threads.

| Split | Policy number | Contains PII | Product | Language |
|---|---:|---:|---:|---:|
| dev | 60/60 | 60/60 | 60/60 | 60/60 |
| test | 120/120 | 120/120 | 120/120 | 120/120 |

## Test Breakdown

Worst fields first:

| Field | Accuracy |
|---|---:|
| urgency | 0.567 |
| sentiment | 0.842 |
| category | 0.883 |
| escalate | 0.883 |
| contains_pii | 1.000 |
| language | 1.000 |
| policy_number | 1.000 |
| product | 1.000 |

Category confusion matrix, rows gold and columns predicted:

```text
                       billing         claims      complaint    information  policy_change      technical
billing                     15              .              .              .              1              .
claims                       .             21              .              .              .              .
complaint                    .              5             10              1              .              .
information                  .              7              .             15              .              .
policy_change                .              .              .              .             22              .
technical                    .              .              .              .              .             23
```

## Error Analysis

| Cluster | Count | What happened | Proposed fix |
|---|---:|---|---|
| Urgency boundaries | 52 | The model/fallback blurred 1 vs 2 account lookup, 2 vs 3 stuck work, and 4 vs 5 Ombudsman tense. | Add contrastive examples and a deterministic override for add-member requests, upload defects, and "going to" vs "am filing". |
| Sentiment boundary | 19 | Emoji/shouting and defect words made first-time requests look frustrated or angry; signatures with "Thanks" are not satisfaction. | Strip signatures for tone and make prior failure mandatory for `frustrated`. |
| Complaint vs underlying task | 14 | Complaints about claims were sometimes labelled `claims`; happy claim follow-ups were sometimes labelled `claims` instead of `information`. | Keep `complaint` limited to Aurora conduct and treat "confirm whether" follow-ups as information. |

One thing that did not work: relying on the model for `product` and `language`.
They are deterministic in this dataset, so moving them into code improved
auditability and made them 1.000 without spending output tokens.

## Economics

Measured test cost was `$0.045819 / 120 = $0.000381825` per ticket. Using
`1 USD = Rs 95.172` from the 1 Sep 2026 USD/INR rate published by FX-Rate
(https://www.fx-rate.net/historical/?c_input=USD&cp_input=INR):

```text
annual_llm_cost = 0.000381825 * 10,000 * 365 = $1,393.66
annual_llm_cost_inr = 1,393.66 * 95.172 = Rs 132,629
agent_hours = 10,000 * 40 * 365 / 3600 = 40,555.56 hours
annual_agent_cost = 40,555.56 * Rs 300 = Rs 12,166,668
llm_cost_per_ticket_inr = 0.000381825 * 95.172 = Rs 0.036
saved_agent_cost_per_ticket = 40/3600 * 300 = Rs 3.33
break_even_record_accuracy = 0.036 / 3.33 = 1.09%
```

The measured record accuracy, 45.83%, is far above the narrow cost-only
break-even point. The deployment question is therefore not model cost; it is
whether the remaining wrong urgency/category records can be reviewed cheaply
enough and safely enough.
