# Lab 2 experiment status

The experiment harness and seven configurations are implemented. The complete
dev grid is pending because the configured Gemini `MAIN` free-tier model returned
`RESOURCE_EXHAUSTED` after reaching its daily limit of 20 requests. A partial
grid would make the prompt and model comparisons misleading, so no results table,
annual cost recommendation, or `reports/lab2_grid.json` is claimed here yet.

## Few-shot selection

| Dev ticket | What the example teaches |
|---|---|
| T0097 | A double debit is billing even when the customer threatens the Ombudsman. |
| T0054 | Agent mis-selling is a conduct complaint, and an absent policy number is null. |
| T0200 | Transliterated Hindi mixed with English still counts as `hi-en`; a settlement deduction belongs to claims. |
| T0029 | A satisfied customer can ask a routine information question after a claim settles. |
| T0238 | An `SR-*` reply reference is not an `AUR-*` policy number, and quoted reply history is not the live request. |
| T0010 | A formal claim rejection with a same-day deadline reaches urgency 5. |

Selecting labelled examples from dev and then scoring those same examples is
answer leakage. The runner holds out all six selected ticket IDs from every
configuration whenever a few-shot variant is in the dev comparison. All seven
rows therefore use the same remaining 54 tickets. The test split is reserved
for final validation and was not used to choose examples or tune variants.

Run the complete experiment when the provider limit resets:

```bash
python labs/lab2/grid.py --all --split dev --save reports/lab2_grid.json
```

The runner reports record and field accuracy, schema validity, repair and
fallback rates, estimated deployment cost including cached calls, and ticket
latency. It stops if more than 20% of a variant falls back to heuristics, so
provider failures cannot silently look like model results. The cascade also
records the small-model answer and sample agreement to assess whether its
disagreement trigger predicts errors.
