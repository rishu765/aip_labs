#!/usr/bin/env python3
"""Lab 7 — the observability dashboard, read from local traces.

    streamlit run labs/lab7/dashboard.py

`aip.tracing` writes one JSONL file per run to .aip_traces/. This page reads
them back. It is a teaching-scale stand-in for Langfuse / LangSmith / Phoenix;
the concept -- structured spans with a run id and a parent id -- is identical.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.config import settings  # noqa: E402

st.set_page_config(page_title="Aurora Assistant — Ops", page_icon="📈", layout="wide")
st.title("Aurora Policy Assistant — operations")

runs = sorted(settings.trace_dir.glob("*.jsonl"), reverse=True)
if not runs:
    st.info(f"No traces yet in {settings.trace_dir}. Run some queries first.")
    st.stop()

chosen = st.sidebar.multiselect("runs", [p.stem for p in runs],
                                default=[runs[0].stem])
rows = [json.loads(l) for p in runs if p.stem in chosen
        for l in p.open(encoding="utf-8") if l.strip()]
if not rows:
    st.stop()

df = pd.DataFrame(rows)
df["ts"] = pd.to_datetime(df["ts"], unit="s")

c = st.columns(5)
c[0].metric("spans", len(df))
c[1].metric("total cost", f"${df.get('cost_usd', pd.Series([0])).fillna(0).sum():.4f}")
llm = df[df["name"] == "llm.call"]
c[2].metric("model calls", len(llm))
if len(llm):
    c[3].metric("cache hit rate", f"{llm.get('cached', pd.Series([False])).fillna(False).mean():.0%}")
c[4].metric("errors", int((df.get("status") == "error").sum()))

st.subheader("Latency by stage")
# TODO C3: p50/p95 per span name. This is the table that answers
#          "which stage should I optimise?" -- see Lab 7 Part B4.
stage = (df.groupby("name")["duration_ms"]
         .agg(n="count", p50="median",
              p95=lambda s: s.quantile(0.95), total="sum")
         .sort_values("total", ascending=False))
st.dataframe(stage, use_container_width=True)

st.subheader("Cost over time")
if "cost_usd" in df:
    cum = df.sort_values("ts").assign(cum=lambda d: d["cost_usd"].fillna(0).cumsum())
    st.line_chart(cum.set_index("ts")["cum"])

st.subheader("Errors")
errs = df[df.get("status") == "error"]
st.dataframe(errs[["ts", "name", "error"]] if len(errs) else pd.DataFrame(),
             use_container_width=True)

# TODO C4: add one alert condition and state what you would do when it fires.
#          Candidates: p95 above SLO for 5 minutes; cost per hour above budget;
#          refusal rate doubling (which usually means the index broke).
st.subheader("Operational Alerts & Runbook Checks")
ask_spans = df[df["name"].isin(["http.ask", "http.ask.stream"])]
if len(ask_spans):
    # Alert 1: Refusal Rate Spike
    refusals = ask_spans[ask_spans.get("refused", pd.Series([False])).fillna(False)]
    refusal_rate = len(refusals) / len(ask_spans)
    if refusal_rate >= 0.35:
        st.error(
            f"🚨 CRITICAL ALERT: Refusal Rate Spike ({refusal_rate:.1%}). "
            f"Expected baseline is <20%. A refusal rate doubling indicates the retriever index is broken or missing documents!"
        )
        st.markdown("""
        **Runbook Action When Fired:**
        1. Run regression gate to test retrieval index health: `python labs/lab7/gate.py`.
        2. Check vector index status and chunk count via `GET /health`.
        3. If chunk count is 0 or degraded, rebuild index with `python labs/lab3/search.py` and restart service.
        """)
    else:
        st.success(f"✅ Refusal rate normal ({refusal_rate:.1%}, alert threshold: 35%).")

    # Alert 2: P95 Latency SLO breach
    p95_lat = ask_spans["duration_ms"].quantile(0.95)
    if p95_lat > 6000:
        st.warning(
            f"⚠️ LATENCY ALERT: p95 latency is {p95_lat:.0f} ms (SLO ceiling ≤ 6,000 ms). "
            f"Upstream provider may be experiencing throttling or embedding network latency."
        )
        st.markdown("""
        **Runbook Action When Fired:**
        1. Check provider status and error logs for 429/503 rate-limit retries.
        2. Check cache hit rate in `/metrics`. If low, verify cache disk space.
        3. Activate semantic cache with threshold ≥ 0.95 to absorb repeated query traffic.
        """)
    else:
        st.success(f"✅ p95 latency within SLO ({p95_lat:.0f} ms ≤ 6,000 ms).")
else:
    st.info("No `/ask` requests recorded in selected run(s) yet.")
