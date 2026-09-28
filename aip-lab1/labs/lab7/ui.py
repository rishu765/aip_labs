#!/usr/bin/env python3
"""Lab 7 — Streamlit front end.

    streamlit run labs/lab7/ui.py

Requires the service to be running:
    uvicorn labs.lab7.service:app --port 8000

The one non-negotiable UI requirement: **citations must be expandable to show
the source text.** Grounding the user cannot check is decoration.
"""
from __future__ import annotations

import json
from pathlib import Path

import requests
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
REVIEW_QUEUE = ROOT / "reports/review_queue.jsonl"

st.set_page_config(page_title="Aurora Policy Assistant", page_icon="🛡️", layout="wide")

API = st.sidebar.text_input("Service URL", "http://localhost:8000")
mode = st.sidebar.selectbox("Mode", ["rag", "tools"], index=0,
                            help="rag: direct retrieval + answer. tools: agent with policy, customer & premium tools.")
stream_mode = st.sidebar.checkbox("Stream answer (SSE)", value=False)
top_k = st.sidebar.slider("Top K documents", min_value=1, max_value=15, value=5)

st.title("Aurora Policy Assistant")
st.caption("Answers come only from Aurora's policy documents. "
           "Every claim is cited. When the documents do not cover a question, "
           "the assistant says so instead of guessing.")

q = st.text_input("Ask a question",
                  placeholder="How long do I have to file a reimbursement claim?")

if st.button("Ask", type="primary") and q:
    with st.spinner("Processing request..."):
        try:
            if stream_mode:
                # SSE Streaming path
                r = requests.post(f"{API}/ask/stream",
                                  json={"question": q, "top_k": top_k, "mode": mode},
                                  stream=True, timeout=60)
                r.raise_for_status()
                data = {}
                placeholder = st.empty()
                accumulated = ""
                for line in r.iter_lines():
                    if line:
                        decoded = line.decode("utf-8")
                        if decoded.startswith("data: "):
                            raw_payload = decoded[6:]
                            try:
                                event_payload = json.loads(raw_payload)
                                if "token" in event_payload:
                                    accumulated += event_payload["token"]
                                    placeholder.markdown(accumulated)
                                elif "answer" in event_payload:
                                    data = event_payload
                            except Exception:
                                pass
            else:
                # Direct JSON path
                r = requests.post(f"{API}/ask",
                                  json={"question": q, "top_k": top_k, "mode": mode},
                                  timeout=60)
                r.raise_for_status()
                data = r.json()

        except requests.HTTPError as exc:
            st.error(f"HTTP {exc.response.status_code}: {exc.response.text[:300]}")
            st.stop()
        except requests.RequestException as exc:
            st.error(f"Service unreachable: {exc}")
            st.stop()

    if data.get("refused"):
        st.warning(data["answer"])
    else:
        st.markdown(data.get("answer", ""))

    # TODO A4: render citations as expanders showing the source excerpt.
    # Grounding the user cannot check is decoration.
    citations = data.get("citations", [])
    if citations:
        st.subheader("Sources & Grounding Excerpts")
        for c in citations:
            with st.expander(f"[{c['index']}] {c['doc_id']}"):
                st.text(c["excerpt"])

    cols = st.columns(4)
    cols[0].metric("latency", f"{data.get('latency_ms', 0):.0f} ms")
    cols[1].metric("cost", f"${data.get('cost_usd', 0):.5f}")
    cols[2].metric("cached", "yes" if data.get("cached") else "no")
    cols[3].metric("sources", len(citations))
    st.caption(f"trace: `{data.get('trace_id', '')}`")

    # Feedback review queue (Stretch Goal)
    st.write("---")
    fcols = st.columns([1, 1, 8])
    if fcols[0].button("👍 Helpful"):
        st.success("Feedback recorded. Thank you!")
    if fcols[1].button("👎 Review needed"):
        REVIEW_QUEUE.parent.mkdir(parents=True, exist_ok=True)
        with REVIEW_QUEUE.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({
                "question": q,
                "answer": data.get("answer"),
                "trace_id": data.get("trace_id"),
                "refused": data.get("refused"),
                "mode": mode,
            }) + "\n")
        st.warning("Case flagged and appended to evaluation review queue (`reports/review_queue.jsonl`).")
