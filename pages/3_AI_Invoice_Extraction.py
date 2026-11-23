import os

import pandas as pd
import streamlit as st

from app_utils import ROOT, hero, insight, kv, setup
from invoice_extractor import RuleBasedExtractor

setup("AI Invoice Extraction", "🧾")
hero("🧾 AI Invoice Extraction", "Turn supplier invoices into clean, checked data for accounting, with an LLM and business rules.",
     ["Claude API · structured output", "Pydantic validation", "Human review queue", "Docker · CI"])

SAMPLES = ROOT / "03-ai-invoice-extraction" / "samples"


def api_key() -> str | None:
    try:
        return st.secrets.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


key = api_key()
engines = ["Rules (offline)"] + (["Claude (LLM)"] if key else [])

left, right = st.columns([1, 1], gap="large")
with left:
    st.markdown("##### 1 · Choose an invoice")
    samples = {p.name: p.read_text(encoding="utf-8") for p in sorted(SAMPLES.glob("*.txt"))}
    choice = st.radio("Sample", list(samples) + ["Paste my own text"], horizontal=True, label_visibility="collapsed")
    text = st.text_area("Invoice text", samples.get(choice, ""), height=380,
                        placeholder="Paste the text of an invoice here…")
    engine = st.segmented_control("Extraction engine", engines, default=engines[-1])
    if not key:
        st.caption("Claude mode is available when an ANTHROPIC_API_KEY is configured in the app secrets. "
                   "The demo uses the offline rule-based engine, which follows the same schema and checks.")
    run = st.button("Extract data", type="primary", use_container_width=True)

with right:
    st.markdown("##### 2 · Extracted and checked data")
    if run and text.strip():
        try:
            if engine == "Claude (LLM)":
                os.environ["ANTHROPIC_API_KEY"] = key
                from invoice_extractor import ClaudeExtractor
                with st.spinner("Asking Claude…"):
                    inv = ClaudeExtractor().extract(text)
            else:
                inv = RuleBasedExtractor().extract(text)
        except Exception as exc:
            st.error(f"Extraction failed: {exc}")
            st.stop()

        if inv.needs_review:
            st.error("⚠️ To review: this invoice does not pass the consistency checks.")
        else:
            st.success("✅ OK: all consistency checks passed, ready for accounting.")

        cur = inv.currency
        a, b = st.columns(2)
        a.markdown(kv("Supplier", inv.supplier_name.replace(" (fictional)", "")), unsafe_allow_html=True)
        b.markdown(kv("Invoice number", inv.invoice_number), unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.markdown(kv("Total excl. tax", f"{inv.total_excl_tax:,.2f} {cur}"), unsafe_allow_html=True)
        b.markdown(kv("VAT", f"{inv.tax_amount:,.2f} {cur}"), unsafe_allow_html=True)
        c.markdown(kv("Total incl. tax", f"{inv.total_incl_tax:,.2f} {cur}"), unsafe_allow_html=True)
        a, b = st.columns(2)
        a.markdown(kv("Invoice date", f"{inv.invoice_date:%d %b %Y}"), unsafe_allow_html=True)
        b.markdown(kv("Due date", inv.due_date.strftime("%d %b %Y") if inv.due_date else "not found"),
                   unsafe_allow_html=True)

        if inv.line_items:
            st.dataframe(pd.DataFrame([li.model_dump() for li in inv.line_items]).rename(columns={
                "description": "Description", "quantity": "Qty", "unit_price": "Unit price", "total": "Line total"}),
                         hide_index=True, use_container_width=True)

        lines_total = round(sum(li.total for li in inv.line_items), 2)
        checks = [
            ("Excl. tax + VAT = incl. tax", abs(inv.total_excl_tax + inv.tax_amount - inv.total_incl_tax) <= 0.02),
            ("Sum of lines = total excl. tax", not inv.line_items or abs(lines_total - inv.total_excl_tax) <= 0.02),
            ("Due date after invoice date", not inv.due_date or inv.due_date >= inv.invoice_date),
        ]
        st.markdown("**Consistency checks**")
        for label, ok in checks:
            st.markdown(f"{'🟢' if ok else '🔴'} {label}")

        with st.expander("JSON sent to accounting"):
            st.json(inv.model_dump(mode="json"))
    else:
        st.info("Pick an invoice on the left and click **Extract data**. Try `invoice_003.txt`: it contains "
                "deliberate errors.")

st.divider()
insight("<b>Design principle: the LLM extracts, the rules decide.</b> Claude is forced to answer through a tool whose "
        "schema is the invoice model, so its output is always valid structured data. Deterministic checks then flag "
        "any inconsistency for a human instead of silently passing it to accounting.")
