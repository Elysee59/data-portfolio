import json
import os
import sys
import threading

import streamlit as st
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

from app_utils import ROOT, hero, insight, setup

sys.path.insert(0, str(ROOT / "04-hr-multi-agent" / "src"))
from hr_agents import SPECIALISTS, HRAgentTeam, run_offline  # noqa: E402

setup("HR Agent Team", "🤖")
hero("🤖 HR Agent Team", "Ask a question about the workforce. A coordinator agent briefs three specialist agents, "
     "they query the HR data in parallel, and the coordinator writes a memo for the HR director.",
     ["Multi-agent (Claude API)", "Tool use", "Parallel sub-agents", "Aggregates only · GDPR"])


def secret(name: str) -> str | None:
    try:
        return st.secrets.get(name) or os.environ.get(name)
    except Exception:
        return os.environ.get(name)


API_KEY, MODEL = secret("ANTHROPIC_API_KEY"), secret("ANTHROPIC_MODEL")
ICONS = {"turnover": "📉", "pay_equity": "⚖️", "retention_risk": "🎯"}
SUGGESTIONS = [
    "Give me a full HR health check for 2025.",
    "Is there a gender pay gap, and is it significant once we compare like with like?",
    "Which departments are most at risk of losing people, and why?",
    "Où perdons-nous le plus de collaborateurs et que faire en priorité ?",
]

left, right = st.columns([3, 2], gap="large")
with right:
    st.markdown("##### How the team works")
    st.graphviz_chart("""
digraph {
  rankdir=LR; bgcolor="transparent"; node [shape=box style="rounded,filled" fontname="Helvetica" fontsize=14 color="#94A3B8"];
  Q [label="Question\\n(HR director)" fillcolor="#F1F5F9"];
  C [label="Coordinator\\nHead of People Analytics" fillcolor="#DBEAFE"];
  T [label="Turnover &\\nAbsence" fillcolor="#FFEDD5"]; P [label="Pay Equity" fillcolor="#FFEDD5"];
  R [label="Retention\\nRisk" fillcolor="#FFEDD5"];
  D [label="HR data tools\\n(aggregates)" shape=cylinder fillcolor="#F1F5F9"];
  M [label="Memo" fillcolor="#DCFCE7"];
  Q -> C; C -> T; C -> P; C -> R; T -> D; P -> D; R -> D; C -> M;
}""", use_container_width=True)
    if API_KEY:
        st.success(f"Live mode: agents run on Claude ({MODEL or 'default model'}).")
    else:
        st.info("**Offline demo mode.** No API key is configured, so the specialists run their standard tool plan "
                "and the memo is assembled from a template. Same team, same tools, same trace, without an LLM.")

with left:
    st.markdown("##### Ask the team")
    if "question" not in st.session_state:
        st.session_state.question = SUGGESTIONS[0]
    cols = st.columns(2)
    for i, s in enumerate(SUGGESTIONS):
        if cols[i % 2].button(s, key=f"sugg{i}", use_container_width=True):
            st.session_state.question = s
    question = st.text_area("Question", key="question", height=90, label_visibility="collapsed")
    run = st.button("Run the team", type="primary", use_container_width=True)

st.divider()

if run and question.strip():
    st.markdown("#### Team at work")
    boxes: dict[str, any] = {}
    lock = threading.Lock()

    def on_event(e: dict) -> None:
        agent = e.get("agent")
        with lock:
            if e["type"] == "specialist_start":
                boxes[agent] = st.status(f"{ICONS.get(agent, '🔎')} {e['name']} specialist: working…", expanded=True)
                boxes[agent].markdown(f"**Assignment:** {e['task']}")
            elif e["type"] == "tool_call" and agent in boxes:
                args = ", ".join(f"{k}={v}" for k, v in (e["input"] or {}).items())
                boxes[agent].markdown(f"🛠️ `{e['tool']}({args})`")
            elif e["type"] == "tool_result" and agent in boxes:
                with boxes[agent].expander(f"Result of {e['tool']}"):
                    st.json(e["result"], expanded=False)
            elif e["type"] == "specialist_done" and agent in boxes:
                boxes[agent].markdown(f"**Report:**\n\n{e['report']}")
                boxes[agent].update(label=f"{ICONS.get(agent, '🔎')} {e['name']} specialist: done",
                                    state="complete", expanded=False)

    ctx = get_script_run_ctx()
    try:
        if API_KEY:
            os.environ["ANTHROPIC_API_KEY"] = API_KEY
            team = HRAgentTeam(model=MODEL, on_event=on_event,
                               thread_setup=lambda: add_script_run_ctx(threading.current_thread(), ctx))
            with st.spinner("The coordinator is briefing the team…"):
                result = team.run(question)
        else:
            result = run_offline(question, on_event=on_event)
    except Exception as exc:  # network, quota, invalid key…
        st.error(f"The team could not finish: {exc}")
        st.stop()

    st.markdown("#### 📝 Memo for the HR director")
    with st.container(border=True):
        st.markdown(result.memo)
    c1, c2 = st.columns([1, 2])
    c1.download_button("Download the memo (.md)", result.memo, file_name="hr_memo.md", use_container_width=True)
    if result.input_tokens:
        c2.caption(f"Tokens used: {result.input_tokens:,} in / {result.output_tokens:,} out · "
                   f"{sum(e['type'] == 'tool_call' for e in result.events)} tool calls")
    with st.expander("Full event trace (JSON)"):
        st.code(json.dumps(result.events, default=str, indent=1)[:20000], language="json")
else:
    st.caption("Pick a suggested question or write your own, then click **Run the team**.")

insight("<b>Design choices.</b> Each specialist is a separate agent with its own prompt, its own tools and its own "
        "context, so the coordinator only reads short reports. Specialists run in parallel. Tools return aggregates "
        "only and suppress groups under 10 people, so the agents cannot single out an employee. Every figure in the "
        "memo must come from a tool result.")
