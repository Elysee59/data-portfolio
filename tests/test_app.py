"""Smoke tests: every page of the Streamlit app renders without an exception."""
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
PAGES = ["Home.py", "pages/1_HR_Analytics.py", "pages/2_Demand_Forecasting.py", "pages/3_AI_Invoice_Extraction.py",
         "pages/4_HR_Agent_Team.py"]


@pytest.mark.parametrize("page", PAGES)
def test_page_renders(page):
    at = AppTest.from_file(str(ROOT / page), default_timeout=120).run()
    assert not at.exception, at.exception


def test_invoice_flags_inconsistent_sample():
    at = AppTest.from_file(str(ROOT / "pages/3_AI_Invoice_Extraction.py"), default_timeout=60).run()
    at.radio[0].set_value("invoice_003.txt").run()
    at.button[0].click().run()
    assert any("To review" in e.value for e in at.error)


def test_agent_team_offline_run():
    at = AppTest.from_file(str(ROOT / "pages/4_HR_Agent_Team.py"), default_timeout=120).run()
    run_btn = [b for b in at.button if b.label == "Run the team"][0]
    run_btn.click().run()
    assert not at.exception, at.exception
    assert any("Offline mode" in m.value for m in at.markdown)


def test_agent_team_live_mode_with_fake_claude(monkeypatch):
    """Live path (parallel specialists streaming into the UI) with a fake Anthropic SDK: no network, no cost."""
    import json
    import sys
    import types
    from types import SimpleNamespace as NS

    def resp(blocks, stop):
        return NS(content=blocks, stop_reason=stop, usage=NS(input_tokens=100, output_tokens=20))

    class FakeClient:
        def __init__(self, *a, **k):
            self.messages = NS(create=self.create)

        def create(self, *, system, tools, messages, **kw):
            last = messages[-1]["content"]
            if "Head of People Analytics for" in system:
                if isinstance(last, str):
                    return resp([NS(type="tool_use", id=f"d{i}", name="delegate",
                                    input={"specialist": k, "task": f"analyse {k}"})
                                 for i, k in enumerate(["turnover", "pay_equity", "retention_risk"])], "tool_use")
                return resp([NS(type="text", text="LIVE MEMO: " + " / ".join(r["content"] for r in last))], "end_turn")
            if isinstance(last, str):
                return resp([NS(type="tool_use", id="t1", name=tools[0]["name"], input={})], "tool_use")
            return resp([NS(type="text", text="figures: " + ", ".join(sorted(json.loads(last[0]["content"]))[:2]))],
                        "end_turn")

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=FakeClient))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    at = AppTest.from_file(str(ROOT / "pages/4_HR_Agent_Team.py"), default_timeout=120).run()
    [b for b in at.button if b.label == "Run the team"][0].click().run()
    assert not at.exception, at.exception
    assert any("LIVE MEMO" in m.value for m in at.markdown)
    assert len(at.status) == 3
    assert any("Cost of this answer" in m.value for m in at.markdown)
    # Same question again: served from the cache, no new API call
    [b for b in at.button if b.label == "Run the team"][0].click().run()
    assert not at.exception, at.exception
    assert any("Saved answer reused" in i.value for i in at.info)
